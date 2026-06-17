#!/usr/bin/env python3
"""
extract_tasks.py

Query the npcsh conversation history database and use an LLM to generalize observed
interaction patterns into new benchmark tasks. Outputs tasks in the standard CSV format
so they can be appended to npcsh/benchmark/tasks.csv.

Usage:
    python scripts/extract_tasks.py --since "7 days" --limit 50 --output tasks_new.json
    python scripts/extract_tasks.py --db ~/.npcsh/npcsh_history.db --since "24 hours" --dry-run
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path


def get_db_connection(db_path: str):
    return sqlite3.connect(db_path)


def fetch_recent_conversations(conn, since: str, limit: int):
    """Fetch recent conversation turns from the database."""
    # Parse since into a timestamp
    now = datetime.now()
    if since.endswith(" days") or since.endswith(" day"):
        num = int(since.split()[0])
        cutoff = now - timedelta(days=num)
    elif since.endswith(" hours") or since.endswith(" hour"):
        num = int(since.split()[0])
        cutoff = now - timedelta(hours=num)
    else:
        cutoff = now - timedelta(days=7)

    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT message_id, timestamp, role, content, model, npc
        FROM conversation_history
        WHERE timestamp > ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (cutoff_str, limit),
    )
    rows = cursor.fetchall()

    conversations = []
    for row in rows:
        conversations.append(
            {
                "message_id": row[0],
                "timestamp": row[1],
                "role": row[2],
                "content": row[3],
                "model": row[4],
                "npc": row[5],
            }
        )
    return conversations


def fetch_jinx_executions(conn, since: str):
    """Fetch recent jinx executions to see what tools were used."""
    now = datetime.now()
    if since.endswith(" days") or since.endswith(" day"):
        num = int(since.split()[0])
        cutoff = now - timedelta(days=num)
    else:
        cutoff = now - timedelta(days=7)

    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    cursor = conn.cursor()
    # Discover actual columns since schema varies across installs
    cursor.execute("PRAGMA table_info(jinx_executions)")
    cols = {c[1] for c in cursor.fetchall()}
    select_cols = [c for c in ["jinx_name", "input", "timestamp", "npc"] if c in cols]
    if not select_cols:
        return []

    cursor.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM jinx_executions
        WHERE timestamp > ?
        ORDER BY timestamp DESC
        """,
        (cutoff_str,),
    )
    rows = cursor.fetchall()

    executions = []
    for row in rows:
        rec = dict(zip(select_cols, row))
        text = rec.get("input", "")
        rec["input"] = text if len(text) < 500 else text[:500] + "..."
        executions.append(rec)
    return executions


def summarize_for_task_generation(conversations: list, executions: list) -> str:
    """Build a prompt-friendly summary of recent activity."""
    lines = ["Recent npcsh activity summary:"]

    # Group by approximate session (every 10 min gap)
    sessions = []
    current = []
    last_ts = None
    for c in sorted(conversations, key=lambda x: x["timestamp"]):
        ts = datetime.strptime(c["timestamp"], "%Y-%m-%d %H:%M:%S")
        if last_ts and (ts - last_ts).total_seconds() > 600:
            sessions.append(current)
            current = []
        current.append(c)
        last_ts = ts
    if current:
        sessions.append(current)

    for i, session in enumerate(sessions[-10:]):
        user_msgs = [s for s in session if s["role"] == "user"]
        [s for s in session if s["role"] == "assistant"]
        if user_msgs:
            first_user = user_msgs[0]["content"][:200]
            lines.append(f"\nSession {i + 1}: {first_user}")

    if executions:
        lines.append("\nRecent tool usage:")
        tool_counts = {}
        for e in executions:
            tool_counts[e["jinx_name"]] = tool_counts.get(e["jinx_name"], 0) + 1
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {tool}: {count} times")

    return "\n".join(lines)


def generate_tasks_with_llm(
    summary: str, num_tasks: int = 5, model: str = "qwen3.5:4b"
) -> list:
    """Use an LLM to generate benchmark tasks from the activity summary."""
    prompt = f"""You are a benchmark task designer for an AI agent shell toolkit called npcsh.

Below is a summary of real user activity. Your job is to generalize the observed patterns into new, concrete benchmark tasks that test an AI agent's ability to use shell commands, write code, manipulate files, and reason about system tasks.

Each task must include:
- id: short kebab-case ID (e.g. "shell-new-01")
- category: one of [shell, file-ops, python, data, system, text, debug, git, multi-step, scripting]
- difficulty: one of [easy, medium, hard]
- setup_cmd: bash command to prepare the environment (or empty)
- instruction: what the agent should do
- verify_cmd: bash command that returns 0 if correct, non-zero if wrong
- description: one-line human description

The tasks should be NEW and different from standard cookbook examples. Draw from real patterns in the user's work. Make them executable in /tmp. Keep instructions clear and unambiguous.

Activity summary:
{summary}

Generate exactly {num_tasks} tasks as a JSON array."""

    try:
        from npcpy.llm_funcs import get_llm_response

        response = get_llm_response(
            prompt, model=model, provider="ollama", format="json"
        )
        text = response.get("response", "")
        # Try to extract JSON array
        if "[" in text and "]" in text:
            start = text.index("[")
            end = text.rindex("]") + 1
            tasks = json.loads(text[start:end])
            return tasks
        return []
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return []


def validate_task(task: dict) -> bool:
    """Basic validation that a task has required fields."""
    required = [
        "id",
        "category",
        "difficulty",
        "instruction",
        "verify_cmd",
        "description",
    ]
    for field in required:
        if field not in task or not task[field]:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract benchmark tasks from npcsh conversations"
    )
    parser.add_argument(
        "--db", default="~/.npcsh/npcsh_history.db", help="Path to npcsh history DB"
    )
    parser.add_argument(
        "--since",
        default="7 days",
        help="How far back to look (e.g. '7 days', '24 hours')",
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Max conversation turns to fetch"
    )
    parser.add_argument(
        "--num-tasks", type=int, default=5, help="Number of new tasks to generate"
    )
    parser.add_argument("--model", default="qwen3.5:4b", help="LLM for task generation")
    parser.add_argument("--output", help="Output JSON file for new tasks")
    parser.add_argument("--append-csv", help="Append validated tasks to this CSV file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print tasks instead of saving"
    )
    args = parser.parse_args()

    db_path = os.path.expanduser(args.db)
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print(f"Fetching conversations from {db_path} (since: {args.since})...")
    conn = get_db_connection(db_path)
    conversations = fetch_recent_conversations(conn, args.since, args.limit)
    executions = fetch_jinx_executions(conn, args.since)
    conn.close()

    print(
        f"Found {len(conversations)} conversation turns, {len(executions)} jinx executions"
    )

    if len(conversations) < 5:
        print("Not enough conversation data to generate meaningful tasks.")
        sys.exit(0)

    summary = summarize_for_task_generation(conversations, executions)
    print("\nGenerating tasks with LLM...")
    tasks = generate_tasks_with_llm(summary, args.num_tasks, args.model)

    valid_tasks = [t for t in tasks if validate_task(t)]
    print(f"Generated {len(tasks)} tasks, {len(valid_tasks)} valid")

    if args.dry_run:
        print(json.dumps(valid_tasks, indent=2))
        return

    if args.output:
        with open(args.output, "w") as f:
            json.dump(valid_tasks, f, indent=2)
        print(f"Saved to {args.output}")

    if args.append_csv:
        import csv as csv_mod

        csv_path = Path(args.append_csv)
        fieldnames = [
            "id",
            "category",
            "difficulty",
            "setup_cmd",
            "instruction",
            "verify_cmd",
            "description",
        ]
        exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            for task in valid_tasks:
                row = {k: task.get(k, "") for k in fieldnames}
                writer.writerow(row)
        print(f"Appended {len(valid_tasks)} tasks to {csv_path}")


if __name__ == "__main__":
    main()
