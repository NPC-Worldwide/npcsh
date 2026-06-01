#!/usr/bin/env python3
"""
extract_memories.py

Query the npcsh conversation history database and extract recurring patterns,
successful strategies, common mistakes, and novel insights. Outputs a structured
memory file that can be consumed by npcsh's knowledge graph or team context.

Usage:
    python scripts/extract_memories.py --since "6 hours" --model qwen3.5:4b
    python scripts/extract_memories.py --db ~/.npcsh/npcsh_history.db --since "24 hours" --output ~/.npcsh/memories.json
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def fetch_conversations(db_path: str, since: str, limit: int = 500):
    """Fetch recent conversation turns."""
    now = datetime.now()
    if since.endswith(" days") or since.endswith(" day"):
        num = int(since.split()[0])
        cutoff = now - timedelta(days=num)
    elif since.endswith(" hours") or since.endswith(" hour"):
        num = int(since.split()[0])
        cutoff = now - timedelta(hours=num)
    else:
        cutoff = now - timedelta(hours=6)

    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_path)
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
    conn.close()

    return [
        {
            "message_id": r[0],
            "timestamp": r[1],
            "role": r[2],
            "content": r[3],
            "model": r[4],
            "npc": r[5],
        }
        for r in rows
    ]


def fetch_jinx_executions(db_path: str, since: str):
    """Fetch jinx execution stats."""
    now = datetime.now()
    if since.endswith(" days") or since.endswith(" day"):
        num = int(since.split()[0])
        cutoff = now - timedelta(days=num)
    else:
        cutoff = now - timedelta(hours=6)

    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Discover actual columns since schema varies across installs
    cursor.execute("PRAGMA table_info(jinx_executions)")
    cols = {c[1] for c in cursor.fetchall()}
    if "jinx_name" not in cols:
        conn.close()
        return []

    group_cols = [c for c in ["jinx_name", "npc"] if c in cols]
    cursor.execute(
        f"""
        SELECT {', '.join(group_cols)}, COUNT(*) as count
        FROM jinx_executions
        WHERE timestamp > ?
        GROUP BY {', '.join(group_cols)}
        """,
        (cutoff_str,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        dict(zip(group_cols + ["count"], r))
        for r in rows
    ]


def summarize_conversations(conversations: list, executions: list) -> str:
    """Build a compact summary for the LLM."""
    # Group by session
    sessions = []
    current = []
    for c in sorted(conversations, key=lambda x: x["timestamp"]):
        current.append(c)
        if len(current) >= 10:
            sessions.append(current)
            current = []
    if current:
        sessions.append(current)

    lines = [f"Analyzed {len(conversations)} conversation turns across {len(sessions)} sessions."]

    # Tool usage summary
    tool_stats = {}
    for e in executions:
        name = e.get("jinx_name", "unknown")
        if name not in tool_stats:
            tool_stats[name] = {"total": 0}
        tool_stats[name]["total"] += e.get("count", 0)

    if tool_stats:
        lines.append("\nTool usage:")
        for name, stats in sorted(tool_stats.items(), key=lambda x: -x[1]["total"])[:10]:
            lines.append(f"  {name}: {stats['total']} calls")

    # Sample user intents
    user_msgs = [c for c in conversations if c["role"] == "user"]
    lines.append(f"\nSample user requests ({min(10, len(user_msgs))} shown):")
    for m in user_msgs[:10]:
        text = m["content"].replace("\n", " ")[:120]
        lines.append(f"  - {text}")

    return "\n".join(lines)


def extract_memories_with_llm(summary: str, model: str = "qwen3.5:4b"):
    """Use LLM to extract structured memories."""
    prompt = f"""You are a memory extraction system for an AI agent toolkit called npcsh.

Analyze the following activity summary and extract structured memories. Return JSON with these keys:
- patterns: list of recurring user behavior patterns (e.g. "user often asks for X after Y")
- preferences: list of inferred user preferences (e.g. "prefers concise outputs", "likes matplotlib over seaborn")
- strategies: list of successful problem-solving approaches observed
- mistakes: list of common errors or failures and their likely causes
- gaps: list of capabilities the user seems to need but currently lack
- tasks: list of 3-5 concrete benchmark task ideas derived from the patterns

Activity summary:
{summary}

Return ONLY valid JSON."""

    try:
        from npcpy.llm_funcs import get_llm_response
        response = get_llm_response(prompt, model=model, provider="ollama", format="json")
        text = response.get("response", "")
        # Extract JSON block
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
    except Exception as e:
        print(f"LLM extraction failed: {e}")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Extract memories from npcsh conversations")
    parser.add_argument("--db", default="~/.npcsh/npcsh_history.db")
    parser.add_argument("--since", default="6 hours")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--output", default="~/.npcsh/extracted_memories.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = os.path.expanduser(args.db)
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    print(f"Fetching data from {db_path} (since: {args.since})...")
    conversations = fetch_conversations(db_path, args.since, args.limit)
    executions = fetch_jinx_executions(db_path, args.since)

    print(f"Found {len(conversations)} turns, {len(executions)} jinx execution groups")
    if len(conversations) < 5:
        print("Not enough data to extract meaningful memories.")
        return

    summary = summarize_conversations(conversations, executions)
    print("\nExtracting memories with LLM...")
    memories = extract_memories_with_llm(summary, args.model)

    if args.dry_run:
        print(json.dumps(memories, indent=2))
        return

    output_path = Path(os.path.expanduser(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(memories, indent=2))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
