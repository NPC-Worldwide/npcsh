#!/usr/bin/env python3
"""
generate_tasks.py

Rule-based benchmark task generator from conversation history.
Extracts high-quality user instructions and converts them into proper
benchmark task definitions with setup_cmd, verify_cmd, etc.

Usage:
    python scripts/generate_tasks.py --max-tasks 15 --append-csv npcsh/benchmark/tasks.csv
"""

import argparse
import csv
import os
import re
import sqlite3
from collections import defaultdict
from difflib import SequenceMatcher


def normalize(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def load_existing_tasks(path: str):
    tasks = []
    instructions = set()
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
            instructions.add(normalize(row["instruction"])[:40])
    return tasks, instructions


def extract_candidates(db_path: str, limit: int = 3000):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT content FROM conversation_history
        WHERE role = 'user'
        AND length(content) BETWEEN 25 AND 350
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (limit,),
    )
    texts = [row[0].strip() for row in cursor.fetchall()]
    conn.close()
    return texts


def is_good_task(text: str):
    t = text.lower()
    if len(text) < 25 or len(text) > 350:
        return False
    skip = [
        "/Users/caug/", "~/.npcsh", "File:", "%File:", "---TRACE---", "/tmp/.tmp",
        "gemma", "qwen", "claude", "npcshrc", "jinx", "/sample", "/search",
        "/delegate", "/convene", "/npcpy", "fucking", "retard", "shit",
        "why ", "how ", "what is", "what are", "what does", "explain",
        "tell me", "show me", "please", "thank", "sorry", "stop ", "dont",
        "no i", "yes ", "okay ", "ok ", "lol", "haha", "wait.", "wait,",
        "hold on", "never mind", "nvm", "actually", "maybe", "probably",
        "definitely", "chapter", "meeting", "seo ", "style", "paragraph",
        "essay", "criticize", "undo", "commit", "history", "branch", "push",
        "mobile", "phone", "app for me", "install it", "gcloud", "container",
        "docker", "kubernetes", "aws", "azure", "gcp",
    ]
    if any(p in t for p in skip):
        return False
    if "/tmp/" not in text:
        return False
    verbs = [
        "create", "write", "generate", "make", "build", "list", "count",
        "find", "search", "convert", "sort", "filter", "compute", "calculate",
        "extract", "parse", "summarize", "compare", "analyze", "install",
        "configure", "download", "fetch", "run", "execute", "test",
        "validate", "check", "plot", "draw", "show", "display", "add ",
        "remove ", "delete ", "update ", "modify ",
    ]
    first = " ".join(t.split()[:6])
    return any(v in first for v in verbs)


def categorize(text: str):
    t = text.lower()
    if "buggy" in t or ("fix" in t.split()[:3] and ".py" in t):
        return "debug"
    if "csv" in t or "json" in t or ("data" in t and any(x in t for x in ["read", "parse", "merge", "join", "aggregate"])):
        return "data"
    if ".py" in t and any(x in t for x in ["write", "create", "make", "implement"]):
        return "python"
    if any(w in t for w in ["sort", "filter", "extract", "parse", "count", "find ", "replace", "search ", "regex"]):
        return "text"
    if any(w in t for w in ["users", "processes", "disk", "hostname", "kernel", "cpu", "memory", "free space", "system", "proc"]):
        return "system"
    if any(w in t for w in ["directory", "mkdir", "touch", "copy", "move", "rename", "tree", "recursive"]):
        return "file-ops"
    if any(w in t for w in ["bash", "shell", "command", "grep", "awk", "sed", "find ", "chmod", "chown"]):
        return "shell"
    if any(w in t for w in ["git ", "repo", "commit", "branch", "merge", "clone", "push", "pull"]):
        return "git"
    return "text"


def difficulty(text: str, category: str):
    t = text.lower()
    hard = ["recursive", "regex", "merge", "sort by", "complex", "advanced", "multiple", "nested", "implement"]
    if any(w in t for w in hard):
        return "hard"
    medium = ["script", "function", "then write", "and then", "calculate", "compute", "transform"]
    if any(w in t for w in medium):
        return "medium"
    if category in ["debug", "python", "git"]:
        return "medium"
    return "easy"


def build_verify_cmd(text: str, category: str):
    tmp_files = re.findall(r"/tmp/[a-zA-Z0-9_.]+", text)
    if not tmp_files:
        return ""
    main = tmp_files[0]
    verifications = [f"test -f {main}"]
    if ".py" in main:
        verifications.append(f"python3 {main}")
    elif ".sh" in main:
        verifications.append(f"bash {main}")
    elif ".txt" in main or ".csv" in main or ".json" in main or ".md" in main or ".html" in main:
        verifications.append(f"test -s {main}")
    return " && ".join(verifications)


def build_setup_cmd(text: str, category: str):
    t = text.lower()
    if "buggy" in t:
        return ""
    return ""


def build_description(text: str):
    desc = text.split(".")[0]
    if len(desc) > 80:
        desc = desc[:77] + "..."
    return desc


def generate_tasks(candidates, existing_instructions, max_tasks: int = 15):
    seen = set()
    tasks = []
    cat_counts = defaultdict(int)

    for instr in existing_instructions:
        pass

    for text in candidates:
        key = normalize(text)[:40]
        if key in seen:
            continue
        dup = False
        for ex in existing_instructions:
            if similarity(text, ex) > 0.7:
                dup = True
                break
        if dup:
            continue
        seen.add(key)

        cat = categorize(text)
        cat_counts[cat] += 1
        task_id = f"{cat}-{cat_counts[cat] + 100:02d}"

        task = {
            "id": task_id,
            "category": cat,
            "difficulty": difficulty(text, cat),
            "setup_cmd": build_setup_cmd(text, cat),
            "instruction": text,
            "verify_cmd": build_verify_cmd(text, cat),
            "description": build_description(text),
        }
        if task["verify_cmd"]:
            tasks.append(task)
        if len(tasks) >= max_tasks:
            break
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.path.expanduser("~/npcsh_history.db"))
    parser.add_argument("--tasks-csv", default="/Users/caug/npcww/npc-core/npcsh/npcsh/benchmark/tasks.csv")
    parser.add_argument("--max-tasks", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    existing_tasks, existing_instructions = load_existing_tasks(args.tasks_csv)
    print(f"Loaded {len(existing_tasks)} existing tasks")

    candidates = extract_candidates(args.db, limit=3000)
    filtered = [t for t in candidates if is_good_task(t)]
    print(f"Found {len(filtered)} candidate tasks")

    new_tasks = generate_tasks(filtered, existing_instructions, args.max_tasks)
    print(f"Generated {len(new_tasks)} new task definitions")

    for t in new_tasks:
        print(f"\n{t['id']} ({t['category']}/{t['difficulty']})")
        print(f"  {t['instruction'][:90]}...")
        print(f"  verify: {t['verify_cmd']}")

    if not args.dry_run and new_tasks:
        with open(args.tasks_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "category", "difficulty", "setup_cmd", "instruction", "verify_cmd", "description"])
            for task in new_tasks:
                writer.writerow(task)
        print(f"\nAppended {len(new_tasks)} tasks to {args.tasks_csv}")
    elif args.dry_run:
        print("\nDry run - no changes written")


if __name__ == "__main__":
    main()
