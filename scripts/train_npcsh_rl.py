#!/usr/bin/env python3
"""
train_npcsh_rl.py

Active RL training loop for npcsh.

1. Collects traces by running benchmark tasks through npcsh
2. Extracts clean (instruction, response) pairs from traces
3. Trains with DPO, GRPO, or PPO via npcpy.ft.rl

Usage:
    # Collect fresh traces and train DPO
    python scripts/train_npcsh_rl.py dpo --model mlx-community/Qwen3-4B-4bit --provider omlx

    # Use existing benchmark CSVs, train GRPO on hard tasks only
    python scripts/train_npcsh_rl.py grpo --model mlx-community/Qwen3-4B-4bit --csv-dir ~/.npcsh/benchmarks/local --hard-only

    # Active loop: evaluate current adapter, collect traces, RL train, repeat
    python scripts/train_npcsh_rl.py grpo --model mlx-community/Qwen3-4B-4bit --active-loop --iterations 3
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_from_csv import parse_trace, _compute_task_difficulty


def load_tasks(csv_path: str = None, category: str = None, difficulty: str = None, task_id: str = None):
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "npcsh" / "benchmark" / "tasks.csv"
    tasks = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if task_id and row["id"] != task_id:
                continue
            if category and row["category"] != category:
                continue
            if difficulty and row["difficulty"] != difficulty:
                continue
            tasks.append(row)
    return tasks


def run_npcsh_task(task: dict, model: str, provider: str, work_dir: str) -> dict:
    """Run a single task through npcsh and return a trace dict."""
    instruction = task["instruction"]
    verify_cmd = task["verify_cmd"]
    setup_cmd = task.get("setup_cmd", "") or ""

    if setup_cmd:
        subprocess.run(["bash", "-c", setup_cmd], capture_output=True, cwd=work_dir)

    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    env["NPCSH_STREAM_OUTPUT"] = "0"

    start = time.time()
    try:
        proc = subprocess.run(
            ["npcsh", "-c", instruction],
            capture_output=True,
            text=True,
            cwd=work_dir,
            env=env,
            timeout=120,
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        output = "TIMEOUT"

    duration = time.time() - start

    try:
        verify = subprocess.run(
            ["bash", "-c", verify_cmd],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=work_dir,
        )
        passed = verify.returncode == 0
    except Exception:
        passed = False

    # Extract clean instruction/response from full output
    parsed = parse_trace(output) or {}
    clean_instruction = parsed.get("instruction", instruction)
    clean_response = parsed.get("response", "")
    # Fallback: if parse_trace couldn't extract, use raw output as response
    if not clean_response:
        clean_response = output

    return {
        "task_id": task["id"],
        "instruction": instruction,
        "output": output,
        "passed": passed,
        "duration": duration,
        "category": task["category"],
        "difficulty": task["difficulty"],
        "task_prompt": clean_instruction,
        "final_output": clean_response,
    }


def collect_traces(tasks: list, model: str, provider: str, attempts: int = 3, keep_all: bool = True) -> list:
    """Run tasks through npcsh, return trace dicts.

    If keep_all=True, every attempt is kept (better for DPO/GRPO).
    If keep_all=False, only the best trace per task is kept.
    """
    traces = []
    for task in tasks:
        print(f"\nTask: {task['id']} ({task['category']}/{task['difficulty']})")
        task_traces = []
        for attempt in range(attempts):
            work_dir = tempfile.mkdtemp(prefix=f"npcsh_rl_{task['id']}_")
            trace = run_npcsh_task(task, model, provider, work_dir)

            reward = 1.0 if trace["passed"] else -0.5
            reward += 0.15 * (attempts - attempt - 1) / attempts
            trace["reward"] = reward
            trace["attempt"] = attempt + 1
            trace["total_iterations"] = attempt + 1
            trace["completed"] = trace["passed"]

            status = "PASS" if trace["passed"] else "FAIL"
            print(f"  attempt {attempt + 1}: {status} reward={reward:.2f} ({trace['duration']:.1f}s)")

            task_traces.append(trace)
            subprocess.run(["rm", "-rf", work_dir], capture_output=True)

        if keep_all:
            traces.extend(task_traces)
        else:
            best = max(task_traces, key=lambda t: t["reward"])
            traces.append(best)

    return traces


def load_csv_traces(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False) -> list:
    """Load traces from existing benchmark CSVs."""
    csv.field_size_limit(10**7)
    task_rates = _compute_task_difficulty(csv_dir, pattern)
    traces = []
    for csv_file in sorted(Path(csv_dir).glob(pattern)):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row["task_id"]
                if hard_only and task_rates.get(tid, 0.5) >= 0.5:
                    continue
                trace = parse_trace(row.get("output", ""))
                if not trace:
                    continue
                passed = row.get("passed", "").lower() == "true"
                attempts = int(row.get("attempts", "1") or 1)
                reward = 1.0 if passed else -0.5
                reward += max(0, 0.15 * (3 - attempts) / 3)
                traces.append({
                    "task_id": tid,
                    "instruction": row.get("instruction", ""),
                    "output": row.get("output", ""),
                    "passed": passed,
                    "duration": float(row.get("duration", "0") or 0),
                    "category": row.get("category", ""),
                    "difficulty": row.get("difficulty", ""),
                    "task_prompt": trace["instruction"],
                    "final_output": trace["response"],
                    "reward": reward,
                    "attempt": attempts,
                    "total_iterations": attempts,
                    "completed": passed,
                })
    print(f"Loaded {len(traces)} traces from {csv_dir} ({pattern})")
    return traces


def train_dpo(traces: list, args):
    from npcpy.ft.rl import RLConfig, train_with_dpo
    config = RLConfig(
        base_model_name=args.model,
        adapter_path=args.output,
        device=args.device,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        beta=args.beta,
        max_pairs=args.max_pairs,
        logging_steps=max(1, len(traces) // 20),
        save_steps=max(1, len(traces) // 5),
    )
    adapter = train_with_dpo(traces, config=config)
    return adapter


def train_grpo(traces: list, args):
    from npcpy.ft.rl import RLConfig, train_with_grpo

    # Group traces by task_id for GRPO
    by_task = {}
    for t in traces:
        by_task.setdefault(t["task_id"], []).append(t)

    groups = []
    for tid, task_traces in by_task.items():
        if len(task_traces) < 2:
            continue
        prompt = task_traces[0]["task_prompt"]
        responses = [(t["final_output"], t["reward"]) for t in task_traces]
        groups.append({"prompt": prompt, "responses": responses})

    print(f"GRPO: {len(groups)} groups from {len(by_task)} tasks")
    if not groups:
        print("Need tasks with multiple traces for GRPO.")
        return None

    config = RLConfig(
        base_model_name=args.model,
        adapter_path=args.output,
        device=args.device,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        group_size=min(args.group_size, min(len(g["responses"]) for g in groups)),
        max_length=1024,
        logging_steps=max(1, len(groups) // 10),
        save_steps=max(1, len(groups) // 3),
    )
    adapter = train_with_grpo(groups, config)
    return adapter


def train_ppo(traces: list, args):
    from npcpy.ft.rl import RLConfig, train_with_ppo

    config = RLConfig(
        base_model_name=args.model,
        adapter_path=args.output,
        device=args.device,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        beta=args.beta,
        clip_eps=args.clip_eps,
        group_size=args.group_size,
        max_length=1024,
        logging_steps=max(1, len(traces) // 10),
        save_steps=max(1, len(traces) // 3),
    )
    adapter = train_with_ppo(traces, config)
    return adapter


def run_training(traces: list, args):
    method = args.method
    print(f"\nTraining with {method.upper()}...")

    if method == "dpo":
        adapter = train_dpo(traces, args)
    elif method == "grpo":
        adapter = train_grpo(traces, args)
    elif method == "ppo":
        adapter = train_ppo(traces, args)
    else:
        print(f"Unknown method: {method}")
        return None

    if adapter:
        print(f"\nAdapter saved to: {adapter}")
        meta = {
            "base_model": args.model,
            "adapter_path": str(adapter),
            "num_traces": len(traces),
            "passed_baseline": sum(1 for t in traces if t["passed"]),
            "total_tasks": len({t["task_id"] for t in traces}),
            "method": method,
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "beta": args.beta,
            },
        }
        meta_path = Path(args.output) / "rl_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
    return adapter


def active_loop(args):
    """Run collect → train → evaluate → repeat."""
    for iteration in range(args.iterations):
        print(f"\n{'=' * 60}")
        print(f"ACTIVE LOOP ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'=' * 60}")

        tasks = load_tasks(category=args.category, difficulty=args.difficulty, task_id=args.task_id)
        if not tasks:
            print("No tasks matched.")
            break

        print(f"Collecting traces with {args.model}/{args.provider}...")
        traces = collect_traces(tasks, args.model, args.provider, attempts=args.attempts, keep_all=True)
        passed = sum(1 for t in traces if t["passed"])
        print(f"Baseline: {passed}/{len(traces)} traces passed")

        # Merge with any existing CSV traces
        if args.csv_dir:
            csv_traces = load_csv_traces(os.path.expanduser(args.csv_dir), hard_only=args.hard_only)
            traces.extend(csv_traces)
            print(f"Merged {len(csv_traces)} CSV traces. Total: {len(traces)}")

        # Save traces
        ts = time.strftime("%Y%m%d_%H%M%S")
        trace_file = Path(args.output) / f"rl_traces_{ts}_iter{iteration}.json"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_file, "w") as f:
            json.dump(traces, f, indent=2)
        print(f"Traces saved to {trace_file}")

        # Train
        adapter = run_training(traces, args)
        if not adapter:
            print("Training failed, stopping loop.")
            break

        # Evaluate: run a quick benchmark subset
        if args.eval_on_loop:
            print("\nEvaluating on first 10 tasks...")
            eval_tasks = tasks[:10]
            eval_traces = collect_traces(eval_tasks, args.model, args.provider, attempts=1, keep_all=False)
            eval_passed = sum(1 for t in eval_traces if t["passed"])
            print(f"Eval: {eval_passed}/{len(eval_traces)} passed")

    print("\nActive loop complete.")


def main():
    parser = argparse.ArgumentParser(description="Train npcsh with RL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--output", default="models/npcsh_rl")
    common.add_argument("--device", default="mlx", choices=["mlx", "cuda", "cpu"])
    common.add_argument("--category", help="Filter tasks by category")
    common.add_argument("--difficulty", help="Filter tasks by difficulty")
    common.add_argument("--task-id", help="Single task for debugging")
    common.add_argument("--epochs", type=int, default=5)
    common.add_argument("--lr", type=float, default=1e-5)
    common.add_argument("--lora-r", type=int, default=16)
    common.add_argument("--beta", type=float, default=0.1)
    common.add_argument("--max-pairs", type=int, default=500)
    common.add_argument("--provider", default="ollama")
    common.add_argument("--attempts", type=int, default=3)
    common.add_argument("--csv-dir", help="Also load traces from existing benchmark CSVs")
    common.add_argument("--hard-only", action="store_true", help="Only use hard tasks from CSVs")

    dpo_p = sub.add_parser("dpo", parents=[common])
    dpo_p.add_argument("--active-loop", action="store_true")
    dpo_p.add_argument("--iterations", type=int, default=3)
    dpo_p.add_argument("--eval-on-loop", action="store_true")

    grpo_p = sub.add_parser("grpo", parents=[common])
    grpo_p.add_argument("--group-size", type=int, default=4)
    grpo_p.add_argument("--active-loop", action="store_true")
    grpo_p.add_argument("--iterations", type=int, default=3)
    grpo_p.add_argument("--eval-on-loop", action="store_true")

    ppo_p = sub.add_parser("ppo", parents=[common])
    ppo_p.add_argument("--group-size", type=int, default=4)
    ppo_p.add_argument("--clip-eps", type=float, default=0.2)
    ppo_p.add_argument("--active-loop", action="store_true")
    ppo_p.add_argument("--iterations", type=int, default=3)
    ppo_p.add_argument("--eval-on-loop", action="store_true")

    args = parser.parse_args()

    if args.cmd in ("dpo", "grpo", "ppo"):
        args.method = args.cmd

    if getattr(args, "active_loop", False):
        active_loop(args)
        return

    # One-shot training
    traces = []
    if args.csv_dir:
        traces.extend(load_csv_traces(os.path.expanduser(args.csv_dir), hard_only=args.hard_only))

    if not args.task_id and not args.category and not args.difficulty and traces:
        # CSV-only mode
        print(f"Training from {len(traces)} CSV traces")
    else:
        tasks = load_tasks(category=args.category, difficulty=args.difficulty, task_id=args.task_id)
        if not tasks:
            print("No tasks matched.")
            sys.exit(1)
        print(f"Collecting traces from {len(tasks)} tasks...")
        fresh_traces = collect_traces(tasks, args.model, args.provider, attempts=args.attempts, keep_all=True)
        traces.extend(fresh_traces)
        passed = sum(1 for t in fresh_traces if t["passed"])
        print(f"Baseline: {passed}/{len(fresh_traces)} traces passed")

    if len(traces) < 5:
        print("Need >= 5 traces.")
        sys.exit(1)

    run_training(traces, args)


if __name__ == "__main__":
    main()
