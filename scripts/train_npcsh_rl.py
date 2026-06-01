#!/usr/bin/env python3
"""
train_npcsh_rl.py

Train an npcsh model with RL (DPO via npcpy.ft.rl) using the benchmark suite as the environment.

The agent is given benchmark tasks. Reward = +1 for pass, -0.5 for fail, +0.2 for fewer attempts.
Traces are collected, preference pairs built, and DPO training runs via MLX or torch.

Usage:
    python scripts/train_npcsh_rl.py --model mlx-community/Qwen3-0.6B-4bit --output models/npcsh_rl --device mlx --category shell
    python scripts/train_npcsh_rl.py --model mlx-community/Qwen3-1.7B-4bit --output models/npcsh_rl --device mlx --task-id shell-01
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def load_tasks(csv_path: str = None, category: str = None, difficulty: str = None, task_id: str = None):
    """Load benchmark tasks from the npcsh tasks.csv."""
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


def verify_task(task: dict, work_dir: str) -> dict:
    """Run a task through npcsh and verify the result. Returns trace dict."""
    instruction = task["instruction"]
    verify_cmd = task["verify_cmd"]
    setup_cmd = task.get("setup_cmd", "") or ""

    # Run setup
    if setup_cmd:
        subprocess.run(["bash", "-c", setup_cmd], capture_output=True, cwd=work_dir)

    # Build npcsh command
    env = os.environ.copy()
    model = env.get("NPCSH_RL_MODEL", "qwen3.5:0.8b")
    provider = env.get("NPCSH_RL_PROVIDER", "ollama")

    cmd = ["npcsh", "-c", instruction]
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    env["NPCSH_STREAM_OUTPUT"] = "0"

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
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

    # Verify
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

    return {
        "task_id": task["id"],
        "instruction": instruction,
        "output": output,
        "passed": passed,
        "duration": duration,
        "category": task["category"],
        "difficulty": task["difficulty"],
    }


def collect_npcsh_traces(tasks: list, model: str, provider: str, attempts_per_task: int = 3) -> list:
    """Run each task N times through npcsh, return traces with rewards."""
    traces = []
    os.environ["NPCSH_RL_MODEL"] = model
    os.environ["NPCSH_RL_PROVIDER"] = provider

    for task in tasks:
        print(f"\nTask: {task['id']} ({task['category']}/{task['difficulty']})")
        best_trace = None
        best_reward = -999

        for attempt in range(attempts_per_task):
            work_dir = tempfile.mkdtemp(prefix=f"npcsh_rl_{task['id']}_")
            trace = verify_task(task, work_dir)

            # Reward: +1 pass, -0.5 fail, +0.2 per fewer attempt index
            reward = 1.0 if trace["passed"] else -0.5
            reward += 0.2 * (attempts_per_task - attempt - 1) / attempts_per_task
            trace["reward"] = reward
            trace["attempt"] = attempt + 1

            status = "PASS" if trace["passed"] else "FAIL"
            print(f"  attempt {attempt + 1}: {status} reward={reward:.2f} ({trace['duration']:.1f}s)")

            if reward > best_reward:
                best_reward = reward
                best_trace = trace

            # Cleanup
            subprocess.run(["rm", "-rf", work_dir], capture_output=True)

        if best_trace:
            traces.append(best_trace)

    return traces


def npcsh_reward_fn(trace: dict) -> float:
    """Reward function for npcpy.ft.rl."""
    return trace.get("reward", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train npcsh with RL on benchmark tasks")
    parser.add_argument("--model", required=True, help="Base model (e.g. mlx-community/Qwen3-0.6B-4bit)")
    parser.add_argument("--output", default="models/npcsh_rl", help="Output adapter path")
    parser.add_argument("--device", default="mlx", choices=["mlx", "cuda", "cpu"])
    parser.add_argument("--category", help="Filter tasks by category")
    parser.add_argument("--difficulty", help="Filter tasks by difficulty")
    parser.add_argument("--task-id", help="Train on single task (for debugging)")
    parser.add_argument("--attempts", type=int, default=3, help="Attempts per task for trace collection")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--provider", default="ollama", help="Provider for npcsh execution during trace collection")
    parser.add_argument("--eval-only", action="store_true", help="Just evaluate, don't train")
    args = parser.parse_args()

    tasks = load_tasks(category=args.category, difficulty=args.difficulty, task_id=args.task_id)
    if not tasks:
        print("No tasks matched the filter.")
        sys.exit(1)

    print(f"Loaded {len(tasks)} tasks")
    print(f"Collecting traces with model={args.model} provider={args.provider}...")

    traces = collect_npcsh_traces(tasks, args.model, args.provider, attempts_per_task=args.attempts)

    passed = sum(1 for t in traces if t["passed"])
    print(f"\nBaseline: {passed}/{len(traces)} tasks passed")

    if args.eval_only:
        return

    # Convert traces to the format npcpy.ft.rl expects
    rl_traces = []
    for t in traces:
        rl_traces.append({
            "task_prompt": t["instruction"],
            "final_output": t["output"],
            "total_iterations": t.get("attempt", 1),
            "completed": t["passed"],
            "task_metadata": t,
            "reward": t["reward"],
        })

    # Save traces
    ts = time.strftime("%Y%m%d_%H%M%S")
    trace_file = f"rl_traces_npcsh_{ts}.json"
    with open(trace_file, "w") as f:
        json.dump(rl_traces, f, indent=2)
    print(f"Traces saved to {trace_file}")

    print("\nTraining with DPO...")

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
        logging_steps=5,
        save_steps=20,
    )

    adapter_path = train_with_dpo(rl_traces, config=config)

    if adapter_path:
        print(f"\nAdapter saved to: {adapter_path}")
        meta = {
            "base_model": args.model,
            "adapter_path": adapter_path,
            "num_traces": len(rl_traces),
            "passed_baseline": passed,
            "total_tasks": len(tasks),
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "beta": args.beta,
            },
        }
        meta_path = Path(args.output) / "rl_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
    else:
        print("Training failed — not enough preference pairs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
