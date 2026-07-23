#!/usr/bin/env python3
"""Agent-loop eval that runs through the real npcsh binary.

Previous versions hand-rolled a Python tool loop with only shell/python tools,
which is not what production npcsh uses. This version shells out to `npcsh -c`
so the model under test sees the exact same tool schema, system prompts, and
multi-turn logic as a real user. Conversations are saved to npcsh_history.db
and can be retrieved for rating/training.

Usage:
    python scripts/eval_mlx_agent_loop.py --n-per-category 1 --output-dir ~/.npcsh/benchmarks/ratings
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# npcsh must be importable so we can reuse the benchmark harness.
scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
sys.path.insert(0, str(repo_root))
from npcsh.benchmark.local_runner import run_task as _run_npcsh_task


BASE_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
DEFAULT_ADAPTER = "adapters/npcsh_sft_0.8b_recent"


def run_task(task, model, timeout=120, max_attempts=3):
    """Run one benchmark task through the real `npcsh -c` path.

    `model` is either the base MLX model name or a local adapter directory.
    npcpy's serve path detects adapter directories by adapter_config.json.
    """
    os.environ["NPCSH_CHAT_MODEL"] = model
    os.environ["NPCSH_CHAT_PROVIDER"] = "mlx"
    os.environ["NPCSH_ACCEPT_PERMISSIONS"] = "1"

    result = _run_npcsh_task(
        task,
        model=model,
        provider="mlx",
        timeout=timeout,
        max_attempts=max_attempts,
        framework="npcsh",
    )

    # Extract a rough turn count from the saved transcript.
    turns = 0
    try:
        turns = result.conversation_dump.count("[assistant]")
    except Exception:
        pass

    return {
        "task_id": result.task_id,
        "category": result.category,
        "difficulty": result.difficulty,
        "model": model,
        "passed": result.passed,
        "duration": round(result.duration, 2),
        "turns": turns,
        "attempts": result.attempts,
        "final_response": result.npcsh_output[:500],
        "verify_output": (result.error or "")[:500],
        "transcript": result.conversation_dump[:4000],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-category", type=int, default=1)
    bench_root = Path(os.environ.get("NPCSH_BENCHMARK_DIR", "~/.npcsh")).expanduser()
    p.add_argument("--output-dir", default=str(bench_root / "benchmarks" / "ratings"))
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-task wall-clock budget in seconds")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Retry attempts per task")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help="Path to the LoRA adapter to compare against the base model")
    p.add_argument("--base-model", default=BASE_MODEL,
                    help="Base MLX model name or path")
    args = p.parse_args()

    tasks = pd.read_csv(Path(__file__).resolve().parent.parent / "npcsh" / "benchmark" / "tasks.csv")
    sampled_rows = []
    for cat, group in tasks.groupby("category"):
        sampled_rows.append(group.sample(min(args.n_per_category, len(group)), random_state=42))
    sampled = pd.concat(sampled_rows, ignore_index=True)

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"eval_mlx_agent_loop_{ts}.csv"

    rows = []
    for model in [args.base_model, args.adapter]:
        for _, task in sampled.iterrows():
            print(f"\n[{model}] {task['id']} {task['category']} {task['difficulty']}")
            result = run_task(task.to_dict(), model, args.timeout, args.max_attempts)
            rows.append(result)
            print("  passed:", result["passed"], "duration:", result["duration"], "turns:", result["turns"])
            pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nWrote {len(rows)} results to {out_path}")


if __name__ == "__main__":
    main()