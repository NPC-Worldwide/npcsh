#!/usr/bin/env python3
"""
evaluate_adapter.py

Evaluate a trained adapter (or baseline model) on the benchmark suite.
Uses the npcsh benchmark API directly — no subprocesses, no server management.

Usage:
    python scripts/evaluate_adapter.py --model mlx-community/Qwen3-4B-4bit --adapter models/npcsh_sft_toolcalls_all
    python scripts/evaluate_adapter.py --model mlx-community/Qwen3-4B-4bit --adapter models/npcsh_sft_toolcalls_all --category python
    python scripts/evaluate_adapter.py --fused-model models/npcsh_sft_toolcalls_all_fused
"""

import argparse
import json
import os
from pathlib import Path


def summarize(report) -> dict:
    """Build summary dict from BenchmarkReport."""
    total = report.total
    passed = report.passed
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print(f"{'='*50}")

    cats = {}
    for r in report.results:
        cat = r.category
        cats.setdefault(cat, {"total": 0, "passed": 0})
        cats[cat]["total"] += 1
        if r.passed:
            cats[cat]["passed"] += 1

    for cat in sorted(cats.keys()):
        stats = cats[cat]
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {cat:12s} {bar} {stats['passed']:3d}/{stats['total']:3d} ({100*rate:3.0f}%)")

    diffs = {}
    for r in report.results:
        d = r.difficulty
        diffs.setdefault(d, {"total": 0, "passed": 0})
        diffs[d]["total"] += 1
        if r.passed:
            diffs[d]["passed"] += 1

    print("\nBy difficulty:")
    for d in sorted(diffs.keys()):
        stats = diffs[d]
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {d:8s} {stats['passed']:3d}/{stats['total']:3d} ({100*rate:3.0f}%)")

    return {
        "passed": passed,
        "total": total,
        "categories": cats,
        "difficulty": diffs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained adapter on benchmark tasks")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--fused-model", default=None, help="Path to fused model (instead of adapter)")
    parser.add_argument("--provider", default="omlx")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--category", default=None)
    parser.add_argument("--output-json", default=None, help="Save summary to JSON file")
    args = parser.parse_args()

    if args.fused_model:
        eval_model = args.fused_model
    else:
        eval_model = args.model

    # If we have an adapter and provider is omlx, we need to tell npcpy
    # where the adapter lives.  The simplest way is via env var.
    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = eval_model
    env["NPCSH_CHAT_PROVIDER"] = args.provider
    if args.adapter:
        env["NPCSH_MLX_ADAPTER_PATH"] = str(args.adapter)

    print(f"\nEvaluating {eval_model} ({args.provider})...")
    if args.adapter:
        print(f"  adapter: {args.adapter}")
    if args.category:
        print(f"  category: {args.category}")
    print(f"  timeout: {args.timeout}s")
    print("")

    # Run benchmark directly via the API
    from npcsh.benchmark.local_runner import run_benchmark

    report = run_benchmark(
        model=eval_model,
        provider=args.provider,
        category=args.category,
        timeout=args.timeout,
    )

    summary = summarize(report)
    summary["model"] = eval_model
    summary["adapter"] = args.adapter
    summary["csv_path"] = str(report.checkpoint_file) if hasattr(report, "checkpoint_file") else None

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        print(f"\nSummary saved to {args.output_json}")


if __name__ == "__main__":
    main()
