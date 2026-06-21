#!/usr/bin/env python3
"""
fuse_and_eval.py

Fuse a trained MLX LoRA adapter with its base model, then evaluate on benchmark tasks.
Uses direct API calls — no subprocesses, no server management.

Usage:
    python scripts/fuse_and_eval.py --adapter models/npcsh_qwen3_4b --tasks 20
"""

import argparse
import os


def fuse_adapter(adapter_path: str, output_path: str):
    """Fuse adapter with base model."""
    print(f"Fusing {adapter_path} → {output_path}")
    try:
        from mlx_lm.lora import fuse

        fuse(adapter_path=adapter_path, save_path=output_path, dequantize=False)
        print(f"Fused model saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Fuse failed: {e}")
        return None


def evaluate_model(
    model_path: str,
    num_tasks: int = 20,
    category: str = None,
    model: str = None,
    provider: str = None,
):
    """Run quick benchmark evaluation via direct API."""
    from npcsh.benchmark.local_runner import run_benchmark

    eval_model = model or os.environ.get("NPCSH_CHAT_MODEL", "mlx-community/Qwen3-4B-4bit")
    eval_provider = provider or os.environ.get("NPCSH_CHAT_PROVIDER", "omlx")

    print(f"\nEvaluating {model_path} ({eval_provider}) on up to {num_tasks} tasks...")

    if model_path != eval_model:
        os.environ["NPCSH_MLX_ADAPTER_PATH"] = str(model_path)

    report = run_benchmark(
        model=eval_model,
        provider=eval_provider,
        category=category,
        timeout=90,
    )

    for r in report.results[:num_tasks]:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {r.task_id} ({r.category}/{r.difficulty}): {status} ({r.duration:.1f}s)"
        )

    passed = sum(1 for r in report.results if r.passed)
    total = len(report.results)
    avg_time = sum(r.duration for r in report.results) / total if total else 0
    print(
        f"\nResult: {passed}/{total} passed ({100*passed/total:.0f}%)  avg={avg_time:.1f}s"
    )
    return passed, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to trained adapter")
    parser.add_argument("--fuse-out", default=None, help="Fused model output path")
    parser.add_argument(
        "--model",
        default=None,
        help="Base model name for evaluation (defaults to NPCSH_CHAT_MODEL env)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider for evaluation (defaults to NPCSH_CHAT_PROVIDER env)",
    )
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--category", default=None)
    parser.add_argument("--skip-fuse", action="store_true")
    args = parser.parse_args()

    if not args.skip_fuse:
        fused_path = args.fuse_out or args.adapter + "_fused"
        model_path = fuse_adapter(args.adapter, fused_path)
        if not model_path:
            print("Fuse failed, evaluating with adapter directly")
            model_path = args.adapter
    else:
        model_path = args.adapter

    evaluate_model(
        model_path, args.tasks, args.category, args.model, args.provider
    )


if __name__ == "__main__":
    main()
