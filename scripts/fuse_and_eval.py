#!/usr/bin/env python3
"""
fuse_and_eval.py

Fuse a trained MLX LoRA adapter with its base model, then evaluate on benchmark tasks.

Usage:
    python scripts/fuse_and_eval.py --adapter models/npcsh_qwen3_4b --tasks 20
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


def evaluate_model(model_path: str, num_tasks: int = 20, category: str = None):
    """Run quick benchmark evaluation."""
    task_file = Path(__file__).parent.parent / "npcsh" / "benchmark" / "tasks.csv"
    tasks = []
    with open(task_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if category and row["category"] != category:
                continue
            tasks.append(row)
            if len(tasks) >= num_tasks:
                break

    passed = 0
    times = []
    for task in tasks:
        work_dir = tempfile.mkdtemp(prefix=f"eval_{task['id']}_")
        setup_cmd = task.get("setup_cmd", "") or ""
        if setup_cmd:
            subprocess.run(["bash", "-c", setup_cmd], capture_output=True, cwd=work_dir)

        env = os.environ.copy()
        env["NPCSH_CHAT_MODEL"] = "mlx-community/Qwen3-4B-4bit"
        env["NPCSH_CHAT_PROVIDER"] = "omlx"
        env["NPCSH_STREAM_OUTPUT"] = "0"

        start = time.time()
        try:
            proc = subprocess.run(
                ["npcsh", "-c", task["instruction"]],
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=env,
                timeout=90,
            )
            time.sleep(0.5)
            verify = subprocess.run(
                ["bash", "-c", task["verify_cmd"]],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=work_dir,
            )
            ok = verify.returncode == 0
        except Exception:
            ok = False

        duration = time.time() - start
        times.append(duration)
        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(f"  {task['id']} ({task['category']}/{task['difficulty']}): {status} ({duration:.1f}s)")
        subprocess.run(["rm", "-rf", work_dir], capture_output=True)

    avg_time = sum(times) / len(times) if times else 0
    print(f"\nResult: {passed}/{len(tasks)} passed ({100*passed/len(tasks):.0f}%)  avg={avg_time:.1f}s")
    return passed, len(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to trained adapter")
    parser.add_argument("--fuse-out", default=None, help="Fused model output path")
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

    print(f"\nEvaluating {model_path} on {args.tasks} tasks...")
    evaluate_model(model_path, args.tasks, args.category)


if __name__ == "__main__":
    main()
