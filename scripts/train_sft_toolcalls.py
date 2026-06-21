#!/usr/bin/env python3
"""
train_sft_toolcalls.py

Train SFT on traces that contain proper tool calls.
Falls back to tasks.csv for instruction when trace doesn't include user message.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_from_csv import parse_trace, _compute_task_difficulty


def _load_task_instructions():
    """Load instruction text for each task_id from tasks.csv."""
    task_file = Path(__file__).parent.parent / "npcsh" / "benchmark" / "tasks.csv"
    instructions = {}
    with open(task_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instructions[row["id"]] = row["instruction"].strip()
    return instructions


def load_toolcall_records(csv_dir: str, hard_only: bool = False):
    csv.field_size_limit(10**7)
    task_rates = _compute_task_difficulty(csv_dir, "*.csv")
    task_instructions = _load_task_instructions()
    records = []
    for csv_file in sorted(Path(csv_dir).glob("*.csv")):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("passed", "").lower() != "true":
                    continue
                tid = row["task_id"]
                if hard_only and task_rates.get(tid, 0.5) >= 0.5:
                    continue
                trace = parse_trace(row.get("output", ""))
                if not trace:
                    continue
                instruction = trace["instruction"]
                if not instruction and tid in task_instructions:
                    instruction = task_instructions[tid]
                if not instruction:
                    continue
                response = trace["response"]
                if not response or "<tool_call>" not in response:
                    continue
                records.append({"instruction": instruction, "response": response, "task_id": tid})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="~/.npcsh/benchmarks/local")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--output", default="adapters/npcsh_sft_toolcalls")
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--device", default="mlx", choices=["mlx", "cuda", "cpu"])
    args = parser.parse_args()

    csv_dir = os.path.expanduser(args.csv_dir)
    records = load_toolcall_records(csv_dir, hard_only=args.hard_only)
    print(f"Loaded {len(records)} tool-call traces" + (" (hard-only)" if args.hard_only else ""))

    if len(records) < 5:
        print("Need >= 5 traces.")
        sys.exit(1)

    X = []
    y = []
    for rec in records:
        X.append(f"<|im_start|>user\n{rec['instruction']}\n<|im_end|>\n<|im_start|>assistant\n")
        y.append(f"{rec['response']}\n<|im_end|>\n")

    from npcpy.ft.sft import run_sft, SFTConfig
    cfg = SFTConfig(
        base_model_name=args.model,
        output_model_path=args.output,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        learning_rate=args.lr,
        max_length=1024,
        logging_steps=max(1, len(X) // 20),
        save_steps=max(1, len(X) // 5),
    )
    adapter = run_sft(X, y, config=cfg, format_style="qwen3")
    print(f"SFT adapter saved to {adapter}")


if __name__ == "__main__":
    main()
