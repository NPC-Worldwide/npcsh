#!/usr/bin/env python3
"""
benchmark_to_sft.py

Convert npcsh benchmark CSV traces into SFT training data and run fine-tuning
via npcpy.ft.sft (MLX on Apple Silicon or torch on CUDA).

Uses the canonical parse_trace from train_from_csv.py to properly reconstruct
Qwen3 <tool_call> JSON blocks.

Usage:
    python scripts/benchmark_to_sft.py --csv ~/.npcsh/benchmarks/local/npcsh_ollama_qwen3.5_0.8b_20260428_122811.csv --model mlx-community/Qwen3-0.6B-4bit --output models/npcsh_sft
    python scripts/benchmark_to_sft.py --csv-dir ~/.npcsh/benchmarks/local/ --pattern "npcsh_ollama_qwen3.5*" --model mlx-community/Qwen3-1.7B-4bit --output models/npcsh_sft --hard-only
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_from_csv import parse_trace, _compute_task_difficulty


def traces_from_csv(csv_path: str, hard_only: bool = False, task_rates: dict = None):
    """Yield parsed traces from a benchmark CSV, filtering for passed tasks."""
    csv.field_size_limit(10**7)
    if task_rates is None:
        task_rates = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            passed = row.get("passed", "").lower()
            if passed not in ("true", "1"):
                continue
            tid = row.get("task_id", "")
            if hard_only and task_rates.get(tid, 0.5) >= 0.5:
                continue
            trace = parse_trace(row.get("output", ""))
            if trace and trace["instruction"] and trace["response"]:
                trace["task_id"] = tid
                trace["category"] = row.get("category", "")
                trace["difficulty"] = row.get("difficulty", "")
                yield trace


def traces_from_dir(csv_dir: str, pattern: str = "npcsh_*.csv", hard_only: bool = False):
    """Yield traces from all matching CSVs in a directory."""
    path = Path(csv_dir)
    task_rates = _compute_task_difficulty(csv_dir, pattern) if hard_only else {}
    for csv_file in sorted(path.glob(pattern)):
        print(f"Reading {csv_file.name}...")
        yield from traces_from_csv(str(csv_file), hard_only=hard_only, task_rates=task_rates)


def build_sft_examples(traces, format_style: str = "qwen3"):
    """Build (X, y) lists for npcpy.ft.sft.run_sft.

    The response includes reconstructed <tool_call> JSON blocks so the model
    learns proper tool-calling syntax.
    """
    X = []
    y = []

    for trace in traces:
        instruction = trace["instruction"]
        response = trace["response"]

        if not instruction or not response:
            continue

        if format_style == "qwen3":
            prompt_text = f"<|im_start|>user\n{instruction}ocide\n<|im_start|>assistant\n"
            output_text = f"{response}ocide"
        elif format_style == "gemma":
            prompt_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
            output_text = f"{response}<end_of_turn>"
        else:
            prompt_text = f"Input: {instruction}\nOutput: "
            output_text = response

        X.append(prompt_text)
        y.append(output_text)

    return X, y


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark traces to SFT training data")
    parser.add_argument("--csv", help="Single benchmark CSV file")
    parser.add_argument("--csv-dir", help="Directory of benchmark CSVs")
    parser.add_argument("--pattern", default="npcsh_*.csv", help="Glob pattern for CSV files")
    parser.add_argument("--model", required=True, help="Base model name (e.g. mlx-community/Qwen3-0.6B-4bit)")
    parser.add_argument("--output", default="models/npcsh_sft", help="Output adapter path")
    parser.add_argument("--device", default="mlx", choices=["mlx", "cuda", "cpu"], help="Training backend")
    parser.add_argument("--format-style", default="qwen3", choices=["qwen3", "gemma", "llama", "raw"])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--hard-only", action="store_true", help="Train only on tasks with <50% success rate")
    parser.add_argument("--save-jsonl", help="Also save training examples to JSONL file")
    parser.add_argument("--skip-sft", action="store_true", help="Only compile data, don't train")
    args = parser.parse_args()

    if not args.csv and not args.csv_dir:
        print("Error: provide --csv or --csv-dir")
        sys.exit(1)

    traces = []
    if args.csv:
        traces = list(traces_from_csv(args.csv, hard_only=args.hard_only))
    else:
        traces = list(traces_from_dir(args.csv_dir, args.pattern, hard_only=args.hard_only))

    print(f"Collected {len(traces)} successful traces" + (" (hard-only)" if args.hard_only else ""))
    if len(traces) < 5:
        print("Need at least 5 traces to train. Run more benchmarks.")
        sys.exit(1)

    X, y = build_sft_examples(traces, args.format_style)

    if args.save_jsonl:
        with open(args.save_jsonl, "w") as f:
            for xi, yi in zip(X, y):
                f.write(json.dumps({"prompt": xi, "completion": yi}) + "\n")
        print(f"Saved {len(X)} examples to {args.save_jsonl}")

    if args.skip_sft:
        print("Skipping SFT (--skip-sft)")
        return

    print(f"Training SFT on {len(X)} examples...")

    from npcpy.ft.sft import run_sft, SFTConfig

    config = SFTConfig(
        base_model_name=args.model,
        output_model_path=args.output,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        logging_steps=max(1, len(X) // args.batch_size // 10),
        save_steps=max(1, len(X) // args.batch_size),
    )

    adapter_path = run_sft(X, y, config=config, format_style=args.format_style)
    print(f"Adapter saved to: {adapter_path}")

    meta = {
        "base_model": args.model,
        "adapter_path": adapter_path,
        "num_examples": len(X),
        "format_style": args.format_style,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
    }
    meta_path = Path(args.output) / "training_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
