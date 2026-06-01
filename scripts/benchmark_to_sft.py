#!/usr/bin/env python3
"""
benchmark_to_sft.py

Convert npcsh benchmark CSV traces into SFT training data and run fine-tuning
via npcpy.ft.sft (MLX on Apple Silicon or torch on CUDA).

Usage:
    python scripts/benchmark_to_sft.py --csv ~/.npcsh/benchmarks/local/npcsh_ollama_qwen3.5_0.8b_20260428_122811.csv --model mlx-community/Qwen3-0.6B-4bit --output models/npcsh_sft
    python scripts/benchmark_to_sft.py --csv-dir ~/.npcsh/benchmarks/local/ --pattern "npcsh_ollama_qwen3.5*" --model mlx-community/Qwen3-1.7B-4bit --output models/npcsh_sft
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


def parse_trace(trace_str: str):
    """Extract (system_prompt, instruction, assistant_response, tool_calls) from a benchmark trace."""
    if not trace_str or "---TRACE---" not in trace_str:
        return None

    parts = trace_str.split("---TRACE---", 1)
    output_before = parts[0].strip()
    trace = parts[1].strip()

    system_prompt = ""
    instruction = ""
    assistant_response = ""
    tool_calls = []

    # Extract system prompt: everything between [system] and [user]
    sys_match = re.search(r"\[system\] (.*?) \[user\]", trace, re.DOTALL)
    if sys_match:
        system_prompt = sys_match.group(1).strip()

    # Extract user instruction (first [user] content)
    user_match = re.search(r"\[user\] (.*?) (?:\[assistant\]|\[tool_call\])", trace, re.DOTALL)
    if user_match:
        instruction = user_match.group(1).strip()
        # Remove the "User Provided Context" boilerplate
        instruction = re.sub(r"User Provided Context:.*", "", instruction, flags=re.DOTALL).strip()

    # Extract assistant response (first [assistant] content)
    assistant_match = re.search(r"\[assistant\] (.*?) (?:\[tool_call\]|\[user\]|\Z)", trace, re.DOTALL)
    if assistant_match:
        assistant_response = assistant_match.group(1).strip()

    # Extract tool calls
    for tc_match in re.finditer(r"\[tool_call\] ([\w_]+)\((.*?)\)", trace):
        tool_calls.append(f"{tc_match.group(1)}({tc_match.group(2)})")

    return {
        "system_prompt": system_prompt,
        "instruction": instruction,
        "assistant_response": assistant_response,
        "tool_calls": tool_calls,
        "output_before": output_before,
    }


def traces_from_csv(csv_path: str, min_reward: float = 0.5):
    """Yield parsed traces from a benchmark CSV, filtering for passed tasks."""
    field_size = csv.field_size_limit(10**7)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            passed = row.get("passed", "").lower()
            if passed not in ("true", "1"):
                continue
            trace = parse_trace(row.get("output", ""))
            if trace and trace["instruction"] and trace["assistant_response"]:
                trace["task_id"] = row.get("task_id", "")
                trace["category"] = row.get("category", "")
                yield trace


def traces_from_dir(csv_dir: str, pattern: str = "npcsh_*.csv"):
    """Yield traces from all matching CSVs in a directory."""
    path = Path(csv_dir)
    for csv_file in sorted(path.glob(pattern)):
        print(f"Reading {csv_file.name}...")
        yield from traces_from_csv(str(csv_file))


def build_sft_examples(traces, format_style: str = "qwen3"):
    """Build (X, y) lists for npcpy.ft.sft.run_sft."""
    X = []
    y = []

    for trace in traces:
        instruction = trace["instruction"]
        response = trace["assistant_response"]

        if format_style == "qwen3":
            prompt_text = f"<|im_start|>system\n{trace['system_prompt']}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            output_text = f"{response}<|im_end|>"
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
    parser.add_argument("--save-jsonl", help="Also save training examples to JSONL file")
    args = parser.parse_args()

    if not args.csv and not args.csv_dir:
        print("Error: provide --csv or --csv-dir")
        sys.exit(1)

    traces = []
    if args.csv:
        traces = list(traces_from_csv(args.csv))
    else:
        traces = list(traces_from_dir(args.csv_dir, args.pattern))

    print(f"Collected {len(traces)} successful traces")
    if len(traces) < 5:
        print("Need at least 5 traces to train. Run more benchmarks.")
        sys.exit(1)

    X, y = build_sft_examples(traces, args.format_style)

    if args.save_jsonl:
        with open(args.save_jsonl, "w") as f:
            for xi, yi in zip(X, y):
                f.write(json.dumps({"prompt": xi, "completion": yi}) + "\n")
        print(f"Saved {len(X)} examples to {args.save_jsonl}")

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

    # Write a small metadata file
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
