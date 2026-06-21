#!/usr/bin/env python3
"""
train_from_csv.py

Read benchmark CSVs directly and train via npcpy.ft.

Usage:
    python scripts/train_from_csv.py sft   --csv-dir ~/.npcsh/benchmarks/local --model mlx-community/Qwen3-4B-4bit
    python scripts/train_from_csv.py dpo   --csv-dir ~/.npcsh/benchmarks/local --model mlx-community/Qwen3-4B-4bit
    python scripts/train_from_csv.py grpo  --csv-dir ~/.npcsh/benchmarks/local --model mlx-community/Qwen3-4B-4bit --group-size 4
    python scripts/train_from_csv.py ppo   --csv-dir ~/.npcsh/benchmarks/local --model mlx-community/Qwen3-4B-4bit --beta 0.1
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path


def parse_trace(trace_str: str):
    if not trace_str or "---TRACE---" not in trace_str:
        return None
    trace = trace_str.split("---TRACE---", 1)[1]
    user_match = re.search(r"\[user\] (.*?) (?:\[assistant\]|\[tool_call\])", trace, re.DOTALL)
    instruction = ""
    if user_match:
        instruction = user_match.group(1).strip()
        instruction = re.sub(r"User Provided Context:.*", "", instruction, flags=re.DOTALL).strip()

    assistant_match = re.search(r"\[assistant\] (.*?) (?:\[tool_call\]|\[user\]|\Z)", trace, re.DOTALL)
    response = assistant_match.group(1).strip() if assistant_match else ""

    import json
    for m in re.finditer(r"\[tool_call\]\s+(\w+)\((\{.*?\})\)", trace):
        fname = m.group(1)
        args_raw = m.group(2)
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            try:
                import ast
                args = ast.literal_eval(args_raw)
            except (ValueError, SyntaxError):
                args = {}

        if fname == "sh":
            fname = "shell"
        elif fname in ("py", "python"):
            fname = "shell"
            if "python_code" in args:
                args["bash_command"] = args.pop("python_code")
        elif fname in ("Charlie", "Alice", "Bob", "Diana", "Eve", "Frank", "Alex", "chat"):
            continue

        tc = json.dumps({"name": fname, "arguments": args}, ensure_ascii=False)
        response += f"\n<tool_call>\n{tc}\n</tool_call>"

    return {"instruction": instruction, "response": response}


def load_csv_records(csv_dir: str, pattern: str = "*.csv"):
    csv.field_size_limit(10**7)
    for csv_file in sorted(Path(csv_dir).glob(pattern)):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace = parse_trace(row.get("output", ""))
                if trace and trace["instruction"] and trace["response"]:
                    yield {
                        "task_id": row["task_id"],
                        "instruction": trace["instruction"],
                        "response": trace["response"],
                        "passed": row.get("passed", "").lower() in ("true", "1"),
                        "attempts": int(row.get("attempts", "1") or 1),
                        "duration": float(row.get("duration", "0") or 0),
                    }


def build_sft_data(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False):
    X, y = [], []
    task_rates = _compute_task_difficulty(csv_dir, pattern)
    count = 0
    for rec in load_csv_records(csv_dir, pattern):
        if not rec["passed"]:
            continue
        tid = rec["task_id"]
        if hard_only and task_rates.get(tid, 0.5) >= 0.5:
            continue
        X.append(f"<|im_start|>user\n{rec['instruction']}<|im_end|>\n<|im_start|>assistant\n")
        y.append(f"{rec['response']}<|im_end|>\n")
        count += 1
    print(f"SFT: {count} passed traces" + (" (hard-only)" if hard_only else ""))
    return X, y


def build_dpo_data(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False):
    from datasets import Dataset

    task_rates = _compute_task_difficulty(csv_dir, pattern)
    by_task = {}
    for rec in load_csv_records(csv_dir, pattern):
        tid = rec["task_id"]
        if hard_only and task_rates.get(tid, 0.5) >= 0.5:
            continue
        by_task.setdefault(tid, []).append(rec)

    pairs = []
    for tid, traces in by_task.items():
        passed = [t for t in traces if t["passed"]]
        failed = [t for t in traces if not t["passed"]]
        if not passed or not failed:
            continue
        for p in passed:
            for f in failed:
                pairs.append({
                    "prompt": p["instruction"],
                    "chosen": p["response"],
                    "rejected": f["response"],
                })

    print(f"DPO: {len(pairs)} pairs from {len(by_task)} tasks" + (" (hard-only)" if hard_only else ""))
    if len(pairs) < 5:
        return None
    return Dataset.from_list(pairs)


def _compute_task_difficulty(csv_dir: str, pattern: str = "*.csv"):
    """Compute per-task success rate for difficulty weighting."""
    from collections import defaultdict
    by_task = defaultdict(list)
    for rec in load_csv_records(csv_dir, pattern):
        by_task[rec["task_id"]].append(rec["passed"])
    rates = {}
    for tid, results in by_task.items():
        rates[tid] = sum(results) / len(results)
    return rates


def build_grpo_data(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False):
    task_rates = _compute_task_difficulty(csv_dir, pattern)
    by_task = {}
    for rec in load_csv_records(csv_dir, pattern):
        tid = rec["task_id"]
        base_rate = task_rates.get(tid, 0.5)
        if hard_only and base_rate >= 0.5:
            continue
        difficulty_weight = 1.0 / (base_rate + 0.1)
        reward = (1.0 if rec["passed"] else -0.5) * difficulty_weight
        if rec["passed"]:
            reward += max(0, 0.3 * (3 - rec["attempts"]) / 3) * difficulty_weight
        rec["reward"] = reward
        by_task.setdefault(tid, []).append(rec)

    groups = []
    for tid, traces in by_task.items():
        if len(traces) < 2:
            continue
        prompt = traces[0]["instruction"]
        responses = [(t["response"], t["reward"]) for t in traces]
        groups.append({"prompt": prompt, "responses": responses})

    print(f"GRPO: {len(groups)} groups" + (" (hard-only)" if hard_only else ""))
    return groups


def build_ppo_data(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False):
    task_rates = _compute_task_difficulty(csv_dir, pattern)
    records = []
    for rec in load_csv_records(csv_dir, pattern):
        tid = rec["task_id"]
        base_rate = task_rates.get(tid, 0.5)
        if hard_only and base_rate >= 0.5:
            continue
        difficulty_weight = 1.0 / (base_rate + 0.1)
        reward = (1.0 if rec["passed"] else -0.5) * difficulty_weight
        if rec["passed"]:
            reward += max(0, 0.3 * (3 - rec["attempts"]) / 3) * difficulty_weight
        rec["reward"] = reward
        records.append(rec)
    print(f"PPO: {len(records)} traces" + (" (hard-only)" if hard_only else ""))
    return records


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv-dir", default="~/.npcsh/benchmarks/local")
    common.add_argument("--pattern", default="*.csv")
    common.add_argument("--model", required=True)
    common.add_argument("--output", default="adapters/npcsh_trained")
    common.add_argument("--device", default="mlx", choices=["mlx", "cuda", "cpu"])
    common.add_argument("--epochs", type=int, default=3)
    common.add_argument("--lr", type=float, default=2e-5)
    common.add_argument("--lora-r", type=int, default=16)
    common.add_argument("--hard-only", action="store_true", help="Train only on tasks with <50% success rate")

    sub.add_parser("sft", parents=[common])

    dpo_p = sub.add_parser("dpo", parents=[common])
    dpo_p.add_argument("--beta", type=float, default=0.5)

    grpo_p = sub.add_parser("grpo", parents=[common])
    grpo_p.add_argument("--group-size", type=int, default=4)

    ppo_p = sub.add_parser("ppo", parents=[common])
    ppo_p.add_argument("--beta", type=float, default=0.1)
    ppo_p.add_argument("--clip-eps", type=float, default=0.2)
    ppo_p.add_argument("--group-size", type=int, default=4)

    args = parser.parse_args()
    csv_dir = os.path.expanduser(args.csv_dir)

    if args.cmd == "sft":
        from npcpy.ft import run_sft, SFTConfig

        X, y = build_sft_data(csv_dir, args.pattern, hard_only=args.hard_only)
        if len(X) < 5:
            print("Need >= 5 passed traces.")
            sys.exit(1)
        cfg = SFTConfig(
            base_model_name=args.model,
            output_model_path=args.output,
            device=args.device,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            learning_rate=args.lr,
            max_length=2048,
            logging_steps=max(1, len(X) // 20),
            save_steps=max(1, len(X) // 5),
        )
        adapter = run_sft(X, y, config=cfg, format_style="qwen3")
        print(f"SFT adapter: {adapter}")

    elif args.cmd == "dpo":
        from npcpy.ft.rl import RLConfig, _train_dpo_mlx

        pairs = build_dpo_data(csv_dir, args.pattern, hard_only=args.hard_only)
        if pairs is None or len(pairs) < 5:
            print("Need >= 5 preference pairs.")
            sys.exit(1)
        pair_list = [
            {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in pairs
        ]
        cfg = RLConfig(
            base_model_name=args.model,
            adapter_path=args.output,
            device=args.device,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            beta=args.beta,
            max_pairs=len(pair_list),
            logging_steps=5,
            save_steps=20,
        )
        adapter = _train_dpo_mlx(pair_list, cfg)
        print(f"DPO adapter: {adapter}")

    elif args.cmd == "grpo":
        from npcpy.ft.rl import RLConfig, train_with_grpo

        groups = build_grpo_data(csv_dir, args.pattern, hard_only=args.hard_only)
        if not groups:
            print("Need tasks with multiple traces for GRPO.")
            sys.exit(1)
        cfg = RLConfig(
            base_model_name=args.model,
            adapter_path=args.output,
            device=args.device,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            group_size=args.group_size,
            max_length=2048,
            logging_steps=5,
            save_steps=20,
        )
        adapter = train_with_grpo(groups, cfg)
        print(f"GRPO adapter: {adapter}")

    elif args.cmd == "ppo":
        from npcpy.ft.rl import RLConfig, train_with_ppo

        records = build_ppo_data(csv_dir, args.pattern, hard_only=args.hard_only)
        if len(records) < 10:
            print("Need >= 10 traces for PPO.")
            sys.exit(1)
        cfg = RLConfig(
            base_model_name=args.model,
            adapter_path=args.output,
            device=args.device,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            beta=args.beta,
            clip_eps=args.clip_eps,
            group_size=args.group_size,
            max_length=2048,
            logging_steps=5,
            save_steps=20,
        )
        adapter = train_with_ppo(records, cfg)
        print(f"PPO adapter: {adapter}")


if __name__ == "__main__":
    main()
