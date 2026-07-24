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

import pandas as pd


def _bench_root() -> Path:
    return Path(os.environ.get("NPCSH_BENCHMARK_DIR", "~/.npcsh")).expanduser()


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


def load_ratings(ratings_path: str):
    """Read ratings CSV file(s) produced by rate_traces.py into a DataFrame.
    ratings_path may be a file or a glob/dir of CSVs. Returns None if none found."""
    import glob
    p = os.path.expanduser(ratings_path)
    files = []
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.csv")))
    elif any(c in p for c in "*?["):
        files = sorted(glob.glob(p))
    elif os.path.exists(p):
        files = [p]
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "composite" in df.columns:
        df["composite"] = pd.to_numeric(df["composite"], errors="coerce")
    return df


def _group_key(row):
    """Stable per-trace grouping key: task_id for benchmark traces, else
    conversation_id, else a hash of the instruction."""
    tid = row.get("task_id")
    if pd.notna(tid) and str(tid).strip():
        return str(tid)
    cid = row.get("conversation_id")
    if pd.notna(cid) and str(cid).strip():
        return str(cid)
    return f"hist-{abs(hash(str(row.get('instruction', ''))))}"


def load_rated_records(ratings_df):
    """Yield training records straight from the ratings CSV, bypassing the
    stale parse_trace path (current benchmark CSVs have no ---TRACE---)."""
    for _, row in ratings_df.iterrows():
        instruction = str(row.get("instruction", "") or "").strip()
        response = str(row.get("response", "") or "").strip()
        composite = row.get("composite")
        if not instruction or not response or pd.isna(composite):
            continue
        passed = row.get("passed")
        if isinstance(passed, str):
            passed = passed.strip().lower() in ("true", "1")
        yield {
            "task_id": _group_key(row),
            "instruction": instruction,
            "response": response,
            "passed": bool(passed) if pd.notna(passed) else None,
            "attempts": 1,
            "duration": 0.0,
            "composite": float(composite),
            "failure_mode": str(row.get("failure_mode", "") or ""),
        }


def _iter_records(csv_dir, pattern, ratings_df):
    """Records from ratings if provided, else from raw benchmark CSVs."""
    if ratings_df is not None:
        yield from load_rated_records(ratings_df)
    else:
        yield from load_csv_records(csv_dir, pattern)


def _chunk_trace(instruction: str, response: str, max_chars: int) -> list:
    """Split a long tool-call trace into multiple SFT examples at natural
    boundaries so we never truncate. Each chunk keeps the instruction prefix
    so the model always has task context."""
    # Rough token-to-char heuristic: ~4 chars/token.
    if len(instruction) + len(response) <= max_chars:
        return [(instruction, response)]
    # Split on tool call boundaries, then double newlines, then single newlines.
    splitters = ["</tool_call>", "\n\n", "\n"]
    parts = [response]
    for sep in splitters:
        new_parts = []
        for part in parts:
            if len(part) > max_chars:
                new_parts.extend([p.strip() for p in part.split(sep) if p.strip()])
            else:
                new_parts.append(part)
        parts = new_parts
        if all(len(p) + len(instruction) <= max_chars for p in parts):
            break
    chunks = []
    prefix = instruction
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        if i > 0:
            prefix = "(continued from previous turn) " + instruction[:200]
        chunks.append((prefix, part))
    return chunks if chunks else [(instruction, response)]


def build_sft_data(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False,
                   ratings_df=None, sft_threshold: float = 0.7, max_length: int = 512):
    # Heuristic: ~4 chars per token; leave headroom for chat template tokens.
    max_chars = max(256, max_length * 4 - 64) if max_length else 0
    X, y = [], []
    task_rates = _compute_task_difficulty(csv_dir, pattern, ratings_df)
    count = 0
    truncated = 0
    for rec in _iter_records(csv_dir, pattern, ratings_df):
        keep = rec["composite"] >= sft_threshold if ratings_df is not None else rec["passed"]
        if not keep:
            continue
        tid = rec["task_id"]
        if hard_only and task_rates.get(tid, 0.5) >= 0.5:
            continue
        instruction = rec["instruction"]
        response = rec["response"]
        if max_length and len(instruction) + len(response) > max_chars:
            chunks = _chunk_trace(instruction, response, max_chars)
            truncated += max(0, len(chunks) - 1)
        else:
            chunks = [(instruction, response)]
        for inst_chunk, resp_chunk in chunks:
            X.append(f"<|im_start|>user\n{inst_chunk}<|im_end|>\n<|im_start|>assistant\n")
            y.append(f"{resp_chunk}<|im_end|>\n")
            count += 1
    src = "rated" if ratings_df is not None else "passed"
    extra = " (hard-only)" if hard_only else ""
    extra += f", split {truncated} long traces into chunks" if truncated else ""
    print(f"SFT: {count} {src} examples{extra}")
    return X, y


def build_dpo_data(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False,
                   ratings_df=None, dpo_gap: float = 0.0, dpo_max_per_task: int = 0):
    task_rates = _compute_task_difficulty(csv_dir, pattern, ratings_df)
    by_task = {}
    for rec in _iter_records(csv_dir, pattern, ratings_df):
        tid = rec["task_id"]
        if hard_only and task_rates.get(tid, 0.5) >= 0.5:
            continue
        by_task.setdefault(tid, []).append(rec)

    pairs = []
    if ratings_df is not None:
        # Use ALL rated traces per task as pairwise preferences, not just a
        # high-gap subset. With 5+ judge composites per task we get a real
        # ranking signal; throwing away pairs via a gap filter is wasting data.
        # dpo_max_per_task (if > 0) is a hard cap for quick experiments.
        for tid, traces in by_task.items():
            if len(traces) < 2:
                continue
            traces.sort(key=lambda t: t.get("composite", 0.0), reverse=True)
            emitted = 0
            for i, hi in enumerate(traces):
                for lo in traces[i + 1:]:
                    gap = hi.get("composite", 0.0) - lo.get("composite", 0.0)
                    if gap < dpo_gap:
                        continue
                    pairs.append({
                        "prompt": hi["instruction"],
                        "chosen": hi["response"],
                        "rejected": lo["response"],
                        "chosen_composite": hi["composite"],
                        "rejected_composite": lo["composite"],
                    })
                    emitted += 1
                    if 0 < dpo_max_per_task <= emitted:
                        break
                if 0 < dpo_max_per_task <= emitted:
                    break
        src = "rated"
    else:
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
        src = "pass/fail"

    print(f"DPO: {len(pairs)} {src} pairs from {len(by_task)} tasks" + (" (hard-only)" if hard_only else ""))
    if len(pairs) < 5:
        return None
    return pairs


def _compute_task_difficulty(csv_dir: str, pattern: str = "*.csv", ratings_df=None):
    """Per-task base rate for difficulty weighting: mean composite when ratings
    are present, else empirical pass rate."""
    from collections import defaultdict
    by_task = defaultdict(list)
    for rec in _iter_records(csv_dir, pattern, ratings_df):
        if ratings_df is not None:
            by_task[rec["task_id"]].append(rec.get("composite", 0.5))
        else:
            by_task[rec["task_id"]].append(rec["passed"])
    rates = {}
    for tid, results in by_task.items():
        rates[tid] = sum(results) / len(results)
    return rates


def _trace_reward(rec, base_rate, reward_mode):
    """Continuous reward. Binary mode is byte-identical to the original formula;
    judge mode uses the LLM composite; hybrid blends both."""
    dw = 1.0 / (base_rate + 0.1)
    passed = rec.get("passed")
    composite = rec.get("composite", 0.0)
    if reward_mode == "judge" or (reward_mode == "hybrid" and passed is None):
        base = composite
        success_signal = composite
    elif reward_mode == "hybrid":
        binary = 1.0 if passed else -0.5
        base = 0.5 * binary + 0.5 * composite
        success_signal = 0.5 * (1.0 if passed else 0.0) + 0.5 * composite
    else:  # binary
        base = 1.0 if passed else -0.5
        success_signal = 1.0 if passed else 0.0
    reward = base * dw
    reward += max(0.0, 0.3 * (3 - rec.get("attempts", 1)) / 3) * success_signal * dw
    return reward


def build_grpo_data(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False,
                    ratings_df=None, reward_mode: str = "binary"):
    task_rates = _compute_task_difficulty(csv_dir, pattern, ratings_df)
    by_task = {}
    for rec in _iter_records(csv_dir, pattern, ratings_df):
        tid = rec["task_id"]
        base_rate = task_rates.get(tid, 0.5)
        if hard_only and base_rate >= 0.5:
            continue
        rec["reward"] = _trace_reward(rec, base_rate, reward_mode)
        by_task.setdefault(tid, []).append(rec)

    groups = []
    for tid, traces in by_task.items():
        if len(traces) < 2:
            continue
        prompt = traces[0]["instruction"]
        responses = [(t["response"], t["reward"]) for t in traces]
        groups.append({"prompt": prompt, "responses": responses})

    src = f" ({reward_mode})" if ratings_df is not None else ""
    print(f"GRPO: {len(groups)} groups{src}" + (" (hard-only)" if hard_only else ""))
    return groups


def build_ppo_data(csv_dir: str, pattern: str = "*.csv", hard_only: bool = False,
                   ratings_df=None, reward_mode: str = "binary"):
    task_rates = _compute_task_difficulty(csv_dir, pattern, ratings_df)
    records = []
    for rec in _iter_records(csv_dir, pattern, ratings_df):
        tid = rec["task_id"]
        base_rate = task_rates.get(tid, 0.5)
        if hard_only and base_rate >= 0.5:
            continue
        rec["reward"] = _trace_reward(rec, base_rate, reward_mode)
        records.append(rec)
    src = f" ({reward_mode})" if ratings_df is not None else ""
    print(f"PPO: {len(records)} traces{src}" + (" (hard-only)" if hard_only else ""))
    return records


def _dump_groups(groups, path):
    """Flatten GRPO groups [{prompt, responses:[(resp, reward),...]}] to CSV."""
    rows = []
    for g in groups:
        for response, reward in g["responses"]:
            rows.append({"prompt": g["prompt"], "response": response, "reward": reward})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv-dir", default=str(_bench_root() / "benchmarks" / "local"))
    common.add_argument("--pattern", default="*.csv")
    common.add_argument("--model", required=True)
    common.add_argument("--output", default="adapters/npcsh_trained")
    common.add_argument("--device", default="mlx", choices=["mlx", "cuda", "cpu"])
    common.add_argument("--epochs", type=int, default=3)
    common.add_argument("--lr", type=float, default=2e-5)
    common.add_argument("--lora-r", type=int, default=16)
    common.add_argument("--batch-size", type=int, default=2)
    common.add_argument("--max-length", type=int, default=512)
    common.add_argument("--hard-only", action="store_true", help="Train only on tasks with <50% success rate")
    common.add_argument("--ratings", default=None,
                        help="Ratings CSV/dir/glob from rate_traces.py; enables graded rewards")
    common.add_argument("--reward-mode", default="binary", choices=["binary", "judge", "hybrid"],
                        help="Reward formula when --ratings is set")
    common.add_argument("--dry-run", action="store_true",
                        help="Build the dataset and dump to CSV, do not train")
    common.add_argument("--dump-dir", default=str(_bench_root() / "benchmarks" / "datasets"),
                        help="Where --dry-run writes the built dataset CSV")

    sft_p = sub.add_parser("sft", parents=[common])
    sft_p.add_argument("--sft-threshold", type=float, default=0.7,
                       help="Min composite to include a trace in SFT (with --ratings)")

    dpo_p = sub.add_parser("dpo", parents=[common])
    dpo_p.add_argument("--beta", type=float, default=0.5)
    dpo_p.add_argument("--dpo-gap", type=float, default=0.0,
                       help="Min composite gap to emit a DPO pair, 0 = use all ordered pairs (with --ratings)")
    dpo_p.add_argument("--dpo-max-per-task", type=int, default=0,
                       help="Max DPO pairs per task, 0 = unlimited (with --ratings)")

    grpo_p = sub.add_parser("grpo", parents=[common])
    grpo_p.add_argument("--group-size", type=int, default=4)

    ppo_p = sub.add_parser("ppo", parents=[common])
    ppo_p.add_argument("--beta", type=float, default=0.1)
    ppo_p.add_argument("--clip-eps", type=float, default=0.2)
    ppo_p.add_argument("--group-size", type=int, default=4)

    args = parser.parse_args()
    csv_dir = os.path.expanduser(args.csv_dir)
    ratings_df = load_ratings(args.ratings) if args.ratings else None
    if ratings_df is not None:
        print(f"[ratings] loaded {len(ratings_df)} rated traces -> reward-mode={args.reward_mode}")
    dump_dir = Path(os.path.expanduser(args.dump_dir))
    if args.dry_run:
        dump_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "sft":
        X, y = build_sft_data(csv_dir, args.pattern, hard_only=args.hard_only,
                              ratings_df=ratings_df, sft_threshold=args.sft_threshold,
                              max_length=args.max_length)
        if len(X) < 5:
            print("Need >= 5 traces.")
            sys.exit(1)
        if args.dry_run:
            pd.DataFrame({"prompt": X, "response": y}).to_csv(
                dump_dir / "sft_data.csv", index=False)
            print(f"[dry-run] dumped {len(X)} SFT rows -> {dump_dir/'sft_data.csv'}")
            return
        from npcpy.ft import run_sft, SFTConfig
        cfg = SFTConfig(
            base_model_name=args.model,
            output_model_path=args.output,
            device=args.device,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
            logging_steps=max(1, len(X) // 20),
            save_steps=max(1, len(X) // 5),
        )
        adapter = run_sft(X, y, config=cfg, format_style="qwen3")
        print(f"SFT adapter: {adapter}")

    elif args.cmd == "dpo":
        pairs = build_dpo_data(csv_dir, args.pattern, hard_only=args.hard_only,
                               ratings_df=ratings_df, dpo_gap=args.dpo_gap,
                               dpo_max_per_task=args.dpo_max_per_task)
        if pairs is None or len(pairs) < 5:
            print("Need >= 5 preference pairs.")
            sys.exit(1)
        if args.dry_run:
            pd.DataFrame(list(pairs)).to_csv(dump_dir / "dpo_pairs.csv", index=False)
            print(f"[dry-run] dumped {len(pairs)} DPO pairs -> {dump_dir/'dpo_pairs.csv'}")
            return
        from npcpy.ft.rl import RLConfig, _train_dpo_mlx
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
            max_length=args.max_length,
            logging_steps=5,
            save_steps=20,
        )
        adapter = _train_dpo_mlx(pair_list, cfg)
        print(f"DPO adapter: {adapter}")

    elif args.cmd == "grpo":
        groups = build_grpo_data(csv_dir, args.pattern, hard_only=args.hard_only,
                                 ratings_df=ratings_df, reward_mode=args.reward_mode)
        if not groups:
            print("Need tasks with multiple traces for GRPO.")
            sys.exit(1)
        if args.dry_run:
            _dump_groups(groups, dump_dir / "grpo_groups.csv")
            print(f"[dry-run] dumped {len(groups)} GRPO groups -> {dump_dir/'grpo_groups.csv'}")
            return
        from npcpy.ft.rl import RLConfig, train_with_grpo
        cfg = RLConfig(
            base_model_name=args.model,
            adapter_path=args.output,
            device=args.device,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            group_size=args.group_size,
            max_length=args.max_length,
            logging_steps=5,
            save_steps=20,
        )
        adapter = train_with_grpo(groups, cfg)
        print(f"GRPO adapter: {adapter}")

    elif args.cmd == "ppo":
        records = build_ppo_data(csv_dir, args.pattern, hard_only=args.hard_only,
                                 ratings_df=ratings_df, reward_mode=args.reward_mode)
        if len(records) < 10:
            print("Need >= 10 traces for PPO.")
            sys.exit(1)
        if args.dry_run:
            pd.DataFrame(records).to_csv(dump_dir / "ppo_records.csv", index=False)
            print(f"[dry-run] dumped {len(records)} PPO records -> {dump_dir/'ppo_records.csv'}")
            return
        from npcpy.ft.rl import RLConfig, train_with_ppo
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
            max_length=args.max_length,
            logging_steps=5,
            save_steps=20,
        )
        adapter = train_with_ppo(records, cfg)
        print(f"PPO adapter: {adapter}")


if __name__ == "__main__":
    main()
