#!/usr/bin/env python3
"""
publish_dataset.py

Collate judge-rated npcsh traces + benchmark task definitions + built RL
datasets (SFT / DPO / GRPO / PPO) + analysis report into a HuggingFace
dataset repo for the enpisi-coder family, so the data is loadable via
`datasets.load_dataset` alongside the training scripts.

Layout written to the dataset repo:
    README.md          dataset card (YAML configs -> load_dataset splits)
    tasks.csv          benchmark task definitions
    rated_traces.csv   judge ratings from rate_traces.py
    sft_data.csv       supervised traces (composite >= --sft-threshold)
    dpo_pairs.csv      quality-ranked preference pairs
    grpo_groups.csv    GRPO groups with graded rewards
    ppo_records.csv    PPO records with graded rewards
    analysis.md        RL-prep report from analyze_ratings.py
    summary.csv        per-task summary

Usage:
    python scripts/publish_dataset.py --ratings ~/.npcsh/benchmarks/ratings/ --skip-upload
    python scripts/publish_dataset.py --ratings ~/.npcsh/benchmarks/ratings/ratings_*.csv \
        --repo-id npc-worldwide/enpisi-coder-data --reward-mode judge
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from train_from_csv import (
    build_sft_data, build_dpo_data, build_grpo_data, build_ppo_data,
    load_ratings, _dump_groups,
)
from analyze_ratings import analyze, load_ratings as load_ratings_df

TASKS_CSV = Path(__file__).resolve().parent.parent / "npcsh" / "benchmark" / "tasks.csv"


def _dataset_card(stats):
    configs = []
    for split in ["rated_traces", "tasks", "sft", "dpo", "grpo", "ppo"]:
        configs.append(f"  - config_name: {split}\n    data_files: {split}.csv")
    card = f"""---
license: mit
tags:
  - npcsh
  - rl
  - preference-data
  - agent-traces
  - enpisi-coder
configs:
{chr(10).join(configs)}
---

# enpisi-coder RL dataset

Judge-rated npcsh agent traces and derived RL training data for the
[enpisi-coder](https://huggingface.co/npc-worldwide/enpisi-coder) model family.

Produced by `scripts/rate_traces.py` (LLM-as-judge) and
`scripts/analyze_ratings.py`; built into SFT/DPO/GRPO/PPO splits by
`scripts/train_from_csv.py`.

## Splits

| Split | Rows | Description |
|-------|------|-------------|
| rated_traces | {stats.get('rated_traces', 0)} | Per-trace judge scores (correctness, tool_selection, efficiency, clarity, partial_credit, composite) |
| tasks | {stats.get('tasks', 0)} | Benchmark task definitions (instruction, verify_cmd, setup_cmd, difficulty) |
| sft | {stats.get('sft', 0)} | High-quality traces (composite >= threshold) formatted for SFT |
| dpo | {stats.get('dpo', 0)} | Preference pairs ranked by judge composite (chosen vs rejected) |
| grpo | {stats.get('grpo', 0)} | GRPO groups with graded rewards |
| ppo | {stats.get('ppo', 0)} | PPO records with graded rewards |

## Reward

`composite` is a continuous judge score in [0, 1] replacing the binary
`verify_cmd` pass/fail. GRPO/PPO rewards are `composite * difficulty_weight`
(judge mode) or a binary/composite blend (hybrid mode); see
`scripts/train_from_csv.py`.

## Load

```python
from datasets import load_dataset
ds = load_dataset("npc-worldwide/enpisi-coder-data", "dpo")
```
"""
    return card


def build_stage(args, ratings_df):
    stage = Path(os.path.expanduser(args.output_dir))
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    stats = {}
    # raw ratings
    ratings_df.to_csv(stage / "rated_traces.csv", index=False)
    stats["rated_traces"] = len(ratings_df)

    # task definitions
    if args.tasks_csv and Path(args.tasks_csv).exists():
        shutil.copy(args.tasks_csv, stage / "tasks.csv")
        stats["tasks"] = len(pd.read_csv(args.tasks_csv))

    csv_dir = os.path.expanduser(args.csv_dir)

    # SFT
    X, y = build_sft_data(csv_dir, ratings_df=ratings_df, sft_threshold=args.sft_threshold)
    pd.DataFrame({"prompt": X, "response": y}).to_csv(stage / "sft.csv", index=False)
    stats["sft"] = len(X)

    # DPO
    pairs = build_dpo_data(csv_dir, ratings_df=ratings_df, dpo_gap=args.dpo_gap,
                           dpo_max_per_task=args.dpo_max_per_task)
    pair_rows = pairs if pairs else []
    pd.DataFrame(pair_rows).to_csv(stage / "dpo.csv", index=False)
    stats["dpo"] = len(pair_rows)

    # GRPO
    groups = build_grpo_data(csv_dir, ratings_df=ratings_df, reward_mode=args.reward_mode)
    _dump_groups(groups, stage / "grpo.csv")
    stats["grpo"] = sum(len(g["responses"]) for g in groups)

    # PPO
    records = build_ppo_data(csv_dir, ratings_df=ratings_df, reward_mode=args.reward_mode)
    pd.DataFrame(records).to_csv(stage / "ppo.csv", index=False)
    stats["ppo"] = len(records)

    # analysis
    report, per_task = analyze(ratings_df, min_attempts=args.min_attempts)
    (stage / "analysis.md").write_text(report)
    if per_task is not None:
        per_task.to_csv(stage / "summary.csv", index=False)

    (stage / "README.md").write_text(_dataset_card(stats))
    return stage, stats


def main():
    parser = argparse.ArgumentParser(description="Collate rated traces into a HF dataset repo")
    parser.add_argument("--ratings", required=True, help="Ratings CSV file/dir/glob")
    parser.add_argument("--csv-dir", default="~/.npcsh/benchmarks/local",
                        help="Benchmark CSV dir (for RL builders + parse_trace fallback)")
    parser.add_argument("--tasks-csv", default=str(TASKS_CSV))
    parser.add_argument("--repo-id", default="npc-worldwide/enpisi-coder-data",
                        help="HF dataset repo id")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF API token")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--reward-mode", default="judge", choices=["binary", "judge", "hybrid"])
    parser.add_argument("--sft-threshold", type=float, default=0.7)
    parser.add_argument("--dpo-gap", type=float, default=0.3)
    parser.add_argument("--dpo-max-per-task", type=int, default=4,
                        help="Max DPO pairs emitted per task (with --ratings)")
    parser.add_argument("--min-attempts", type=int, default=2)
    parser.add_argument("--output-dir", default="~/.npcsh/benchmarks/hf_stage",
                        help="Local staging directory")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Build the staging dir, do not upload to HF")
    args = parser.parse_args()

    ratings_df = load_ratings(args.ratings)
    if ratings_df is None or ratings_df.empty:
        print("No ratings found.")
        return
    print(f"[ratings] {len(ratings_df)} rated traces")

    stage, stats = build_stage(args, ratings_df)
    print(f"[stage] built at {stage}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.skip_upload:
        print("[skip-upload] staging dir ready, not uploading.")
        return

    if not args.token:
        print("No HF_TOKEN set; pass --token or export HF_TOKEN. Aborting upload.")
        return
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        print(f"huggingface_hub not installed: {e}")
        return

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    print(f"[upload] -> https://huggingface.co/datasets/{args.repo_id}")
    api.upload_folder(folder_path=str(stage), repo_id=args.repo_id,
                      repo_type="dataset",
                      commit_message=f"Update enpisi-coder RL dataset ({stats})")
    print(f"[upload] done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()