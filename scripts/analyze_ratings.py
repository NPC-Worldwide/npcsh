#!/usr/bin/env python3
"""
analyze_ratings.py

Aggregate judge ratings from rate_traces.py into an RL-prep decision report:
per-task / per-category / per-model breakdowns, failure-mode distribution,
hardest tasks, and judge-vs-verifier disagreements (the highest-value RL
targets). Writes a markdown report + summary.csv.

Usage:
    python scripts/analyze_ratings.py --ratings '~/.npcsh/benchmarks/ratings/ratings_*.csv'
    python scripts/analyze_ratings.py --ratings ~/.npcsh/benchmarks/ratings/ --min-attempts 3
"""

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

RATINGS_DIR = Path("~/.npcsh/benchmarks/ratings").expanduser()
SCORE_COLS = ["correctness", "effectiveness", "tool_selection", "efficiency",
              "clarity", "partial_credit", "composite"]


def load_ratings(paths):
    files = []
    for p in paths:
        p = os.path.expanduser(p)
        if any(c in p for c in "*?["):
            files.extend(glob.glob(p))
        elif os.path.exists(p):
            files.append(p)
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in sorted(files)]
    df = pd.concat(dfs, ignore_index=True)
    for c in SCORE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "passed" in df.columns:
        df["passed"] = df["passed"].map(
            lambda x: True if str(x).strip().lower() in ("true", "1.0", "1")
            else False if str(x).strip().lower() in ("false", "0.0", "0")
            else None
        )
    return df


def _md_table(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def analyze(df, min_attempts=2, disagree_low=0.6, disagree_high=0.4):
    """Return (report_markdown, per_task_summary_df_or_None)."""
    lines = []
    lines.append("# npcsh ratings analysis\n")
    lines.append(f"Traces rated: {len(df)} across {df['source'].nunique()} source(s), "
                 f"{df['model'].nunique()} model(s).\n")

    if "model" in df.columns and df["model"].notna().any():
        per_model = (df.groupby("model")["composite"]
                       .agg(["count", "mean", "median", "std"])
                       .round(3).sort_values("mean", ascending=False))
        per_model.columns = ["n", "mean_composite", "median", "std"]
        lines.append("## Per model\n")
        lines.append(_md_table(per_model.reset_index()) + "\n")

    cat_df = df[(df["category"].notna()) & (df["category"] != "")]
    if len(cat_df):
        per_cat = (cat_df.groupby("category")["composite"]
                     .agg(["count", "mean", "std"]).round(3)
                     .sort_values("mean"))
        per_cat.columns = ["n", "mean_composite", "std"]
        lines.append("## Per category (benchmark traces)\n")
        lines.append(_md_table(per_cat.reset_index()) + "\n")

    if "failure_mode" in df.columns:
        fm = df["failure_mode"].fillna("none").value_counts().reset_index()
        fm.columns = ["failure_mode", "count"]
        fm["pct"] = (fm["count"] / len(df) * 100).round(1)
        lines.append("## Failure-mode distribution\n")
        lines.append(_md_table(fm) + "\n")

    per_task = None
    task_df = df[(df["task_id"].notna()) & (df["task_id"] != "")]
    if len(task_df):
        per_task = (task_df.groupby("task_id")
                    .agg(n=("composite", "count"),
                         mean_composite=("composite", "mean"),
                         pass_rate=("passed", lambda s: s.dropna().mean() if s.dropna().size else None))
                    .reset_index())
        per_task["mean_composite"] = per_task["mean_composite"].round(3)
        per_task["pass_rate"] = per_task["pass_rate"].round(3)
        if "model" in task_df.columns:
            per_task = per_task.merge(
                task_df.groupby("task_id")["model"].agg(
                    lambda s: s.mode().iat[0] if not s.mode().empty else ""),
                on="task_id", how="left")
        if "failure_mode" in task_df.columns:
            per_task = per_task.merge(
                task_df.groupby("task_id")["failure_mode"].agg(
                    lambda s: s.mode().iat[0] if not s.mode().empty else ""),
                on="task_id", how="left").rename(
                    columns={"failure_mode": "dominant_failure_mode"})

        hardest = per_task[per_task["n"] >= min_attempts].sort_values("mean_composite").head(20)
        lines.append(f"## Hardest tasks (n >= {min_attempts}, lowest composite)\n")
        lines.append(_md_table(hardest) + "\n")

        judged = task_df[task_df["passed"].notna()]
        if len(judged):
            disagree_cols = [c for c in
                             ["task_id", "model", "passed", "composite", "failure_mode", "rationale"]
                             if c in judged.columns]
            disagree = judged[
                ((judged["passed"] == True) & (judged["composite"] < disagree_low)) |
                ((judged["passed"] == False) & (judged["composite"] > disagree_high))
            ][disagree_cols]
            lines.append("## Judge / verifier disagreements (high-value RL targets)\n")
            lines.append(f"Verifier passed but composite < {disagree_low}, or verifier "
                         f"failed but composite > {disagree_high}.\n")
            if len(disagree):
                lines.append(_md_table(disagree.head(30)) + "\n")
            else:
                lines.append("None.\n")

    return "\n".join(lines), per_task


def main():
    parser = argparse.ArgumentParser(description="Aggregate judge ratings into an RL-prep report")
    parser.add_argument("--ratings", nargs="+", default=["~/.npcsh/benchmarks/ratings/ratings_*.csv"],
                        help="Ratings CSV file(s) or glob")
    parser.add_argument("--min-attempts", type=int, default=2)
    parser.add_argument("--disagree-low", type=float, default=0.6)
    parser.add_argument("--disagree-high", type=float, default=0.4)
    parser.add_argument("--output-dir", default="~/.npcsh/benchmarks/ratings")
    parser.add_argument("--dry-run", action="store_true", help="Print report, write nothing")
    args = parser.parse_args()

    df = load_ratings(args.ratings)
    if df.empty:
        print("No ratings found.")
        return
    print(f"Loaded {len(df)} ratings over {df['model'].nunique()} model(s).")

    report, per_task = analyze(df, args.min_attempts, args.disagree_low, args.disagree_high)

    if args.dry_run:
        print(report)
        return

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"analysis_{ts}.md"
    md_path.write_text(report)
    written = [str(md_path)]
    if per_task is not None:
        csv_path = out_dir / f"summary_{ts}.csv"
        per_task.to_csv(csv_path, index=False)
        written.append(str(csv_path))
    print("Wrote " + ", ".join(written))


if __name__ == "__main__":
    main()