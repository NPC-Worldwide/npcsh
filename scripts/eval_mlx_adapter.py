#!/usr/bin/env python3
"""
eval_mlx_adapter.py

Smoke-test eval for a trained MLX LoRA adapter, driven ENTIRELY through npcpy's
native `get_llm_response` with the new provider="mlx". npcpy loads the MLX
base (+ adapter) and generates; this script never imports mlx/mlx_lm.

Generates completions for held-out instructions with the BASE mlx model and
with the ADAPTER (provider="mlx" for both), judge-rates both with the same
5-judge ensemble as rate_traces.py, and compares mean composite.

Honest framing: this is single-turn generation (not the full npcsh agent loop
— get_mlx_response doesn't parse tool calls yet), and every task_id was seen
during training. So it's a train-set, single-turn signal: does imitating the
judge-chosen traces move the judge score vs the untouched base on the same
instructions.

Usage:
    python scripts/eval_mlx_adapter.py \\
        --adapter adapters/npcsh_dpo_0.8b_smoke \\
        --base-model mlx-community/Qwen3.5-0.8B-4bit \\
        --ratings ~/.npcsh/benchmarks/ratings/ratings_20260712_141645.csv \\
        --n-instructions 30 --jobs 8
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from npcpy.llm_funcs import get_llm_response
from rate_traces import (
    DEFAULT_JUDGE_PANEL,
    aggregate_judges,
    load_judge_npcs,
    rate_with_judge,
)


def sample_instructions(ratings_path, n):
    df = pd.read_csv(os.path.expanduser(ratings_path))
    df = df[(df["instruction"].notna()) & (df["instruction"].str.strip() != "")]
    uniq = (
        df.sort_values("composite", ascending=False)
        .drop_duplicates(subset=["task_id"], keep="first")
        [["task_id", "instruction", "category"]]
        .reset_index(drop=True)
    )
    cats = uniq["category"].dropna().unique().tolist()
    picks = []
    per_cat = max(1, n // max(1, len(cats)))
    for c in cats:
        picks.append(uniq[uniq["category"] == c].head(per_cat))
    picks = pd.concat(picks) if picks else uniq.head(0)
    if len(picks) < n:
        remaining = uniq[~uniq["task_id"].isin(picks["task_id"])]
        picks = pd.concat([picks, remaining.head(n - len(picks))])
    return picks.head(n).reset_index(drop=True)


def gen(model, instruction, max_tokens):
    """One generation through npcpy's native mlx provider. npcpy owns mlx."""
    r = get_llm_response(
        instruction,
        model=model,
        provider="mlx",
        max_tokens=max_tokens,
    )
    resp = r.get("response")
    if isinstance(resp, str):
        return resp.strip()
    return str(resp or "").strip()


def judge_rate(instruction, response, judges, jobs):
    trace = {"instruction": instruction, "response": response, "passed": None}
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {
            ex.submit(rate_with_judge, trace, npc, name, model, [False]): (name, model)
            for name, model, npc in judges
        }
        for f in as_completed(futs):
            results.append(f.result())
    order = {name: i for i, (name, _, _) in enumerate(judges)}
    results.sort(key=lambda r: order.get(r["judge_name"], 99))
    return aggregate_judges(results), results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="adapters/npcsh_dpo_0.8b_smoke")
    p.add_argument("--base-model", default="mlx-community/Qwen3.5-0.8B-4bit")
    p.add_argument("--ratings", required=True)
    p.add_argument("--n-instructions", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--judge-panel", default=None)
    p.add_argument("--judge-provider", default="ollama")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    panel = (
        [(n.split(":")[0], n.split(":")[1]) for n in args.judge_panel.split(",")]
        if args.judge_panel else DEFAULT_JUDGE_PANEL
    )
    print(f"[judge] loading 5-judge ensemble: {panel}")
    judges = load_judge_npcs(panel, args.judge_provider)

    instrs = sample_instructions(args.ratings, args.n_instructions)
    print(f"[eval] {len(instrs)} held-out instructions across "
          f"{instrs['category'].nunique()} categories")
    print(f"[gen] base={args.base_model}  adapter={args.adapter}  (provider=mlx, via npcpy)")

    # warm both loads once so the first timed call isn't cold
    gen(args.base_model, "warmup", 4)
    gen(args.adapter, "warmup", 4)

    rows = []
    for i, r in instrs.iterrows():
        instr = r["instruction"]
        cat = r.get("category", "")
        tid = r.get("task_id", "")
        base_resp = gen(args.base_model, instr, args.max_tokens)
        ada_resp = gen(args.adapter, instr, args.max_tokens)
        base_agg, _ = judge_rate(instr, base_resp, judges, args.jobs)
        ada_agg, _ = judge_rate(instr, ada_resp, judges, args.jobs)
        d = ada_agg["composite"] - base_agg["composite"]
        print(f"[{i+1}/{len(instrs)}] {tid} ({cat})  "
              f"base={base_agg['composite']:.2f}  adapter={ada_agg['composite']:.2f}  Δ={d:+.2f}")
        rows.append({
            "task_id": tid, "category": cat, "instruction": instr,
            "base_response": base_resp, "adapter_response": ada_resp,
            "base_composite": base_agg["composite"],
            "adapter_composite": ada_agg["composite"],
            "base_std": base_agg["composite_std"],
            "adapter_std": ada_agg["composite_std"],
            "delta": d,
            "base_rationale": base_agg["rationale"][:300],
            "adapter_rationale": ada_agg["rationale"][:300],
        })

    df = pd.DataFrame(rows)
    print("\n================ SMOKE-TEST RESULT (provider=mlx, via npcpy) ================")
    print(f"Instructions: {len(df)}")
    print(f"Base    mean composite: {df['base_composite'].mean():.3f}  (std {df['base_composite'].std():.3f})")
    print(f"Adapter mean composite: {df['adapter_composite'].mean():.3f}  (std {df['adapter_composite'].std():.3f})")
    print(f"Mean Δ (adapter - base): {df['delta'].mean():+.3f}")
    won = (df["delta"] > 0.02).sum()
    lost = (df["delta"] < -0.02).sum()
    tied = len(df) - won - lost
    print(f"Per-instruction: improved {won} / regressed {lost} / tied {tied}")
    print("\nPer category (mean Δ):")
    print(df.groupby("category")["delta"].agg(["count", "mean"]).round(3).to_string())

    out = args.output or os.path.expanduser(
        f"~/.npcsh/benchmarks/ratings/eval_mlx_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} eval rows -> {out}")


if __name__ == "__main__":
    main()