#!/usr/bin/env python3
"""
rate_traces.py

LLM-as-judge rater for npcsh traces. Grades each trace on a rubric
(correctness, tool_selection, efficiency, clarity, partial_credit) and a
composite score in [0, 1], producing a continuous reward signal to replace
the binary verify_cmd pass/fail used by the RL training scripts.

Two trace sources feed one judge, writing a unified CSV sidecar
(~/.npcsh/benchmarks/ratings/ratings_<ts>.csv):

  1. Benchmark CSVs  (--csv-dir, default ~/.npcsh/benchmarks/local)
     Reuses the npcsh/benchmark/tasks.csv to join the task instruction +
     verify_cmd by task_id. The verify pass/fail is given to the judge as
     calibration context, but the judge still grades partial progress.

  2. npcsh history DB (--db, default ~/npcsh_history.db)
     Mines real (user -> assistant + tool_calls + tool_results) traces from
     conversation_history. No verifier exists here, so the judge is the sole
     signal.

Usage:
    python scripts/rate_traces.py --csv-dir ~/.npcsh/benchmarks/local --limit 5 --dry-run
    python scripts/rate_traces.py --db ~/npcsh_history.db --since "7 days" --limit 20
    python scripts/rate_traces.py --csv-dir ~/.npcsh/benchmarks/local --judge-model qwen3.5:122b-cloud
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from npcpy.llm_funcs import get_llm_response
from npcpy.npc_compiler import NPC

_BENCH_ROOT = Path(os.environ.get("NPCSH_BENCHMARK_DIR", "~/.npcsh")).expanduser()
RATINGS_DIR = Path(os.environ.get("NPCSH_RATINGS_DIR", _BENCH_ROOT / "benchmarks" / "ratings")).expanduser()
TASKS_CSV = Path(__file__).resolve().parent.parent / "npcsh" / "benchmark" / "tasks.csv"

RATING_COLUMNS = [
    "trace_key", "source", "task_id", "conversation_id", "branch_id", "model",
    "category", "difficulty", "instruction", "response", "passed",
    "correctness", "effectiveness", "tool_selection", "efficiency", "clarity",
    "partial_credit", "composite", "composite_std",
    "judge_composites", "judge_panel", "judge_models",
    "rationale", "ts",
]

# 5-judge ensemble: one npcsh team NPC persona pinned to one cloud model each.
# (npc_name, ollama cloud model). Personas are read from ~/.npcsh/npc_team/<name>.npc.
DEFAULT_JUDGE_PANEL = [
    ("corca", "kimi-k2.7-code:cloud"),
    ("frederic", "deepseek-v4-pro:cloud"),
    ("kadiefa", "glm-5.2:cloud"),
    ("alicanto", "qwen3.5:cloud"),
    ("sibiji", "minimax-m3:cloud"),
]
NPC_TEAM_DIR = Path(os.environ.get("NPCSH_NPC_TEAM_DIR", "~/.npcsh/npc_team")).expanduser()

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# --------------------------------------------------------------------------- #
# Trace loading + cleaning
# --------------------------------------------------------------------------- #

def clean_transcript(raw, max_total=200000):
    """Strip ANSI codes and [attempt N] markers from raw npcsh stdout. Full
    fidelity — the judge needs the actual shell results to score correctness,
    so we do NOT truncate command outputs. max_total is only a safety cap
    against pathological multi-MB outputs."""
    if not raw or not isinstance(raw, str):
        return ""
    s = ANSI_RE.sub("", raw)
    blocks = [p.strip() for p in re.split(r"\[attempt \d+\]", s) if p.strip()]
    transcript = "\n---attempt---\n".join(blocks)
    if len(transcript) > max_total:
        transcript = transcript[:max_total] + "\n...[truncated]"
    return transcript


def parse_model_from_filename(path):
    """npcsh_ollama_deepseek-v4-flash_cloud_20260711_155412.csv ->
    ('npcsh', 'ollama', 'deepseek-v4-flash_cloud'). Best-effort."""
    stem = Path(path).stem
    parts = stem.split("_")
    framework = parts[0] if parts else ""
    provider = parts[1] if len(parts) > 1 else ""
    if parts and (parts[-1].isdigit() or parts[-1] == "running"):
        model = "_".join(parts[2:-1])
    else:
        model = "_".join(parts[2:])
    return framework, provider, model


def load_tasks_index(tasks_csv=TASKS_CSV):
    """task_id -> {instruction, verify_cmd, category, difficulty}."""
    if not Path(tasks_csv).exists():
        return {}
    df = pd.read_csv(tasks_csv)
    idx = {}
    for _, row in df.iterrows():
        idx[str(row["id"])] = {
            "instruction": str(row.get("instruction", "")),
            "verify_cmd": str(row.get("verify_cmd", "")),
            "category": str(row.get("category", "")),
            "difficulty": str(row.get("difficulty", "")),
        }
    return idx


def load_csv_traces(csv_dir, pattern="*.csv", limit=None, min_category_size=6):
    """Yield benchmark traces joined with task instructions, in an order that
    spreads MODEL DIVERSITY across tasks early.

    Skips task_ids not present in tasks.csv (no instruction to judge against —
    these are the small/unsupported categories like audio-gen, image-gen,
    web-search, delegation, tool-chain that the benchmark CSVs reference but
    tasks.csv doesn't define) and skips any category with fewer than
    `min_category_size` tasks (drops 5-task sets, keeps the 10-task ones).

    Each CSV file is one model's full run over all tasks. If we walked
    file-by-file, a partial run would rate every task from only the first
    model — useless for DPO/GRPO, which need several models' traces per task.
    So instead we group every trace by task_id across ALL files, then yield
    round-robin: pass 1 emits task[0] model[0], task[1] model[0], ...; pass 2
    emits task[0] model[1], ...; etc. No per-task cap — every trace is rated
    eventually — but a Ctrl-C partial still leaves each task with ratings from
    several models instead of one. `limit` bounds the total count (0 = unlimited)."""
    tasks = load_tasks_index()
    valid_ids = set()
    cat_sizes = defaultdict(int)
    for tid, t in tasks.items():
        if t.get("instruction"):
            valid_ids.add(tid)
            cat_sizes[re.sub(r"-\d+$", "", tid)] += 1

    csv_dir = Path(csv_dir).expanduser()
    files = sorted(csv_dir.glob(pattern))
    # task_id -> list of traces, preserved in file-sorted (model) order
    by_task = {}
    n = 0
    for csv_file in files:
        framework, provider, model = parse_model_from_filename(csv_file)
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"[skip] {csv_file.name}: {e}")
            continue
        if "output" not in df.columns or "task_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            task_id = str(row.get("task_id", "")).strip()
            if not task_id or task_id not in valid_ids:
                continue
            cat = re.sub(r"-\d+$", "", task_id)
            if cat_sizes.get(cat, 0) < min_category_size:
                continue
            task = tasks.get(task_id, {})
            instruction = task.get("instruction") or ""
            verify_cmd = task.get("verify_cmd") or ""
            passed = row.get("passed")
            if isinstance(passed, str):
                passed = passed.strip().lower() in ("true", "1")
            by_task.setdefault(task_id, []).append({
                "source": "benchmark",
                "task_id": task_id,
                "conversation_id": "",
                "branch_id": "",
                "model": model,
                "instruction": instruction,
                "response": clean_transcript(str(row.get("output", ""))),
                "passed": bool(passed) if pd.notna(passed) else None,
                "verify_cmd": verify_cmd,
                "category": task.get("category", ""),
                "difficulty": task.get("difficulty", ""),
                "_key": f"benchmark|{csv_file.stem}|{task_id}|{n}",
            })
            n += 1

    # round-robin across tasks so each task gains a new model's rating per pass
    task_ids = sorted(by_task)
    emitted = 0
    pos = 0
    while True:
        progressed = False
        for tid in task_ids:
            bucket = by_task[tid]
            if pos < len(bucket):
                yield bucket[pos]
                emitted += 1
                progressed = True
                if limit and emitted >= limit:
                    return
        if not progressed:
            return
        pos += 1


def _is_substantive_user_msg(text):
    t = str(text).strip().lower()
    if len(t) < 15:
        return False
    chatter = ["hey", "hello", "hi ", "ok", "okay", "lol", "thanks", "thank you",
               "yes", "no ", "sure", "cool", "nice", "woo", "yay", "hmm", "wait"]
    if t in chatter or t.rstrip("!?.") in chatter:
        return False
    return True


def fetch_conversation_traces(conn, since, limit):
    """Group conversation_history by (conversation_id, branch_id); emit real
    user -> assistant (+ tool_calls + tool_results) traces."""
    cutoff = _cutoff(since)
    query = """
        SELECT conversation_id, branch_id, role, content, tool_calls,
               tool_results, model, provider, timestamp
        FROM conversation_history
        WHERE timestamp > ? AND role IN ('user', 'assistant')
        ORDER BY conversation_id, branch_id, timestamp
    """
    rows = conn.execute(query, (cutoff,)).fetchall()
    by_conv = {}
    for cid, bid, role, content, tc, tr, model, provider, ts in rows:
        by_conv.setdefault((cid, bid), []).append(
            {"role": role, "content": content, "tool_calls": tc,
             "tool_results": tr, "model": model, "timestamp": ts}
        )
    traces = []
    for (cid, bid), msgs in by_conv.items():
        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        if not user_msgs or not asst_msgs:
            continue
        instruction = str(user_msgs[0]["content"] or "").strip()
        if not _is_substantive_user_msg(instruction):
            continue
        a = asst_msgs[-1]
        content = str(a["content"] or "").strip()
        tool_calls = str(a["tool_calls"] or "").strip()
        tool_results = str(a["tool_results"] or "").strip()
        # keep only traces with real tool use or a substantive assistant reply
        if not tool_calls and len(content) < 50:
            continue
        response_parts = [p for p in [content, f"[tool_calls] {tool_calls}",
                                      f"[tool_results] {tool_results}"] if p and p != "[tool_calls] " and p != "[tool_results] "]
        response = "\n".join(response_parts)
        traces.append({
            "source": "history",
            "task_id": "",
            "conversation_id": cid or "",
            "branch_id": bid or "",
            "model": a["model"] or "",
            "instruction": instruction,
            "response": clean_transcript(response),
            "passed": None,
            "verify_cmd": "",
            "category": "",
            "difficulty": "",
            "_key": f"history|{cid}|{bid}",
        })
        if limit and len(traces) >= limit:
            break
    return traces


def _cutoff(since):
    if since.endswith("days") or since.endswith("day"):
        num = int(re.findall(r"\d+", since)[0])
        cutoff = datetime.now() - timedelta(days=num)
    elif since.endswith("hours") or since.endswith("hour"):
        num = int(re.findall(r"\d+", since)[0])
        cutoff = datetime.now() - timedelta(hours=num)
    else:
        cutoff = datetime.now() - timedelta(days=7)
    return cutoff.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

def extract_primary_directive(npc_name):
    """Pull the primary_directive block out of a team .npc file by indentation.
    We avoid yaml/jinja-parsing the whole file (plonk has {% for %} jinja that
    breaks yaml.safe_load); only the persona text is needed for judging."""
    path = NPC_TEAM_DIR / f"{npc_name}.npc"
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("primary_directive:") and "|" in line:
            start = i + 1
            break
    if start is None:
        return ""
    block = []
    for line in lines[start:]:
        if line.strip() == "":
            block.append("")
            continue
        if line.startswith("  "):
            block.append(line[2:])
        else:
            break
    return "\n".join(block).strip()


def load_judge_npcs(panel, provider):
    """Build lightweight judge NPCs: each carries its team persona as the
    system prompt (primary_directive) + a pinned cloud model. No jinxes are
    compiled — get_llm_response only reads primary_directive + model/provider,
    so we skip the fragile .npc file loader entirely."""
    rubric = f"""You are also serving as a STRICT, CRITICAL judge rating an AI agent's trace for an agentic shell task. Your job is to find flaws, not to rubber-stamp. Be demanding.

CRITICAL RULE — do NOT defer to the verifier: the verifier (verify_cmd) only checks that an output file exists. It does NOT validate HOW the task was done, whether the approach was any good, or whether the agent actually ran the commands it claims. A trace that "passes" the verifier can still be mediocre. DO NOT give full marks just because the verifier passed. Grade the trace on its own merits.

Score each dimension in [0.0, 1.0], and look hard for reasons to dock points EVEN ON CORRECT OUTCOMES:
- correctness: did the trace actually accomplish the instruction — and is it robust, or barely scraping by?
- effectiveness: did the agent achieve the goal in a way that generalizes, or via a fragile/roundabout path? Was the right approach chosen for the job? Does the solution actually meet the spirit of the task, not just the literal file check?
- tool_selection: were the chosen commands/tools the most appropriate and idiomatic, or clunky/wasteful? Penalize using a hammer for a thumbtack.
- efficiency: were steps economical? Penalize redundant attempts, re-running the same command, unnecessary intermediate steps, retrying after a non-error, or an overcomplicated solution to a simple problem.
- clarity: was the agent's reasoning shown and coherent? Penalize terse summary-only traces that don't show the actual commands/results, self-contradictory explanations, or missing reasoning.
- partial_credit: how close to a correct outcome, even if it failed?
- composite: your overall weighted score in [0.0, 1.0].

Reserve a composite of 1.0 ONLY for traces that are correct, effective, efficient, well-tooled, AND clearly reasoned. A bare-minimum pass with a terse claim and no shown commands is at most ~0.5-0.6. A passing trace with redundant attempts or an overcomplicated approach MUST lose efficiency/effectiveness points even though the output is correct. A trace that claims success without showing the commands/results does not deserve full marks on any dimension.

Return ONLY JSON with keys: correctness, effectiveness, tool_selection, efficiency, clarity, partial_credit, composite, rationale. The rationale MUST cite the specific flaw(s) you found, or state explicitly that the trace was excellent on every dimension (which is rare)."""
    judges = []
    for npc_name, model in panel:
        persona = extract_primary_directive(npc_name)
        npc = NPC(
            name=npc_name,
            primary_directive=f"{persona}\n\n{rubric}" if persona else rubric,
            model=model,
            provider=provider,
        )
        judges.append((npc_name, model, npc))
    return judges


def build_judge_prompt(trace):
    instruction = trace["instruction"]
    response = trace["response"]
    passed = trace["passed"]
    verify_cmd = trace.get("verify_cmd", "")
    verifier_line = ""
    if passed is not None:
        verifier_line = f"Verifier (verify_cmd) result: {'PASSED' if passed else 'FAILED'}\nverify_cmd: {verify_cmd}\n"
    schema_example = (
        "Return ONLY JSON. Example of a mediocre trace that technically passed but wasted steps:\n"
        + """{"correctness": 0.85, "effectiveness": 0.55, """
        + """"tool_selection": 0.50, "efficiency": 0.40, "clarity": 0.60, """
        + """"partial_credit": 0.90, "composite": 0.62, """
        + """"rationale": "The file was created, but the agent ran three redundant ls commands and used a python one-liner where a simple cp would suffice."}\n\n"""
        + "Now rate this trace. Return ONLY JSON with the same keys."
    )
    prompt = f"""{verifier_line}Instruction:
{instruction}

Agent trace (cleaned npcsh stdout / tool calls + results):
{response}

{schema_example}"""
    return prompt


def _coerce_score(data, key):
    try:
        return float(data.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def rate_with_judge(trace, npc, npc_name, model, _err_seen=None):
    """One judge's rating. The NPC's primary_directive (persona + rubric) is
    the system prompt; the pinned model/provider come from the NPC itself."""
    prompt = build_judge_prompt(trace)
    try:
        response = get_llm_response(prompt, npc=npc, format="json")
        data = response["response"]
        # npcpy parses format="json" into a dict for some providers (e.g. deepseek)
        # but the ollama provider returns the raw JSON string — parse it once here.
        if isinstance(data, str):
            data = json.loads(data)
    except Exception as e:
        if _err_seen is not None and not _err_seen[0]:
            _err_seen[0] = True
            print(f"[judge error:{npc_name}] {type(e).__name__}: {e}")
        return {
            "judge_name": npc_name, "model": model, "error": True,
            "correctness": None, "effectiveness": None, "tool_selection": None,
            "efficiency": None, "clarity": None, "partial_credit": None,
            "composite": None,
            "rationale": f"judge error: {type(e).__name__}: {e}",
        }
    composite = _coerce_score(data, "composite")
    if composite == 0.0:
        composite = (
            0.4 * _coerce_score(data, "correctness")
            + 0.2 * _coerce_score(data, "effectiveness")
            + 0.15 * _coerce_score(data, "tool_selection")
            + 0.15 * _coerce_score(data, "efficiency")
            + 0.1 * _coerce_score(data, "clarity")
        )
    return {
        "judge_name": npc_name, "model": model, "error": False,
        "correctness": _coerce_score(data, "correctness"),
        "effectiveness": _coerce_score(data, "effectiveness"),
        "tool_selection": _coerce_score(data, "tool_selection"),
        "efficiency": _coerce_score(data, "efficiency"),
        "clarity": _coerce_score(data, "clarity"),
        "partial_credit": _coerce_score(data, "partial_credit"),
        "composite": round(composite, 4),
        "rationale": str(data.get("rationale", "")),
    }


def aggregate_judges(judge_results):
    """Mean across judges per dimension + composite; std of the composite is a
    disagreement signal. Failed judges (composite None) are dropped from the
    mean but kept in judge_composites as ERR."""
    dims = ["correctness", "effectiveness", "tool_selection", "efficiency",
            "clarity", "partial_credit", "composite"]
    agg = {}
    for d in dims:
        vals = [r[d] for r in judge_results if r.get(d) is not None]
        agg[d] = round(sum(vals) / len(vals), 4) if vals else 0.0
    comp_vals = [r["composite"] for r in judge_results if r.get("composite") is not None]
    agg["composite_std"] = round(float(pd.Series(comp_vals).std()), 4) if len(comp_vals) > 1 else 0.0
    agg["judge_composites"] = ",".join(
        f"{r['judge_name']}={r['composite']:.2f}" if r.get("composite") is not None
        else f"{r['judge_name']}=ERR" for r in judge_results)
    rsorted = sorted(judge_results, key=lambda r: len(r.get("rationale", "")), reverse=True)
    agg["rationale"] = rsorted[0].get("rationale", "") if rsorted else ""
    return agg


# --------------------------------------------------------------------------- #
# Output / resume
# --------------------------------------------------------------------------- #

def load_seen(output_path):
    if not output_path.exists():
        return set()
    df = pd.read_csv(output_path)
    if "trace_key" not in df.columns:
        return set()
    return set(str(k) for k in df["trace_key"].dropna())


def trace_seen_key(trace):
    return trace.get("_key", "")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge rater for npcsh traces")
    parser.add_argument("--csv-dir", default=str(_BENCH_ROOT / "benchmarks" / "local"),
                        help="Benchmark CSV dir (set empty to skip CSV source)")
    parser.add_argument("--db", default=os.environ.get("NPCSH_HISTORY_DB", str(_BENCH_ROOT / "npcsh_history.db")),
                        help="npcsh history DB path (set empty to skip DB source)")
    parser.add_argument("--since", default="7 days", help="DB lookback window")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max traces to rate per source (0 = unlimited)")
    parser.add_argument("--min-category-size", type=int, default=6,
                        help="Skip task categories with fewer than this many tasks "
                             "(drops 5-task sets like audio-gen; 0 = keep all)")
    parser.add_argument("--judge-panel", default=None,
                        help="Comma list of npc:model pairs, e.g. "
                             "corca:kimi-k2.7-code:cloud,frederic:deepseek-v4-pro:cloud. "
                             "Defaults to the 5-judge ensemble from the npcsh team.")
    parser.add_argument("--judge-provider", default="ollama", help="Judge provider")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent judge calls")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--resume", action="store_true", help="Skip traces already in --output")
    parser.add_argument("--dry-run", action="store_true", help="Rate and print, write nothing")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).expanduser() if args.output else RATINGS_DIR / f"ratings_{ts}.csv"

    traces = []
    seen = load_seen(output_path) if (args.resume and not args.dry_run and output_path.exists()) else set()
    if args.csv_dir:
        csv_dir = Path(args.csv_dir).expanduser()
        if csv_dir.exists():
            loaded = list(load_csv_traces(csv_dir, limit=0,
                                          min_category_size=args.min_category_size))
            print(f"[csv] loaded {len(loaded)} benchmark traces from {csv_dir}")
            if seen:
                loaded = [t for t in loaded if trace_seen_key(t) not in seen]
                print(f"[resume] {len(loaded)} benchmark traces remain after skipping seen")
            if args.limit:
                loaded = loaded[:args.limit]
            traces.extend(loaded)
        else:
            print(f"[csv] dir not found: {csv_dir}")
    if args.db:
        db_path = Path(args.db).expanduser()
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            db_limit = args.limit if not args.csv_dir else 0
            db_traces = fetch_conversation_traces(conn, args.since, db_limit)
            conn.close()
            if seen:
                db_traces = [t for t in db_traces if trace_seen_key(t) not in seen]
                if args.limit:
                    db_traces = db_traces[:args.limit]
            print(f"[db] loaded {len(db_traces)} history traces from {db_path}")
            traces.extend(db_traces)
        else:
            print(f"[db] not found: {db_path}")

    if not traces:
        print("No traces to rate.")
        return

    todo = traces
    print(f"[batch] rating {len(todo)} traces")
    panel = []
    if args.judge_panel:
        for pair in args.judge_panel.split(","):
            pair = pair.strip()
            if not pair:
                continue
            npc_name, model = pair.split(":", 1)
            panel.append((npc_name.strip(), model.strip()))
    else:
        panel = DEFAULT_JUDGE_PANEL
    judges = load_judge_npcs(panel, args.judge_provider)
    n_judges = len(judges)
    print(f"[judges] {n_judges} judges: " + ", ".join(f"{n}({m})" for n, m, _ in judges))
    print(f"Rating {len(todo)} traces x {n_judges} judges = {len(todo) * n_judges} calls "
          f"({args.jobs} jobs)...")

    def _write_records(records, output_path):
        if output_path.exists():
            old = pd.read_csv(output_path)
            df = pd.concat([old, pd.DataFrame(records, columns=RATING_COLUMNS)],
                           ignore_index=True)
        else:
            df = pd.DataFrame(records, columns=RATING_COLUMNS)
        df.to_csv(output_path, index=False)
        return len(df)

    records = []
    done = 0
    total_written = 0
    _err_seen = [False]
    interrupted = False
    results = defaultdict(list)  # trace index -> list of judge result dicts
    ex = ThreadPoolExecutor(max_workers=args.jobs)
    futures = {}
    for ti, trace in enumerate(todo):
        for npc_name, model, npc_obj in judges:
            fut = ex.submit(rate_with_judge, trace, npc_obj, npc_name, model, _err_seen)
            futures[fut] = (ti, npc_name)
    try:
        for fut in as_completed(futures):
            ti, npc_name = futures[fut]
            res = fut.result()
            trace = todo[ti]
            tid = trace['task_id'] or trace['conversation_id'] or trace.get('_key', '')
            comp_str = f"{res['composite']:.2f}" if res.get('composite') is not None else "ERR"
            print(f"  [t{ti} {tid}] {npc_name} {res['model']} composite={comp_str} | "
                  f"{res['rationale'][:160]}")
            results[ti].append(res)
            if len(results[ti]) == n_judges:
                agg = aggregate_judges(results[ti])
                rec = {
                    "trace_key": trace.get("_key", ""),
                    "source": trace["source"],
                    "task_id": trace["task_id"],
                    "conversation_id": trace["conversation_id"],
                    "branch_id": trace["branch_id"],
                    "model": trace["model"],
                    "category": trace.get("category", ""),
                    "difficulty": trace.get("difficulty", ""),
                    "instruction": trace["instruction"],
                    "response": trace["response"],
                    "passed": trace["passed"],
                    "correctness": agg["correctness"],
                    "effectiveness": agg["effectiveness"],
                    "tool_selection": agg["tool_selection"],
                    "efficiency": agg["efficiency"],
                    "clarity": agg["clarity"],
                    "partial_credit": agg["partial_credit"],
                    "composite": agg["composite"],
                    "composite_std": agg["composite_std"],
                    "judge_composites": agg["judge_composites"],
                    "judge_panel": ",".join(n for n, _, _ in judges),
                    "judge_models": ",".join(m for _, m, _ in judges),
                    "rationale": agg["rationale"],
                    "ts": ts,
                }
                records.append(rec)
                done += 1
                print(f"  => [{done}/{len(todo)}] {trace['source']} {tid} "
                      f"AGG composite={agg['composite']:.2f} std={agg['composite_std']:.2f} "
                      f"[{agg['judge_composites']}]")
                # write EVERY row to disk as it completes — a crash only ever
                # costs the in-flight trace, never anything already rated
                total_written = _write_records(records, output_path)
                records.clear()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupt] cancelling pending judge calls, saving complete traces...")
        for f in futures:
            f.cancel()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    if args.dry_run:
        df = pd.DataFrame(records, columns=RATING_COLUMNS)
        print(df[["source", "task_id", "model", "composite",
                  "rationale"]].to_string(index=False))
        print(f"\n[dry-run] rated {len(records)} traces; nothing written.")
        return

    if records:
        total_written = _write_records(records, output_path)
    tag = " [partial, interrupted]" if interrupted else ""
    print(f"\nWrote {done} ratings -> {output_path} ({total_written} total){tag}")
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()