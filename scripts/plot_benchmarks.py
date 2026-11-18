#!/usr/bin/env python3
"""Regenerate the ALARA/HAXD benchmark figures (Figs. 2-6) from a benchmark datacore.

The figures in ``papers/research/ai/multi-agent/HAXD_2026_ALARA/haxd_2026_npcsh.tex``
are produced by a paper-side ``make_plots.py`` that hardcodes a list of models and
depends on the ``chroptiks`` plotting package. This script inverts that: it reads a
benchmark **datacore** (the CSVs the benchmark runner already writes) and emits the
same figures, so the plots and the README scores table stay in sync with the runs.

Figures produced (names match the paper's ``\\includegraphics`` targets):

    scaling.pdf                  Fig. 2  score vs parameter count, per family
    npcsh_vs_mmlu.pdf            Fig. 3  agency vs intelligence, regression + 1 sigma
    tool_calls_vs_agency.pdf     Fig. 4  mean tool calls/task vs agency
    retry_value_by_category.pdf  Fig. 5  first-attempt pass rate + retry gain by category
    family_category_heatmap.pdf  Fig. 6  pass rate by model x task category

Three further scaling variants share the same axes (parameters x score) but are
separate files, so none replaces ``scaling.pdf``:

    scaling_versions.pdf         two side-by-side panels, one per npcsh version
                                 bucket (< v2.1.10 / >= v2.1.10), family markers,
                                 with the signed change annotated beside each
                                 model measured on both sides
    scaling_country.pdf          one point per model, marker keyed by country of
                                 origin rather than by family, so the panel reads
                                 as a country comparison on a common size axis

Figure names accepted by ``--only``: scaling, scaling_versions,
scaling_country, npcsh_vs_mmlu, tool_calls_vs_agency,
retry_value_by_category, family_category_heatmap.

Datacore layout (all parts optional except the runs):

    <datacore>/
        runs/                 # per-model CSVs written by npcsh/benchmark/local_runner.py
            npcsh_ollama_qwen3.5_4b_20260307_203901.csv
        model_meta.csv        # model,family,params_b,maker[,country,mmlu,mmlu_pro]
                              #   (see --write-meta-template)
        trace.csv             # optional per-task rows w/ explicit n_tool_calls/n_attempts

Country of origin is resolved maker -> country, never guessed from a model or
family name: ``model_meta.csv`` may supply ``country`` directly, otherwise a
missing ``maker`` is filled from the family's real company and the country is
looked up from that. Models with no resolvable country are reported and omitted
from ``scaling_country.pdf`` rather than binned as "unknown".

The per-run CSV columns come straight from ``local_runner.run_benchmark``:
``task_id,category,difficulty,passed,duration,error,output[,attempts]``.
When ``attempts`` / tool-call counts are absent they are recovered from the
``output`` transcript. Two trace formats are handled:

  * paper-era: a ``---TRACE---`` block with ``[tool]`` / ``[assistant]`` markers
  * current:   ANSI-coloured stdout with one ``\u26a1 <tool>:`` line per call,
    no ``---TRACE---`` block. The ``attempts`` column exists in these files but
    is entirely empty, so attempts fall back to ``[attempt N]`` markers.

Runs are selected per model, not globally: the most complete run wins, and a
1-task crash fragment can never beat a 100-task run. Runs whose best attempt
covers under half the expected tasks are dropped and reported.

Usage
-----
    # figures only, from a datacore dir
    python scripts/plot_benchmarks.py --datacore ~/.npcsh --outdir figures

    # additionally compare the figures' source data to the README scores table
    python scripts/plot_benchmarks.py --datacore ~/.npcsh --readme README.md

    # scaffold a model_meta.csv for models that have runs but no metadata
    python scripts/plot_benchmarks.py --datacore ~/.npcsh --write-meta-template

    # Fig. 3 (npcsh_vs_mmlu) needs an intelligence score in model_meta.csv.
    # It uses the `mmlu` column when >=3 models have one, otherwise falls back
    # to `mmlu_pro` -- most 2026 model cards publish MMLU-Pro, not plain MMLU.
    # Missing values are never imputed; a panel with too little data is skipped.

    # validate the whole pipeline on synthetic data (no runs needed)
    python scripts/plot_benchmarks.py --demo --outdir /tmp/benchfigs

This script never modifies README.md. Its scores table is curated per npcsh
release; a datacore is a snapshot of whatever runs you point it at, which may be
an older suite with a different task count. --readme only reports whether the
two agree.

Exit codes: 0 on success, 2 when no usable runs were found.
"""

import argparse
import math
import os
import re
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Load the paper's own matplotlibrc instead of inventing rcParams. The paper's
# look -- serif + LaTeX text, 20pt axis labels, inward ticks on all four sides,
# plasma default cmap -- comes entirely from that file, and make_plots.py loads
# it with matplotlib.rc_file before doing anything else.
# Vendored copies live next to this script so the figure styling travels with the
# repo. The paper directory is only a fallback for a machine that has it checked
# out; without either, matplotlib falls back to its defaults and the figures come
# out in the wrong typeface.
_PAPER_DIR = Path(
    os.environ.get(
        "NPCSH_PAPER_DIR",
        "/home/caug/npcww/papers/research/ai/multi-agent/HAXD_2026_ALARA",
    )
)
_ASSET_DIR = Path(__file__).resolve().parent / "assets"


def _resolve_asset(name, paper_relpath):
    """Prefer the vendored asset, then the paper checkout, else None."""
    for cand in (_ASSET_DIR / name, _PAPER_DIR / paper_relpath):
        if Path(cand).exists():
            return Path(cand)
    return None


_PAPER_CSV = _resolve_asset(
    "external_benchmarks.csv", "benchmark_data/external_benchmarks.csv"
)
_RC_PATH = Path(
    os.environ.get("NPCSH_MATPLOTLIBRC")
    or (_resolve_asset("matplotlibrc", "matplotlibrc") or "")
)
if _RC_PATH.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_RC_PATH.parent))
    matplotlib.rc_file(str(_RC_PATH))
    RC_SOURCE = str(_RC_PATH)
else:
    RC_SOURCE = "matplotlib defaults (paper matplotlibrc not found)"

if os.environ.get("NPCSH_FIG_USETEX") == "0":
    matplotlib.rcParams["text.usetex"] = False


def _usetex_works():
    """Probe LaTeX once; disable cleanly rather than dying mid-figure."""
    if not matplotlib.rcParams.get("text.usetex"):
        return False
    try:
        f, a = plt.subplots(figsize=(1, 1))
        a.set_xlabel("probe 100%")
        f.canvas.draw()
        plt.close(f)
        return True
    except Exception:  # noqa: BLE001
        plt.close("all")
        matplotlib.rcParams["text.usetex"] = False
        return False


USETEX = _usetex_works()

RUN_COLUMNS = [
    "task_id",
    "category",
    "difficulty",
    "passed",
    "duration",
    "output",
    "attempts",
]
# Directory name used inside a datacore for the paper's own historical runs. It
# is excluded from run/trace/meta detection and loaded as pre-release data.
_PAPER_RUNS_DIRNAME = "paper_runs"

DEFAULT_EXCLUDE_CATS = ("audio-gen", "image-gen")

# Models whose earlier runs predate fixes that materially change their tool-call
# behaviour. Their pre-cutoff runs are discarded so a stale run can never win
# the "most complete run" selection. Not a general outlier rule: a model that
# is merely slow or verbose (e.g. lfm3) is kept.
DEFAULT_STALE_MODELS = ("qwen3.5:0.8b", "qwen3.5:2b")

# v2.1.10 shipped on 2026-07-31 (git tag v2.1.10). Runs are bucketed by the date
# encoded in their filename, because the trace CSVs carry no version column.
# The cut is clean: every historical run here is <= 2026-07-28 and every current
# one is >= 2026-11-15.
ERA_CUTOFF = date(2026, 7, 31)
ERA_LABELS = {False: r"$<$v2.1.10", True: r"$\geq$v2.1.10"}

# Models dropped from every figure by name. Not a statistical rule: a
# deliberate, reviewable list, each entry with its reason.
#   qwen3.5:0.8b    median 58 tool calls/task in its newest run (field median
#                   4) -- it does not terminate, so its call counts say
#                   nothing about the tool-call / score relationship.
#   gemma4:31b-mlx  quantized MLX build scoring 41% against 92% for the same
#                   model served as gemma4:31b-cloud -- a deployment artifact,
#                   not a capability measurement.
#   ornith:9b       Jul 13 run scoring 57%, predating this week's baseline.
#   ornith:35b      Jul 19 run; the 97% result is not from the current build.
#   lfm3            median 41 tool calls/task, same non-convergence pattern.
DEFAULT_EXCLUDE_MODELS = (
    "qwen3.5:0.8b",
    "gemma4:31b-mlx",
    "ornith:9b",
    "ornith:35b",
    "lfm3",
)

# Marker written by local_runner between attempts and the conversation transcript.
TRACE_MARKER = "---TRACE---"


# --------------------------------------------------------------------------- #
# Model name / parameter parsing
# --------------------------------------------------------------------------- #

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bBtT])\b")


def parse_params_b(raw):
    """'0.8b' -> 0.8, '428b (23b active)' -> 428.0, '1t (32b active)' -> 1000.0.

    Always takes the *first* size token, which is the total parameter count in
    every naming convention we care about ('816b total / 42b active' -> 816).
    Returns nan when nothing parses.
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)) and not (
        isinstance(raw, float) and math.isnan(raw)
    ):
        return float(raw)
    s = str(raw).strip()
    if not s or s.lower() == "unknown":
        return float("nan")
    m = _SIZE_RE.search(s)
    if not m:
        return float("nan")
    val, unit = float(m.group(1)), m.group(2).lower()
    return val * 1000.0 if unit == "t" else val


def safe_model_slug(model):
    """Inverse of local_runner's `model.replace('/', '_').replace(':', '_')`."""
    return str(model).replace("/", "_").replace(":", "_")


def resolve_model_name(filename_stem, known_models=()):
    """Recover the model id from a runner CSV stem.

    Runner filenames look like ``npcsh_ollama_qwen3.5_4b_20260307_203901``. The
    ``:`` in ``qwen3.5:4b`` was flattened to ``_``, so we prefer an exact match
    against a known model list, then fall back to a heuristics that re-inserts
    the tag separator before a trailing size token.
    """
    stem = str(filename_stem)
    # Strip .csv if the caller passed a filename.
    if stem.lower().endswith(".csv"):
        stem = stem[:-4]
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)  # timestamp
    stem = re.sub(r"_running$", "", stem)  # checkpoint files
    for prefix in ("npcsh_", "npcsh-local_", "npc_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    provider = None
    for prov in (
        "ollama",
        "llamacpp",
        "anthropic",
        "openai",
        "gemini",
        "deepseek",
        "moonshot",
        "openrouter",
        "perplexity",
        "lmstudio",
    ):
        if stem.startswith(prov + "_"):
            provider = prov
            stem = stem[len(prov) + 1 :]
            break

    # 1) exact match against a known model id, comparing on the flattened form
    lowered = stem.lower()
    for known in known_models:
        if safe_model_slug(known).lower() == lowered:
            return str(known), provider

    # 2) registry match allowing a provider-ish suffix (e.g. ':latest' dropped)
    for known in known_models:
        slug = safe_model_slug(known).lower()
        if lowered.startswith(slug):
            return str(known), provider

    # 3) heuristic: '<base>_<tag>' -> '<base>:<tag>'. The tag can be a size
    #    ('4b', '0.8b', '675b-cloud', '31b-mlx'), an expert size ('e2b'), or a
    #    deployment suffix ('cloud', 'latest') as used in the HF export.
    m = re.match(
        r"^(?P<base>.+?)_(?P<tag>"
        r"\d+(?:\.\d+)?[bBtT](?:[-_][A-Za-z0-9_.\-]+)*"
        r"|e\d+b"
        r"|cloud|latest"
        r")$",
        stem,
    )
    if m:
        return f"{m.group('base')}:{m.group('tag').replace('_', '-')}", provider
    return stem, provider


# --------------------------------------------------------------------------- #
# Datacore loading
# --------------------------------------------------------------------------- #


# Two transcript formats exist in the wild:
#   <= 2.1.1x  "[tool]" / "[assistant]" role markers inside a ---TRACE--- block
#   >= 2.1.2x  ANSI-coloured stdout, one "\u26a1 <tool>:" line per tool call, no markers
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOLT = "\u26a1"  # rendered before every tool invocation in newer transcripts


def _count_trace(blob):
    """Return (n_tool_calls, n_assistant_turns_or_nan) for one `output` cell.

    Turns come back NaN when the transcript has no role markers, because the
    newer format has no way to distinguish an assistant message from its tool
    call; callers then fall back to a tool-calls-per-attempt basis instead of
    reporting a meaningless ratio of 1.0.
    """
    text = _ANSI_RE.sub("", str(blob) if blob is not None else "")
    if TRACE_MARKER in text:
        text = text.split(TRACE_MARKER)[-1]

    roles = re.findall(r"\[(system|user|assistant|tool)(?:[^\]]*)\]", text)
    role_tools = roles.count("tool")
    bolt_tools = text.count(_BOLT)
    tools = max(role_tools, bolt_tools)

    if roles:
        turns = roles.count("assistant")
    else:
        turns = np.nan
    return tools, turns


def _count_attempts(blob):
    """`attempts` fallback: number of `[attempt N]` markers, min 1."""
    # Each retry's stdout is joined with " ||| ", so every "[attempt N]" marker
    # in the cell belongs to this task and must all be counted.
    text = _ANSI_RE.sub("", str(blob) if blob is not None else "")
    if TRACE_MARKER in text:
        text = text.split(TRACE_MARKER)[-1]
    n = len(re.findall(r"\[attempt\s+\d+\]", text))
    return max(n, 1)


_RUN_TS_RE = re.compile(r"_(\d{8})_\d{6}\.csv$")


def run_date(path):
    """Date encoded in a runner filename (`..._YYYYMMDD_HHMMSS.csv`), or None."""
    m = _RUN_TS_RE.search(str(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def era_of(d):
    """True when a run postdates the v2.1.10 release, else False."""
    if d is None:
        return True
    return d >= ERA_CUTOFF


def load_run_csv(path, model=None, known_models=()):
    """Read one runner CSV into a normalized per-task frame."""
    df = pd.read_csv(path, dtype={"error": str, "output": str})
    if df.empty:
        return pd.DataFrame()

    if model is None:
        if "model" in df.columns and df["model"].notna().any():
            model = str(df["model"].dropna().iloc[0])
        else:
            model, _ = resolve_model_name(Path(path).name, known_models)
    provider = None
    if "provider" in df.columns and df["provider"].notna().any():
        provider = str(df["provider"].dropna().iloc[0])
    if provider is None:
        _, provider = resolve_model_name(Path(path).name, known_models)

    out = pd.DataFrame(
        {
            "model": model,
            "provider": provider or "unknown",
            "task_id": df["task_id"].astype(str),
            "category": df["category"].astype(str) if "category" in df else "unknown",
            "difficulty": df["difficulty"].astype(str)
            if "difficulty" in df
            else "unknown",
            "duration": pd.to_numeric(df.get("duration"), errors="coerce"),
        }
    )

    if "passed" in df:
        out["passed"] = (
            df["passed"]
            .map(
                lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes", "pass")
            )
            .astype(bool)
        )
    else:
        out["passed"] = False

    # The `attempts` column is often present but empty (NaN) in newer exports,
    # so blank cells are recovered from the "[attempt N]" transcript markers.
    attempt_col = (
        "attempts"
        if "attempts" in df
        else ("n_attempts" if "n_attempts" in df else None)
    )
    parsed_attempts = (
        df["output"].map(_count_attempts).astype(int)
        if "output" in df
        else pd.Series(1, index=df.index)
    )
    if attempt_col is not None:
        col = pd.to_numeric(df[attempt_col], errors="coerce")
        out["n_attempts"] = col.fillna(parsed_attempts).fillna(1).astype(int)
    else:
        out["n_attempts"] = parsed_attempts

    if "n_tool_calls" in df:
        out["n_tool_calls"] = (
            pd.to_numeric(df["n_tool_calls"], errors="coerce").fillna(0).astype(int)
        )
    elif "output" in df:
        counts = df["output"].map(_count_trace)
        out["n_tool_calls"] = [c[0] for c in counts]
        out["n_turns"] = [c[1] for c in counts]
    else:
        out["n_tool_calls"] = 0

    if "n_turns" not in out:
        out["n_turns"] = np.nan
    return out


def _dedupe_keep_best(long, run_col="run_file"):
    """A model may have several run files. Keep the most complete run *per model*.

    Selection is per model, not global, so a model benchmarked on a 60-task
    subset survives when another model was run on all 100 tasks. Earlier partial
    or checkpoint runs would otherwise drag a model's score down.
    """
    if long.empty or run_col not in long.columns:
        return long

    keys = ["model", run_col]
    if "era" in long.columns:
        # Keep one run per model *per version bucket*. Grouping on model alone
        # would discard a model's July run whenever it also has a November one,
        # which is exactly the comparison the version split needs.
        keys = ["model", "era", run_col]

    counts = long.groupby(keys)["task_id"].nunique().reset_index()
    # Most unique tasks wins; ties break toward the lexically last filename,
    # which sorts by the %Y%m%d_%H%M%S timestamp suffix.
    group_cols = keys[:-1]
    counts = counts.sort_values(keys + ["task_id"])
    chosen = counts.groupby(group_cols).tail(1)
    if len(group_cols) == 1:
        sel = chosen.set_index("model")[run_col]
        kept = long[long[run_col] == long["model"].map(sel)]
    else:
        pair = set(zip(chosen["model"], chosen[run_col]))
        kept = long[[(m, r) in pair for m, r in zip(long["model"], long[run_col])]]

    kept = kept.drop_duplicates(subset=keys + ["task_id"], keep="last")
    return kept


def find_datacore_files(datacore, recursive=True):
    """Bucket datacore CSVs into (runs, trace, meta)."""
    root = Path(datacore).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"datacore not found: {root}")

    if root.is_file():
        # A single long-format CSV is treated as the per-task source.
        return {"dir": root.parent, "runs": [root], "trace": None, "meta": None}

    globber = root.rglob if recursive else root.glob
    runs, trace, meta = [], None, None
    meta_candidates = []
    for csv in sorted(globber("*.csv")):
        name = csv.name.lower()
        # paper_runs/ holds the paper's own history. It is loaded separately as
        # pre-release data, and must not be mistaken for this datacore's current
        # trace file or metadata -- otherwise its trace_analysis.csv would
        # short-circuit the run CSVs entirely.
        if _PAPER_RUNS_DIRNAME in csv.parts:
            continue
        if name == "trace.csv":
            trace = csv
        elif name in (
            "model_meta.csv",
            "models.csv",
            "model_registry.csv",
            "external_benchmarks.csv",
        ):
            meta_candidates.append(csv)
        else:
            runs.append(csv)

    # Prefer model_meta.csv over the other accepted names; it is the canonical
    # registry and the only one that may carry a country column.
    for preferred in (
        "model_meta.csv",
        "models.csv",
        "model_registry.csv",
        "external_benchmarks.csv",
    ):
        for cand in meta_candidates:
            if cand.name.lower() == preferred:
                meta = cand
                break
        if meta is not None:
            break

    return {"dir": root, "runs": runs, "trace": trace, "meta": meta}


def load_model_meta(path):
    if path is None or not Path(path).exists():
        return pd.DataFrame(
            columns=[
                "model",
                "family",
                "params_b",
                "maker",
                "country",
                "mmlu",
                "mmlu_pro",
            ]
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "model" not in df.columns:
        raise ValueError(f"{path}: model_meta.csv needs a 'model' column")
    if "params" in df.columns and "params_b" not in df.columns:
        df["params_b"] = df["params"].map(parse_params_b)
    if "params_b" in df.columns:
        df["params_b"] = df["params_b"].map(parse_params_b)
    else:
        df["params_b"] = df["model"].map(parse_params_b)
    if "params_raw" not in df.columns and "params" in df.columns:
        df["params_raw"] = df["params"].astype(str)
    for col in ("family", "maker", "country", "mmlu", "mmlu_pro"):
        if col not in df.columns:
            df[col] = np.nan
    # Blank CSV cells must become NaN, not the empty string.
    for col in ("mmlu", "mmlu_pro"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["model", "family", "params_b", "maker", "country", "mmlu", "mmlu_pro"]
    if "params_raw" in df.columns:
        keep.append("params_raw")
    return df[keep].drop_duplicates("model")


# Canonical family labels for models whose tag does not name a family, plus the
# spellings that should collapse onto the paper's family names.
_FAMILY_CANON = {
    # Generation-agnostic families. Splitting Qwen3.5 from Qwen3.6 (or Gemma 3
    # from Gemma 4) made the legend illegible and implied a distinction the
    # benchmark does not make, so each maker's line is one family here.
    "qwen": "Qwen",
    "qwen2": "Qwen",
    "qwen2.5": "Qwen",
    "qwen3": "Qwen",
    "qwen3.5": "Qwen",
    "qwen3.6": "Qwen",
    "qwen3.8": "Qwen",
    "gemma": "Gemma",
    "gemma2": "Gemma",
    "gemma3": "Gemma",
    "gemma4": "Gemma",
    "granite": "Granite",
    "granite4.1": "Granite",
    "glm": "GLM",
    "glm4": "GLM",
    "glm-4.7-flash": "GLM",
    "glm-4.6v-flash": "GLM",
    "gpt-oss": "GPT-OSS",
    "gptoss": "GPT-OSS",
    "phi": "Phi",
    "phi4": "Phi",
    "phi-4": "Phi",
    "olmo2": "OLMo2",
    "olmo": "OLMo2",
    "llama": "Llama",
    "llama3": "Llama",
    "llama3.1": "Llama",
    "llama3.2": "Llama",
    "mistral": "Mistral",
    "mistral-small3.2": "Mistral",
    "ministral-3": "Mistral",
    "ministral": "Mistral",
    "devstral": "Mistral",
    "ornith": "Ornith",
    "nemotron": "Nemotron",
    "laguna": "Laguna",
    "lfm3": "LFM",
    "lfm": "LFM",
    "mimo-v2.6": "MiMo",
    "mimo": "MiMo",
    "inkling": "Inkling",
    "beam": "Beam",
    "bonsai": "Bonsai",
    "muse": "Muse",
    "kimi": "Kimi",
    "minimax": "Minimax",
}


# Country of origin by maker, as it appears in model_meta.csv. Checked against
# company sites / model cards rather than inferred from the model name.
#
# Two entries are worth noting because the metadata labels are misleading:
#   "North AI Labs" is Cohere's Northern family (Canada), not a US lab.
#   "Muse" is Meta's open-weight line (USA).
_MAKER_COUNTRY = {
    "Alibaba": "China",
    "Allen Institute for AI": "USA",
    "Cohere": "Canada",
    "DeepSeek": "China",
    "Google": "USA",
    "IBM": "USA",
    "Liquid AI": "USA",
    "Meta": "USA",
    "Microsoft": "USA",
    "Minimax": "China",
    "Mistral AI": "France",
    "Moonshot AI": "China",
    "NVIDIA": "USA",
    "OpenAI": "USA",
    "Ornith": "China",
    "Poolside": "USA",
    "Prism ML": "USA",
    "Reflection AI": "USA",
    "Thinking Machines": "USA",
    "Xiaomi": "China",
    "Zhipu AI": "China",
}

# True maker for each canonical family, used when the registry has no maker.
# The paper's external_benchmarks.csv carries model+family only, so every
# historical row arrives without a maker and needs this to be attributed at all.
# This maps family -> *company*; the country is then looked up in
# _MAKER_COUNTRY, so a model's origin never depends on guessing from its name.
_FAMILY_MAKER = {
    "Qwen": "Alibaba",
    "GLM": "Zhipu AI",
    "Kimi": "Moonshot AI",
    "Minimax": "Minimax",
    "MiMo": "Xiaomi",
    "DeepSeek": "DeepSeek",
    "Ornith": "Ornith",
    "Gemma": "Google",
    "Granite": "IBM",
    "Nemotron": "NVIDIA",
    "Bonsai": "Prism ML",
    "Muse": "Meta",
    "Beam": "Reflection AI",
    "Laguna": "Poolside",
    "LFM": "Liquid AI",
    "Inkling": "Thinking Machines",
    "GPT-OSS": "OpenAI",
    "Phi": "Microsoft",
    "OLMo2": "Allen Institute for AI",
    "Llama": "Meta",
    "Mistral": "Mistral AI",
    "North Mini Code": "Cohere",
}


# model_meta.csv predates this table and some rows name the product line rather
# than the company. Map those onto the real maker so they still resolve.
_MAKER_ALIAS = {
    "Muse": "Meta",
    "North AI Labs": "Cohere",
}


def infer_maker(maker, family):
    """Resolve a maker, falling back to the family's true maker."""
    if pd.notna(maker) and str(maker).strip():
        return _MAKER_ALIAS.get(str(maker).strip(), str(maker).strip())
    return _FAMILY_MAKER.get(str(family), np.nan)


def infer_country(maker, family):
    """Country of origin for a model, derived from its maker.

    Country is never read off the model name or the family label: a family maps
    to a company, and the company maps to a country. That keeps things like
    Phi -> Microsoft -> USA explicit instead of accidental.
    """
    country = _MAKER_COUNTRY.get(str(maker)) if pd.notna(maker) else None
    if country is None:
        resolved = infer_maker(maker, family)
        country = _MAKER_COUNTRY.get(str(resolved)) if pd.notna(resolved) else None
    return country if country else np.nan


def infer_family(model):
    """Fallback family key when the registry has no entry.

    'qwen3.5:4b' -> 'Qwen3.5', 'glm-4.7-flash' -> 'GLM', 'llama3.1:8b' -> 'Llama'.
    """
    stem = str(model).split(":")[0].split("/")[-1].lower()
    # Strip quantization / instruction suffixes that are not part of the family.
    stem = re.sub(r"-(?:q\d\w*|gguf|instruct|it|chat|latest|preview|cloud)$", "", stem)

    for key in (stem, stem.replace("_", "-")):
        if key in _FAMILY_CANON:
            return _FAMILY_CANON[key]

    # Longest registry prefix wins: 'qwen3.5' before 'qwen3'.
    for key in sorted(_FAMILY_CANON, key=len, reverse=True):
        if stem.startswith(key):
            return _FAMILY_CANON[key]

    base = stem.replace("-", " ").replace("_", " ").strip().title()
    return base or str(model)


def canon_family(label):
    """Fold a resolved family label onto its canonical, generation-agnostic name.

    model_meta.csv pins values like 'Qwen3.5' and 'Gemma 4'; those would defeat
    the consolidation in _FAMILY_CANON unless run through it as well. Anything
    unrecognised passes through unchanged.
    """
    raw = str(label).strip()
    if not raw or raw.lower() == "nan":
        return raw

    def norm(x):
        return re.sub(r"[^a-z0-9]", "", str(x).lower())

    key = norm(raw)
    # Exact match first, against keys normalised the same way ('qwen3.5' and
    # 'Qwen3.5' both reduce to 'qwen35').
    for k, v in _FAMILY_CANON.items():
        if norm(k) == key:
            return v
    # Longest matching prefix wins, so 'qwen3.5' is tried before 'qwen3'.
    for k in sorted(_FAMILY_CANON, key=len, reverse=True):
        if key.startswith(norm(k)):
            return _FAMILY_CANON[k]
    return raw


def load_paper_trace_csvs(path):
    """Load per-task paper-era results from a trace_analysis.csv.

    The paper's ``benchmark_data/`` directory also holds ``trace_analysis.csv``:
    one row per (model, task) with ``passed``, ``n_attempts``, ``n_tool_calls``
    and ``category``. That is richer than the summary CSV and uses the same
    schema as a modern ``trace.csv``, so it is preferred when present -- it
    carries the tool-call and attempt numbers that the summary drops.
    """
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"model", "task_id", "category", "passed"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    out["passed"] = out["passed"].astype(str).str.lower().isin(("true", "1", "yes"))
    for col in ("n_attempts", "n_tool_calls", "duration"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out["n_attempts"] = out["n_attempts"].fillna(1)
    out["n_tool_calls"] = out["n_tool_calls"].fillna(0)
    out["n_turns"] = np.nan
    out["provider"] = "ollama"
    out["era"] = False
    out["source"] = "paper"
    out["run_file"] = Path(path).name
    return out


def load_paper_datacore(path):
    """Load the paper's summary CSV (external_benchmarks.csv) as pre-release rows.

    The paper's 22 models were benchmarked on 115 tasks in Feb-Mar 2026, i.e.
    long before v2.1.10. They are the historical baseline the version split is
    meant to show, and they include the very low scores (olmo2:7b 5%, qwen3:0.6b
    4%) that the current 100-task set no longer produces.

    Returns a per-model frame in this script's summary schema.
    """
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"model", "npcsh_score", "npcsh_total"}
    if not need.issubset(df.columns):
        print(f"  skip paper data: {Path(path).name} lacks {need - set(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "model": df["model"].astype(str),
            "score_pct": 100.0 * df["npcsh_score"] / df["npcsh_total"],
            "passed": df["npcsh_score"].astype(int),
            "n_tasks": df["npcsh_total"].astype(int),
            "family": df.get("family", pd.Series(dtype=str)),
            "params_b": df["params_b"].map(parse_params_b)
            if "params_b" in df
            else df["model"].map(parse_params_b),
        }
    )
    out["family"] = out["family"].map(canon_family)
    # The paper CSV carries no maker, so country has to come from the family.
    # Without this every historical row would be "unknown origin" and the
    # country panel would only describe the current release.
    out["maker"] = [
        infer_maker(mk, fam)
        for mk, fam in zip(
            df["maker"] if "maker" in df.columns else [np.nan] * len(out),
            out["family"],
        )
    ]
    out["country"] = [
        infer_country(mk, fam) for mk, fam in zip(out["maker"], out["family"])
    ]
    out["era"] = False
    out["source"] = "paper"
    return out


def load_datacore(
    datacore,
    exclude_cats=DEFAULT_EXCLUDE_CATS,
    meta_path=None,
    stale_before=None,
    stale_models=DEFAULT_STALE_MODELS,
    exclude_models=DEFAULT_EXCLUDE_MODELS,
    paper_path=None,
    include_paper_tasks=False,
):
    """Load a datacore into (long per-task frame, per-model summary frame).

    `stale_before` (a date) discards runs dated earlier than it, but only for
    the models listed in `stale_models`. This keeps a stale run from winning
    the per-model "most complete run" selection while leaving every other
    model's history untouched.
    """
    found = find_datacore_files(datacore)

    meta = load_model_meta(meta_path or found["meta"])
    known = meta["model"].tolist() if not meta.empty else []

    # A long-format CSV (model column present) is the simplest case.
    if found["trace"] is not None:
        trace = pd.read_csv(found["trace"])
        trace.columns = [c.strip().lower() for c in trace.columns]
        long = _from_trace_table(trace)
        if not long.empty:
            print(
                f"  loaded {len(long)} per-task rows from {Path(found['trace']).name}"
            )
            return _finalize(long, meta, exclude_cats, exclude_models)

    frames = []
    for path in found["runs"]:
        try:
            frame = load_run_csv(path, known_models=known)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {Path(path).name}: {exc}", file=sys.stderr)
            continue
        if frame.empty:
            continue
        # Ignore checkpoint/partial files: they carry no signal about final score.
        if "_running" in Path(path).name:
            print(f"  skip checkpoint {Path(path).name}", file=sys.stderr)
            continue
        frame["run_file"] = Path(path).name
        frame["era"] = era_of(run_date(path))

        if stale_before is not None and stale_models:
            rd = run_date(path)
            model_name = str(frame["model"].iloc[0]) if not frame.empty else None
            if rd is not None and rd < stale_before and model_name in stale_models:
                print(
                    f"  skip stale {Path(path).name}: {rd} < {stale_before} "
                    f"for {model_name}",
                    file=sys.stderr,
                )
                continue

        frames.append(frame)
        print(f"  loaded {Path(path).name}: {len(frame)} tasks")

    if not frames:
        raise SystemExit(
            f"no usable run CSVs under {Path(datacore)}\n"
            "Run the benchmark first (npcsh/benchmark/local_runner.py) or use --demo."
        )

    long = pd.concat(frames, ignore_index=True)

    # Paper-era per-task rows, if a trace_analysis.csv sits next to the summary.
    # These join `long` so both the version split and the retry/heatmap figures
    # see the historical runs, not just the summary scores.
    paper_long = pd.DataFrame()
    # A paper_runs/ directory inside the datacore wins: it travels with the
    # dataset, so a fresh clone reproduces the historical rows without needing
    # the paper checkout or the vendored copy.
    in_datacore = Path(datacore) / _PAPER_RUNS_DIRNAME
    if in_datacore.is_dir():
        paper_dir = in_datacore
        cand_csv = in_datacore / "external_benchmarks.csv"
        if cand_csv.exists():
            paper_path = str(cand_csv)
    else:
        paper_dir = Path(paper_path).parent if paper_path else None
    if include_paper_tasks and paper_dir is not None and paper_dir.exists():
        trace_cand = paper_dir / "trace_analysis.csv"
        paper_long = load_paper_trace_csvs(trace_cand)
        if not paper_long.empty:
            print(
                f"  loaded {len(paper_long)} per-task rows from "
                f"{trace_cand.parent.name}/{trace_cand.name} "
                f"({paper_long['model'].nunique()} models)"
            )

    if not paper_long.empty and include_paper_tasks:
        # trace_analysis.csv carries no parameter counts or families. Add those
        # models to the metadata lookup rather than to `long`, so there is only
        # one merge path into the per-task frame -- adding the columns to
        # `paper_long` instead would collide with the metadata merge inside
        # _finalize and split them into family_x / family_y.
        paper_summary = load_paper_datacore(paper_path)
        if not paper_summary.empty:
            add = paper_summary[
                [
                    c
                    for c in ("model", "family", "params_b", "maker", "country")
                    if c in paper_summary.columns
                ]
            ].drop_duplicates("model")
            if meta.empty:
                meta = add
            else:
                add = add[~add["model"].isin(set(meta["model"]))]
                meta = pd.concat([meta, add], ignore_index=True)
        # Opt-in: merging the historical per-task rows widens Figs 3-6 to models
        # that have no modern run, which changes their regressions. Off by
        # default so the published figures stay reproducible; on when you want
        # the full history in every panel.
        long = pd.concat([long, paper_long], ignore_index=True)

    long = _dedupe_keep_best(long)
    long, summary = _finalize(long, meta, exclude_cats, exclude_models)

    # Only fold the summary in when the per-task rows were not merged, otherwise
    # those models would be counted twice (once from `long`, once from the CSV).
    if not include_paper_tasks:
        paper = load_paper_datacore(paper_path)
        if not paper.empty:
            if exclude_models:
                paper = paper[~paper["model"].isin(exclude_models)]
            summary["source"] = "runs"
            summary = pd.concat([summary, paper], ignore_index=True)
            print(f"  added {len(paper)} pre-v2.1.10 rows from {Path(paper_path).name}")
    else:
        print("  paper per-task rows merged into the datacore (history in all figures)")

    return long, summary


def _from_trace_table(trace):
    """Normalize a long/trace-shaped table (model,task_id,category,...)."""
    colmap = {
        "n_tool_calls": "n_tool_calls",
        "tool_calls": "n_tool_calls",
        "n_attempts": "n_attempts",
        "attempts": "n_attempts",
        "n_iterations": "n_turns",
    }
    df = trace.rename(columns={k: v for k, v in colmap.items() if k in trace.columns})
    if "model" not in df.columns or "task_id" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "model": df["model"].astype(str),
            "provider": df["provider"].astype(str) if "provider" in df else "unknown",
            "task_id": df["task_id"].astype(str),
            "category": df["category"].astype(str) if "category" in df else "unknown",
            "difficulty": df["difficulty"].astype(str)
            if "difficulty" in df
            else "unknown",
            "duration": pd.to_numeric(df.get("duration"), errors="coerce"),
            "passed": df["passed"]
            .map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
            .astype(bool)
            if "passed" in df
            else False,
        }
    )
    out["n_attempts"] = (
        pd.to_numeric(df["n_attempts"], errors="coerce").fillna(1).astype(int)
        if "n_attempts" in df
        else 1
    )
    out["n_tool_calls"] = (
        pd.to_numeric(df["n_tool_calls"], errors="coerce").fillna(0).astype(int)
        if "n_tool_calls" in df
        else 0
    )
    out["n_turns"] = (
        pd.to_numeric(df["n_turns"], errors="coerce") if "n_turns" in df else np.nan
    )
    return out


def _drop_partial_models(long, min_fraction=0.5):
    """Drop models whose only run is a tiny fragment (crashed/interrupted).

    Several exports contain 1-task stub files sitting next to full 100-task
    runs. Keeping them would show a real model as 0%. A model is kept when its
    best run covers at least `min_fraction` of the modal task count.
    """
    if long.empty:
        return long, []
    per_model = long.groupby("model")["task_id"].nunique()
    mode_n = per_model.mode().iloc[0] if not per_model.empty else 0
    if not mode_n:
        return long, []
    threshold = max(1, int(mode_n * min_fraction))
    dropped = sorted(per_model[per_model < threshold].index)
    if not dropped:
        return long, []
    kept = long[~long["model"].isin(dropped)]
    for m in dropped:
        print(
            f"  drop {m}: best run has {per_model[m]}/{mode_n} tasks "
            f"(< {threshold}); treating as an incomplete run",
            file=sys.stderr,
        )
    return kept, dropped


def _finalize(long, meta, exclude_cats, exclude_models=()):
    long, _ = _drop_partial_models(long)
    if long.empty:
        raise SystemExit("every run was too small to be a complete benchmark run")

    if exclude_models:
        # Snapshot BEFORE filtering, otherwise the "missing" check below sees
        # the models it just removed and reports every one as absent.
        present = set(long["model"])
        dropped = [m for m in exclude_models if m in present]
        missing = [m for m in exclude_models if m not in present]
        if dropped:
            long = long[~long["model"].isin(dropped)]
            print(f"  excluded by name: {', '.join(dropped)}", file=sys.stderr)
        if missing:
            print(
                f"  note: --exclude-models names not in data: {', '.join(missing)}",
                file=sys.stderr,
            )
        if long.empty:
            raise SystemExit("every model was excluded; check --exclude-models")

    if exclude_cats:
        before = len(long)
        long = long[~long["category"].isin(exclude_cats)]
        if len(long) != before:
            print(
                f"  excluded categories {', '.join(exclude_cats)} ({before - len(long)} rows)"
            )

    if long.empty:
        raise SystemExit("every row was filtered out; check --exclude-cats")

    # Join model metadata, defaulting gracefully.
    if not meta.empty:
        long = long.merge(meta, on="model", how="left")
    else:
        long["family"] = np.nan
        long["params_b"] = np.nan
        long["maker"] = np.nan
        long["country"] = np.nan
        long["mmlu"] = np.nan
        long["mmlu_pro"] = np.nan
    if "params_raw" not in long.columns:
        long["params_raw"] = np.nan

    if "era" not in long.columns:
        # The trace.csv path has no filename to date, so treat those as current.
        long["era"] = True
    long["era"] = long["era"].astype(bool)

    long["family"] = long["family"].fillna(long["model"].map(infer_family))
    # Fold meta-provided labels onto canonical families so Qwen3.5/3.6/3.8
    # are one legend entry rather than three.
    long["family"] = long["family"].map(canon_family)
    long["params_b"] = long["params_b"].fillna(long["model"].map(parse_params_b))
    if "country" not in long.columns:
        long["country"] = np.nan
    # Resolve a missing maker from the family's real company first, then derive
    # the country from the maker, so the two columns never disagree and no
    # country is read off a model or family name.
    long["maker"] = pd.Series(
        [infer_maker(mk, fam) for mk, fam in zip(long["maker"], long["family"])],
        index=long.index,
        dtype=object,
    )
    derived_country = pd.Series(
        [
            infer_country(maker, fam)
            for maker, fam in zip(long["maker"], long["family"])
        ],
        index=long.index,
        dtype=object,
    )
    long["country"] = long["country"].where(long["country"].notna(), derived_country)
    long["n_turns"] = pd.to_numeric(long["n_turns"], errors="coerce")

    summary = summarize(long)
    return long, summary


def summarize(long):
    """Per-model aggregates used by every figure."""
    n_tasks = len(long)
    if n_tasks == 0:
        raise SystemExit("no rows to summarize")

    def _agg(group):
        total = len(group)
        passed = int(group["passed"].sum())
        first = ((group["passed"]) & (group["n_attempts"] == 1)).sum()
        tools_total = group["n_tool_calls"].sum()
        turns_total = group["n_turns"].sum(min_count=1)  # NaN-safe
        mean_tools = group["n_tool_calls"].mean()

        # Both candidates are computed here; a single basis is chosen globally
        # after the groupby so every model on an axis is directly comparable.
        attempts_total = group["n_attempts"].sum()
        per_turn = (
            tools_total / turns_total if turns_total and turns_total > 0 else np.nan
        )
        per_attempt = tools_total / attempts_total if attempts_total else np.nan

        return pd.Series(
            {
                "n_tasks": total,
                "passed": passed,
                "failed": total - passed,
                "score_pct": 100.0 * passed / total if total else np.nan,
                "first_attempt_pct": 100.0 * first / total if total else np.nan,
                "retry_gain": 100.0 * (passed - first) / total if total else np.nan,
                "mean_tool_calls": mean_tools,
                "tool_calls_per_turn": per_turn,
                "tool_calls_per_attempt": per_attempt,
                "mean_attempts": group["n_attempts"].mean(),
                "mean_duration": group["duration"].mean(),
                "pass_tool_calls": group.loc[group["passed"], "n_tool_calls"].mean(),
                "fail_tool_calls": group.loc[~group["passed"], "n_tool_calls"].mean(),
            }
        )

    # Group by version bucket too when one is present: a model can appear in
    # both eras and each needs its own summary row for the split plot.
    group_keys = ["model", "era"] if "era" in long.columns else ["model"]

    summary = (
        long.groupby(group_keys, dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    # Carry model-level attributes through.
    attrs = (
        long.groupby(group_keys, dropna=False)
        .agg(
            family=("family", "first"),
            params_b=("params_b", "first"),
            params_raw=("params_raw", "first"),
            maker=("maker", "first"),
            country=("country", "first"),
            mmlu=("mmlu", "first"),
            mmlu_pro=("mmlu_pro", "first"),
            provider=("provider", "first"),
        )
        .reset_index()
    )
    summary = summary.merge(attrs, on=group_keys, how="left")
    summary = summary.sort_values("score_pct", ascending=False).reset_index(drop=True)

    # Pick one turn-rate basis for the whole figure. Older trace formats logged
    # n_iterations=0, so if any model lacks turn data we fall back to tool calls
    # per attempt for *every* model rather than mixing the two on one axis.
    has_turns = summary["tool_calls_per_turn"].notna().all()
    if has_turns:
        summary["tool_rate"] = summary["tool_calls_per_turn"]
        summary.attrs["tool_rate_basis"] = "turn"
    else:
        summary["tool_rate"] = summary["tool_calls_per_attempt"]
        summary.attrs["tool_rate_basis"] = "attempt"
        n_missing = int(summary["tool_calls_per_turn"].isna().sum())
        print(
            f"  note: {n_missing}/{len(summary)} models have no turn counts; "
            f"using tool calls per attempt for all models"
        )
    return summary


# --------------------------------------------------------------------------- #
# Plot helpers -- mirroring the paper's make_plots.py
# --------------------------------------------------------------------------- #

# Exact per-figure styling taken from make_plots.py: axis-label size, tick-label
# size, legend size/placement. The paper sets these per axes, not globally, so
# we do the same rather than trying to bake them into rcParams.
STYLE = {
    "scaling": {"lab": 14, "tick": 11, "leg": 7, "figsize": (8.4, 4.0)},
    "mmlu": {"lab": 14, "tick": 11, "leg": 8, "figsize": (5.5, 5.5)},
    "toolcalls": {"lab": 12, "tick": 11, "leg": 7, "figsize": (5.5, 4.5)},
    "retry": {"lab": 12, "tick": 9, "leg": 9, "figsize": (6.0, 4.5)},
    "heatmap": {"lab": 10, "tick": 8, "leg": 8, "figsize": (10.0, 5.0)},
    "scaling_versions": {"lab": 14, "tick": 11, "leg": 7, "figsize": (11.0, 3.9)},
    "scaling_country": {"lab": 12, "tick": 10, "leg": 7, "figsize": (7.6, 5.2)},
}

# The paper hardcodes xlim=(0.4, 50) and ticks 0.5..32 because its largest model
# is 35B. This datacore reaches 1T, so those limits would clip half the points.
# Pick ticks from the data instead, keeping the paper's 0.5/1/2/4/8 cadence.
_POW_CANDIDATES = (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def _param_axis(ax, params):
    """Log x-axis scaled to the data, with paper-style decade ticks."""
    p = pd.Series(params).dropna()
    if p.empty:
        return
    lo, hi = float(p.min()), float(p.max())
    pad_lo = max(0.2, lo / 2.0)
    pad_hi = hi * 1.8
    ax.set_xscale("log")
    ax.set_xlim(pad_lo, pad_hi)
    ticks = [t for t in _POW_CANDIDATES if pad_lo <= t <= pad_hi]
    if len(ticks) >= 2:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])


def _family_styles(summary):
    """Stable family -> (color, marker).

    The paper uses plt.cm.Set1 and a fixed marker table keyed by family name.
    Reuse that mapping where it applies so colours match the paper for the
    families it covered, and extend it deterministically for the rest.
    """
    paper_markers = {
        "Qwen3.5": "o",
        "Qwen3": "s",
        "GLM": "D",
        "GPT-OSS": "^",
        "Gemma3": "v",
        "Mistral": "P",
        "Llama": "X",
        "Phi": "*",
        "OLMo2": "h",
    }
    families = list(dict.fromkeys(summary["family"].fillna("unknown")))
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(families), 3)))
    extra = ["d", "8", "p", "<", ">", "H", "1", "2", "3", "4"]
    styles, ei = {}, 0
    for i, fam in enumerate(families):
        marker = paper_markers.get(fam)
        if marker is None:
            marker = extra[ei % len(extra)]
            ei += 1
        styles[fam] = (cmap[i % len(cmap)], marker)
    return styles


def _makefig(figsize):
    """The paper's makefig from chroptiks -- create a single-axes figure."""
    try:
        from chroptiks.plotting_utils import makefig

        return makefig(figsize=figsize)
    except Exception:  # noqa: BLE001
        return plt.subplots(figsize=figsize)


def _finish(fig, ax, style, outdir, name, xlabel, ylabel):
    """Apply the paper's label/tick/legend treatment and save."""
    ax.grid(False)
    ax.set_xlabel(xlabel, fontsize=style["lab"])
    ax.set_ylabel(ylabel, fontsize=style["lab"])
    ax.tick_params(labelsize=style["tick"])
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf")


def _legend_lower_right(ax, fontsize, ncol=1):
    """Legend at loc=4 (lower right), with the handle/label order preserved.

    Kept as a named helper so every figure uses the same placement, and so the
    paper's per-figure fontsize still applies.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(fontsize=fontsize, frameon=False, loc=4, ncol=ncol)


def _linfit(x, y):
    """Least-squares fit returning (slope, intercept, r, p, sigma_of_residuals)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0:
        return None
    try:
        from scipy import stats

        res = stats.linregress(x, y)
        slope, intercept, r, p = res.slope, res.intercept, res.rvalue, res.pvalue
    except Exception:  # noqa: BLE001
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r = math.sqrt(max(0.0, 1 - ss_res / ss_tot)) if ss_tot else 0.0
        p = float("nan")
    sigma = float(np.std(y - (slope * x + intercept)))
    return slope, intercept, float(r), float(p), sigma


def _regression(ax, sub, xcol, ycol, styles, style, xlabel):
    """The paper's regression treatment: black fit line, grey 1-sigma band,
    family-coloured markers, model labels with sigma-based bold weight.

    Legend placement is chosen to avoid the points: the paper hardcodes
    loc='lower right' / 'upper left' per figure, which works because its data
    fills a predictable corner. Here we pick the emptiest corner instead, which
    is the specific complaint about legends colliding with the data.
    """
    fit = _linfit(sub[xcol], sub[ycol])
    if fit is not None:
        slope, intercept, r, p, sigma = fit
        xs = sub[xcol].astype(float)
        lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
        pad = 0.05 * (hi - lo or 1.0)
        grid = np.linspace(lo - pad, hi + pad, 100)
        line = slope * grid + intercept
        ax.plot(grid, line, "k-", linewidth=1.5, zorder=1, label=rf"$r={r:.2f}$")
        ax.fill_between(
            grid,
            line - sigma,
            line + sigma,
            color="gray",
            alpha=0.15,
            zorder=0,
            label=rf"$1\sigma = {sigma:.1f}\%$",
        )

    for fam, (color, marker) in styles.items():
        fam_sub = sub[sub["family"] == fam]
        fam_sub = fam_sub[np.isfinite(fam_sub[xcol]) & np.isfinite(fam_sub[ycol])]
        if fam_sub.empty:
            continue
        ax.scatter(
            fam_sub[xcol],
            fam_sub[ycol],
            c=[color],
            marker=marker,
            s=80,
            label=fam,
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )
        for _, row in fam_sub.iterrows():
            if fit is not None:
                delta = row[ycol] - (fit[0] * row[xcol] + fit[1])
                weight = "bold" if abs(delta) > fit[4] else "normal"
            else:
                weight = "normal"
            ax.annotate(
                str(row["model"]),
                (row[xcol], row[ycol]),
                fontsize=6,
                fontweight=weight,
                xytext=(4, 4),
                textcoords="offset points",
            )

    _legend_lower_right(ax, style["leg"])
    return fit


# --------------------------------------------------------------------------- #
# Figures -- one single-axes panel each, as in the paper
# --------------------------------------------------------------------------- #


def _makefig_panels(figsize, ncols=2, sharey=True):
    """Side-by-side panels via the paper's makefig, else plain subplots."""
    try:
        from chroptiks.plotting_utils import makefig

        fig, axes = makefig(shape=(1, ncols), figsize=figsize, sharey=sharey)
        axes = np.atleast_1d(np.asarray(axes)).ravel()
        # makefig tiles with wspace=0, which collides the right panel's y tick
        # labels against the left panel's frame.
        fig.subplots_adjust(wspace=0.10)
        return fig, axes
    except Exception:  # noqa: BLE001
        fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=sharey)
        return fig, np.atleast_1d(np.array(axes)).ravel()


def _finish_panels(fig, axes, style, outdir, name, xlabel, ylabel, handles=None):
    """Label and save a multi-panel figure, sharing one legend."""
    for ax in axes:
        ax.grid(False)
        ax.set_xlabel(xlabel, fontsize=style["lab"])
        ax.tick_params(labelsize=style["tick"])
    axes[0].set_ylabel(ylabel, fontsize=style["lab"])
    if handles:
        fig.legend(
            handles=handles,
            fontsize=style["leg"],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=8,
            handletextpad=0.4,
            columnspacing=1.0,
        )
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf")


def latest_per_model(summary):
    """One summary row per model, preferring the post-v2.1.10 measurement.

    Only Fig. 2 splits by version. The remaining figures each describe a model
    once, so leaving the era-split summary in place would plot the same model
    two or three times (a model with a paper row, a pre-release run and a
    current run appears three times) and distort the regressions.
    """
    if "era" not in summary.columns:
        return summary
    ordered = summary.sort_values("era")  # False before True
    return ordered.groupby("model", as_index=False).tail(1).reset_index(drop=True)


def latest_long(long):
    """Per-task rows for the newest era only, one set per model.

    Figs 5 and 6 aggregate task rows, so leaving both eras in place would count
    three models' 100 tasks twice and shift the per-category pass rates.
    """
    if long.empty or "era" not in long.columns:
        return long
    chosen = (
        long[["model", "era"]]
        .drop_duplicates()
        .sort_values("era")  # False before True
        .groupby("model", as_index=False)
        .tail(1)
    )
    keep = set(zip(chosen["model"], chosen["era"]))
    return long[[(m, e) in keep for m, e in zip(long["model"], long["era"])]]


def latest_pairs(summary):
    """The (model, era) pairs selected by latest_per_model, for filtering `long`."""
    s = latest_per_model(summary)
    return set(zip(s["model"], s["era"]))


def fig_scaling(summary, styles, outdir):
    """Fig. 2: benchmark score versus parameter count, one point per model.

    This is the plain scaling panel: every model plotted individually, marker
    shape and colour encoding its family. It is the version the paper shows and
    the one to read when you want to see the models themselves.
    """
    st = STYLE["scaling"]
    sub = latest_per_model(summary)
    sub = sub[np.isfinite(sub["params_b"])].copy()
    dropped = len(summary) - len(sub)
    if dropped:
        print(f"  note: {dropped} row(s) have no parameter count; omitted")
    if len(sub) < 2:
        print("  skip scaling: fewer than 2 models have a parameter count")
        return

    fig, ax = _makefig(st["figsize"])
    for fam, (color, marker) in styles.items():
        fam_sub = sub[sub["family"] == fam].sort_values("params_b")
        if fam_sub.empty:
            continue
        ax.scatter(
            fam_sub["params_b"],
            fam_sub["score_pct"],
            c=[color],
            marker=marker,
            s=30,
            label=fam,
            zorder=3,
            edgecolors="black",
            linewidths=0.4,
        )
    _param_axis(ax, sub["params_b"])
    ax.set_ylim(0, 105)
    _legend_lower_right(ax, st["leg"], ncol=2)
    _finish(fig, ax, st, outdir, "scaling", r"Parameters (B)", r"Score (%)")
    print(f"    scaling: {len(sub)} models")


def fig_scaling_versions(summary, styles, outdir):
    """Score versus parameters, split into two panels by npcsh version.

    Side by side so the models evaluated in each release can be compared
    directly. Marker shape and colour encode family and are identical across
    panels; only the population differs. Models measured on both sides get the
    signed change annotated beside their point in the newer panel.
    """
    st = STYLE["scaling_versions"]
    sub = summary[np.isfinite(summary["params_b"])].copy()
    if "era" not in sub.columns or sub["era"].nunique() < 2:
        print("  skip scaling_versions: need both a pre and post version bucket")
        return

    # A model can appear in the pre bucket twice: once as a paper summary row
    # (115-task) and once as an actual run (100-task). Plotting both would show
    # the same model at two different scores, so collapse to one row per
    # (model, era). The real run wins, because the post bucket is also 100-task
    # and comparing like with like is the point of this panel.
    if "source" in sub.columns:
        sub = sub.sort_values(
            "source", key=lambda s: s.map({"paper": 0, "runs": 1}).fillna(1)
        )
    sub = sub.drop_duplicates(["model", "era"], keep="last")

    pre = sub[~sub["era"]]
    post = sub[sub["era"]]
    if pre.empty or post.empty:
        print("  skip scaling_versions: one version bucket is empty")
        return

    from matplotlib.lines import Line2D

    fig, axes = _makefig_panels(st["figsize"], ncols=2, sharey=True)
    shared = set(pre["model"]) & set(post["model"])

    for ax, part, label in (
        (axes[0], pre, ERA_LABELS[False]),
        (axes[1], post, ERA_LABELS[True]),
    ):
        for fam, (color, marker) in styles.items():
            fam_sub = part[part["family"] == fam].sort_values("params_b")
            if fam_sub.empty:
                continue
            ax.scatter(
                fam_sub["params_b"],
                fam_sub["score_pct"],
                c=[color],
                marker=marker,
                s=30,
                zorder=3,
                edgecolors="black",
                linewidths=0.4,
            )
        _param_axis(ax, sub["params_b"])
        ax.set_ylim(0, 105)
        ax.set_title(f"{label} ({len(part)} models)", fontsize=st["lab"] - 2)

    # Deltas go on the right, beside the newer number, and only for models that
    # appear on both sides -- otherwise the annotation would imply a change that
    # was never measured.
    for model in sorted(shared):
        a = pre[pre["model"] == model].iloc[0]
        b = post[post["model"] == model].iloc[0]
        axes[1].annotate(
            f"{b['score_pct'] - a['score_pct']:+.0f}",
            (b["params_b"], b["score_pct"]),
            fontsize=st["tick"] - 2,
            color="0.25",
            xytext=(6, -3),
            textcoords="offset points",
            zorder=4,
        )

    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=fam,
        )
        for fam, (color, marker) in styles.items()
        if fam in set(sub["family"])
    ]
    _finish_panels(
        fig,
        axes,
        st,
        outdir,
        "scaling_versions",
        r"Parameters (B)",
        r"Score (%)",
        handles=handles,
    )
    print(
        f"    scaling_versions: {len(pre)} pre vs {len(post)} post, {len(shared)} shared"
    )


def fig_vs_mmlu(summary, styles, outdir):
    """Fig. 3: agency versus intelligence.

    Prefers plain MMLU (what the paper plotted) and falls back to MMLU-Pro,
    which is what 2026 model cards actually publish. The two are different
    scales, so they are never mixed in one panel.
    """
    st = STYLE["mmlu"]
    mmlu_n = int(summary["mmlu"].notna().sum())
    pro_n = int(summary["mmlu_pro"].notna().sum()) if "mmlu_pro" in summary else 0

    if mmlu_n >= 3:
        col, label = "mmlu", r"Intelligence (MMLU %)"
    elif pro_n >= 3:
        col, label = "mmlu_pro", r"Intelligence (MMLU-Pro %)"
        print(f"  note: using MMLU-Pro, which {pro_n} models report")
    else:
        print(
            f"  skip npcsh_vs_mmlu: need >=3 models with MMLU or MMLU-Pro "
            f"(have mmlu={mmlu_n}, mmlu_pro={pro_n})"
        )
        return

    sub = latest_per_model(summary).dropna(subset=[col])
    if len(sub) < 3:
        print(f"  skip npcsh_vs_mmlu: only {len(sub)} models have {col}")
        return

    fig, ax = _makefig(st["figsize"])
    fit = _regression(ax, sub, col, "score_pct", styles, st, label)
    if fit:
        print(
            f"    npcsh_vs_mmlu ({col}): r={fit[2]:.3f} p={fit[3]:.2e} "
            f"sigma={fit[4]:.1f}"
        )
    _finish(
        fig,
        ax,
        st,
        outdir,
        "npcsh_vs_mmlu",
        label,
        r"Agency (npcsh Score %)",
    )


def fig_tool_calls_vs_agency(summary, styles, outdir):
    """Fig. 4: mean tool calls per task versus agency score."""
    st = STYLE["toolcalls"]
    sub = latest_per_model(summary).dropna(subset=["mean_tool_calls", "score_pct"])
    if len(sub) < 3:
        print("  skip tool_calls_vs_agency: fewer than 3 models have tool-call data")
        return

    fig, ax = _makefig(st["figsize"])
    fit = _regression(
        ax,
        sub,
        "mean_tool_calls",
        "score_pct",
        styles,
        st,
        r"Mean tool calls per task",
    )
    # The paper leaves this axis on auto-scale. Scores here span only ~81-98,
    # so a fixed 0-105 range would squash every point into the top fifth and
    # waste most of the panel. Keep auto-scaling and reserve a slice at the
    # bottom for the legend instead, measured to clear every marker.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.25 * (y1 - y0), y1)
    _legend_lower_right(ax, st["leg"], ncol=2)
    if fit:
        print(f"    tool_calls_vs_agency: r={fit[2]:.3f} p={fit[3]:.2e} n={len(sub)}")
    _finish(
        fig,
        ax,
        st,
        outdir,
        "tool_calls_vs_agency",
        r"Mean tool calls per task",
        r"Agency (npcsh Score %)",
    )


def fig_retry_value(long, outdir):
    """Fig. 5: first-attempt pass rate and retry gain, by category."""
    st = STYLE["retry"]
    if "category" not in long:
        return
    rows = []
    for cat, grp in long.groupby("category"):
        first = ((grp["passed"]) & (grp["n_attempts"] == 1)).sum()
        total = len(grp)
        rows.append(
            {
                "category": cat,
                "first_attempt_pct": 100.0 * first / total if total else 0.0,
                "retry_gain": 100.0 * (grp["passed"].sum() - first) / total
                if total
                else 0.0,
            }
        )
    cat_retry = pd.DataFrame(rows).sort_values("retry_gain", ascending=True)
    if cat_retry.empty:
        return

    fig, ax = _makefig(st["figsize"])
    y = np.arange(len(cat_retry))
    ax.barh(
        y,
        cat_retry["first_attempt_pct"],
        height=0.7,
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.4,
        label="First attempt",
        alpha=0.85,
    )
    ax.barh(
        y,
        cat_retry["retry_gain"],
        height=0.7,
        left=cat_retry["first_attempt_pct"],
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.4,
        label="Retry gain",
        alpha=0.85,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(cat_retry["category"], fontsize=8)
    ax.set_xlim(0, 105)
    _legend_lower_right(ax, st["leg"])
    for i, (_, row) in enumerate(cat_retry.iterrows()):
        if row["retry_gain"] > 1:
            ax.text(
                row["first_attempt_pct"] + row["retry_gain"] + 1,
                i,
                rf"+{row['retry_gain']:.0f}%",
                va="center",
                fontsize=6,
                color="#ff7f0e",
            )
    _finish(fig, ax, st, outdir, "retry_value_by_category", r"Pass rate (%)", "")

    passed = long[long["passed"]]
    if len(passed):
        share = 100.0 * (passed["n_attempts"] == 1).sum() / len(passed)
        print(f"    share of successful tasks passed on attempt 1: {share:.1f}%")
    overall = 100.0 * ((long["passed"]) & (long["n_attempts"] == 1)).sum() / len(long)
    print(f"    first-attempt pass rate (all tasks): {overall:.1f}%")


def fig_family_category_heatmap(long, summary, outdir):
    """Fig. 6: pass rate by model and task category."""
    st = STYLE["heatmap"]
    pivot = long.pivot_table(
        index="category", columns="model", values="passed", aggfunc="mean"
    )
    if pivot.empty:
        return
    order = [m for m in summary["model"].tolist() if m in pivot.columns]
    pivot = pivot[order]
    # Categories ordered by mean difficulty, hardest at the top (as in paper).
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    n_models = len(pivot.columns)
    # Paper uses a fixed 10x5 for 22 models; scale the width to model count.
    fig, ax = _makefig((max(7.0, 0.34 * n_models + 3.0), 5.0))
    im = ax.imshow(
        pivot.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1
    )
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.grid(False)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="black" if 0.3 < val < 0.8 else "white",
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"Pass Rate", fontsize=st["lab"])
    _finish(fig, ax, st, outdir, "family_category_heatmap", "", "")


def fig_scaling_country(summary, styles, outdir):
    """Score versus parameters, one point per model, marker keyed by country.

    Country is the *series*, not the x-axis. Plotting it as a categorical axis
    only said which countries were represented; putting parameters back on x
    makes the panel quantitative and comparable to Fig. 2, and lets you see
    whether a country's scores come from its models being larger or from
    something else. Marker colour and shape are keyed by country here; size
    carries no information, so nothing is double-encoded.

    Models with no known country, or no parameter count, are reported and
    omitted rather than dropped into an implied "unknown" bucket.
    """
    from matplotlib.lines import Line2D

    st = STYLE["scaling_country"]
    sub = latest_per_model(summary)
    if "country" not in sub.columns:
        print("  skip scaling_country: no country column in summary")
        return

    unknown = sub[sub["country"].isna()]
    if len(unknown):
        names = ", ".join(sorted(unknown["model"]))
        print(f"  note: {len(unknown)} model(s) have no country of origin: {names}")

    sub = sub[sub["country"].notna()].copy()
    no_params = sub[~np.isfinite(sub["params_b"])]
    sub = sub[np.isfinite(sub["params_b"])].copy()
    if len(no_params):
        names = ", ".join(sorted(no_params["model"]))
        print(f"  note: {len(no_params)} model(s) have no parameter count: {names}")
    if len(sub) < 2:
        print(
            "  skip scaling_country: fewer than 2 models have both a country and a size"
        )
        return

    # Order countries by median score so the legend reads strongest-first.
    order = (
        sub.groupby("country")["score_pct"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Fixed marker per country, independent of the family table used elsewhere,
    # so a country keeps the same symbol from run to run.
    _MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h")
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(order), 3)))
    country_style = {
        c: (cmap[i % len(cmap)], _MARKERS[i % len(_MARKERS)])
        for i, c in enumerate(order)
    }

    # Two different models can land on the exact same (size, score). Nudge only
    # those apart, multiplicatively so the offset looks even on the log axis;
    # every other point is drawn at its true value.
    key = list(
        zip(
            sub["params_b"].round(3),
            sub["score_pct"].round(2),
        )
    )
    counts = pd.Series(key).value_counts()
    seen = {}
    xplot = []
    for k in key:
        n = int(counts[k])
        i = seen.get(k, 0)
        seen[k] = i + 1
        # centre the duplicates: -0.5, +0.5 for a pair, etc.
        shift = 0.0 if n == 1 else (i - (n - 1) / 2) * 0.06
        xplot.append(float(k[0]) * (10**shift))
    sub = sub.assign(_xplot=xplot)

    fig, ax = _makefig(st["figsize"])
    for country in order:
        col = sub[sub["country"] == country].sort_values("params_b")
        color, marker = country_style[country]
        ax.scatter(
            col["_xplot"],
            col["score_pct"],
            c=[color],
            marker=marker,
            s=34,
            zorder=3,
            edgecolors="black",
            linewidths=0.4,
        )

    _param_axis(ax, sub["params_b"])
    ax.set_ylim(0, 105)

    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=country_style[c][1],
            markersize=4.5,
            markerfacecolor=country_style[c][0],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=f"{c} (n={int((sub['country'] == c).sum())})",
        )
        for c in order
    ]
    if handles:
        ax.legend(
            handles=handles,
            fontsize=st["leg"],
            frameon=False,
            loc=4,
            ncol=1,
            handletextpad=0.4,
        )

    _finish(fig, ax, st, outdir, "scaling_country", r"Parameters (B)", r"Score (%)")
    print(f"    scaling_country: {len(sub)} models across {len(order)} countries")


# --------------------------------------------------------------------------- #
# README consistency check (read-only)
# --------------------------------------------------------------------------- #


def parse_readme_table(text):
    """Parse the existing `### Scores` HTML table into ordered row dicts.

    Used only to *compare* the committed README against the datacore. Nothing
    here writes to the file: the README scores are curated per npcsh release
    and are not derivable from an older benchmark datacore.
    """
    idx = text.find("### Scores (")
    if idx == -1:
        return [], None
    heading = text[idx : text.find("\n", idx)].strip()
    m = re.search(r"### Scores \((\d+) tasks\)", heading)
    total_tasks = int(m.group(1)) if m else None

    table_start = text.find("<table>", idx)
    table_end = text.find("</table>", table_start)
    if table_start == -1 or table_end == -1:
        return [], total_tasks

    rows = []
    cur_maker, cur_family = None, None
    for tr in re.findall(r"<tr>(.*?)</tr>", text[table_start:table_end], re.S):
        cells = re.findall(r"<td([^>]*)>(.*?)</td>", tr, re.S)
        if not cells:
            continue  # header row is <th>, not <td>
        vals = [re.sub(r"<[^>]+>", "", v).strip() for _, v in cells]
        n = len(vals)
        if n == 6:
            maker, family, model, params, ver, score = vals
        elif n == 5:
            maker = cur_maker
            family, model, params, ver, score = vals
        elif n == 4:
            maker, family = cur_maker, cur_family
            model, params, ver, score = vals
        else:
            continue
        cur_maker, cur_family = maker, family
        rows.append(
            {
                "maker": maker,
                "family": family,
                "model": model,
                "params": params,
                "version": ver,
                "score": score,
            }
        )
    return rows, total_tasks


def check_readme(readme_path, summary):
    """Compare the committed README table against the datacore. READ-ONLY.

    The README is the release-facing record: its scores come from whatever
    npcsh version was current when those rows were measured. A datacore is a
    snapshot of some (possibly older) set of runs. They are expected to
    disagree, so this only reports the difference and never rewrites anything.

    Returns a list of human-readable findings.
    """
    path = Path(readme_path)
    if not path.exists():
        return [f"readme: {path} not found, skipping consistency check"]

    rows, readme_tasks = parse_readme_table(path.read_text(encoding="utf-8"))
    if not rows:
        return [f"readme: no `### Scores` table found in {path}"]

    datacore_tasks = int(summary["n_tasks"].max()) if not summary.empty else 0
    findings = []

    if readme_tasks is not None and datacore_tasks and readme_tasks != datacore_tasks:
        findings.append(
            f"task-count mismatch: README reports {readme_tasks} tasks, datacore has "
            f"{datacore_tasks}. These measure different suites; figures reflect the "
            f"datacore only."
        )

    readme_versions = sorted({r["version"] for r in rows if r["version"]})
    if readme_versions:
        findings.append(f"README rows were measured on: {', '.join(readme_versions)}")

    # Count how many datacore models even appear in the README, by bare tag.
    datacore_tags = {str(m).split(":")[-1].lower() for m in summary["model"]}
    datacore_tags |= {str(m).lower() for m in summary["model"]}
    overlap = sum(
        1
        for r in rows
        if r["model"].lower() in datacore_tags
        or str(r["params"]).lower() in datacore_tags
    )
    findings.append(
        f"{overlap}/{len(rows)} README rows correspond to a model present in the "
        f"datacore ({len(summary)} models)."
    )
    findings.append(
        "README left unchanged. Update it only when you re-run the benchmark on "
        "the current npcsh release."
    )
    return findings


# --------------------------------------------------------------------------- #
# Meta template + demo
# --------------------------------------------------------------------------- #


def write_meta_template(summary, path):
    """Emit a model_meta.csv skeleton for models that lack metadata."""
    df = summary[["model", "family", "params_b", "maker", "mmlu"]].copy()
    df["maker"] = df["maker"].where(df["maker"].notna(), "Unknown")
    df["mmlu"] = df["mmlu"].where(df["mmlu"].notna(), "")
    df["params_b"] = df["params_b"].map(lambda v: f"{v:g}" if np.isfinite(v) else "")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote meta template -> {path} ({len(df)} models)")


def _demo_datacore(directory, n_families=5, seed=0):
    """Synthetic datacore so the pipeline can be validated without real runs."""
    rng = np.random.default_rng(seed)
    root = Path(directory)
    (root / "runs").mkdir(parents=True, exist_ok=True)

    families = {
        "Qwen3.5": [(0.8, 12), (2, 72), (4, 67), (9, 90), (35, 101)],
        "Qwen3": [(0.6, 5), (1.7, 32), (4, 84), (8, 75), (30, 93)],
        "Gemma3": [(4, 30), (12, 67), (27, 65)],
        "Llama": [(3, 17), (8, 50)],
        "OLMo2": [(7, 6), (13, 37)],
    }
    families = dict(list(families.items())[:n_families])
    cats = [
        "shell",
        "file-ops",
        "python",
        "data",
        "system",
        "text",
        "debug",
        "git",
        "multi-step",
        "scripting",
    ]
    n_tasks = 100
    meta_rows, run_frames = [], []

    for fam, specs in families.items():
        for params, passed_n in specs:
            model = f"{fam.lower().replace('.', '.')}:{params:g}b"
            score = passed_n / 115.0
            rows = []
            for i in range(n_tasks):
                cat = cats[i % len(cats)]
                passed = rng.random() < score
                attempts = (
                    int(rng.integers(1, 2))
                    if passed and rng.random() < 0.8
                    else int(rng.integers(1, 6))
                )
                tools = int(max(1, rng.poisson(1 + 4 * score)))
                turns = int(max(1, rng.poisson(1 + 1.5 * score)))
                rows.append(
                    {
                        "task_id": f"{cat}-{i:02d}",
                        "category": cat,
                        "difficulty": "easy" if i % 3 else "medium",
                        "passed": passed,
                        "duration": float(rng.uniform(5, 300)),
                        "error": "",
                        "output": "",
                        "attempts": attempts,
                        "n_tool_calls": tools,
                        "n_turns": turns,
                    }
                )
            frame = pd.DataFrame(rows)
            run_frames.append(frame.assign(model=model, provider="ollama"))
            meta_rows.append(
                {
                    "model": model,
                    "family": fam,
                    "params_b": params,
                    "maker": "Demo Labs",
                    "mmlu": round(
                        min(95, 45 + 20 * math.log10(max(params, 0.5) * 4 + 1)), 1
                    ),
                }
            )

    for i, frame in enumerate(run_frames):
        frame.drop(columns=["model", "provider"]).to_csv(
            root / "runs" / f"npcsh_ollama_demo_{i:02d}_20260101_120000.csv",
            index=False,
        )
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(root / "model_meta.csv", index=False)
    long = pd.concat(run_frames, ignore_index=True)
    long.to_csv(root / "trace.csv", index=False)
    return root


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Regenerate ALARA/HAXD benchmark figures (Figs. 2-6) from a datacore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----")[-1],
    )
    p.add_argument(
        "--datacore",
        default=os.path.expanduser("~/.npcsh"),
        help="datacore dir with runs/*.csv [+ model_meta.csv, trace.csv] "
        "or a single long-format CSV (default: ~/.npcsh)",
    )
    p.add_argument(
        "--meta",
        default=None,
        help="explicit model_meta.csv (overrides <datacore>/model_meta.csv)",
    )
    p.add_argument(
        "--outdir", default="figures", help="figure output dir (default: figures)"
    )
    p.add_argument(
        "--exclude-cats",
        default=",".join(DEFAULT_EXCLUDE_CATS),
        help="comma-separated categories to drop (default: audio-gen,image-gen)",
    )
    p.add_argument(
        "--exclude-models",
        default=",".join(DEFAULT_EXCLUDE_MODELS),
        help="comma-separated models to drop from all figures. Explicit list, "
        f"no statistics (default: {','.join(DEFAULT_EXCLUDE_MODELS) or 'none'})",
    )
    p.add_argument(
        "--stale-before",
        default=None,
        metavar="YYYY-MM-DD",
        help="discard runs older than this date for --stale-models only. "
        "Defaults to 7 days ago, so a stale run cannot win run selection.",
    )
    p.add_argument(
        "--stale-models",
        default=",".join(DEFAULT_STALE_MODELS),
        help="comma-separated models subject to --stale-before "
        f"(default: {','.join(DEFAULT_STALE_MODELS)})",
    )
    p.add_argument(
        "--paper-data",
        default=str(_PAPER_CSV) if _PAPER_CSV else "",
        help="paper summary CSV (115-task, pre-v2.1.10) folded in as historical "
        "rows for the version split. Pass an empty string to disable.",
    )
    p.add_argument(
        "--include-paper-tasks",
        action="store_true",
        help="merge the paper_runs/ per-task rows into the datacore so Figs 3-6 "
        "cover the historical models too. Off by default, because those models "
        "have no modern run and including them shifts the regressions.",
    )
    p.add_argument(
        "--readme",
        default=None,
        help="README.md to compare against the datacore (read-only; never written). "
        "Default: ./README.md when present.",
    )
    p.add_argument(
        "--write-meta-template",
        default=None,
        metavar="PATH",
        help="write a model_meta.csv skeleton to PATH and exit",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="generate a synthetic datacore and run the full pipeline",
    )
    p.add_argument(
        "--only",
        default=None,
        help="comma-separated figure names to draw "
        "(scaling,scaling_versions,scaling_country,npcsh_vs_mmlu,"
        "tool_calls_vs_agency,retry_value_by_category,"
        "family_category_heatmap)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.demo:
        demo_dir = Path(
            args.datacore
            if os.path.expanduser(args.datacore) != os.path.expanduser("~/.npcsh")
            else tempfile.mkdtemp(prefix="benchcore_")
        )
        if demo_dir.name == "npcsh" or not str(demo_dir):
            demo_dir = Path(tempfile.mkdtemp(prefix="benchcore_"))
        demo_dir = demo_dir / "demo_datacore"
        print(f"demo: building synthetic datacore at {demo_dir}")
        _demo_datacore(demo_dir)
        args.datacore = str(demo_dir)
        if args.outdir == "figures":
            args.outdir = str(demo_dir / "figures")

    exclude = tuple(c.strip() for c in args.exclude_cats.split(",") if c.strip())

    stale_models = tuple(m.strip() for m in args.stale_models.split(",") if m.strip())
    if args.stale_before:
        try:
            stale_before = datetime.strptime(args.stale_before, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit(
                f"--stale-before must be YYYY-MM-DD, got {args.stale_before!r}"
            )
    else:
        stale_before = date.today() - timedelta(days=7)

    print(f"datacore: {args.datacore}")
    if stale_models:
        print(f"  ignoring runs before {stale_before} for: {', '.join(stale_models)}")

    exclude_models = tuple(
        m.strip() for m in args.exclude_models.split(",") if m.strip()
    )

    long, summary = load_datacore(
        args.datacore,
        exclude_cats=exclude,
        meta_path=args.meta,
        stale_before=stale_before,
        stale_models=stale_models,
        exclude_models=exclude_models,
        paper_path=args.paper_data or None,
        include_paper_tasks=args.include_paper_tasks,
    )

    n_models = summary["model"].nunique()
    n_rows = len(summary)
    if n_rows != n_models:
        print(
            f"\n{n_models} models ({n_rows} model/version rows), {len(long)} task rows"
        )
    else:
        print(f"\n{n_models} models, {len(long)} task rows")
    show = ["model", "family", "params_b", "n_tasks", "score_pct", "tool_rate"]
    show = [c for c in show if c in summary.columns]
    print(summary[show].to_string(index=False))

    if args.write_meta_template:
        write_meta_template(summary, args.write_meta_template)
        return 0

    outdir = Path(args.outdir)
    styles = _family_styles(summary)
    wanted = None
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}

    def _want(name):
        return wanted is None or name in wanted

    print("\nfigures:")
    if _want("scaling"):
        fig_scaling(summary, styles, outdir)
    if _want("scaling_versions"):
        fig_scaling_versions(summary, styles, outdir)
    if _want("scaling_country"):
        fig_scaling_country(summary, styles, outdir)
    if _want("npcsh_vs_mmlu"):
        fig_vs_mmlu(summary, styles, outdir)
    if _want("tool_calls_vs_agency"):
        fig_tool_calls_vs_agency(summary, styles, outdir)
    if _want("retry_value_by_category"):
        fig_retry_value(latest_long(long), outdir)
    if _want("family_category_heatmap"):
        fig_family_category_heatmap(
            latest_long(long), latest_per_model(summary), outdir
        )

    readme = Path(args.readme) if args.readme else Path("README.md")
    findings = check_readme(readme, summary)
    print("\nreadme check (read-only):")
    for line in findings:
        print(f"  {line}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
