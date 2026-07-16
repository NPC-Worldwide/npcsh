#!/usr/bin/env python3
"""Run the npcsh 100-task benchmark across all locally-downloaded Ollama models.

This skips any model whose tag ends in `:cloud` (those are routed separately by
`run_ollama_cloud_models.py`), and also skips embeddings / image-only models.
Resume logic uses the same checkpoint/final-CSV naming convention as
`npcsh.benchmark.local_runner`.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Models that are not text-completion LLMs and should never be benchmarked here.
EXCLUDED_MODELS = frozenset({
    "nomic-embed-text:latest",
    "x/z-image-turbo:latest",
})


def _is_excluded(tag: str) -> bool:
    """Skip embed/image models and tinytim HF downloads."""
    if tag in EXCLUDED_MODELS:
        return True
    if tag.startswith("hf.co/"):
        return True
    return False


def _list_local_models() -> list[str]:
    """Return all local Ollama model tags that are not cloud endpoints."""
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("Failed to run `ollama list`:", result.stderr, file=sys.stderr)
        return []

    models = []
    for line in result.stdout.strip().splitlines()[1:]:
        tag = line.split()[0] if line.split() else ""
        if not tag:
            continue
        if tag.endswith(":cloud") or tag.endswith("-cloud"):
            continue
        if _is_excluded(tag):
            continue
        models.append(tag)
    return sorted(models)


def _final_csv_path(model: str, report_dir: Path) -> Path | None:
    """Return the newest CSV local_runner has finished for this model."""
    safe_model = model.replace("/", "_").replace(":", "_")
    final_pattern = re.compile(rf"^npcsh_ollama_{re.escape(safe_model)}_\d{{8}}_\d{{6}}$")
    running_pattern = re.compile(rf"^npcsh_ollama_{re.escape(safe_model)}_running$")
    candidates = [
        p for p in report_dir.glob("*.csv")
        if final_pattern.match(p.stem) or running_pattern.match(p.stem)
    ]
    for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            df = pd.read_csv(p, dtype={"task_id": str})
            if len(df) >= 100:
                return p
        except Exception:
            continue
    return None


def run_model(model: str, timeout: int, python: str) -> str:
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    report_dir.mkdir(parents=True, exist_ok=True)
    log = report_dir / f"local_{model.replace('/', '_').replace(':', '_')}.log"

    final_csv = _final_csv_path(model, report_dir)
    if final_csv is not None:
        msg = f"[SKIP]  {model} - already finished: {final_csv}"
        print(msg, flush=True)
        with open(log, "a", buffering=1) as f:
            f.write(msg + "\n")
        return msg

    cmd = [
        python,
        "-m",
        "npcsh.benchmark.local_runner",
        "--model",
        model,
        "--provider",
        "ollama",
        "--timeout",
        str(timeout),
        "--resume",
    ]

    with open(log, "w", buffering=1) as f:
        print(f"[START] {model}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            for line in proc.stdout:
                print(f"[{model}] {line}", end="", flush=True)
                f.write(line)
            proc.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            raise
        if proc.returncode == 0:
            msg = f"[DONE]  {model} - see {log}"
        else:
            msg = f"[FAIL]  {model} - see {log}"
        print(msg, flush=True)
        f.write(msg + "\n")
    return msg


def main():
    parser = argparse.ArgumentParser(
        description="Run npcsh benchmark across all local Ollama models"
    )
    parser.add_argument("timeout", nargs="?", type=int, default=1200)
    parser.add_argument("--jobs", "-j", type=int, default=1)
    parser.add_argument(
        "--python",
        default=os.environ.get(
            "NPCSH_PYTHON", "/Users/caug/.pyenv/versions/3.12.0/envs/npc/bin/python3"
        ),
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of model tags to run (default: all local non-cloud)",
    )
    args = parser.parse_args()

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = _list_local_models()
        if not models:
            print("No local non-cloud models found.", file=sys.stderr)
            return
        print(f"Discovered {len(models)} local non-cloud model(s):", flush=True)
        for m in models:
            print(f"  - {m}", flush=True)

    if args.jobs == 1:
        for model in models:
            try:
                run_model(model, args.timeout, args.python)
            except KeyboardInterrupt:
                print("\nInterrupted", flush=True)
                sys.exit(130)
            except Exception as e:
                print(f"[ERROR] {model}: {e}", flush=True)
        return

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_model, model, args.timeout, args.python): model
            for model in models
        }
        try:
            for future in as_completed(futures):
                model = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] {model}: {e}", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted", flush=True)
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(130)


if __name__ == "__main__":
    main()
