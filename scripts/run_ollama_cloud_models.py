#!/usr/bin/env python3
"""Run the npcsh 100-task benchmark across Ollama cloud models in parallel,
resuming from any existing checkpoint or completed CSV."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


MODELS = [
    "deepseek-v4-flash:cloud",
    "devstral-2:123b-cloud",
    "devstral-small-2:24b-cloud",
    "gemma4:31b-cloud",
    "kimi-k2.7-code:cloud",
    "minimax-m3:cloud",
    "mistral-large-3:675b-cloud",
    "nemotron-3-super:cloud",
    "qwen3.5:cloud",
]


def _checkpoint_path(model: str, report_dir: Path) -> Path:
    """Exact same filename local_runner uses for checkpoints."""
    safe_model = model.replace("/", "_")
    return report_dir / f"npcsh_ollama_{safe_model}_running.csv"


def _final_csv_path(model: str, report_dir: Path) -> Path | None:
    """Return the newest final CSV local_runner writes for this model."""
    safe_model = model.replace("/", "_")
    candidates = [
        p for p in report_dir.glob("*.csv")
        if p.stem.startswith(f"npcsh_ollama_{safe_model}_") and "_running" not in p.stem
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_model(model: str, timeout: int, binary: str, python: str) -> str:
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    report_dir.mkdir(parents=True, exist_ok=True)
    log = report_dir / f"{model.replace('/', '_')}.log"

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
        "--binary",
        binary,
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
        )
        for line in proc.stdout:
            print(f"[{model}] {line}", end="", flush=True)
            f.write(line)
        proc.wait()
        if proc.returncode == 0:
            msg = f"[DONE]  {model} - see {log}"
        else:
            msg = f"[FAIL]  {model} - see {log}"
        print(msg, flush=True)
        f.write(msg + "\n")
    return msg


def main():
    parser = argparse.ArgumentParser(description="Run npcsh benchmark across Ollama cloud models")
    parser.add_argument("timeout", nargs="?", type=int, default=120)
    parser.add_argument("--jobs", "-j", type=int, default=3)
    parser.add_argument("--binary", default=os.environ.get("NPCSH_BIN", "/Users/caug/npcww/npc-core/npcsh/rust/target/release/npcsh"))
    parser.add_argument("--python", default=os.environ.get("NPCSH_PYTHON", "/Users/caug/.pyenv/versions/3.12.0/envs/npc/bin/python3"))
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_model, model, args.timeout, args.binary, args.python): model
            for model in MODELS
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] {model}: {e}", flush=True)


if __name__ == "__main__":
    main()
