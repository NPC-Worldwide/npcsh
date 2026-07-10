"""
Benchmark runner using the Rust npcsh binary.

Passes each task instruction to `npcsh` as a positional argument, then verifies with verify_cmd.

Usage:
    python -m npcsh.benchmark.rust_runner
    python -m npcsh.benchmark.rust_runner --model gemini-2.5-flash --provider gemini
    python -m npcsh.benchmark.rust_runner --category shell --limit 5
    python -m npcsh.benchmark.rust_runner --task-id shell-01
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd


NPCSH_BIN = os.path.expanduser("~/.npcsh/bin/npcsh")

def _set_binary(path):
    global NPCSH_BIN
    NPCSH_BIN = path


@dataclass
class TaskResult:
    task_id: str
    category: str
    difficulty: str
    passed: bool
    duration: float
    error: Optional[str] = None
    output: str = ""


def load_tasks(category=None, difficulty=None, task_id=None, limit=None):
    task_file = Path(__file__).parent / "tasks.csv"
    tasks = pd.read_csv(task_file).to_dict('records')

    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if difficulty:
        tasks = [t for t in tasks if t["difficulty"] == difficulty]
    if limit:
        tasks = tasks[:limit]

    return tasks


def clean_task_artifacts(task):
    paths = set()
    for fld in ("verify_cmd", "instruction"):
        text = task.get(fld, "") or ""
        for m in re.findall(r'/tmp/[\w.*/-]+', text):
            paths.add(m.rstrip(")'\"`;,"))

    for d in ["/tmp/mydir", "/tmp/myrepo", "/tmp/project", "/tmp/rentest",
              "/tmp/webapp", "/tmp/mathpkg"]:
        shutil.rmtree(d, ignore_errors=True)

    for p in paths:
        if "*" in p:
            import glob
            for f in glob.glob(p):
                try:
                    os.remove(f)
                except (OSError, IsADirectoryError):
                    shutil.rmtree(f, ignore_errors=True)
        else:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
            except (OSError, FileNotFoundError):
                pass


def run_task(task, model, provider, timeout=120, verbose=False):
    task_id = task["id"]
    instruction = task["instruction"]
    verify_cmd = task["verify_cmd"]
    setup_cmd = task.get("setup_cmd", "") or ""

    clean_task_artifacts(task)

    if setup_cmd and isinstance(setup_cmd, str) and setup_cmd.strip():
        try:
            subprocess.run(["bash", "-c", setup_cmd], timeout=15, capture_output=True)
        except Exception as e:
            print(f"  setup failed: {e}")

    start = time.time()
    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    if verbose:
        env["NPCSH_DEBUG"] = "1"

    script_dir = Path(tempfile.mkdtemp(prefix=f"npcsh_bench_{task_id}_"))
    script_path = script_dir / "task.nsh"
    script_path.write_text(instruction, encoding="utf-8")

    log_dir = Path.home() / ".npcsh" / "benchmarks" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_id}.log"

    output_lines = []
    try:
        proc = subprocess.Popen(
            [NPCSH_BIN, str(script_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, cwd="/tmp",
        )
        with open(log_path, "w", encoding="utf-8") as log_f:
            if proc.stdout:
                for line in proc.stdout:
                    output_lines.append(line)
                    log_f.write(line)
                    log_f.flush()
                    if verbose:
                        print(line, end="", flush=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            output_lines.append(f"TIMEOUT after {timeout}s\n")
        output = "".join(output_lines)
    except Exception as e:
        output = f"Error: {e}\n"
    finally:
        shutil.rmtree(script_dir, ignore_errors=True)

    duration = time.time() - start

    try:
        verify = subprocess.run(
            ["bash", "-c", verify_cmd],
            capture_output=True, text=True, timeout=30,
            cwd="/tmp",
        )
        passed = verify.returncode == 0
    except Exception:
        passed = False

    return TaskResult(
        task_id=task_id,
        category=task["category"],
        difficulty=task["difficulty"],
        passed=passed,
        duration=duration,
        output=output[:500],
    )


def run_benchmark(model, provider, tasks, timeout=120, verbose=False):
    results = []
    passed = 0
    total = len(tasks)

    print(f"\n{'='*60}")
    print(f"  npcsh-rs benchmark: {model} ({provider})")
    print(f"  {total} tasks, timeout={timeout}s")
    print(f"  logs: ~/.npcsh/benchmarks/logs/<task_id>.log")
    print(f"{'='*60}\n")

    for i, task in enumerate(tasks):
        task_id = task["id"]
        print(f"[{i+1}/{total}] {task_id} ({task['category']}/{task['difficulty']})...", end=" ", flush=True)

        result = run_task(task, model, provider, timeout, verbose=verbose)
        results.append(result)

        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} ({result.duration:.1f}s)")
        if not verbose:
            log_path = Path.home() / ".npcsh" / "benchmarks" / "logs" / f"{task_id}.log"
            print(f"         log: {log_path}")

        if result.passed:
            passed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} ({100*passed/total:.0f}%)")
    print(f"{'='*60}")

    by_cat = {}
    for r in results:
        if r.category not in by_cat:
            by_cat[r.category] = {"passed": 0, "total": 0}
        by_cat[r.category]["total"] += 1
        if r.passed:
            by_cat[r.category]["passed"] += 1

    print("\nBy category:")
    for cat, stats in sorted(by_cat.items()):
        p, t = stats["passed"], stats["total"]
        print(f"  {cat:<15} {p}/{t} ({100*p/t:.0f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run npcsh-rs benchmark")
    parser.add_argument("--model", "-m", default="gemini-2.5-flash")
    parser.add_argument("--provider", "-pr", default="gemini")
    parser.add_argument("--category", "-c", default=None)
    parser.add_argument("--difficulty", "-d", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Stream npcsh output to stdout and enable NPCSH_DEBUG")
    parser.add_argument("--binary", default=NPCSH_BIN, help="Path to npcsh binary")
    args = parser.parse_args()

    _set_binary(args.binary)

    if not os.path.exists(NPCSH_BIN):
        print(f"Error: npcsh binary not found at {NPCSH_BIN}")
        print("Install it: curl -fsSL https://enpisi.com/install-npcsh.sh | sh")
        print("Or build it: cd npcsh/rust && cargo build --release")
        return

    tasks = load_tasks(
        category=args.category,
        difficulty=args.difficulty,
        task_id=args.task_id,
        limit=args.limit,
    )

    if not tasks:
        print("No tasks matched the filters")
        return

    run_benchmark(args.model, args.provider, tasks, args.timeout, verbose=args.verbose)


if __name__ == "__main__":
    main()
