"""
Standalone benchmark runner for npcsh.

Runs a set of tasks through npcsh -c with a local model and verifies
results by checking file system state and command output.

Usage:
    python -m npcsh.benchmark.local_runner
    python -m npcsh.benchmark.local_runner --model mistral-small3.2 --provider ollama
    python -m npcsh.benchmark.local_runner --category shell
    python -m npcsh.benchmark.local_runner --difficulty easy
    python -m npcsh.benchmark.local_runner --task-id shell-pipe-01
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TaskResult:
    task_id: str
    category: str
    difficulty: str
    passed: bool
    duration: float
    error: Optional[str] = None
    npcsh_output: str = ""


@dataclass
class BenchmarkReport:
    model: str
    provider: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    duration: float = 0.0
    results: List[TaskResult] = field(default_factory=list)
    by_category: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)


def load_tasks(
    task_file: Optional[str] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    task_id: Optional[str] = None,
) -> list:
    if task_file is None:
        task_file = Path(__file__).parent / "tasks.json"

    with open(task_file) as f:
        tasks = json.load(f)

    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if difficulty:
        tasks = [t for t in tasks if t["difficulty"] == difficulty]

    return tasks


def clean_task_artifacts():
    """Remove /tmp files created by tasks so runs don't bleed into each other."""
    import glob
    patterns = [
        "/tmp/result.txt", "/tmp/pyfiles.txt", "/tmp/uname.txt", "/tmp/nums.txt",
        "/tmp/dirs.txt", "/tmp/comment_count.txt", "/tmp/largest.txt",
        "/tmp/hello.txt", "/tmp/person.json", "/tmp/config.ini", "/tmp/env.sh",
        "/tmp/fib.py", "/tmp/rev.py", "/tmp/calc.py", "/tmp/wordcount.py",
        "/tmp/sample.txt", "/tmp/wc_result.json", "/tmp/fizzbuzz.py",
        "/tmp/data.csv", "/tmp/analyze.py", "/tmp/stats.json",
        "/tmp/scores.csv", "/tmp/inventory.json", "/tmp/total.py",
        "/tmp/sysinfo.txt", "/tmp/env_info.txt", "/tmp/path_vars.txt",
        "/tmp/log.txt", "/tmp/errors.txt", "/tmp/fruits.txt", "/tmp/sorted_fruits.txt",
        "/tmp/words.txt", "/tmp/unique_counts.txt",
        "/tmp/broken.py", "/tmp/buggy.py",
        "/tmp/report.txt", "/tmp/users.json",
        "/tmp/backup.sh", "/tmp/backup.tar.gz",
        "/tmp/todo.py", "/tmp/todos.txt",
        "/tmp/sunset.png", "/tmp/cat.png", "/tmp/generated.png",
        "/tmp/forest.png", "/tmp/img_info.py",
        "/tmp/welcome.wav", "/tmp/welcome.mp3",
        "/tmp/pangram.wav", "/tmp/pangram.mp3",
        "/tmp/search_results.txt", "/tmp/linux_creator.txt", "/tmp/japan_pop.txt",
        "/tmp/languages.txt", "/tmp/rank.py",
        "/tmp/primes.py", "/tmp/fib_research.py",
        "/tmp/disk_usage.txt", "/tmp/file_count.txt",
    ]
    import shutil
    for p in patterns:
        try:
            os.remove(p)
        except (OSError, FileNotFoundError):
            pass
    for d in ["/tmp/mydir", "/tmp/myrepo", "/tmp/project"]:
        shutil.rmtree(d, ignore_errors=True)


def run_task(task: dict, model: str, provider: str, timeout: int = 120) -> TaskResult:
    """Run a single task through npcsh -c and verify the result."""

    task_id = task["id"]
    instruction = task["instruction"]
    verify_cmd = task["verify_cmd"]

    # Per-task timeout override (some tasks like image gen need more time)
    task_timeout = task.get("timeout", timeout)
    verify_timeout = task.get("verify_timeout", 30)

    # Clean up before each task
    clean_task_artifacts()

    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    env["NPCSH_STREAM_OUTPUT"] = "0"

    if provider == "ollama":
        env.setdefault("OLLAMA_HOST", "http://localhost:11434")
        env["NPCSH_OLLAMA_NUM_CTX"] = "16384"

    start = time.time()

    # Run npcsh
    try:
        result = subprocess.run(
            ["npcsh", "-c", instruction],
            capture_output=True,
            text=True,
            timeout=task_timeout,
            env=env,
        )
        npcsh_output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return TaskResult(
            task_id=task_id,
            category=task["category"],
            difficulty=task["difficulty"],
            passed=False,
            duration=time.time() - start,
            error="timeout",
        )
    except Exception as e:
        return TaskResult(
            task_id=task_id,
            category=task["category"],
            difficulty=task["difficulty"],
            passed=False,
            duration=time.time() - start,
            error=str(e),
        )

    duration = time.time() - start

    # Small delay to let filesystem sync (especially for image/audio files on macOS)
    time.sleep(0.5)

    # Verify
    try:
        verify = subprocess.run(
            ["bash", "-c", verify_cmd],
            capture_output=True,
            text=True,
            timeout=verify_timeout,
        )
        passed = verify.returncode == 0
    except Exception as e:
        passed = False
        npcsh_output += f"\nVerify error: {e}"

    return TaskResult(
        task_id=task_id,
        category=task["category"],
        difficulty=task["difficulty"],
        passed=passed,
        duration=duration,
        npcsh_output=npcsh_output[:2000],
    )


def run_benchmark(
    model: str = "mistral-small3.2",
    provider: str = "ollama",
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    task_id: Optional[str] = None,
    timeout: int = 120,
) -> BenchmarkReport:

    tasks = load_tasks(category=category, difficulty=difficulty, task_id=task_id)
    report = BenchmarkReport(model=model, provider=provider, total=len(tasks))

    print(f"\nnpcsh benchmark: {provider}/{model}", flush=True)
    print(f"Tasks: {len(tasks)}", flush=True)
    print("=" * 60, flush=True)

    for i, task in enumerate(tasks):
        tid = task["id"]
        print(f"\n[{i+1}/{len(tasks)}] {tid} ({task['category']}/{task['difficulty']})", flush=True)
        print(f"  {task['description']}", flush=True)

        result = run_task(task, model, provider, timeout)
        report.results.append(result)

        if result.passed:
            report.passed += 1
            print(f"  PASS ({result.duration:.1f}s)", flush=True)
        elif result.error:
            report.errors += 1
            report.failed += 1
            print(f"  ERROR: {result.error} ({result.duration:.1f}s)", flush=True)
        else:
            report.failed += 1
            print(f"  FAIL ({result.duration:.1f}s)", flush=True)

        report.duration += result.duration

        # Track by category
        cat = task["category"]
        if cat not in report.by_category:
            report.by_category[cat] = {"total": 0, "passed": 0}
        report.by_category[cat]["total"] += 1
        if result.passed:
            report.by_category[cat]["passed"] += 1

        # Track by difficulty
        diff = task["difficulty"]
        if diff not in report.by_difficulty:
            report.by_difficulty[diff] = {"total": 0, "passed": 0}
        report.by_difficulty[diff]["total"] += 1
        if result.passed:
            report.by_difficulty[diff]["passed"] += 1

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total: {report.total}  Passed: {report.passed}  Failed: {report.failed}  Errors: {report.errors}")
    if report.total > 0:
        print(f"Score: {report.passed}/{report.total} ({100*report.passed/report.total:.0f}%)")
    print(f"Duration: {report.duration:.1f}s")

    print("\nBy category:")
    for cat, stats in sorted(report.by_category.items()):
        print(f"  {cat:<15} {stats['passed']}/{stats['total']}")

    print("\nBy difficulty:")
    for diff, stats in sorted(report.by_difficulty.items()):
        print(f"  {diff:<10} {stats['passed']}/{stats['total']}")

    # Save report
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"{provider}_{model}_{ts}.json"

    with open(report_file, "w") as f:
        json.dump({
            "model": model,
            "provider": provider,
            "timestamp": ts,
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "errors": report.errors,
            "score": report.passed / report.total if report.total else 0,
            "duration": report.duration,
            "by_category": report.by_category,
            "by_difficulty": report.by_difficulty,
            "results": [
                {
                    "task_id": r.task_id,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "passed": r.passed,
                    "duration": r.duration,
                    "error": r.error,
                }
                for r in report.results
            ],
        }, f, indent=2)

    print(f"\nReport saved: {report_file}")
    return report


def compare_models(
    models: List[tuple],
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    timeout: int = 120,
) -> dict:
    """Run benchmark across multiple models and print comparison."""
    all_results = {}

    for model, provider in models:
        key = f"{provider}/{model}"
        print(f"\n{'='*60}")
        print(f"  MODEL: {key}")
        print(f"{'='*60}")
        report = run_benchmark(
            model=model,
            provider=provider,
            category=category,
            difficulty=difficulty,
            timeout=timeout,
        )
        all_results[key] = report

    # Comparison table
    print(f"\n{'='*60}", flush=True)
    print("MODEL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Model':<30} {'Score':>8} {'Pass':>6} {'Fail':>6} {'Time':>8}", flush=True)
    print("-" * 60, flush=True)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].passed,
        reverse=True,
    )

    for key, report in sorted_results:
        pct = 100 * report.passed / report.total if report.total else 0
        print(
            f"{key:<30} {pct:>7.0f}% {report.passed:>5}/{report.total:<5} "
            f"{report.duration:>7.0f}s",
            flush=True,
        )

    # Category breakdown per model
    all_cats = set()
    for r in all_results.values():
        all_cats.update(r.by_category.keys())

    print(f"\nCategory breakdown:", flush=True)
    header = f"{'Category':<15}"
    for key in [k for k, _ in sorted_results]:
        short = key.split("/")[-1][:12]
        header += f" {short:>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for cat in sorted(all_cats):
        row = f"{cat:<15}"
        for key, _ in sorted_results:
            report = all_results[key]
            stats = report.by_category.get(cat, {"passed": 0, "total": 0})
            row += f" {stats['passed']:>5}/{stats['total']:<5}"
        print(row, flush=True)

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="npcsh local benchmark")
    parser.add_argument("--model", "-m", default="mistral-small3.2")
    parser.add_argument("--provider", "-p", default="ollama")
    parser.add_argument("--category", "-c", default=None)
    parser.add_argument("--difficulty", "-d", default=None)
    parser.add_argument("--task-id", "-t", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple local models")

    args = parser.parse_args()

    if args.compare:
        models = [
            ("mistral-small3.2", "ollama"),
            ("qwen3:8b", "ollama"),
            ("gemma3:12b", "ollama"),
            ("phi4", "ollama"),
            ("llama3.1:8b", "ollama"),
        ]
        compare_models(
            models,
            category=args.category,
            difficulty=args.difficulty,
            timeout=args.timeout,
        )
    else:
        run_benchmark(
            model=args.model,
            provider=args.provider,
            category=args.category,
            difficulty=args.difficulty,
            task_id=args.task_id,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
