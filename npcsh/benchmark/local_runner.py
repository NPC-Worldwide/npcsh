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

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import concurrent.futures 
from concurrent.futures import ThreadPoolExecutor, TimeoutError


import pandas as pd
from npcsh.routes import router


from npcsh._state import (
    initial_state,
    execute_command,
    setup_shell,
)
@dataclass
class TaskResult:
    task_id: str
    category: str
    difficulty: str
    passed: bool
    duration: float
    attempts: int = 1
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
        task_file = Path(__file__).parent / "tasks.csv"

    tasks = pd.read_csv(task_file).to_dict('records')

    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if difficulty:
        tasks = [t for t in tasks if t["difficulty"] == difficulty]

    return tasks


def clean_task_artifacts():
    """Remove /tmp files created by tasks so runs don't bleed into each other."""
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


def _setup_state(model: str, provider: str, state):
    """One-time setup for a task: load team, register jinxes, configure model."""
    command_history, team, default_npc = setup_shell()
    if team and hasattr(team, 'jinxs_dict'):
        for jinx_name, jinx_obj in team.jinxs_dict.items():
            router.register_jinx(jinx_obj)
    state.npc = default_npc
    state.npc.model = model
    state.npc.provider = provider
    state.team = team
    state.team.model = model
    state.team.provider = provider
    state.chat_model = model
    state.chat_provider = provider
    state.command_history = command_history
    state.current_path = os.getcwd()
    state.messages = []
    state._max_iterations = 10
    return command_history


def _run_attempt(instruction: str, state, command_history) -> tuple:
    """Execute one instruction within an existing session."""
    msg_count_before = len(state.messages)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(execute_command, instruction, state, router=router, command_history=command_history)
        try:
            final_state, output = future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            print("Model inference timed out after 120 seconds!", flush=True)
            final_state = state
            output = {"output": "Model inference timed out after 120 seconds! No response received."}
        except Exception as e:
            print(f"Exception during execute_command: {e}", flush=True)
            final_state = state
            output = {"output": f"Exception during execute_command: {e}"}

    output_str = ""
    if isinstance(output, dict):
        output_str = output.get('output') or output.get('response') or str(output)
        print(output_str)
    elif final_state.stream_output and output is not None:
        chunks = []
        for chunk in output:
            s = str(chunk)
            chunks.append(s)
            print(s, end='')
        print()
        output_str = "".join(chunks)
    elif output is not None:
        output_str = str(output)
        print(output_str)

    # Capture full conversation trace: every message the model sent/received
    new_msgs = final_state.messages[msg_count_before:]
    trace_parts = []
    for msg in new_msgs:
        role = msg.get("role", "?")
        content = str(msg.get("content", "")).replace("\n", "\\n")
        tool_calls = msg.get("tool_calls", [])
        if content:
            trace_parts.append(f"[{role}] {content}")
        for tc in tool_calls:
            fn = tc.get("function", {})
            args = str(fn.get("arguments", "")).replace("\n", "\\n")
            trace_parts.append(f"[tool_call] {fn.get('name', '?')}({args})")
    trace = " | ".join(trace_parts)

    full_output = f"{output_str} ---TRACE--- {trace}" if trace else output_str
    return final_state, full_output


def run_task(task: dict,
             model: str,
             provider: str,
             timeout: int = 120,
             max_attempts: int = 5,
             startup_overhead: float = 0.0) -> TaskResult:
    """Run a task with retries until it passes or the timeout/attempt budget is exhausted."""

    task_id = task["id"]
    instruction = task["instruction"]
    verify_cmd = task["verify_cmd"]
    verify_timeout = task.get("verify_timeout", 30)

    deadline = time.time() + timeout
    start = time.time()
    attempt = 0
    passed = False
    all_outputs: list = []
    last_output = ""

    # Set up once per task — fresh messages, same session for retries
    command_history = _setup_state(model, provider, initial_state)

    while time.time() < deadline and attempt < max_attempts:
        attempt += 1
        clean_task_artifacts()

        if attempt == 1:
            current_instruction = instruction
        else:
            remaining = int(deadline - time.time())
            current_instruction = (
                f"The verification check failed. Here is what it produced:\n"
                f"{last_output[:800]}\n\n"
                f"Fix it. {remaining}s remaining."
            )

        print(f"  [attempt {attempt}]", flush=True)

        try:
            _, output_str = _run_attempt(
                current_instruction, initial_state, command_history
            )
        except Exception as e:
            output_str = f"Exception: {e}"

        all_outputs.append(f"[attempt {attempt}] {output_str}")
        last_output = output_str

        # Filesystem sync delay
        time.sleep(5)

        # Verify
        try:
            verify = subprocess.run(
                ["bash", "-c", verify_cmd],
                capture_output=True, text=True, timeout=verify_timeout,
            )
            passed = verify.returncode == 0
        except Exception as e:
            passed = False
            all_outputs.append(f"Verify error: {e}")

        if passed:
            break

        remaining = deadline - time.time()
        if remaining <= 0:
            break
        print(f"  attempt {attempt} failed, {int(remaining)}s left — retrying", flush=True)

    duration = time.time() - start
    return TaskResult(
        task_id=task_id,
        category=task["category"],
        difficulty=task["difficulty"],
        passed=passed,
        duration=max(0, duration - startup_overhead),
        attempts=attempt,
        npcsh_output=" ||| ".join(all_outputs),
    )


def run_benchmark(
    model:str,
    provider:str,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    task_id: Optional[str] = None,
    timeout: int = 120,
    resume: bool = False,
) -> BenchmarkReport:

    tasks = load_tasks(category=category, difficulty=difficulty, task_id=task_id)
    report = BenchmarkReport(model=model, provider=provider, total=len(tasks))

    # Resume: load completed tasks from checkpoint CSV
    import csv as csv_mod
    csv_mod.field_size_limit(10**7)
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    checkpoint_file = report_dir / f"{provider}_{model}_running.csv"
    completed_ids = set()
    if resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for row in csv_mod.DictReader(f):
                completed_ids.add(row["task_id"])
                report.results.append(TaskResult(
                    task_id=row["task_id"],
                    category=row["category"],
                    difficulty=row["difficulty"],
                    passed=row["passed"].lower() == "true",
                    duration=float(row.get("duration", 0)),
                    attempts=int(row.get("attempts", 1)),
                    error=row.get("error") or None,
                    npcsh_output=row.get("output", ""),
                ))
                if row["passed"].lower() == "true":
                    report.passed += 1
                else:
                    report.failed += 1
                report.duration += float(row.get("duration", 0))
                cat = row["category"]
                if cat not in report.by_category:
                    report.by_category[cat] = {"total": 0, "passed": 0}
                report.by_category[cat]["total"] += 1
                if row["passed"].lower() == "true":
                    report.by_category[cat]["passed"] += 1
                diff = row["difficulty"]
                if diff not in report.by_difficulty:
                    report.by_difficulty[diff] = {"total": 0, "passed": 0}
                report.by_difficulty[diff]["total"] += 1
                if row["passed"].lower() == "true":
                    report.by_difficulty[diff]["passed"] += 1
        print(f"Resumed {len(completed_ids)} tasks from checkpoint", flush=True)

    # Calibrate startup overhead
    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    env["NPCSH_STREAM_OUTPUT"] = "0"
    env["NPCSH_NO_EMBEDDINGS"] = "1"
    if provider == "ollama":
        env.setdefault("OLLAMA_HOST", "http://localhost:11434")
        env["NPCSH_OLLAMA_NUM_CTX"] = "16384"

    print(f"\nnpcsh benchmark: {provider}/{model}", flush=True)
    print("Calibrating startup overhead...", flush=True)
    cal_times = []
    for _ in range(3):
        cal_start = time.time()
        subprocess.run(
            ["npcsh", "-c", "echo ok"],
            text=True, env=env, timeout=120,
        )
        cal_times.append(time.time() - cal_start)
    startup_overhead = sum(cal_times) / len(cal_times)
    print(f"Startup overhead: {startup_overhead:.1f}s (avg of 3)", flush=True)

    print(f"Tasks: {len(tasks)} ({len(completed_ids)} already done)", flush=True)
    print("=" * 60, flush=True)

    for i, task in enumerate(tasks):
        tid = task["id"]
        if tid in completed_ids:
            print(f"\n[{i+1}/{len(tasks)}] {tid} — skipped (resumed)", flush=True)
            continue

        print(f"\n[{i+1}/{len(tasks)}] {tid} ({task['category']}/{task['difficulty']})", flush=True)
        print(f"  {task['description']}", flush=True)

        result = run_task(task, model, provider, timeout, startup_overhead=startup_overhead)
        report.results.append(result)

        if result.passed:
            report.passed += 1
            print(f"  PASS ({result.duration:.1f}s, {result.attempts} attempt(s))", flush=True)
        elif result.error:
            print(result)
            report.errors += 1
            report.failed += 1
            print(f"  ERROR: {result.error} ({result.duration:.1f}s)", flush=True)
        else:
            report.failed += 1
            print(f"  FAIL ({result.duration:.1f}s, {result.attempts} attempt(s))", flush=True)

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

        # Save after every task so we don't lose progress
        report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
        report_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = report_dir / f"{provider}_{model}_running.csv"
        df = pd.DataFrame([
            {"task_id": r.task_id, "category": r.category, "difficulty": r.difficulty,
             "passed": r.passed, "attempts": r.attempts, "duration": round(r.duration, 1), "error": r.error or "",
             "output": r.npcsh_output}
            for r in report.results
        ])
        df.to_csv(checkpoint_file, index=False)

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

    # Save final report as CSV
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"{provider}_{model}_{ts}.csv"
    df = pd.DataFrame([
        {"task_id": r.task_id, "category": r.category, "difficulty": r.difficulty,
         "passed": r.passed, "duration": round(r.duration, 1), "error": r.error or "",
             "output": r.npcsh_output}
        for r in report.results
    ])
    df.to_csv(report_file, index=False)

    # Remove checkpoint file
    checkpoint = report_dir / f"{provider}_{model}_running.csv"
    if checkpoint.exists():
        checkpoint.unlink()

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

    print("\nCategory breakdown:", flush=True)
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


def rerun_failed(csv_path: str, model: str, provider: str, timeout: int = 120):
    """Re-run only the failed tasks from an existing CSV and overwrite results in-place."""
    import csv as csv_mod
    csv_mod.field_size_limit(10**7)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    failed_ids = [r["task_id"] for r in rows if r.get("passed", "").lower() != "true"]
    print(f"\nRerun failed tasks from {csv_path.name}")
    print(f"Total rows: {len(rows)}, Failed: {len(failed_ids)}")

    if not failed_ids:
        print("No failed tasks to rerun.")
        return

    all_tasks = load_tasks()
    task_lookup = {t["id"]: t for t in all_tasks}

    # Calibrate startup overhead
    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    env["NPCSH_STREAM_OUTPUT"] = "0"
    env["NPCSH_NO_EMBEDDINGS"] = "1"
    if provider == "ollama":
        env.setdefault("OLLAMA_HOST", "http://localhost:11434")
        env["NPCSH_OLLAMA_NUM_CTX"] = "16384"

    print("Calibrating startup overhead...", flush=True)
    cal_times = []
    for _ in range(3):
        cal_start = time.time()
        subprocess.run(
            ["npcsh", "-c", "echo ok"],
            #stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, timeout=120,
        )
        cal_times.append(time.time() - cal_start)
    startup_overhead = sum(cal_times) / len(cal_times)
    print(f"Startup overhead: {startup_overhead:.1f}s", flush=True)

    # Build index: task_id -> row index
    row_index = {}
    for i, r in enumerate(rows):
        row_index[r["task_id"]] = i

    improved = 0
    for j, tid in enumerate(failed_ids):
        if tid not in task_lookup:
            print(f"  [{j+1}/{len(failed_ids)}] {tid} — task definition not found, skipping")
            continue

        task = task_lookup[tid]
        print(f"\n  [{j+1}/{len(failed_ids)}] {tid} ({task['category']}/{task['difficulty']})", flush=True)

        result = run_task(task, model, provider, timeout, startup_overhead)

        if result.passed:
            print(f"    PASS ({result.duration:.1f}s) — upgraded!", flush=True)
            improved += 1
        elif result.error:
            print(f"    ERROR: {result.error} ({result.duration:.1f}s)", flush=True)
        else:
            print(f"    FAIL ({result.duration:.1f}s)", flush=True)

        # Overwrite the row
        idx = row_index[tid]
        rows[idx]["passed"] = str(result.passed)
        rows[idx]["duration"] = str(round(result.duration, 1))
        rows[idx]["error"] = result.error or ""
        rows[idx]["output"] = result.npcsh_output

        # Save after each task
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    total_passed = sum(1 for r in rows if r.get("passed", "").lower() == "true")
    print(f"\nDone. Improved {improved}/{len(failed_ids)} tasks.")
    print(f"New total: {total_passed}/{len(rows)} ({100*total_passed//len(rows)}%)")
    print(f"Saved to: {csv_path}")


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
    parser.add_argument("--rerun-failed", type=str, default=None,
                        help="Path to existing CSV — re-run only failed tasks and overwrite")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from _running.csv checkpoint, skip already-completed tasks")

    args = parser.parse_args()

    if args.rerun_failed:
        rerun_failed(
            csv_path=args.rerun_failed,
            model=args.model,
            provider=args.provider,
            timeout=args.timeout,
        )
    elif args.compare:
        models = [
          
            ("qwen3:8b", "ollama"),
          
            ("qwen3:1.7b", "ollama"),
            ("qwen3:4b", "ollama"),
            ("qwen3:30b", "ollama"),  
            ("qwen3:0.6b", "ollama"),
          
            ('llama3.2:1b', 'ollama'),
            ('llama3.2:3b', 'ollama'),
            ('llama3.1:8b', 'ollama'),
            ('gemma3:1b', 'ollama'),
            ('gemma3:4b', 'ollama'),
            ('gemma3:12b', 'ollama'),
          
            ('gemma3:27b', 'ollama'),
                 
            ("mistral-small3.2:latest", "ollama"),
            ("phi4", "ollama"),
            ('gpt-oss:20b', 'ollama')
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
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
