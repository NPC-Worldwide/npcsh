"""Standalone benchmark runner for npcsh.

Runs a set of tasks through the Rust `npcsh` binary and verifies results by
checking file system state and command output.

Usage:
    python -m npcsh.benchmark.local_runner
    python -m npcsh.benchmark.local_runner --model mistral-small3.2 --provider ollama
    python -m npcsh.benchmark.local_runner --category shell
    python -m npcsh.benchmark.local_runner --difficulty easy
    python -m npcsh.benchmark.local_runner --task-id shell-pipe-01
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


SUPPORTED_FRAMEWORKS = (
    "npcsh",
    "claude", "opencode", "nanocoder",
    "npc-claude", "npc-codex", "npc-opencode",
)
NPCSH_SPECIFIC_CATEGORIES = frozenset({
    "delegation", "tool-chain", "image-gen", "audio-gen", "web-search",
})

NPCSH_BIN = os.path.expanduser("~/.npcsh/bin/npcsh")


def _find_npcsh_bin(path: Optional[str] = None) -> str:
    """Return the path to the npcsh binary, falling back to PATH lookup."""
    if path:
        return path
    if os.path.exists(NPCSH_BIN):
        return NPCSH_BIN
    found = shutil.which("npcsh")
    if found:
        return found
    raise FileNotFoundError("npcsh binary not found; build and install it first")


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
    framework: str = "npcsh",
) -> list:
    if task_file is None:
        task_file = Path(__file__).parent / "tasks.csv"

    tasks = pd.read_csv(task_file).to_dict('records')

    if framework != "npcsh":
        tasks = [t for t in tasks if t["category"] not in NPCSH_SPECIFIC_CATEGORIES]

    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if difficulty:
        tasks = [t for t in tasks if t["difficulty"] == difficulty]

    return tasks


_NOSUDO_DIR = "/tmp/npcsh_bench_nosudo"


def setup_bench_env():
    """Process-level env tweaks for safe, non-interactive benchmarking.

    1. Fake sudo on PATH — prevents password prompts hanging the runner.
    2. MPLBACKEND=Agg — prevents matplotlib NSWindow crash in daemon threads.
    """
    import atexit

    os.makedirs(_NOSUDO_DIR, exist_ok=True)
    fake_sudo = os.path.join(_NOSUDO_DIR, "sudo")
    with open(fake_sudo, "w") as f:
        f.write(
            '#!/bin/sh\n'
            'echo "[bench] sudo intercepted — running without elevation: $*" >&2\n'
            'exec "$@"\n'
        )
    os.chmod(fake_sudo, 0o755)
    current = os.environ.get("PATH", "")
    if _NOSUDO_DIR not in current.split(os.pathsep):
        os.environ["PATH"] = _NOSUDO_DIR + os.pathsep + current
    atexit.register(_remove_sudo_trap)

    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass

    try:
        import ollama as _ollama
        import httpx
        _ollama.chat.__self__._client.timeout = httpx.Timeout(90.0)
        print("Bench env: sudo trap + headless matplotlib + ollama timeout=90s", flush=True)
    except Exception:
        print("Bench env: sudo trap + headless matplotlib", flush=True)


def _remove_sudo_trap():
    shutil.rmtree(_NOSUDO_DIR, ignore_errors=True)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(p for p in parts if p != _NOSUDO_DIR)


def clean_task_artifacts(task: dict = None):
    """Remove /tmp files that a task's verify_cmd and instruction reference."""
    paths = set()
    if task:
        for fld in ("verify_cmd", "instruction"):
            text = task.get(fld, "")
            for m in re.findall(r'/tmp/[\w.*/-]+', text):
                clean = m.rstrip(")'\"`;,")
                paths.add(clean)

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


def _kill_descendants(root_pid: int) -> None:
    """SIGKILL the process group AND every descendant, then poll until reaped."""
    import signal as _signal
    try:
        pgid = os.getpgid(root_pid)
        os.killpg(pgid, _signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    descendants: list = []
    try:
        import psutil
        try:
            root = psutil.Process(root_pid)
            descendants = root.children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in descendants:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.wait_procs(descendants, timeout=3)
    except ImportError:
        try:
            subprocess.run(["pkill", "-9", "-P", str(root_pid)],
                           capture_output=True, timeout=5)
        except Exception:
            pass


def _run_npcsh_attempt(
    instruction: str,
    binary: str,
    model: str,
    provider: str,
    attempt_timeout: float,
    work_dir: str,
    stream: bool = True,
) -> str:
    """Launch the Rust npcsh binary as a subprocess for one task attempt."""
    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    env["NPCSH_DEBUG"] = "1"
    env["NPCSH_STREAM_OUTPUT"] = "1"
    env["NPCSH_ACCEPT_PERMISSIONS"] = "1"
    env["NPCSH_CWD"] = work_dir

    # local_runner passes the instruction as a positional arg. The current Rust binary
    # expects either an .nsh script file or interactive input, so write a temp .nsh.
    import tempfile as _tempfile
    script_dir = Path(_tempfile.mkdtemp(prefix="npcsh_bench_"))
    script_path = script_dir / "task.nsh"
    script_path.write_text(instruction, encoding="utf-8")

    print(f"  [npcsh] {instruction[:80]}... (cwd={work_dir})", flush=True)
    try:
        proc = subprocess.Popen(
            [binary, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=work_dir,
            start_new_session=True,
        )
    except FileNotFoundError:
        shutil.rmtree(script_dir, ignore_errors=True)
        return f"Exception: npcsh binary not found at {binary}"

    output_lines = []
    hard_deadline = time.time() + attempt_timeout + 5.0
    try:
        if proc.stdout:
            for line in proc.stdout:
                output_lines.append(line)
                if stream:
                    print(line, end="", flush=True)
        proc.wait(timeout=attempt_timeout)
    except subprocess.TimeoutExpired:
        pass

    _kill_descendants(proc.pid)
    try:
        proc.kill()
    except Exception:
        pass
    while proc.poll() is None and time.time() < hard_deadline:
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            _kill_descendants(proc.pid)
    if proc.poll() is None:
        try:
            proc.stdout.close()
        except Exception:
            pass

    shutil.rmtree(script_dir, ignore_errors=True)
    return "".join(output_lines) if output_lines else f"Timed out after {attempt_timeout:.0f}s"


def _build_external_command(framework: str, instruction: str, model: str,
                            npc: str = "sibiji") -> tuple:
    """Build (cmd_list, env_dict) for shelling out to a non-npcsh framework."""
    env = os.environ.copy()
    opencode_bin = os.path.expanduser("~/.opencode/bin")
    if opencode_bin not in env.get("PATH", ""):
        env["PATH"] = opencode_bin + ":" + env.get("PATH", "")

    def _claude_env(e):
        e["ANTHROPIC_AUTH_TOKEN"] = "ollama"
        e["ANTHROPIC_BASE_URL"] = "http://localhost:11434"
        e["DISABLE_AUTOUPDATER"] = "1"
        e["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        e.pop("CLAUDECODE", None)
        e.pop("CLAUDE_CODE_ENTRYPOINT", None)
        return e

    if framework == "claude":
        return [
            "claude", "-p", instruction,
            "--dangerously-skip-permissions", "--model", model,
        ], _claude_env(env)
    if framework == "npc-claude":
        return [
            "npc-claude", "--npc", npc,
            "-p", instruction,
            "--dangerously-skip-permissions", "--model", model,
        ], _claude_env(env)
    if framework == "opencode":
        return [
            os.path.expanduser("~/.opencode/bin/opencode"),
            "run", instruction, "-m", f"ollama/{model}",
        ], env
    if framework == "npc-opencode":
        return [
            "npc-opencode", "--npc", npc,
            "run", instruction, "-m", f"ollama/{model}",
        ], env
    if framework == "nanocoder":
        return ["nanocoder", "run", instruction], env
    if framework == "npc-codex":
        return ["npc-codex", "--npc", npc, "exec", instruction], env

    raise ValueError(f"Unsupported external framework: {framework}")


def _run_external_attempt(instruction: str, framework: str, model: str,
                          attempt_timeout: float, work_dir: str) -> str:
    """Subprocess-launch an external framework CLI in `work_dir`."""
    cmd, env = _build_external_command(framework, instruction, model)
    print(f"  [{framework}] {' '.join(cmd[:3])}... (cwd={work_dir})", flush=True)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=work_dir,
            start_new_session=True,
        )
    except FileNotFoundError:
        return f"Exception: {framework} CLI not found on PATH"

    hard_deadline = time.time() + attempt_timeout + 5.0
    try:
        stdout, stderr = proc.communicate(timeout=attempt_timeout)
        return (stdout or "") + (stderr or "")
    except subprocess.TimeoutExpired:
        pass

    _kill_descendants(proc.pid)
    try:
        proc.kill()
    except Exception:
        pass
    while proc.poll() is None and time.time() < hard_deadline:
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            _kill_descendants(proc.pid)
    if proc.poll() is None:
        try:
            proc.stdout.close()
            proc.stderr.close()
        except Exception:
            pass
    return f"Timed out after {attempt_timeout:.0f}s"


def run_task(task: dict,
             model: str,
             provider: str,
             timeout: int = 120,
             max_attempts: int = 5,
             think=None,
             framework: str = "npcsh",
             binary: str = NPCSH_BIN) -> TaskResult:
    """Run a task with retries until it passes or the budget is exhausted."""
    task_id = task["id"]
    instruction = task["instruction"]
    verify_cmd = task["verify_cmd"]
    setup_cmd = task.get("setup_cmd", "") or ""
    if not isinstance(setup_cmd, str):
        setup_cmd = ""
    verify_timeout = task.get("verify_timeout", 30)

    deadline = time.time() + timeout
    start = time.time()
    attempt = 0
    passed = False
    all_outputs: list = []
    last_output = ""

    task_dir = tempfile.mkdtemp(prefix=f"npcsh_bench_{task_id}_")

    while attempt < max_attempts:
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        attempt += 1
        clean_task_artifacts(task)

        if setup_cmd:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                subprocess.run(
                    ["bash", "-c", setup_cmd],
                    timeout=min(remaining, 15), capture_output=True, text=True,
                    cwd=task_dir,
                )
            except Exception as e:
                print(f"  setup_cmd failed: {e}", flush=True)

        remaining = deadline - time.time()
        if remaining <= 0:
            break

        if attempt == 1:
            current_instruction = instruction
        else:
            prev_summary = last_output
            current_instruction = f"""{instruction}

Your previous attempt did not produce the correct result. Here is what happened:
{prev_summary}

Try a different approach. Do not search the web about this."""

        attempt_timeout = remaining
        if attempt_timeout <= 0:
            break

        print(f"  [attempt {attempt}, {attempt_timeout:.0f}s cap]", flush=True)
        if attempt > 1:
            print(f"  [retry prompt] {current_instruction}", flush=True)

        try:
            if framework == "npcsh":
                output_str = _run_npcsh_attempt(
                    current_instruction, binary, model, provider,
                    attempt_timeout=attempt_timeout, work_dir=task_dir,
                )
            else:
                output_str = _run_external_attempt(
                    current_instruction, framework, model,
                    attempt_timeout=attempt_timeout, work_dir=task_dir,
                )
        except Exception as e:
            output_str = f"Exception: {e}"

        all_outputs.append(f"[attempt {attempt}] {output_str}")
        last_output = output_str

        remaining = deadline - time.time()
        if remaining <= 0:
            break

        time.sleep(min(2, remaining))

        remaining = deadline - time.time()
        if remaining <= 0:
            break

        try:
            verify = subprocess.run(
                ["bash", "-c", verify_cmd],
                capture_output=True, text=True,
                timeout=min(remaining, verify_timeout),
                cwd=task_dir,
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

    shutil.rmtree(task_dir, ignore_errors=True)

    return TaskResult(
        task_id=task_id,
        category=task["category"],
        difficulty=task["difficulty"],
        passed=passed,
        duration=duration,
        attempts=attempt,
        npcsh_output=" ||| ".join(all_outputs),
    )


def run_benchmark(
    model: str,
    provider: str,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    task_id: Optional[str] = None,
    timeout: int = 120,
    resume: bool = False,
    think=None,
    framework: str = "npcsh",
    binary: str = NPCSH_BIN,
) -> BenchmarkReport:

    setup_bench_env()
    tasks = load_tasks(category=category, difficulty=difficulty, task_id=task_id,
                       framework=framework)
    report = BenchmarkReport(model=model, provider=provider, total=len(tasks))

    import csv as csv_mod
    csv_mod.field_size_limit(10**7)
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    safe_model = model.replace("/", "_")
    checkpoint_file = report_dir / f"{framework}_{provider}_{safe_model}_running.csv"
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

    print(f"\n{framework} benchmark: {provider}/{model} (timeout={timeout}s per task)", flush=True)
    print(f"Tasks: {len(tasks)} ({len(completed_ids)} already done)", flush=True)
    print("=" * 60, flush=True)

    for i, task in enumerate(tasks):
        tid = task["id"]
        if tid in completed_ids:
            print(f"\n[{i+1}/{len(tasks)}] {tid} — skipped (resumed)", flush=True)
            continue

        print(f"\n[{i+1}/{len(tasks)}] {tid} ({task['category']}/{task['difficulty']})", flush=True)
        print(f"  {task['description']}", flush=True)

        result = run_task(task, model, provider, timeout, think=think,
                          framework=framework, binary=binary)
        report.results.append(result)

        if result.passed:
            report.passed += 1
            print(f"  PASS ({result.duration:.1f}s, {result.attempts} attempt(s))", flush=True)
        elif result.error:
            report.errors += 1
            report.failed += 1
            print(f"  ERROR: {result.error} ({result.duration:.1f}s)", flush=True)
        else:
            report.failed += 1
            print(f"  FAIL ({result.duration:.1f}s, {result.attempts} attempt(s))", flush=True)

        report.duration += result.duration

        cat = task["category"]
        if cat not in report.by_category:
            report.by_category[cat] = {"total": 0, "passed": 0}
        report.by_category[cat]["total"] += 1
        if result.passed:
            report.by_category[cat]["passed"] += 1

        diff = task["difficulty"]
        if diff not in report.by_difficulty:
            report.by_difficulty[diff] = {"total": 0, "passed": 0}
        report.by_difficulty[diff]["total"] += 1
        if result.passed:
            report.by_difficulty[diff]["passed"] += 1

        report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
        report_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_")
        checkpoint_file = report_dir / f"{framework}_{provider}_{safe_model}_running.csv"
        df = pd.DataFrame([
            {"task_id": r.task_id, "category": r.category, "difficulty": r.difficulty,
             "passed": r.passed, "attempts": r.attempts, "duration": round(r.duration, 1), "error": r.error or "",
             "output": r.npcsh_output}
            for r in report.results
        ])
        df.to_csv(checkpoint_file, index=False)

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

    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_").replace(":", "_")
    report_file = report_dir / f"{framework}_{provider}_{safe_model}_{ts}.csv"
    df = pd.DataFrame([
        {"task_id": r.task_id, "category": r.category, "difficulty": r.difficulty,
         "passed": r.passed, "duration": round(r.duration, 1), "error": r.error or "",
             "output": r.npcsh_output}
        for r in report.results
    ])
    df.to_csv(report_file, index=False)

    safe_model = model.replace("/", "_")
    checkpoint = report_dir / f"{framework}_{provider}_{safe_model}_running.csv"
    if checkpoint.exists():
        checkpoint.unlink()

    print(f"\nReport saved: {report_file}")
    return report


def compare_models(
    models: List[tuple],
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    timeout: int = 120,
    think=None,
    framework: str = "npcsh",
    binary: str = NPCSH_BIN,
) -> dict:
    """Run benchmark across multiple models and print comparison."""
    all_results = {}

    for model, provider in models:
        key = f"{provider}/{model}"
        print(f"\n{'='*60}")
        print(f"  MODEL: {key}  ({framework})")
        print(f"{'='*60}")
        report = run_benchmark(
            model=model,
            provider=provider,
            category=category,
            difficulty=difficulty,
            timeout=timeout,
            think=think,
            framework=framework,
            binary=binary,
        )
        all_results[key] = report

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
            f"{key:<30} {pct:>7.0f}% {report.passed:>5}/{report.total:<5} {report.duration:>7.0f}s",
            flush=True,
        )

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


def rerun_failed(csv_path: str, model: str, provider: str, timeout: int = 120,
                 think=None, framework: str = "npcsh", binary: str = NPCSH_BIN):
    """Re-run only the failed tasks from an existing CSV and overwrite results in-place."""
    setup_bench_env()
    import csv as csv_mod
    csv_mod.field_size_limit(10**7)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    failed_ids = [r["task_id"] for r in rows if r.get("passed", "").lower() != "true"]
    print(f"\nRerun failed tasks from {csv_path.name}  ({framework})")
    print(f"Total rows: {len(rows)}, Failed: {len(failed_ids)}")

    if not failed_ids:
        print("No failed tasks to rerun.")
        return

    all_tasks = load_tasks(framework=framework)
    task_lookup = {t["id"]: t for t in all_tasks}

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

        result = run_task(task, model, provider, timeout, think=think,
                          framework=framework, binary=binary)

        if result.passed:
            print(f"    PASS ({result.duration:.1f}s) — upgraded!", flush=True)
            improved += 1
        elif result.error:
            print(f"    ERROR: {result.error} ({result.duration:.1f}s)", flush=True)
        else:
            print(f"    FAIL ({result.duration:.1f}s)", flush=True)

        idx = row_index[tid]
        rows[idx]["passed"] = str(result.passed)
        rows[idx]["duration"] = str(round(result.duration, 1))
        rows[idx]["error"] = result.error or ""
        rows[idx]["output"] = result.npcsh_output

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    total_passed = sum(1 for r in rows if r.get("passed", "").lower() == "true")
    print(f"\nDone. Improved {improved}/{len(failed_ids)} tasks.")
    print(f"New total: {total_passed}/{len(rows)} ({100*total_passed//len(rows)}%)")
    print(f"Saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="npcsh local benchmark")
    parser.add_argument("--model", "-m", default="mistral-small3.2")
    parser.add_argument("--provider", "-p", default="ollama")
    parser.add_argument("--category", "-c", default=None)
    parser.add_argument("--difficulty", "-d", default=None)
    parser.add_argument("--task-id", "-t", default=None)
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-task wall-clock budget in seconds (default: 120)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple local models")
    parser.add_argument("--rerun-failed", type=str, default=None,
                        help="Path to existing CSV — re-run only failed tasks and overwrite")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from _running.csv checkpoint, skip already-completed tasks")
    parser.add_argument("--think", default=None,
                        help="Control thinking mode for ollama models (true/false, default: model default)")
    parser.add_argument("--framework", "-f", default="npcsh",
                        choices=list(SUPPORTED_FRAMEWORKS),
                        help="Which framework runs the task (default: npcsh)")
    parser.add_argument("--binary", default=NPCSH_BIN,
                        help="Path to the npcsh binary (default: ~/.npcsh/bin/npcsh)")

    args = parser.parse_args()

    binary = _find_npcsh_bin(args.binary)

    think_val = None
    if args.think is not None:
        if args.think.lower() in ("true", "1", "yes"):
            think_val = True
        elif args.think.lower() in ("false", "0", "no"):
            think_val = False
        else:
            think_val = args.think

    if args.rerun_failed:
        rerun_failed(
            csv_path=args.rerun_failed,
            model=args.model,
            provider=args.provider,
            timeout=args.timeout,
            think=think_val,
            framework=args.framework,
            binary=binary,
        )
    elif args.compare:
        models = [
            ("qwen3:8b", "ollama"),
            ("qwen3:1.7b", "ollama"),
            ("qwen3:4b", "ollama"),
            ("qwen3:30b", "ollama"),
            ("qwen3:0.6b", "ollama"),
            ("llama3.2:1b", "ollama"),
            ("llama3.2:3b", "ollama"),
            ("llama3.1:8b", "ollama"),
            ("gemma3:1b", "ollama"),
            ("gemma3:4b", "ollama"),
            ("gemma3:12b", "ollama"),
            ("gemma3:27b", "ollama"),
            ("mistral-small3.2:latest", "ollama"),
            ("phi4", "ollama"),
            ("gpt-oss:20b", "ollama"),
        ]
        compare_models(
            models,
            category=args.category,
            difficulty=args.difficulty,
            timeout=args.timeout,
            think=think_val,
            framework=args.framework,
            binary=binary,
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
            think=think_val,
            framework=args.framework,
            binary=binary,
        )


if __name__ == "__main__":
    main()
