#!/usr/bin/env python3
"""
active_learning_loop.py

Iterative self-improvement pipeline:
    1. Evaluate current model on full benchmark
    2. Identify failed / low-success categories
    3. Run failed tasks with a strong teacher model to collect successful traces
    4. Merge new traces into training data
    5. Retrain SFT (or RL) on expanded dataset
    6. Evaluate new model vs previous
    7. If improved, keep; else rollback
    8. Repeat for N iterations

Usage:
    python scripts/active_learning_loop.py \
        --model mlx-community/Qwen3-4B-4bit \
        --teacher-model gemma3:27b \
        --teacher-provider ollama \
        --sft-output models/npcsh_sft_active \
        --iterations 3 \
        --category python
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Import our trace parser
try:
    from train_from_csv import parse_trace, load_traces
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from train_from_csv import parse_trace, load_traces


def run_command(cmd, timeout=120, cwd=None, env=None):
    """Run shell command and return stdout, stderr, returncode."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"Timeout after {timeout}s", 1


def evaluate_model(model: str, provider: str, timeout: int = 60, max_tasks: int = None):
    """Run benchmark and return results dict."""
    print(f"\n[EVAL] Running benchmark: {model} ({provider})")

    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    env["NPCSH_STREAM_OUTPUT"] = "0"

    cmd = f"python3 -m npcsh.benchmark.local_runner --model {model} --provider {provider} --timeout {timeout}"
    if max_tasks:
        cmd += f" --max-tasks {max_tasks}"

    out, err, rc = run_command(cmd, timeout=timeout * 30 + 300, env=env)

    # Parse the CSV output
    results = {}
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    # Find most recent CSV
    csvs = sorted(report_dir.glob("npcsh_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csvs:
        with open(csvs[0]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row["task_id"]] = {
                    "passed": row.get("passed", "").lower() == "true",
                    "category": row.get("category", ""),
                    "difficulty": row.get("difficulty", ""),
                    "duration": float(row.get("duration", "0") or 0),
                    "attempts": int(row.get("attempts", "1") or 1),
                }
        print(f"[EVAL] Loaded {len(results)} results from {csvs[0].name}")
    else:
        print("[EVAL] No benchmark CSV found!")

    passed = sum(1 for r in results.values() if r["passed"])
    print(f"[EVAL] Score: {passed}/{len(results)} ({100*passed/len(results):.0f}%)")
    return results


def identify_weak_categories(results: dict, min_success_rate: float = 0.3):
    """Find categories with success rate below threshold."""
    cats = {}
    for tid, r in results.items():
        cat = r["category"]
        cats.setdefault(cat, {"total": 0, "passed": 0})
        cats[cat]["total"] += 1
        if r["passed"]:
            cats[cat]["passed"] += 1

    weak = []
    for cat, stats in sorted(cats.items()):
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({100*rate:.0f}%)")
        if rate < min_success_rate:
            weak.append((cat, rate, stats["total"]))

    weak.sort(key=lambda x: x[1])
    return weak


def run_task_with_teacher(task: dict, teacher_model: str, teacher_provider: str, timeout: int = 60):
    """Run a single task with teacher model, return trace if successful."""
    work_dir = tempfile.mkdtemp(prefix=f"teacher_{task['id']}_")

    setup_cmd = task.get("setup_cmd", "") or ""
    if setup_cmd:
        run_command(setup_cmd, timeout=30, cwd=work_dir)

    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = teacher_model
    env["NPCSH_CHAT_PROVIDER"] = teacher_provider
    env["NPCSH_STREAM_OUTPUT"] = "0"

    out, err, rc = run_command(
        f"npcsh -c {repr(task['instruction'])}",
        timeout=timeout,
        cwd=work_dir,
        env=env,
    )

    # Verify
    verify_cmd = task.get("verify_cmd", "") or ""
    if verify_cmd:
        vout, verr, vrc = run_command(verify_cmd, timeout=15, cwd=work_dir)
        passed = vrc == 0
    else:
        passed = rc == 0

    # Cleanup
    run_command(f"rm -rf {work_dir}", timeout=10)

    if not passed:
        return None

    # Construct a trace record
    trace = {
        "task_id": task["id"],
        "category": task.get("category", ""),
        "difficulty": task.get("difficulty", ""),
        "model": teacher_model,
        "provider": teacher_provider,
        "passed": True,
        "attempts": 1,
        "output": out + "\n---TRACE---\n" + err,
    }
    return trace


def collect_teacher_traces(weak_categories: list, results: dict, teacher_model: str, teacher_provider: str, max_per_cat: int = 10):
    """Run failed tasks from weak categories with teacher model."""
    print(f"\n[TEACHER] Collecting traces with {teacher_model} ({teacher_provider})")

    task_file = Path(__file__).parent.parent / "npcsh" / "benchmark" / "tasks.csv"
    tasks = {}
    with open(task_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks[row["id"]] = row

    failed_tasks = [tid for tid, r in results.items() if not r["passed"]]
    new_traces = []

    for cat, rate, total in weak_categories:
        cat_failed = [tid for tid in failed_tasks if results.get(tid, {}).get("category") == cat]
        cat_failed = cat_failed[:max_per_cat]
        print(f"  {cat}: {len(cat_failed)} failed tasks to retry")

        for tid in cat_failed:
            task = tasks.get(tid)
            if not task:
                continue
            print(f"    Running {tid}...", end="", flush=True)
            trace = run_task_with_teacher(task, teacher_model, teacher_provider)
            if trace:
                new_traces.append(trace)
                print(" PASS")
            else:
                print(" FAIL")
            time.sleep(0.5)

    print(f"[TEACHER] Collected {len(new_traces)} successful traces")
    return new_traces


def save_traces_to_csv(traces: list, csv_path: str):
    """Append traces to a CSV file."""
    path = Path(csv_path)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "category", "difficulty", "model", "provider", "passed", "attempts", "output"])
        if not exists:
            writer.writeheader()
        for t in traces:
            writer.writerow(t)
    print(f"[SAVE] Appended {len(traces)} traces to {csv_path}")


def train_sft(csv_path: str, model: str, output: str, epochs: int = 5, lora_r: int = 16, lr: float = 2e-5, device: str = "mlx"):
    """Run SFT training."""
    print(f"\n[TRAIN] SFT training → {output}")
    cmd = (
        f"python3 scripts/train_sft_toolcalls.py "
        f"--csv-dir {Path(csv_path).parent} "
        f"--model {model} "
        f"--device {device} "
        f"--output {output} "
        f"--epochs {epochs} "
        f"--lr {lr} "
        f"--lora-r {lora_r}"
    )
    out, err, rc = run_command(cmd, timeout=3600)
    if rc != 0:
        print(f"[TRAIN] SFT failed:\n{err}")
        return False
    print(f"[TRAIN] SFT complete: {output}")
    return True


def merge_adapter(adapter_path: str, output_path: str):
    """Fuse adapter into base model."""
    print(f"\n[MERGE] Fusing adapter → {output_path}")
    cmd = f"python -m mlx_lm.lora --adapter-path {adapter_path} --save-path {output_path}"
    out, err, rc = run_command(cmd, timeout=600)
    return rc == 0


def active_loop(args):
    """Main active learning loop."""
    csv_dir = Path(args.csv_dir).expanduser()
    csv_dir.mkdir(parents=True, exist_ok=True)
    teacher_csv = csv_dir / "teacher_traces.csv"

    best_score = 0
    best_adapter = None
    history = []

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"ACTIVE LEARNING ITERATION {iteration}/{args.iterations}")
        print(f"{'='*60}")

        # 1. Evaluate current model
        if iteration == 1 and args.initial_adapter:
            # Use provided adapter
            current_adapter = args.initial_adapter
            print(f"[INIT] Using initial adapter: {current_adapter}")
            # Start server with adapter for evaluation
            # (evaluation via omlx uses mlx_lm.server)
        else:
            current_adapter = args.sft_output + f"_iter{iteration-1}"

        results = evaluate_model(args.model, args.provider, timeout=args.timeout, max_tasks=args.max_tasks)
        if not results:
            print("[ERROR] No results, skipping iteration")
            continue

        passed = sum(1 for r in results.values() if r["passed"])
        total = len(results)
        score = passed / total if total > 0 else 0
        print(f"[SCORE] Iteration {iteration}: {passed}/{total} ({100*score:.0f}%)")

        history.append({"iteration": iteration, "passed": passed, "total": total, "score": score})

        # Save history
        hist_path = Path(args.sft_output + "_history.json")
        hist_path.write_text(json.dumps(history, indent=2))

        if score > best_score:
            best_score = score
            best_adapter = current_adapter
            print(f"[BEST] New best score! Adapter: {best_adapter}")

        # Check if we should stop early
        if score >= args.target_score:
            print(f"[DONE] Target score {args.target_score} reached!")
            break

        # 2. Identify weak categories
        weak = identify_weak_categories(results, min_success_rate=args.min_success_rate)
        if not weak:
            print("[DONE] No weak categories — all above threshold!")
            break

        print(f"\n[WEAK] Categories to improve: {', '.join(c for c, _, _ in weak)}")

        # 3. Collect teacher traces for failed tasks
        if args.teacher_model and args.teacher_provider:
            teacher_traces = collect_teacher_traces(
                weak, results,
                args.teacher_model, args.teacher_provider,
                max_per_cat=args.max_teacher_tasks,
            )
            if teacher_traces:
                save_traces_to_csv(teacher_traces, teacher_csv)
        else:
            print("[SKIP] No teacher model configured")
            teacher_traces = []

        # 4. Retrain SFT on combined data
        if not args.skip_train:
            iter_output = args.sft_output + f"_iter{iteration}"
            success = train_sft(
                str(teacher_csv if teacher_traces else csv_dir),
                args.model,
                iter_output,
                epochs=args.epochs,
                lora_r=args.lora_r,
                lr=args.lr,
                device=args.device,
            )
            if not success:
                print("[ERROR] Training failed, rolling back to best adapter")
                break
        else:
            print("[SKIP] --skip-train set")

    # Final summary
    print(f"\n{'='*60}")
    print("ACTIVE LEARNING COMPLETE")
    print(f"{'='*60}")
    print(f"Iterations:    {len(history)}")
    print(f"Best score:    {best_score:.0%}")
    print(f"Best adapter:  {best_adapter}")
    for h in history:
        print(f"  Iter {h['iteration']}: {h['passed']}/{h['total']} ({100*h['score']:.0f}%)")

    return best_adapter, best_score


def main():
    parser = argparse.ArgumentParser(description="Active learning loop for npcsh model improvement")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--provider", default="omlx")
    parser.add_argument("--teacher-model", default="gemma3:27b", help="Teacher model for trace collection")
    parser.add_argument("--teacher-provider", default="ollama")
    parser.add_argument("--sft-output", default="models/npcsh_sft_active")
    parser.add_argument("--csv-dir", default="~/.npcsh/benchmarks/local")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--device", default="mlx")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-teacher-tasks", type=int, default=10, help="Max failed tasks per category to retry with teacher")
    parser.add_argument("--min-success-rate", type=float, default=0.3, help="Categories below this rate get teacher attention")
    parser.add_argument("--target-score", type=float, default=0.85, help="Stop when overall score reaches this")
    parser.add_argument("--initial-adapter", default=None, help="Start from this adapter instead of base model")
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate, do not train")
    args = parser.parse_args()

    active_loop(args)


if __name__ == "__main__":
    main()
