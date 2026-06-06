#!/usr/bin/env python3
"""
evaluate_adapter.py

Evaluate a trained adapter (or fused model) on the benchmark suite.
Automatically starts mlx_lm.server if needed, runs benchmark, then kills server.

Usage:
    python scripts/evaluate_adapter.py --adapter models/npcsh_sft_toolcalls_all
    python scripts/evaluate_adapter.py --adapter models/npcsh_sft_toolcalls_all --category python
    python scripts/evaluate_adapter.py --fused-model models/npcsh_sft_toolcalls_all_fused
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


def find_free_port(start=8000, max_port=9000):
    import socket
    for port in range(start, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found")


def wait_for_server(port, timeout=30):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_server(model: str, adapter_path: str = None, port: int = 8000):
    """Start mlx_lm.server in background, return process."""
    cmd = ["python", "-m", "mlx_lm.server", "--model", model, "--port", str(port)]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    print(f"[SERVER] Starting mlx_lm.server on port {port}...")
    if adapter_path:
        print(f"  model={model} adapter={adapter_path}")
    else:
        print(f"  model={model}")

    log_file = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    if not wait_for_server(port, timeout=60):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"[SERVER] Failed to start. Log: {log_file.name}")
        with open(log_file.name) as f:
            print(f.read())
        return None, None

    print(f"[SERVER] Ready on port {port}")
    return proc, log_file.name


def stop_server(proc):
    if proc is None:
        return
    print("[SERVER] Stopping...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("[SERVER] Stopped")


def run_benchmark(model: str, provider: str, port: int, timeout: int = 60, category: str = None, max_tasks: int = None):
    """Run benchmark via subprocess."""
    print(f"\n[BENCH] Running benchmark...")

    env = os.environ.copy()
    env["NPCSH_CHAT_MODEL"] = model
    env["NPCSH_CHAT_PROVIDER"] = provider
    env["NPCSH_STREAM_OUTPUT"] = "0"

    cmd = (
        f"python3 -m npcsh.benchmark.local_runner "
        f"--model {model} --provider {provider} --timeout {timeout}"
    )
    if category:
        cmd += f" --category {category}"
    if max_tasks:
        cmd += f" --max-tasks {max_tasks}"

    proc = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout * 30 + 300,
        env=env,
    )

    if proc.returncode != 0:
        print(f"[BENCH] Benchmark runner exited {proc.returncode}")
        print(proc.stderr[:2000])

    # Find the CSV report
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "local"
    csvs = sorted(report_dir.glob("npcsh_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        print("[BENCH] No benchmark CSV found!")
        return {}

    latest = csvs[0]
    results = {}
    with open(latest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["task_id"]] = {
                "passed": row.get("passed", "").lower() == "true",
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "duration": float(row.get("duration", "0") or 0),
                "attempts": int(row.get("attempts", "1") or 1),
            }

    return results, latest


def summarize(results: dict):
    total = len(results)
    passed = sum(1 for r in results.values() if r["passed"])
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print(f"{'='*50}")

    cats = {}
    for tid, r in results.items():
        cat = r["category"]
        cats.setdefault(cat, {"total": 0, "passed": 0})
        cats[cat]["total"] += 1
        if r["passed"]:
            cats[cat]["passed"] += 1

    for cat in sorted(cats.keys()):
        stats = cats[cat]
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {cat:12s} {bar} {stats['passed']:3d}/{stats['total']:3d} ({100*rate:3.0f}%)")

    # Difficulty breakdown
    diffs = {}
    for tid, r in results.items():
        d = r["difficulty"]
        diffs.setdefault(d, {"total": 0, "passed": 0})
        diffs[d]["total"] += 1
        if r["passed"]:
            diffs[d]["passed"] += 1

    print(f"\nBy difficulty:")
    for d in sorted(diffs.keys()):
        stats = diffs[d]
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {d:8s} {stats['passed']:3d}/{stats['total']:3d} ({100*rate:3.0f}%)")

    return {"passed": passed, "total": total, "categories": cats, "difficulty": diffs}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained adapter on benchmark tasks")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--fused-model", default=None, help="Path to fused model (instead of adapter)")
    parser.add_argument("--provider", default="omlx")
    parser.add_argument("--port", type=int, default=None, help="mlx_lm.server port (auto-find if not set)")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--category", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output-json", default=None, help="Save summary to JSON file")
    args = parser.parse_args()

    if args.fused_model:
        eval_model = args.fused_model
        adapter_path = None
    elif args.adapter:
        eval_model = args.model
        adapter_path = args.adapter
    else:
        print("Error: specify --adapter or --fused-model")
        sys.exit(1)

    port = args.port or find_free_port()
    server_proc, log_file = start_server(eval_model, adapter_path, port)
    if server_proc is None:
        print("Failed to start server")
        sys.exit(1)

    try:
        results, csv_path = run_benchmark(
            eval_model, args.provider, port,
            timeout=args.timeout,
            category=args.category,
            max_tasks=args.max_tasks,
        )
        summary = summarize(results)
        summary["model"] = eval_model
        summary["adapter"] = adapter_path
        summary["csv_path"] = str(csv_path)

        if args.output_json:
            Path(args.output_json).write_text(json.dumps(summary, indent=2))
            print(f"\nSummary saved to {args.output_json}")
    finally:
        stop_server(server_proc)
        if log_file:
            Path(log_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
