"""
Cross-framework benchmark runner.

Runs the same tasks against multiple agentic coding tools, all using
the same Ollama model, to isolate framework quality from model quality.

Outputs a CSV with one row per (framework, task) pair including the
model's actual output so you can inspect what each framework did.

Supported frameworks:
  - npcsh:     npcsh -c "instruction"
  - opencode:  ~/.opencode/bin/opencode run "instruction" -m ollama/model
  - nanocoder: nanocoder run "instruction"
  - claude:    claude -p "instruction" --dangerously-skip-permissions (via Ollama)

Usage:
    python -m npcsh.benchmark.cross_framework_runner
    python -m npcsh.benchmark.cross_framework_runner --frameworks npcsh opencode claude
    python -m npcsh.benchmark.cross_framework_runner --model mistral-small3.2
    python -m npcsh.benchmark.cross_framework_runner --category shell
    python -m npcsh.benchmark.cross_framework_runner --task-id file-create-01
"""

import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional


# Categories that don't require npcsh-specific jinxes
FRAMEWORK_AGNOSTIC_CATEGORIES = {
    "shell", "file-ops", "python", "data", "system",
    "text", "debug", "git", "multi-step", "scripting",
}


def load_tasks(
    task_file: Optional[str] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    task_id: Optional[str] = None,
    agnostic_only: bool = True,
) -> list:
    if task_file is None:
        task_file = Path(__file__).parent / "tasks.json"
    with open(task_file) as f:
        tasks = json.load(f)
    if agnostic_only:
        tasks = [t for t in tasks if t["category"] in FRAMEWORK_AGNOSTIC_CATEGORIES]
    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if difficulty:
        tasks = [t for t in tasks if t["difficulty"] == difficulty]
    return tasks


def clean_task_artifacts():
    """Remove /tmp files created by tasks so runs don't bleed into each other."""
    import shutil
    patterns = [
        # shell
        "/tmp/result.txt", "/tmp/pyfiles.txt", "/tmp/uname.txt", "/tmp/nums.txt",
        "/tmp/dirs.txt", "/tmp/comment_count.txt", "/tmp/largest.txt",
        "/tmp/now.txt", "/tmp/ext_count.txt", "/tmp/big_etc.txt",
        # file-ops
        "/tmp/hello.txt", "/tmp/person.json", "/tmp/config.ini", "/tmp/env.sh",
        "/tmp/colors.txt", "/tmp/requirements.txt", "/tmp/docker-compose.yml",
        "/tmp/Makefile",
        # python
        "/tmp/fib.py", "/tmp/rev.py", "/tmp/calc.py", "/tmp/wordcount.py",
        "/tmp/sample.txt", "/tmp/wc_result.json", "/tmp/fizzbuzz.py",
        "/tmp/even.py", "/tmp/palindrome.py", "/tmp/stats.py",
        "/tmp/matrix.py", "/tmp/tree.py",
        # data
        "/tmp/data.csv", "/tmp/analyze.py", "/tmp/stats.json",
        "/tmp/scores.csv", "/tmp/inventory.json", "/tmp/total.py",
        "/tmp/books.json", "/tmp/temps.csv", "/tmp/convert.py", "/tmp/temps_f.csv",
        "/tmp/sales.csv", "/tmp/top_sales.py", "/tmp/contacts.json",
        "/tmp/employees.csv", "/tmp/dept_avg.py",
        "/tmp/users_a.json", "/tmp/users_b.json", "/tmp/merge.py",
        "/tmp/merged_users.json", "/tmp/weather.csv", "/tmp/weather_report.py",
        "/tmp/weather_report.txt",
        # system
        "/tmp/sysinfo.txt", "/tmp/env_info.txt", "/tmp/path_vars.txt",
        "/tmp/proc_count.txt", "/tmp/disk_free.txt", "/tmp/usernames.txt",
        "/tmp/uptime.txt", "/tmp/top_mem.txt", "/tmp/home_var.txt",
        "/tmp/sys_summary.txt",
        # text
        "/tmp/log.txt", "/tmp/errors.txt", "/tmp/fruits.txt", "/tmp/sorted_fruits.txt",
        "/tmp/words.txt", "/tmp/unique_counts.txt",
        "/tmp/mixed_case.txt", "/tmp/lower.txt",
        "/tmp/addresses.txt", "/tmp/cities.txt",
        "/tmp/poem.txt", "/tmp/line_lengths.txt",
        "/tmp/emails.txt", "/tmp/extracted_emails.txt",
        "/tmp/numbers.txt", "/tmp/even_numbers.txt",
        "/tmp/csv_raw.txt", "/tmp/table.txt",
        "/tmp/paragraph.txt", "/tmp/word_stats.txt",
        # debug
        "/tmp/broken.py", "/tmp/buggy.py",
        "/tmp/fix_indent.py", "/tmp/fix_loop.py", "/tmp/fix_dict.py",
        "/tmp/fix_recursion.py", "/tmp/fix_sort.py", "/tmp/fix_import.py",
        "/tmp/fix_class.py", "/tmp/fix_file.py", "/tmp/debug_test.txt",
        # git
        "/tmp/git_history.txt", "/tmp/my_diff.txt", "/tmp/status_output.txt",
        "/tmp/stash_list.txt",
        # multi-step
        "/tmp/report.txt", "/tmp/users.json",
        "/tmp/backup.sh", "/tmp/backup.tar.gz",
        "/tmp/todo.py", "/tmp/todos.txt",
        "/tmp/disk_usage.txt", "/tmp/file_count.txt",
        "/tmp/animals.txt", "/tmp/animal_results.txt",
        "/tmp/server_check.sh", "/tmp/port_status.txt",
        "/tmp/grades.csv", "/tmp/averages.py", "/tmp/averages.csv",
        "/tmp/test_mathpkg.py",
        "/tmp/log_analyzer.py", "/tmp/app.log", "/tmp/log_summary.json",
        # scripting
        "/tmp/greet.sh", "/tmp/countdown.sh", "/tmp/fileinfo.sh",
        "/tmp/extension_count.sh", "/tmp/rename_ext.sh",
        "/tmp/even_odd.sh", "/tmp/monitor.sh", "/tmp/monitor_log.txt",
        "/tmp/largest_files.sh", "/tmp/csv2json.sh", "/tmp/test_convert.csv",
        "/tmp/converted.json", "/tmp/semver.sh",
        # npcsh-specific: image-gen
        "/tmp/sunset.png", "/tmp/cat.png", "/tmp/generated.png",
        "/tmp/robot.png", "/tmp/city.png",
        # npcsh-specific: audio-gen
        "/tmp/welcome.wav", "/tmp/welcome.mp3",
        "/tmp/pangram.wav", "/tmp/pangram.mp3",
        "/tmp/speech.wav", "/tmp/speech.mp3",
        "/tmp/haiku.wav", "/tmp/haiku.mp3",
        "/tmp/test_audio.wav", "/tmp/test_audio.mp3",
        # npcsh-specific: web-search
        "/tmp/search_results.txt", "/tmp/linux_creator.txt",
        "/tmp/japan_pop.txt", "/tmp/python_year.txt", "/tmp/js_frameworks.txt",
        # npcsh-specific: delegation
        "/tmp/primes.py", "/tmp/fib_research.py",
        "/tmp/sysreport.sh", "/tmp/sysreport.txt",
        "/tmp/sorter.py", "/tmp/validators.py",
        # npcsh-specific: tool-chain
        "/tmp/languages.txt", "/tmp/rank.py", "/tmp/forest.png",
        "/tmp/img_info.py", "/tmp/capital.txt", "/tmp/capital_audio.wav",
        "/tmp/fetch_parse.py", "/tmp/sample_api.json", "/tmp/cheapest.txt",
        "/tmp/dog.png", "/tmp/gallery.html",
    ]
    for p in patterns:
        try:
            os.remove(p)
        except (OSError, FileNotFoundError):
            pass
    for d in [
        "/tmp/mydir", "/tmp/myrepo", "/tmp/project", "/tmp/webapp",
        "/tmp/gittest", "/tmp/gitbranch", "/tmp/gittag", "/tmp/gitlog",
        "/tmp/gitignore", "/tmp/gitdiff", "/tmp/gitmerge", "/tmp/gitstatus",
        "/tmp/gitstash", "/tmp/mathpkg", "/tmp/rentest",
    ]:
        shutil.rmtree(d, ignore_errors=True)


def build_command(framework: str, instruction: str, model: str) -> tuple:
    """Return (cmd_list, env_dict) for a given framework."""
    env = os.environ.copy()
    opencode_bin = os.path.expanduser("~/.opencode/bin")
    if opencode_bin not in env.get("PATH", ""):
        env["PATH"] = opencode_bin + ":" + env.get("PATH", "")

    if framework == "npcsh":
        env["NPCSH_CHAT_MODEL"] = model
        env["NPCSH_CHAT_PROVIDER"] = "ollama"
        env["NPCSH_STREAM_OUTPUT"] = "0"
        env.setdefault("OLLAMA_HOST", "http://localhost:11434")
        env["NPCSH_OLLAMA_NUM_CTX"] = "32768"
        return ["npcsh", "-c", instruction], env

    elif framework == "opencode":
        opencode_path = os.path.expanduser("~/.opencode/bin/opencode")
        return [opencode_path, "run", instruction, "-m", f"ollama/{model}"], env

    elif framework == "nanocoder":
        return ["nanocoder", "run", instruction], env

    elif framework == "claude":
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"
        env["ANTHROPIC_BASE_URL"] = "http://localhost:11434"
        env["DISABLE_AUTOUPDATER"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        # Remove nesting detection so we can launch from inside claude code
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        return [
            "claude", "-p", instruction,
            "--dangerously-skip-permissions",
            "--model", model,
        ], env

    else:
        raise ValueError(f"Unknown framework: {framework}")


def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    # Also try killing children via pkill
    try:
        subprocess.run(["pkill", "-9", "-P", str(pid)],
                       capture_output=True, timeout=5)
    except Exception:
        pass


def run_task(task: dict, framework: str, model: str, timeout: int = 120) -> dict:
    """Run a single task, return a dict with all fields for CSV."""
    task_id = task["id"]
    instruction = task["instruction"]
    verify_cmd = task["verify_cmd"]
    task_timeout = min(max(task.get("timeout", timeout), timeout), 300)  # cap at 5 min
    verify_timeout = task.get("verify_timeout", 30)

    clean_task_artifacts()
    cmd, env = build_command(framework, instruction, model)

    start = time.time()
    output = ""
    error = ""
    passed = False

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=task_timeout)
            output = (stdout or "") + (stderr or "")
        except subprocess.TimeoutExpired:
            kill_process_tree(proc.pid)
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            error = "timeout"
            output = "TIMEOUT after {}s".format(task_timeout)

    except FileNotFoundError:
        error = f"{framework} not found"
    except Exception as e:
        error = str(e)

    duration = time.time() - start

    # Verify if no error
    if not error:
        time.sleep(0.5)
        try:
            verify = subprocess.run(
                ["bash", "-c", verify_cmd],
                capture_output=True, text=True, timeout=verify_timeout,
            )
            passed = verify.returncode == 0
        except Exception as e:
            error = f"verify error: {e}"

    # Classify what the model did
    action_type = "none"
    if error:
        action_type = error
    elif passed:
        action_type = "executed"
    else:
        # Check if model gave advice instead of acting
        lower_out = output.lower()
        if any(phrase in lower_out for phrase in [
            "i'm unable to", "i cannot", "i can't",
            "here's how", "you can use", "you can do",
            "follow these steps", "open a terminal",
            "i don't have the capability",
            "i'm sorry, but i don't",
        ]):
            action_type = "gave_advice"
        elif "error" in lower_out or "traceback" in lower_out:
            action_type = "crashed"
        else:
            action_type = "attempted_but_failed"

    # Truncate output for CSV readability
    output_short = output.strip().replace("\n", " | ")[:500]

    return {
        "framework": framework,
        "task_id": task_id,
        "category": task["category"],
        "difficulty": task["difficulty"],
        "passed": passed,
        "action_type": action_type,
        "duration": round(duration, 1),
        "error": error,
        "output_preview": output_short,
    }


def run_comparison(
    frameworks: List[str],
    model: str = "mistral-small3.2",
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    task_id: Optional[str] = None,
    timeout: int = 120,
) -> str:
    tasks = load_tasks(category=category, difficulty=difficulty, task_id=task_id)

    # CSV output path
    report_dir = Path.home() / ".npcsh" / "benchmarks" / "cross_framework"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = report_dir / f"{model}_{ts}.csv"

    fieldnames = [
        "framework", "task_id", "category", "difficulty",
        "passed", "action_type", "duration", "error", "output_preview",
    ]

    all_rows = []

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for fw in frameworks:
            print(f"\n{'='*60}", flush=True)
            print(f"  {fw} | {model} via ollama | {len(tasks)} tasks", flush=True)
            print(f"{'='*60}", flush=True)

            fw_pass = 0
            fw_total = 0

            for i, task in enumerate(tasks):
                tid = task["id"]
                print(f"  [{i+1}/{len(tasks)}] {tid}", end="", flush=True)

                row = run_task(task, fw, model, timeout)
                writer.writerow(row)
                csvfile.flush()
                all_rows.append(row)

                fw_total += 1
                if row["passed"]:
                    fw_pass += 1

                status = "PASS" if row["passed"] else row["action_type"].upper()
                print(f"  {status}  ({row['duration']}s)", flush=True)

            pct = 100 * fw_pass / fw_total if fw_total else 0
            print(f"\n  {fw}: {fw_pass}/{fw_total} ({pct:.0f}%)", flush=True)

    # Print summary comparison
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY | model: {model} via ollama", flush=True)
    print(f"{'='*70}", flush=True)

    for fw in frameworks:
        fw_rows = [r for r in all_rows if r["framework"] == fw]
        total = len(fw_rows)
        passed = sum(1 for r in fw_rows if r["passed"])
        advice = sum(1 for r in fw_rows if r["action_type"] == "gave_advice")
        crashed = sum(1 for r in fw_rows if r["action_type"] == "crashed")
        timeout_count = sum(1 for r in fw_rows if r["action_type"] == "timeout")
        attempted = sum(1 for r in fw_rows if r["action_type"] == "attempted_but_failed")
        pct = 100 * passed / total if total else 0
        print(f"\n  {fw}: {passed}/{total} ({pct:.0f}%)", flush=True)
        print(f"    executed & passed: {passed}", flush=True)
        print(f"    attempted but failed: {attempted}", flush=True)
        print(f"    gave advice only: {advice}", flush=True)
        print(f"    crashed: {crashed}", flush=True)
        print(f"    timed out: {timeout_count}", flush=True)

    # Per-task comparison table
    print(f"\n{'Task':<25}", end="", flush=True)
    for fw in frameworks:
        print(f" {fw:>12}", end="", flush=True)
    print(flush=True)
    print("-" * (25 + 13 * len(frameworks)), flush=True)

    for task in tasks:
        tid = task["id"]
        print(f"{tid:<25}", end="", flush=True)
        for fw in frameworks:
            row = next((r for r in all_rows if r["framework"] == fw and r["task_id"] == tid), None)
            if row is None:
                label = "--"
            elif row["passed"]:
                label = "PASS"
            else:
                label = row["action_type"][:10]
            print(f" {label:>12}", end="", flush=True)
        print(flush=True)

    print(f"\nCSV saved: {csv_path}", flush=True)
    return str(csv_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-framework benchmark")
    parser.add_argument(
        "--frameworks", "-f", nargs="+",
        default=["npcsh", "opencode", "nanocoder", "claude"],
        help="Frameworks to benchmark",
    )
    parser.add_argument("--model", "-m", default="mistral-small3.2")
    parser.add_argument("--category", "-c", default=None)
    parser.add_argument("--difficulty", "-d", default=None)
    parser.add_argument("--task-id", "-t", default=None)
    parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    run_comparison(
        frameworks=args.frameworks,
        model=args.model,
        category=args.category,
        difficulty=args.difficulty,
        task_id=args.task_id,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
