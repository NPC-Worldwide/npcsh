#!/usr/bin/env python3
"""Proper eval: test adapter through npcsh-style tool call parsing."""

import csv
import subprocess
import time
import re
import json

# Load tasks
tasks = []
with open("/Users/caug/npcww/npc-core/npcsh/npcsh/benchmark/tasks.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["category"] == "shell" and row["difficulty"] == "easy":
            tasks.append(row)

tasks = tasks[:5]

from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.lora import load_adapters
from mlx_lm.generate import make_sampler

# Load models
print("Loading models...")
base, tokenizer = mlx_load("mlx-community/Qwen3.5-2B-4bit")
adapter, _ = mlx_load("mlx-community/Qwen3.5-2B-4bit")
load_adapters(adapter, "adapters/qwen3.5/2b/mlx/")
sampler = make_sampler(temp=0.0)

SYSTEM_MSG = 'You are npcsh, a shell assistant. When asked to do something, use the shell tool by outputting: <tool_call>\\n{"name":"shell","arguments":{"bash_command":"YOUR_COMMAND"}}\\n</tool_call>'


def ask(model, instruction):
    prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    return mlx_generate(
        model, tokenizer, prompt=prompt, max_tokens=100, sampler=sampler, verbose=False
    )


def extract_tool_call(response):
    """Parse npcsh-style tool calls from response."""
    # Pattern: <tool_call>\n{"name":"shell","arguments":{"bash_command":"..."}}\n</tool_call>
    m = re.search(r"<tool_call>\s*(\{.+?\})\s*</tool_call>", response, re.DOTALL)
    if m:
        try:
            tc = json.loads(m.group(1))
            if tc.get("name") == "shell":
                args = tc.get("arguments", {})
                return args.get("bash_command") or args.get("command")
        except:
            pass

    # Fallback: look for ```bash blocks
    m = re.search(r"```(?:bash|sh|shell)?\n(.+?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()

    # Fallback 2: first line that looks like a command
    for line in response.split("\n"):
        line = line.strip()
        if line and any(
            c in line
            for c in [
                "find ",
                "ls ",
                "echo ",
                "cat ",
                "mkdir ",
                "touch ",
                "wc ",
                "grep ",
                "date ",
                "printf ",
                "> ",
                "| ",
            ]
        ):
            return line.strip()
    return None


def run_task(task, model, model_name):
    # Cleanup /tmp
    for prefix in [
        "/tmp/countme",
        "/tmp/result",
        "/tmp/pydir",
        "/tmp/uname",
        "/tmp/nums",
        "/tmp/dirtest",
        "/tmp/comments",
        "/tmp/sizetest",
        "/tmp/exttest",
        "/tmp/bigtest",
        "/tmp/now",
        "/tmp/pyfiles",
        "/tmp/hello",
        "/tmp/mydir",
        "/tmp/person",
        "/tmp/config",
        "/tmp/env",
        "/tmp/colors",
        "/tmp/webapp",
        "/tmp/requirements",
        "/tmp/docker",
    ]:
        subprocess.run(f"rm -rf {prefix}*", shell=True, capture_output=True)

    setup = task.get("setup_cmd", "")
    if setup:
        subprocess.run(setup, shell=True, capture_output=True)

    start = time.time()
    response = ask(model, task["instruction"])
    time.time() - start

    cmd = extract_tool_call(response)
    if cmd:
        subprocess.run(cmd, shell=True, capture_output=True)

    verify = task["verify_cmd"]
    try:
        result = subprocess.run(verify, shell=True, capture_output=True, timeout=5)
        passed = result.returncode == 0
    except:
        passed = False

    print(f"  {model_name}: {'PASS' if passed else 'FAIL'} | cmd={cmd}")
    print(f"    Response: {response[:100]}")

    return passed


# Run
print(f"{'=' * 60}")
print("Testing with npcsh-style tool call prompt")
print(f"{'=' * 60}\n")

base_pass = 0
adapt_pass = 0

for task in tasks:
    print(f"\n{task['id']}: {task['instruction'][:60]}")
    if run_task(task, base, "BASE"):
        base_pass += 1
    if run_task(task, adapter, "ADAPT"):
        adapt_pass += 1

print(f"\n{'=' * 60}")
print("RESULTS")
print(f"{'=' * 60}")
print(f"Base:    {base_pass}/{len(tasks)}")
print(f"Adapter: {adapt_pass}/{len(tasks)}")
