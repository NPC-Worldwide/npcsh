#!/usr/bin/env python3
"""Quick benchmark: test adapter vs base on a few representative tasks."""
import csv
import subprocess
import time
from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.lora import load_adapters
from mlx_lm.generate import make_sampler

print("Loading base Qwen3.5-2B-4bit...")
base_model, tokenizer = mlx_load('mlx-community/Qwen3.5-2B-4bit')

print("Loading adapter...")
adapter_model, _ = mlx_load('mlx-community/Qwen3.5-2B-4bit')
load_adapters(adapter_model, 'adapters/qwen3.5/2b/mlx/')

sampler = make_sampler(temp=0.0)

def ask(model, instruction, max_tokens=100):
    prompt = f"<|im_start|>system\nYou are a helpful assistant that uses shell commands when needed.\n<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n"
    return mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )

def extract_tool_calls(response):
    """Extract shell commands from response."""
    import re
    m = re.search(r'\[tool\](.+?)\[/tool\]', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```(?:bash|sh|shell)?\n(.+?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip().split('\n')[0]
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('*'):
            if any(cmd in line for cmd in ['find', 'ls', 'cat', 'echo', 'mkdir', 'touch', 'wc', 'grep', 'dd', 'date', 'printf']):
                return line
    return None

def run_task(task, model, model_name):
    setup = task.get('setup_cmd', '')
    instruction = task['instruction']
    verify = task['verify_cmd']

    if setup:
        subprocess.run(setup, shell=True, capture_output=True)

    start = time.time()
    response = ask(model, instruction, max_tokens=200)
    duration = time.time() - start

    cmd = extract_tool_calls(response)
    if cmd:
        subprocess.run(cmd, shell=True, capture_output=True)

    try:
        result = subprocess.run(verify, shell=True, capture_output=True)
        passed = result.returncode == 0
    except:
        passed = False

    return {
        'model': model_name,
        'task': task['id'],
        'passed': passed,
        'duration': duration,
        'cmd': cmd,
        'response': response[:80],
    }

tasks = []
with open('/Users/caug/npcww/npc-core/npcsh/npcsh/benchmark/tasks.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tasks.append(row)

for task in tasks[:5]:
    print(f"\n=== {task['id']} ({task['difficulty']}) ===")
    print(f"Instruction: {task['instruction'][:60]}")

    base_result = run_task(task, base_model, "base")
    print(f"  BASE: {'PASS' if base_result['passed'] else 'FAIL'} | {base_result['duration']:.1f}s | cmd={base_result['cmd']}")

    adapter_result = run_task(task, adapter_model, "adapter")
    print(f"  ADPT: {'PASS' if adapter_result['passed'] else 'FAIL'} | {adapter_result['duration']:.1f}s | cmd={adapter_result['cmd']}")

print("\nDone.")
