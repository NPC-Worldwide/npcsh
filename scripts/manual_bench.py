#!/usr/bin/env python3
"""Manual benchmark: test base vs adapter on shell easy tasks."""
import csv
import subprocess
import time
import re

tasks = []
with open('/Users/caug/npcww/npc-core/npcsh/npcsh/benchmark/tasks.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['category'] == 'shell' and row['difficulty'] in ('easy', 'medium'):
            tasks.append(row)

tasks = tasks[:6]

from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.lora import load_adapters
from mlx_lm.generate import make_sampler

print("Loading models...")
base, tokenizer = mlx_load('mlx-community/Qwen3.5-2B-4bit')
adapter, _ = mlx_load('mlx-community/Qwen3.5-2B-4bit')
load_adapters(adapter, 'adapters/qwen3.5/2b/mlx/')
sampler = make_sampler(temp=0.0)

SYSTEM_MSG = "You are a helpful assistant. When asked to perform a task, respond with a shell command that accomplishes it. Output only the command, no explanation."

def ask(model, instruction):
    prompt = f"<|im_start|>system\n{SYSTEM_MSG}\n<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n"
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=80, sampler=sampler, verbose=False)

def extract_cmd(response):
    m = re.search(r'```(?:bash|sh|shell)?\n(.+?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip().split('\n')[0].strip()
    m = re.search(r'`([^`]+)`', response)
    if m:
        return m.group(1).strip()
    for line in response.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('*') and not line.startswith('-'):
            if any(c in line for c in ['find ', 'ls ', 'echo ', 'cat ', 'mkdir ', 'touch ', 'wc ', 'grep ', 'date ', 'printf ', '> ', '| ']):
                return line.strip()
    return None

def run_task(task, model, model_name):
    for prefix in ['/tmp/countme', '/tmp/result', '/tmp/pydir', '/tmp/uname', '/tmp/nums', '/tmp/dirtest', '/tmp/comments', '/tmp/sizetest', '/tmp/exttest', '/tmp/bigtest', '/tmp/now', '/tmp/pyfiles', '/tmp/hello', '/tmp/mydir', '/tmp/person', '/tmp/config', '/tmp/env', '/tmp/colors', '/tmp/webapp', '/tmp/requirements', '/tmp/docker']:
        subprocess.run(f'rm -rf {prefix}*', shell=True, capture_output=True)

    setup = task.get('setup_cmd', '')
    if setup:
        subprocess.run(setup, shell=True, capture_output=True)

    start = time.time()
    response = ask(model, task['instruction'])
    duration = time.time() - start

    cmd = extract_cmd(response)
    if cmd:
        subprocess.run(cmd, shell=True, capture_output=True)

    verify = task['verify_cmd']
    try:
        result = subprocess.run(verify, shell=True, capture_output=True, timeout=5)
        passed = result.returncode == 0
    except:
        passed = False

    return {
        'task': task['id'],
        'passed': passed,
        'cmd': cmd,
        'resp': response[:80],
        'dur': duration,
    }

results = []
for task in tasks:
    print(f"\n=== {task['id']} ===")
    print(f"  Instruction: {task['instruction'][:60]}")

    base_r = run_task(task, base, "base")
    print(f"  BASE:  {'PASS' if base_r['passed'] else 'FAIL'} | cmd={base_r['cmd']}")

    adapter_r = run_task(task, adapter, "adapter")
    print(f"  ADAPT: {'PASS' if adapter_r['passed'] else 'FAIL'} | cmd={adapter_r['cmd']}")

    results.append({'task': task['id'], 'base': base_r['passed'], 'adapter': adapter_r['passed']})

print(f"\n{'='*40}")
print("SUMMARY")
print(f"{'='*40}")
base_total = sum(1 for r in results if r['base'])
adapt_total = sum(1 for r in results if r['adapter'])
print(f"Base:    {base_total}/{len(results)}")
print(f"Adapter: {adapt_total}/{len(results)}")
