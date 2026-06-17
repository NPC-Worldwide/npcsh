#!/usr/bin/env python3
"""Benchmark fixed adapter through npcsh-style tool execution."""
import csv
import subprocess
import time
import re
import json

# Load tasks
tasks = []
with open('/Users/caug/npcww/npc-core/npcsh/npcsh/benchmark/tasks.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['category'] == 'shell' and row['difficulty'] in ('easy', 'medium'):
            tasks.append(row)

# First 8 tasks
tasks = tasks[:8]
print(f"Benchmarking {len(tasks)} shell tasks")

from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.lora import load_adapters
from mlx_lm.generate import make_sampler

# Load models
print("Loading base Qwen3.5-2B-4bit...")
base, tokenizer = mlx_load('mlx-community/Qwen3.5-2B-4bit')

print("Loading fixed adapter...")
adapter, _ = mlx_load('mlx-community/Qwen3.5-2B-4bit')
load_adapters(adapter, 'adapters/qwen3.5/2b/mlx/')
sampler = make_sampler(temp=0.0)

def ask(model, instruction):
    prompt = f"<|im_start|>system\nYou are npcsh. Use tools to accomplish tasks. Output tool calls in format: <|tool_call|>\\n{{\"name\":\"shell\",\"arguments\":{{\"bash_command\":\"...\"}}}}\\n<|/tool_call|>\n<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n"
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=100, sampler=sampler, verbose=False)

def parse_tool_calls(response):
    """Extract shell tool calls from response."""
    calls = []
    # Pattern 1: <tool_call>{...}</tool_call>
    for m in re.finditer(r'<tool_call>\\s*(\\{.+?\\})\\s*</tool_call>', response, re.DOTALL):
        try:
            tc = json.loads(m.group(1))
            if tc.get('name') == 'shell':
                calls.append(tc)
        except:
            pass
    # Pattern 2: raw shell command
    if not calls:
        for line in response.split('\\n'):
            line = line.strip()
            if any(cmd in line for cmd in ['find ', 'ls ', 'echo ', 'cat ', 'mkdir ', 'touch ', 'wc ', 'grep ', 'date ', 'printf ', '> ', '| ']):
                calls.append({'name': 'shell', 'arguments': {'bash_command': line}})
                break
    return calls

def run_task(task, model, label):
    # Cleanup /tmp
    for prefix in ['/tmp/countme', '/tmp/result', '/tmp/pydir', '/tmp/uname', '/tmp/nums', '/tmp/dirtest', '/tmp/comments', '/tmp/sizetest', '/tmp/exttest', '/tmp/bigtest', '/tmp/now', '/tmp/pyfiles', '/tmp/hello', '/tmp/mydir', '/tmp/person', '/tmp/config', '/tmp/env', '/tmp/colors', '/tmp/webapp', '/tmp/requirements', '/tmp/docker']:
        subprocess.run(f'rm -rf {prefix}*', shell=True, capture_output=True)
    
    setup = task.get('setup_cmd', '')
    if setup:
        subprocess.run(setup, shell=True, capture_output=True)
    
    start = time.time()
    response = ask(model, task['instruction'])
    duration = time.time() - start
    
    calls = parse_tool_calls(response)
    for call in calls:
        args = call.get('arguments', {})
        cmd = args.get('bash_command') or args.get('command')
        if cmd:
            subprocess.run(cmd, shell=True, capture_output=True)
    
    verify = task['verify_cmd']
    try:
        result = subprocess.run(verify, shell=True, capture_output=True, timeout=5)
        passed = result.returncode == 0
    except:
        passed = False
    
    print(f"  {label}: {'PASS' if passed else 'FAIL'} | {duration:.1f}s | calls={len(calls)}")
    if not passed:
        print(f"    Response: {response[:100]}")
    
    return passed

# Run benchmark
print(f"\\n{'='*60}")
print("BASE MODEL (no adapter)")
print(f"{'='*60}")
base_pass = sum(run_task(t, base, "BASE") for t in tasks)

print(f"\\n{'='*60}")
print("ADAPTER (fixed tool-calls)")
print(f"{'='*60}")
adapt_pass = sum(run_task(t, adapter, "ADAPT") for t in tasks)

print(f"\\n{'='*60}")
print(f"RESULTS: {len(tasks)} tasks")
print(f"{'='*60}")
print(f"Base:    {base_pass}/{len(tasks)} ({100*base_pass/len(tasks):.0f}%)")
print(f"Adapter: {adapt_pass}/{len(tasks)} ({100*adapt_pass/len(tasks):.0f}%)")
