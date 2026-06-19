#!/usr/bin/env python3
"""Benchmark adapter vs base on first 5 shell easy tasks."""
import csv

# Load tasks
tasks = []
with open('/Users/caug/npcww/npc-core/npcsh/npcsh/benchmark/tasks.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['category'] == 'shell' and row['difficulty'] == 'easy':
            tasks.append(row)

print(f"Tasks: {len(tasks)} shell easy")

# Select first 5
tasks = tasks[:5]
for t in tasks:
    print(f"  {t['id']}: {t['instruction'][:50]}")

print()

# Setup npcsh state
from npcsh._state import setup_shell
result = setup_shell()
if isinstance(result, tuple):
    state = result[0]
else:
    state = result

state.chat_model = "mlx-community/Qwen3.5-2B-4bit"
state.chat_provider = "mlx"

from npcsh.benchmark.local_runner import run_task

def run_on_model(tasks, model, provider, label):
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print(f"{'='*60}")
    passed = 0
    for task in tasks:
        # Clean state between tasks
        state.chat_model = model
        state.chat_provider = provider
        
        result = run_task(task, state, {}, timeout=30)
        status = "PASS" if result.passed else "FAIL"
        print(f"  {task['id']}: {status} | {result.duration:.1f}s | attempts={result.attempts}")
        if result.passed:
            passed += 1
    print(f"  TOTAL: {passed}/{len(tasks)}")
    return passed

# Run base
base_pass = run_on_model(tasks, "mlx-community/Qwen3.5-2B-4bit", "mlx", "BASE (no adapter)")

# Run adapter
adapter_pass = run_on_model(tasks, "adapters/qwen3.5/2b/mlx/", "mlx", "ADAPTER (trained)")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Base:     {base_pass}/{len(tasks)}")
print(f"Adapter:  {adapter_pass}/{len(tasks)}")
