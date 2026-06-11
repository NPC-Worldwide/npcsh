#!/usr/bin/env python3
"""Standalone MLX trainer for Qwen3.5-0.8B."""
import csv, json, os, re, sys
from pathlib import Path

def parse_trace(trace_str):
    if not trace_str or '---TRACE---' not in trace_str:
        return None
    trace = trace_str.split('---TRACE---', 1)[1]
    user_match = re.search(r'\[user\] (.*?) (?:\[assistant\]|\[tool_call\])', trace, re.DOTALL)
    instruction = ''
    if user_match:
        instruction = user_match.group(1).strip()
        instruction = re.sub(r'User Provided Context:.*', '', instruction, flags=re.DOTALL).strip()
    assistant_match = re.search(r'\[assistant\] (.*?) (?:\[tool_call\]|\[user\]|\Z)', trace, re.DOTALL)
    response = assistant_match.group(1).strip() if assistant_match else ''
    for m in re.finditer(r'\[tool_call\]\s+(\w+)\((\{.*?\})\)', trace):
        fname, args_raw = m.group(1), m.group(2)
        try:
            args = json.loads(args_raw)
        except:
            try: args = eval(args_raw)
            except: args = {}
        if fname == 'sh': fname = 'shell'
        elif fname in ('py','python','Charlie','Alice','Bob','Diana','Eve','Frank','Alex','chat'): continue
        tc = json.dumps({'name': fname, 'arguments': args}, ensure_ascii=False)
        response += f'\n<tool_call>\n{tc}\n</tool_call>'
    return {'instruction': instruction, 'response': response}

X, y = [], []
csv.field_size_limit(10**7)
for csv_file in sorted(Path(os.path.expanduser('~/.npcsh/benchmarks/local')).glob('*.csv')):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace = parse_trace(row.get('output', ''))
            if trace and trace['instruction'] and trace['response'] and row.get('passed','').lower() in ('true','1'):
                X.append(f"<|im_start|>user\n{trace['instruction']}\n<|im_end|>\n<|im_start|>assistant\n")
                y.append(f"{trace['response']}\n<|im_end|>\n")

print(f'SFT: {len(X)} passed traces', flush=True)

sys.path.insert(0, '/Users/caug/npcww/npc-core/npcpy')
from npcpy.ft import run_sft, SFTConfig

cfg = SFTConfig(
    base_model_name='mlx-community/Qwen3.5-0.8B-4bit',
    output_model_path='adapters/qwen3.5/0.8b/mlx/',
    device='mlx',
    lora_r=4, lora_alpha=8,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=2e-5,
    max_length=512,
    logging_steps=100,
    save_steps=99999,
)

print('Starting training...', flush=True)
adapter = run_sft(X, y, config=cfg, format_style='qwen3')
print(f'DONE: {adapter}', flush=True)
