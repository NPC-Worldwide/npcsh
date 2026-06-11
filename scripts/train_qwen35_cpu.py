#!/usr/bin/env python3
"""Pure PyTorch + PEFT training for Qwen3.5-0.8B on CPU."""
import csv, json, os, re, sys, math
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

# Load data
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

# Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

# Load base model
base_model = 'Qwen/Qwen3.5-0.8B'
output_path = 'adapters/qwen3.5/0.8b/hf/'
os.makedirs(output_path, exist_ok=True)

print(f'Loading {base_model}...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map='cpu',
)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
print('Model loaded', flush=True)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize
max_length = 512
inputs_list = []
for inp, out in zip(X, y):
    text = inp + out
    tokens = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    # Create labels: mask input part
    input_len = len(tokenizer(inp, add_special_tokens=False)['input_ids'])
    labels = tokens['input_ids'].clone()
    labels[0, :input_len] = -100
    inputs_list.append({
        'input_ids': tokens['input_ids'][0],
        'attention_mask': tokens['attention_mask'][0],
        'labels': labels[0],
    })

# Simple training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

epochs = 3
batch_size = 1
num_batches = math.ceil(len(inputs_list) / batch_size)

print(f'Training {epochs} epochs, {num_batches} batches/epoch', flush=True)

for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(inputs_list), batch_size):
        batch = inputs_list[i:i+batch_size]
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        labels = torch.stack([b['labels'] for b in batch])

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i // batch_size + 1) % 50 == 0:
            avg = total_loss / (i // batch_size + 1)
            print(f'Epoch {epoch+1}/{epochs} Batch {i//batch_size+1}/{num_batches} Loss: {avg:.4f}', flush=True)

    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}', flush=True)

# Save
print(f'Saving to {output_path}...', flush=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f'DONE: {output_path}', flush=True)
