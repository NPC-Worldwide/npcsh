#!/usr/bin/env python3
"""Quick eval: compare base model vs adapter on sample tasks."""
from mlx_lm import load as mlx_load, generate

# Load base
print("Loading base Qwen3.5-2B-4bit...")
model_base, tokenizer = mlx_load('mlx-community/Qwen3.5-2B-4bit')

# Load adapter
print("Loading adapter...")
model_adapter, _ = mlx_load('mlx-community/Qwen3.5-2B-4bit')
from mlx_lm.lora import load_adapters
load_adapters(model_adapter, 'adapters/qwen3.5/2b/mlx/')

# Sample prompts
tests = [
    {
        "q": "List all Python files in the current directory recursively",
        "expected": "shell",
    },
    {
        "q": "Show me the contents of /etc/passwd",
        "expected": "shell",
    },
    {
        "q": "What is 2+2?",
        "expected": "direct_answer",
    },
]

for test in tests:
    prompt = f'<|im_start|>user\n{test["q"]}\n<|im_end|>\n<|im_start|>assistant\n'
    print(f'\n=== {test["q"]} ===')
    
    print('BASE:')
    resp_base = generate(model_base, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
    print(resp_base[:100])
    
    print('ADAPTER:')
    resp_adapter = generate(model_adapter, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
    print(resp_adapter[:100])

print('\nDone.')
