from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.lora import load_adapters
from mlx_lm.generate import make_sampler
import subprocess
import os

model, tokenizer = mlx_load('mlx-community/Qwen3.5-2B-4bit')
load_adapters(model, 'adapters/qwen3.5/2b/mlx/')
sampler = make_sampler(temp=0.0)

# Setup
subprocess.run('rm -rf /tmp/countme /tmp/result.txt', shell=True)
subprocess.run('mkdir -p /tmp/countme && for f in a b c d e f g; do echo "file $f" > /tmp/countme/${f}.txt; done', shell=True)

# Generate
prompt = '<|im_start|>user\nCount the number of files in /tmp/countme and write just the number to /tmp/result.txt<|im_end|>\n<|im_start|>assistant\n'
response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=100, sampler=sampler, verbose=False)
print('RAW RESPONSE:')
print(repr(response))
print()

# Extract command
# Try to find a shell-looking command in the text
candidates = []
for line in response.split('\n'):
    line = line.strip()
    if line and not line.startswith('|') and not line.startswith('*') and not line.startswith('-') and not line.startswith('#'):
        if any(cmd in line for cmd in ['find', 'ls', 'cat', 'echo', 'mkdir', 'touch', 'wc', 'grep', 'dd', 'date', 'printf', '>', '|', '/tmp/']):
            candidates.append(line)

if candidates:
    cmd = candidates[-1]  # usually the last one is the actual command
    print(f'EXTRACTED CMD: {cmd}')
    subprocess.run(cmd, shell=True)
    if os.path.exists('/tmp/result.txt'):
        with open('/tmp/result.txt') as f:
            content = f.read().strip()
        print(f'RESULT: "{content}"')
        print(f'PASS: {content == "7"}')
    else:
        print('FILE NOT CREATED')
else:
    print('NO COMMAND FOUND')
    print('All candidates checked:', candidates)
