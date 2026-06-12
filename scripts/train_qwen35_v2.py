#!/usr/bin/env python3
"""Train Qwen3.5-2B with corrected tool-call data."""
import json, os, sys
from pathlib import Path
from argparse import Namespace

output_dir = Path('adapters/qwen3.5/2b/mlx-v2/')
output_dir.mkdir(parents=True, exist_ok=True)

from mlx_lm.lora import CONFIG_DEFAULTS, run

args = Namespace(**CONFIG_DEFAULTS)
args.model = 'mlx-community/Qwen3.5-2B-4bit'
args.train = True
args.data = '/tmp/qwen35_v2_data'
args.fine_tune_type = 'lora'
args.num_layers = 16
args.batch_size = 1
args.iters = 1950  # 195 examples * 10 epochs
args.val_batches = 0
args.learning_rate = 2e-5
args.steps_per_report = 50
args.steps_per_eval = 0
args.grad_accumulation_steps = 1
args.resume_adapter_file = None
args.adapter_path = str(output_dir)
args.save_every = 99999
args.test = False
args.test_batches = 0
args.max_seq_length = 512
args.config = None
args.grad_checkpoint = False
args.seed = 42
args.mask_prompt = False
args.optimizer = 'adamw'
args.lora_parameters = {"rank": 8, "alpha": 16, "dropout": 0.05, "scale": 2.0}

print('Starting training with corrected data...', flush=True)
run(args)

adapter_config = {
    "model": args.model,
    "fine_tune_type": "lora",
    "num_layers": args.num_layers,
    "lora_parameters": args.lora_parameters,
}
with open(output_dir / 'adapter_config.json', 'w') as f:
    json.dump(adapter_config, f, indent=2)

print(f'DONE: {output_dir}', flush=True)
