#!/usr/bin/env python3
"""MLX LoRA training via mlx_lm.lora.run API."""

import csv
import json
import os
import re
from pathlib import Path
from argparse import Namespace


def parse_trace(trace_str):
    if not trace_str or "---TRACE---" not in trace_str:
        return None
    trace = trace_str.split("---TRACE---", 1)[1]
    user_match = re.search(
        r"\[user\] (.*?) (?:\[assistant\]|\[tool_call\])", trace, re.DOTALL
    )
    instruction = ""
    if user_match:
        instruction = user_match.group(1).strip()
        instruction = re.sub(
            r"User Provided Context:.*", "", instruction, flags=re.DOTALL
        ).strip()
    assistant_match = re.search(
        r"\[assistant\] (.*?) (?:\[tool_call\]|\[user\]|\Z)", trace, re.DOTALL
    )
    response = assistant_match.group(1).strip() if assistant_match else ""
    for m in re.finditer(r"\[tool_call\]\s+(\w+)\((\{.*?\})\)", trace):
        fname, args_raw = m.group(1), m.group(2)
        try:
            args = json.loads(args_raw)
        except:
            try:
                args = eval(args_raw)
            except:
                args = {}
        if fname == "sh":
            fname = "shell"
        elif fname in (
            "py",
            "python",
            "Charlie",
            "Alice",
            "Bob",
            "Diana",
            "Eve",
            "Frank",
            "Alex",
            "chat",
        ):
            continue
        tc = json.dumps({"name": fname, "arguments": args}, ensure_ascii=False)
        response += f"\n<tool_call>\n{tc}\n</tool_call>"
    return {"instruction": instruction, "response": response}


records = []
csv.field_size_limit(10**7)
for csv_file in sorted(
    Path(os.path.expanduser("~/.npcsh/benchmarks/local")).glob("*.csv")
):
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace = parse_trace(row.get("output", ""))
            if (
                trace
                and trace["instruction"]
                and trace["response"]
                and row.get("passed", "").lower() in ("true", "1")
            ):
                records.append(
                    {
                        "messages": [
                            {"role": "user", "content": trace["instruction"]},
                            {"role": "assistant", "content": trace["response"]},
                        ]
                    }
                )

print(f"SFT: {len(records)} passed traces", flush=True)

data_dir = Path("/tmp/qwen35_data")
data_dir.mkdir(exist_ok=True)
with open(data_dir / "train.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
print(f"Data saved to {data_dir}/train.jsonl", flush=True)

output_dir = Path("adapters/qwen3.5/0.8b/mlx/")
output_dir.mkdir(parents=True, exist_ok=True)

from mlx_lm.lora import CONFIG_DEFAULTS

args = Namespace(**CONFIG_DEFAULTS)
args.model = "mlx-community/Qwen3.5-0.8B-4bit"
args.train = True
args.data = str(data_dir)
args.fine_tune_type = "lora"
args.num_layers = 16
args.batch_size = 1
args.iters = 3516
args.val_batches = 0
args.learning_rate = 2e-5
args.steps_per_report = 100
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
args.optimizer = "adamw"
args.lora_parameters = {"rank": 8, "alpha": 16, "dropout": 0.05, "scale": 2.0}

print("Starting mlx_lm.lora.run...", flush=True)
from mlx_lm.lora import run

run(args)

adapter_config = {
    "model": args.model,
    "fine_tune_type": "lora",
    "num_layers": args.num_layers,
    "lora_parameters": args.lora_parameters,
}
with open(output_dir / "adapter_config.json", "w") as f:
    json.dump(adapter_config, f, indent=2)

print(f"DONE: {output_dir}", flush=True)
