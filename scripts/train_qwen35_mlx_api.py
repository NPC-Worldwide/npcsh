#!/usr/bin/env python3
"""MLX LoRA training via mlx_lm.lora.train API."""

import csv
import json
import os
import re
from pathlib import Path


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

from mlx_lm.lora import linear_to_lora_layers
from mlx_lm import load as mlx_load

model_name = "mlx-community/Qwen3.5-0.8B-4bit"
output_dir = Path("adapters/qwen3.5/0.8b/mlx/")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {model_name}", flush=True)
model, tokenizer = mlx_load(model_name)

num_layers = 16
lora_config = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "scale": 2.0,
}
linear_to_lora_layers(model, num_layers, lora_config)
print(f"LoRA applied: r=8, alpha=16, layers={num_layers}", flush=True)

from mlx_lm.tuner.datasets import load_local_dataset


class FakeConfig:
    mask_prompt = False
    prompt_feature = "prompt"
    text_feature = "text"
    completion_feature = "completion"
    chat_feature = "messages"


train_ds, val_ds, test_ds = load_local_dataset(data_dir, tokenizer, FakeConfig())
print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val", flush=True)

from mlx_lm.tuner.trainer import TrainingArgs

args = TrainingArgs(
    batch_size=1,
    iters=3516,
    val_batches=0,
    steps_per_report=100,
    steps_per_eval=0,
    steps_per_save=99999,
    max_seq_length=512,
    adapter_file=str(output_dir / "adapters.safetensors"),
    grad_checkpoint=False,
    grad_accumulation_steps=1,
    learning_rate=2e-5,
)

import mlx.optimizers as mlx_opt

optimizer = mlx_opt.AdamW(learning_rate=2e-5)

from mlx_lm.lora import train_model

print("Starting training...", flush=True)
train_model(
    args=args,
    model=model,
    train_set=train_ds,
    valid_set=None,
)

adapter_config = {
    "model": model_name,
    "fine_tune_type": "lora",
    "num_layers": num_layers,
    "lora_parameters": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "scale": 2.0,
    },
}
with open(output_dir / "adapter_config.json", "w") as f:
    json.dump(adapter_config, f, indent=2)

print(f"DONE: {output_dir}", flush=True)
