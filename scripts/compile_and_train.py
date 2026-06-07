#!/usr/bin/env python3
"""
compile_and_train.py

End-to-end pipeline:
1. Load raw SFT JSONL from benchmark traces
2. Strip redundant system prompts, keep only the instruction + full response chain
3. Save compact training data
4. Train SFT LoRA via npcpy.ft.sft on a real model (Qwen3-4B or Qwen3-8B)
5. Evaluate baseline vs trained on a subset of benchmark tasks

Usage:
    python scripts/compile_and_train.py --model mlx-community/Qwen3-4B-4bit --epochs 5
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def compactify_jsonl(input_path: str, output_path: str, system_template_path: str = None):
    """Strip identical system prompts from each example and keep compact format."""
    print(f"Compactifying {input_path}...")

    examples = []
    with open(input_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if not examples:
        return None, None

    # Extract system prompt (it's identical across all)
    first_prompt = examples[0]["prompt"]
    system_match = re.search(r"<\|im_start\|>system\n(.*?)(?:<\|im_end\|>|\\n<\|im_start\|>user)", first_prompt, re.DOTALL)
    system_prompt = system_match.group(1).strip() if system_match else ""

    # Extract just the user instruction from each prompt
    compact = []
    for ex in examples:
        prompt = ex["prompt"]
        # Find user content
        user_match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
        instruction = user_match.group(1).strip() if user_match else ""
        # Remove "User Provided Context" boilerplate
        instruction = re.sub(r"User Provided Context:.*", "", instruction, flags=re.DOTALL).strip()

        if not instruction or not ex["completion"]:
            continue

        compact.append({
            "instruction": instruction,
            "completion": ex["completion"],
        })

    print(f"Compact: {len(compact)} examples (removed {len(examples) - len(compact)} empty)")

    # Save compact JSONL
    with open(output_path, "w") as f:
        for c in compact:
            f.write(json.dumps(c) + "\n")
    print(f"Saved compact data to {output_path}")

    # Save system template
    if system_template_path and system_prompt:
        Path(system_template_path).write_text(system_prompt)
        print(f"Saved system template ({len(system_prompt)} chars) to {system_template_path}")

    return compact, system_prompt


def build_xy(compact_data: list, system_prompt: str, format_style: str = "qwen3"):
    """Build X/y lists for SFT. System prompt is identical across all examples — omit it
    from training data to avoid wasting tokens and truncation. The inference pipeline
    adds the system prompt at runtime."""
    X = []
    y = []

    for item in compact_data:
        instruction = item["instruction"]
        completion = item["completion"]

        if format_style == "qwen3":
            prompt_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            output_text = f"{completion}<|im_end|>"
        elif format_style == "gemma":
            prompt_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
            output_text = f"{completion}<end_of_turn>"
        else:
            prompt_text = f"Input: {instruction}\nOutput: "
            output_text = completion

        X.append(prompt_text)
        y.append(output_text)

    return X, y


def fuse_adapter(adapter_path: str, output_path: str):
    """Fuse LoRA adapter with base model using mlx_lm.lora.fuse."""
    print(f"Fusing adapter {adapter_path} → {output_path}")
    try:
        from mlx_lm.lora import fuse
        fuse(
            adapter_path=adapter_path,
            save_path=output_path,
            dequantize=False,
        )
        print(f"Fused model saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Fuse failed: {e}")
        print("Trying CLI fallback...")
        try:
            subprocess.run(
                ["python", "-m", "mlx_lm.lora", "--adapter-path", adapter_path, "--save-path", output_path],
                check=True,
            )
            return output_path
        except Exception as e2:
            print(f"CLI fallback also failed: {e2}")
            return None


def evaluate_on_benchmark(model_path: str, provider: str, category: str = None, num_tasks: int = 10):
    """Run a quick benchmark subset to compare."""
    print(f"\nEvaluating {model_path} ({provider}) on {num_tasks} tasks...")

    task_file = Path(__file__).parent.parent / "npcsh" / "benchmark" / "tasks.csv"
    tasks = []
    with open(task_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if category and row["category"] != category:
                continue
            tasks.append(row)
            if len(tasks) >= num_tasks:
                break

    passed = 0
    for task in tasks:
        work_dir = tempfile.mkdtemp(prefix=f"eval_{task['id']}_")
        setup_cmd = task.get("setup_cmd", "") or ""
        if setup_cmd:
            subprocess.run(["bash", "-c", setup_cmd], capture_output=True, cwd=work_dir)

        env = os.environ.copy()
        env["NPCSH_CHAT_MODEL"] = model_path
        env["NPCSH_CHAT_PROVIDER"] = provider
        env["NPCSH_STREAM_OUTPUT"] = "0"

        try:
            proc = subprocess.run(
                ["npcsh", "-c", task["instruction"]],
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=env,
                timeout=60,
            )
            time.sleep(1)
            verify = subprocess.run(
                ["bash", "-c", task["verify_cmd"]],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=work_dir,
            )
            if verify.returncode == 0:
                passed += 1
        except Exception:
            pass
        finally:
            subprocess.run(["rm", "-rf", work_dir], capture_output=True)

    print(f"  Result: {passed}/{len(tasks)} passed")
    return passed, len(tasks)


def main():
    parser = argparse.ArgumentParser(description="Compile traces and train")
    parser.add_argument("--jsonl", default="~/.npcsh/benchmarks/sft_data_all.jsonl")
    parser.add_argument("--compact-out", default="~/.npcsh/benchmarks/sft_compact.jsonl")
    parser.add_argument("--system-template", default="~/.npcsh/benchmarks/system_template.txt")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--output", default="models/npcsh_trained")
    parser.add_argument("--device", default="mlx", choices=["mlx", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--fuse", action="store_true", help="Fuse adapter into merged model after training")
    parser.add_argument("--provider", default="omlx", choices=["omlx", "ollama", "transformers"], help="Provider for evaluation")
    args = parser.parse_args()

    input_path = os.path.expanduser(args.jsonl)
    compact_path = os.path.expanduser(args.compact_out)
    template_path = os.path.expanduser(args.system_template)

    # 1. Compactify
    compact_data, system_prompt = compactify_jsonl(input_path, compact_path, template_path)
    if not compact_data:
        print("No data to train on.")
        sys.exit(1)

    # 2. Build X/y
    print("\nBuilding training examples...")
    X, y = build_xy(compact_data, system_prompt, format_style="qwen3")
    print(f"Built {len(X)} examples (avg prompt len: {sum(len(x) for x in X)//len(X)}, avg completion len: {sum(len(yi) for yi in y)//len(y)})")

    if args.skip_sft:
        print("\nSkipping SFT (--skip-sft)")
        adapter_path = args.output
    else:
        # 3. Train SFT
        print(f"\nTraining SFT: {args.model} → {args.output}")
        print(f"  epochs={args.epochs}, lr={args.lr}, lora_r={args.lora_r}, max_length={args.max_length}")

        from npcpy.ft.sft import run_sft, SFTConfig

        config = SFTConfig(
            base_model_name=args.model,
            output_model_path=args.output,
            device=args.device,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
            logging_steps=max(1, len(X) // args.batch_size // 20),
            save_steps=max(1, len(X) // args.batch_size // 5),
        )

        adapter_path = run_sft(X, y, config=config, format_style="qwen3")
        print(f"Adapter saved to: {adapter_path}")

        meta = {
            "base_model": args.model,
            "adapter_path": adapter_path,
            "num_examples": len(X),
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        }
        (Path(args.output) / "training_metadata.json").write_text(json.dumps(meta, indent=2))

    # 4. Optionally fuse adapter
    fused_path = None
    if args.fuse and not args.skip_sft:
        fused_path = fuse_adapter(adapter_path, args.output + "_fused")

    # 5. Evaluate
    if not args.skip_eval:
        print("\n--- Baseline Evaluation ---")
        baseline_pass, baseline_total = evaluate_on_benchmark(args.model, args.provider, num_tasks=10)

        print("\n--- Trained Evaluation ---")
        eval_model = fused_path or adapter_path
        trained_pass, trained_total = evaluate_on_benchmark(eval_model, args.provider, num_tasks=10)

        print(f"\n{'='*50}")
        print(f"Baseline: {baseline_pass}/{baseline_total}")
        print(f"Trained:  {trained_pass}/{trained_total}")
        delta = trained_pass - baseline_pass
        print(f"Delta:    {delta:+d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
