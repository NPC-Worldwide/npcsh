#!/usr/bin/env python3
"""
publish_model.py

Publish npcsh fine-tuned artifacts to HuggingFace Hub.

Handles:
    - Adapters (LoRA weights) → upload as-is
    - Merged full models → merge + upload
    - GGUF exports → quantize + upload single file
    - MLX conversions → convert + upload

Usage:
    # Upload an adapter
    python scripts/publish_model.py --adapter adapters/npcsh_sft_qwen3 --repo-id myuser/npcsh-qwen3-sft

    # Merge, export GGUF, and upload both
    python scripts/publish_model.py --adapter adapters/npcsh_sft_qwen3 --repo-id myuser/npcsh-qwen3 \
        --merge --gguf --quantization Q4_K_M

    # Full pipeline: merge → GGUF → MLX → upload everything
    python scripts/publish_model.py --adapter adapters/npcsh_sft_qwen3 --repo-id myuser/npcsh-qwen3 \
        --merge --gguf --mlx --quantization Q4_K_M

    # Upload existing merged model directly
    python scripts/publish_model.py --model models/npcsh_sft_qwen3_merged --repo-id myuser/npcsh-qwen3-full
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _infer_name_from_path(path: str) -> str:
    """Derive a clean name from a path like adapters/npcsh_sft_qwen3."""
    name = Path(path).name
    # Strip one common suffix, most specific first
    for suffix in ["_merged", "_gguf", "_mlx", "_sft", "_dpo", "_grpo", "_ppo"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name or "npcsh_model"


def _write_model_card(path: str, meta: dict):
    """Write a README.md model card for HF Hub."""
    card = f"""---
tags:
  - npcsh
  - llm
  - fine-tuned
  - {meta.get('format', 'adapter')}
  - {meta.get('base_model', 'unknown')}
---

# {meta.get('name', 'npcsh Model')}

This is a fine-tuned model for [npcsh](https://github.com/npcsh/npcsh) — the NPC shell for LLM-powered command execution.

## Details

| Attribute | Value |
|-----------|-------|
| Base Model | `{meta.get('base_model', 'unknown')}` |
| Format | `{meta.get('format', 'adapter')}` |
| Training | `{meta.get('training_type', 'SFT')}` |
| Examples | `{meta.get('num_examples', 'unknown')}` |
| LoRA r | `{meta.get('lora_r', 'N/A')}` |
| LoRA alpha | `{meta.get('lora_alpha', 'N/A')}` |
| Epochs | `{meta.get('epochs', 'unknown')}` |
| LR | `{meta.get('lr', 'unknown')}` |

## Usage

### With npcsh (MLX)

```bash
export NPCSH_CHAT_MODEL="{meta.get('repo_id', '')}"
export NPCSH_CHAT_PROVIDER="huggingface"
```

### With Ollama (GGUF)

```bash
ollama create npcsh -f {meta.get('gguf_file', 'model.gguf')}
```

### With transformers (Python)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("{meta.get('base_model', '')}")
model = PeftModel.from_pretrained(base, "{meta.get('repo_id', '')}")
```

## Training Data

Trained on npcsh benchmark traces collected from local task execution.
"""
    readme_path = Path(path) / "README.md"
    readme_path.write_text(card)
    print(f"[card] Wrote model card to {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Publish npcsh models to HuggingFace Hub")
    parser.add_argument("--adapter", help="Path to LoRA adapter directory")
    parser.add_argument("--model", help="Path to merged full model directory")
    parser.add_argument("--repo-id", required=True, help="HF Hub repo ID (e.g. username/model-name)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF API token")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into full model before upload")
    parser.add_argument("--gguf", action="store_true", help="Export GGUF and upload")
    parser.add_argument("--mlx", action="store_true", help="Convert to MLX format and upload")
    parser.add_argument("--quantization", default="Q4_K_M", help="GGUF quantization type")
    parser.add_argument("--base-model", help="Base model ID (inferred from adapter_config if omitted)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for merge step")
    parser.add_argument("--output-dir", default=".", help="Directory for intermediate exports")
    parser.add_argument("--skip-upload", action="store_true", help="Only prepare files, don't upload")
    args = parser.parse_args()

    if not args.adapter and not args.model:
        print("Error: provide --adapter or --model")
        sys.exit(1)

    from npcpy.ft import merge_and_save, export_adapter, convert_to_mlx, upload_to_hub

    source_path = args.adapter or args.model
    base_model = args.base_model
    artifacts = []  # list of (local_path, repo_subfolder, description)

    # Load metadata from adapter if available
    meta = {}
    adapter_config_path = Path(source_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            cfg = json.load(f)
        meta["base_model"] = cfg.get("model", cfg.get("base_model_name_or_path", ""))
        meta["lora_r"] = cfg.get("lora_parameters", {}).get("rank", cfg.get("r", "N/A"))
        meta["lora_alpha"] = cfg.get("lora_parameters", {}).get("alpha", cfg.get("lora_alpha", "N/A"))
        meta["format"] = "adapter"

    training_meta_path = Path(source_path) / "training_metadata.json"
    if training_meta_path.exists():
        with open(training_meta_path) as f:
            training_meta = json.load(f)
        meta.update(training_meta)

    meta["name"] = _infer_name_from_path(source_path)
    meta["repo_id"] = args.repo_id

    # 1. Adapter upload
    if args.adapter:
        print(f"\n[1/4] Adapter: {args.adapter}")
        artifacts.append((args.adapter, "adapter", "LoRA adapter"))

    # 2. Merge full model
    merged_path = None
    if args.merge and args.adapter:
        print(f"\n[2/4] Merging adapter into full model...")
        merged_path = merge_and_save(
            args.adapter,
            base_model=base_model,
            output_path=os.path.join(args.output_dir, f"{meta['name']}_merged"),
            device=args.device,
        )
        _write_model_card(merged_path, {**meta, "format": "merged"})
        artifacts.append((merged_path, "full", "Merged full model"))
        meta["merged_path"] = merged_path

    # 3. GGUF export
    gguf_path = None
    if args.gguf and (args.adapter or merged_path):
        print(f"\n[3/4] Exporting GGUF ({args.quantization})...")
        src = merged_path or args.adapter
        gguf_path = export_adapter(
            src,
            output_path=os.path.join(args.output_dir, f"{meta['name']}_{args.quantization.lower()}.gguf"),
            base_model=base_model,
            format="gguf",
            quantization=args.quantization,
            device=args.device,
        )
        meta["gguf_file"] = Path(gguf_path).name
        artifacts.append((gguf_path, "gguf", f"GGUF {args.quantization}"))

    # 4. MLX conversion
    mlx_path = None
    if args.mlx and args.adapter:
        print(f"\n[4/4] Converting to MLX...")
        mlx_path = convert_to_mlx(
            args.adapter,
            output_path=os.path.join(args.output_dir, f"{meta['name']}_mlx"),
            base_model=base_model,
        )
        _write_model_card(mlx_path, {**meta, "format": "mlx"})
        artifacts.append((mlx_path, "mlx", "MLX adapter"))

    # Upload all artifacts
    if args.skip_upload:
        print("\n[skip-upload] Prepared artifacts:")
        for path, subfolder, desc in artifacts:
            print(f"  {desc}: {path} → (would upload to {subfolder}/)")
        return

    print(f"\n{'='*50}")
    print(f"Uploading to https://huggingface.co/{args.repo_id}")
    print(f"{'='*50}")

    for path, subfolder, desc in artifacts:
        print(f"\n[upload] {desc}: {path} → {subfolder}/")
        try:
            url = upload_to_hub(
                path,
                repo_id=args.repo_id,
                token=args.token,
                private=args.private,
                path_in_repo=subfolder,
                commit_message=f"Upload {desc} for {meta['name']}",
            )
            print(f"[upload] Done: {url}/tree/main/{subfolder}")
        except Exception as e:
            print(f"[upload] ERROR: {e}")

    print(f"\n{'='*50}")
    print(f"All artifacts published to {args.repo_id}")
    print(f"{'='*50}")
    for path, subfolder, desc in artifacts:
        print(f"  {subfolder:12} → {desc}")
    print(f"\nView: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
