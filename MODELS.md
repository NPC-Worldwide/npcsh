# Model Publishing Workflow

## Directory Layout

```
npcsh/
├── adapters/          # LoRA adapters (small, trainable)
│   ├── npcsh_sft_qwen3/
│   ├── npcsh_rl_grpo_gemma/
│   └── .gitkeep
├── models/            # Full merged models (large, serveable)
│   ├── npcsh_sft_qwen3_merged/
│   └── .gitkeep
└── scripts/
    ├── train_from_csv.py
    ├── compile_and_train.py
    ├── benchmark_to_sft.py
    ├── evaluate_adapter.py
    ├── fuse_and_eval.py
    └── publish_model.py
```

## Quick Start

### 1. Train an adapter

```bash
python scripts/train_from_csv.py sft \
    --csv-dir ~/.npcsh/benchmarks/local \
    --model mlx-community/Qwen3-4B-4bit \
    --output adapters/npcsh_sft_qwen3 \
    --device mlx \
    --epochs 5
```

Output goes to `adapters/npcsh_sft_qwen3/` as:
- `adapters.safetensors` — LoRA weights
- `adapter_config.json` — metadata (base model, rank, alpha)
- `training_metadata.json` — training hyperparameters

### 2. Evaluate it

```bash
python scripts/evaluate_adapter.py \
    --model mlx-community/Qwen3-4B-4bit \
    --adapter adapters/npcsh_sft_qwen3 \
    --provider omlx \
    --tasks 20
```

### 3. Merge + export + publish

```bash
# Full pipeline: merge → GGUF → MLX → upload everything
python scripts/publish_model.py \
    --adapter adapters/npcsh_sft_qwen3 \
    --repo-id myusername/npcsh-qwen3-sft \
    --merge \
    --gguf \
    --mlx \
    --quantization Q4_K_M
```

This uploads to HF Hub with subfolders:
- `adapter/` — original LoRA weights
- `full/` — merged full model (transformers format)
- `gguf/` — quantized GGUF for Ollama / LM Studio
- `mlx/` — MLX-converted weights for `npcsh` with `provider=omlx`

### 4. Use published models

**With npcsh (MLX provider):**
```bash
export NPCSH_CHAT_MODEL="hf.co/myusername/npcsh-qwen3-sft/mlx"
export NPCSH_CHAT_PROVIDER="omlx"
```

**With Ollama (GGUF):**
```bash
ollama create npcsh -f ./models/npcsh_sft_qwen3_q4_k_m.gguf
export NPCSH_CHAT_MODEL="npcsh"
export NPCSH_CHAT_PROVIDER="ollama"
```

**With transformers (Python):**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base, "hf.co/myusername/npcsh-qwen3-sft/adapter")
```

## Available Scripts

| Script | Purpose |
|--------|---------|
| `train_from_csv.py` | SFT / DPO / GRPO / PPO from benchmark CSVs |
| `benchmark_to_sft.py` | Convert traces → SFT data → train |
| `compile_and_train.py` | Compact JSONL → train → evaluate |
| `train_npcsh_rl.py` | RL training with active learning loop |
| `active_learning_loop.py` | Self-improvement via teacher model |
| `evaluate_adapter.py` | Benchmark evaluation (direct API) |
| `fuse_and_eval.py` | Fuse + evaluate merged model |
| `publish_model.py` | Merge / export / upload to HF Hub |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace API token for uploads |
| `NPCSH_CHAT_MODEL` | Active model for npcsh shell |
| `NPCSH_CHAT_PROVIDER` | Active provider (omlx, ollama, etc.) |
