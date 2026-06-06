#!/bin/bash
# run_training_pipeline.sh
# End-to-end training pipeline: ingest traces → SFT → RL → evaluate → compare.
#
# Usage:
#   bash scripts/run_training_pipeline.sh mlx-community/Qwen3-4B-4bit mlx
#   bash scripts/run_training_pipeline.sh mlx-community/Qwen3-4B-4bit mlx dpo
#   bash scripts/run_training_pipeline.sh mlx-community/Qwen3-4B-4bit mlx grpo --hard-only

set -e

MODEL="${1:-mlx-community/Qwen3-4B-4bit}"
DEVICE="${2:-mlx}"
METHOD="${3:-grpo}"
shift 3 || true
EXTRA_ARGS="$@"

OUTPUT_BASE="${HOME}/.npcsh/models"
TS=$(date +%Y%m%d_%H%M%S)
CSV_DIR="${HOME}/.npcsh/benchmarks/local"

mkdir -p "$OUTPUT_BASE"

# 1. SFT on clean tool-call traces
echo "=== Phase 1: SFT on tool-call traces ==="
SFT_OUTPUT="${OUTPUT_BASE}/npcsh_sft_toolcalls_${TS}"

if [ -d "$CSV_DIR" ]; then
    python3 scripts/train_sft_toolcalls.py \
        --csv-dir "$CSV_DIR" \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output "$SFT_OUTPUT" \
        --hard-only \
        --epochs 5 \
        --lr 2e-5 \
        --lora-r 16
else
    echo "No benchmark CSVs found at $CSV_DIR. Run benchmarks first."
    exit 1
fi

# 2. RL training starting from SFT adapter
echo "=== Phase 2: ${METHOD} RL training ==="
RL_OUTPUT="${OUTPUT_BASE}/npcsh_rl_${METHOD}_${TS}"

python3 scripts/train_npcsh_rl.py "$METHOD" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --output "$RL_OUTPUT" \
    --csv-dir "$CSV_DIR" \
    --hard-only \
    --epochs 5 \
    --lr 1e-5 \
    --lora-r 16 \
    --provider omlx \
    $EXTRA_ARGS

# 3. Evaluate baseline (base model) vs SFT vs RL
echo "=== Phase 3: Evaluation ==="
EVAL_TASKS=20
EVAL_TIMEOUT=60

echo "--- Baseline (base model) ---"
python3 -m npcsh.benchmark.local_runner \
    --model "$MODEL" --provider omlx --timeout "$EVAL_TIMEOUT" \
    | tee /tmp/bench_baseline_${TS}.log || true

echo "--- SFT adapter ---"
# Serve SFT adapter via mlx_lm.server and evaluate
mlx_lm.server --model "$MODEL" --adapter-path "$SFT_OUTPUT" --port 8000 > /tmp/mlx_sft_${TS}.log 2>&1 &
curl -s http://127.0.0.1:8000/v1/models > /dev/null || sleep 5
NPCSH_CHAT_PROVIDER=omlx NPCSH_CHAT_MODEL="$MODEL" python3 -m npcsh.benchmark.local_runner \
    --model "$MODEL" --provider omlx --timeout "$EVAL_TIMEOUT" \
    | tee /tmp/bench_sft_${TS}.log || true
kill %1 2>/dev/null || true

echo "--- RL adapter ---"
mlx_lm.server --model "$MODEL" --adapter-path "$RL_OUTPUT" --port 8000 > /tmp/mlx_rl_${TS}.log 2>&1 &
curl -s http://127.0.0.1:8000/v1/models > /dev/null || sleep 5
NPCSH_CHAT_PROVIDER=omlx NPCSH_CHAT_MODEL="$MODEL" python3 -m npcsh.benchmark.local_runner \
    --model "$MODEL" --provider omlx --timeout "$EVAL_TIMEOUT" \
    | tee /tmp/bench_rl_${TS}.log || true
kill %1 2>/dev/null || true

# Summary
echo ""
echo "=== Pipeline complete ==="
echo "SFT adapter: $SFT_OUTPUT"
echo "RL adapter:  $RL_OUTPUT"
echo "Logs:        /tmp/bench_*_${TS}.log"
