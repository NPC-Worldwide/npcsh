#!/bin/bash
# run_training_pipeline.sh
# One-shot pipeline: ingest traces → build dataset → SFT → RL → evaluate
# Uses npcpy.ft.sft and npcpy.ft.rl under the hood.
#
# Usage:
#   bash scripts/run_training_pipeline.sh qwen3.5:4b
#   bash scripts/run_training_pipeline.sh mlx-community/Qwen3-1.7B-4bit mlx

set -e

MODEL="${1:-mlx-community/Qwen3-0.6B-4bit}"
DEVICE="${2:-mlx}"
OUTPUT_BASE="${HOME}/.npcsh/models"
TS=$(date +%Y%m%d_%H%M%S)

# 1. Ingest traces from benchmark CSVs
echo "=== Phase 1: Ingesting benchmark traces ==="
CSV_DIR="${HOME}/.npcsh/benchmarks/local"
if [ -d "$CSV_DIR" ]; then
    python3 scripts/benchmark_to_sft.py \
        --csv-dir "$CSV_DIR" \
        --pattern "npcsh_*.csv" \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output "${OUTPUT_BASE}/npcsh_sft_${TS}" \
        --epochs 10 \
        --lr 3e-5 \
        --save-jsonl "${OUTPUT_BASE}/training_data_${TS}.jsonl"
else
    echo "No benchmark CSVs found at $CSV_DIR. Run benchmarks first."
    exit 1
fi

# 2. RL training on benchmark tasks
echo "=== Phase 2: RL training on benchmark tasks ==="
python3 scripts/train_npcsh_rl.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --output "${OUTPUT_BASE}/npcsh_rl_${TS}" \
    --attempts 3 \
    --epochs 10

# 3. Evaluate the trained model
echo "=== Phase 3: Evaluating trained model ==="
# The RL script already prints pass rates. Optionally re-run benchmark here:
# python -m npcsh.benchmark.local_runner --model "${OUTPUT_BASE}/npcsh_rl_${TS}" --provider transformers

echo "=== Pipeline complete ==="
echo "SFT adapter: ${OUTPUT_BASE}/npcsh_sft_${TS}"
echo "RL adapter:  ${OUTPUT_BASE}/npcsh_rl_${TS}"
