#!/usr/bin/env bash
# Pull Ollama cloud variants that are not yet available locally.
set -euo pipefail

MODELS=(
  glm-5.2:cloud
  gemma4:26b-cloud
  gemma4:12b-cloud
  qwen3.5:35b-cloud
  qwen3.5:122b-cloud
  minimax-m2.5:cloud
  deepseek-v4-pro:cloud
  nemotron-3-ultra:cloud
  gpt-oss:120b-cloud
  qwen3-coder:480b-cloud
  glm-4.7:cloud
  gemini-3-flash-preview:cloud
  minimax-m2.1:cloud
  deepseek-v3.2:cloud
  ministral-3:14b-cloud
  nemotron-3-nano:30b-cloud
  deepseek-v3.1:cloud
  gemma3:27b-cloud
  qwen3-coder-next:cloud
)

mkdir -p ~/.npcsh/benchmarks/local
log="$HOME/.npcsh/benchmarks/local/ollama_pull_cloud.log"
: > "$log"

for model in "${MODELS[@]}"; do
  echo "[PULL] $model" | tee -a "$log"
  if ollama pull "$model" >> "$log" 2>&1; then
    echo "[DONE] $model" | tee -a "$log"
  else
    echo "[FAIL] $model" | tee -a "$log"
  fi
done
