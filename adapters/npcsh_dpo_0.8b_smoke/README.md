---
language: en
license: mit
base_model: mlx-community/Qwen3.5-0.8B-4bit
library_name: mlx
tags:
  - mlx
  - lora
  - npcsh
  - enpisi-coder
  - agent
---

# enpisi-coder 0.8B smoke (MLX LoRA)

First smoke-test LoRA adapter for the enpisi-coder family. SFT-on-chosen
over judge-rated npcsh agent traces (the 5-judge ensemble's highest-composite
trace per task), trained 60 iterations on a 0.8B base.

- **Base model:** `mlx-community/Qwen3.5-0.8B-4bit`
- **Fine-tune type:** LoRA
- **LoRA:** rank 16, alpha 16, dropout 0.1, scale 1.0, 24 layers
- **Data:** judge-rated npcsh agent traces (`npc-worldwide/npcsh-traces`),
  chosen = highest `composite` trace per task, DPO-style pairs capped at 4/task
- **Trainer:** `npcpy.ft.rl._train_dpo_mlx` (SFT on `f"{prompt}\n{chosen}"`)

## Use with npcpy

```python
from npcpy.llm_funcs import get_llm_response

r = get_llm_response(
    "Use list_files to list /tmp.",
    model="npc-worldwide/enpisi-coder",   # or local path to this folder
    provider="mlx",                         # npcpy loads base + adapter
    tools=tools, tool_map=tool_map,
    auto_process_tool_calls=True,
)
```

npcpy's `provider="mlx"` route loads the base model and applies this adapter,
parses + executes the Qwen3 tool-call blocks the model emits, and runs the
agent loop natively. Callers never import `mlx_lm` themselves.

## Honest eval caveats

- 60 / 800 iterations (stopped early for the smoke run).
- SFT-on-chosen, not a true preference optimization (the `_train_dpo_mlx`
  trainer trains on the chosen response and ignores `rejected`).
- Every task_id was seen during training (no held-out generalization split).
- Single-judge-turn eval showed a small lift over base (~+0.029 composite,
  16 improved / 10 regressed / 4 tied across 30 stratified instructions),
  strongest in file-ops / git / debug / shell categories.
- The real eval is a full npcsh agent-loop run with `NPCSH_CHAT_PROVIDER=mlx`
  on held-out tasks, judge-rated base vs adapter (not yet run).