---
name: deep_research
description: Deep research mode - multi-agent hypothesis exploration with approval-gated
  TUI pipeline
---

# deep_research

Deep research mode - multi-agent hypothesis exploration with approval-gated TUI pipeline

## Inputs

- `query` (default: `None`)
- `num_npcs` (default: `3`)
- `model` (default: `None`)
- `provider` (default: `None`)
- `max_steps` (default: `10`)
- `num_cycles` (default: `3`)
- `format` (default: `'report'`)

## Steps

- `alicanto_research` → [`alicanto_research.py`](./alicanto_research.py)

## Usage

```
/run_jinx jinx_ref=deep_research input_values={"query": null, "num_npcs": 3, "model": null, "provider": null, "max_steps": 10, "num_cycles": 3, "format": "report"}
```
