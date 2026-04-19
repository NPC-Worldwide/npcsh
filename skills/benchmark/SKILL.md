---
name: benchmark
description: Run the npcsh benchmark suite against a model
---

# benchmark

Run the npcsh benchmark suite against a model

## Inputs

- `model` (default: `''`)
- `provider` (default: `''`)
- `category` (default: `''`)
- `difficulty` (default: `''`)
- `task_id` (default: `''`)
- `timeout` (default: `'120'`)

## Steps

- `run_benchmark` → [`run_benchmark.py`](./run_benchmark.py)

## Usage

```
/run_jinx jinx_ref=benchmark input_values={"model": "", "provider": "", "category": "", "difficulty": "", "task_id": "", "timeout": "120"}
```
