---
name: compress
description: Manages conversation and knowledge context - compress, flush, sleep,
  dream
---

# compress

Manages conversation and knowledge context - compress, flush, sleep, dream

## Inputs

- `flush` (default: `''`)
- `sleep` (default: `False`)
- `dream` (default: `False`)
- `ops` (default: `''`)
- `context` (default: `''`)
- `model` (default: `''`)
- `provider` (default: `''`)

## Steps

- `manage_context_and_memory` → [`manage_context_and_memory.py`](./manage_context_and_memory.py)

## Usage

```
/run_jinx jinx_ref=compress input_values={"flush": "", "sleep": false, "dream": false, "ops": "", "context": "", "model": "", "provider": ""}
```
