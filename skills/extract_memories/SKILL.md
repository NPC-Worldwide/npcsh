---
name: extract_memories
description: Extract memories from recent conversations and store as pending_approval.
  Runs non-interactively for scheduled jobs.
---

# extract_memories

Extract memories from recent conversations and store as pending_approval. Runs non-interactively for scheduled jobs.

## Inputs

- `limit` (default: `'50'`)
- `context` (default: `''`)
- `model` (default: `''`)
- `provider` (default: `''`)

## Steps

- `extract_recent_memories` → [`extract_recent_memories.py`](./extract_recent_memories.py)

## Usage

```
/run_jinx jinx_ref=extract_memories input_values={"limit": "50", "context": "", "model": "", "provider": ""}
```
