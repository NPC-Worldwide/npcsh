---
name: sleep
description: Evolve knowledge graph. Use --dream for creative synthesis, --backfill
  to import approved memories.
---

# sleep

Evolve knowledge graph. Use --dream for creative synthesis, --backfill to import approved memories.

## Inputs

- `dream` (default: `False`)
- `backfill` (default: `False`)
- `ops` (default: `''`)
- `context` (default: `''`)
- `model` (default: `''`)
- `provider` (default: `''`)

## Steps

- `evolve_knowledge_graph` → [`evolve_knowledge_graph.py`](./evolve_knowledge_graph.py)

## Usage

```
/run_jinx jinx_ref=sleep input_values={"dream": false, "backfill": false, "ops": "", "context": "", "model": "", "provider": ""}
```
