---
name: kg_search_connected
description: Traverse the knowledge graph from a seed node, returning all nodes reachable
  within N hops. Use this when you already know a starting fact or concept and want
  to explore its neighborhood (related facts, linked concepts).
---

# kg_search_connected

Traverse the knowledge graph from a seed node, returning all nodes reachable within N hops. Use this when you already know a starting fact or concept and want to explore its neighborhood (related facts, linked concepts).

## Inputs

- `seed` (default: `''`)
- `max_depth` (default: `'2'`)
- `max_per_hop` (default: `'10'`)

## Steps

- `traverse` → [`traverse.py`](./traverse.py)

## Usage

```
/run_jinx jinx_ref=kg_search_connected input_values={"seed": "", "max_depth": "2", "max_per_hop": "10"}
```
