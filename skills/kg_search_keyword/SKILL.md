---
name: kg_search_keyword
description: Search the knowledge graph by keyword overlap. Returns facts and concept
  names whose text overlaps the query. Fast, deterministic, no LLM required. Use this
  when the query uses specific terms that should appear verbatim in the KG.
---

# kg_search_keyword

Search the knowledge graph by keyword overlap. Returns facts and concept names whose text overlaps the query. Fast, deterministic, no LLM required. Use this when the query uses specific terms that should appear verbatim in the KG.

## Inputs

- `query` (default: `''`)
- `limit` (default: `'15'`)
- `type` (default: `'both'`)

## Steps

- `search` → [`search.py`](./search.py)

## Usage

```
/run_jinx jinx_ref=kg_search_keyword input_values={"query": "", "limit": "15", "type": "both"}
```
