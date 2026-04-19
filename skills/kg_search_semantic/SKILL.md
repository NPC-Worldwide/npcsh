---
name: kg_search_semantic
description: Search the knowledge graph by embedding similarity. Returns facts most
  semantically similar to the query using the embedding collection synced from kg_facts.
  Use this when the query's intent matters more than its surface terms.
---

# kg_search_semantic

Search the knowledge graph by embedding similarity. Returns facts most semantically similar to the query using the embedding collection synced from kg_facts. Use this when the query's intent matters more than its surface terms.

## Inputs

- `query` (default: `''`)
- `limit` (default: `'10'`)
- `embedding_model` (default: `'nomic-embed-text'`)
- `embedding_provider` (default: `'ollama'`)

## Steps

- `semantic_search` → [`semantic_search.py`](./semantic_search.py)

## Usage

```
/run_jinx jinx_ref=kg_search_semantic input_values={"query": "", "limit": "10", "embedding_model": "nomic-embed-text", "embedding_provider": "ollama"}
```
