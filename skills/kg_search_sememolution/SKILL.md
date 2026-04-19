---
name: kg_search_sememolution
description: Query a Sememolution population. Samples multiple KG-individuals from
  the population, each searches its own sub-graph with its own Poisson-sampled traversal,
  responses are ranked, and fitness updates. Returns the ranked candidates. Use this
  when the population is large enough and you want diverse, ranked perspectives on
  the same query.
---

# kg_search_sememolution

Query a Sememolution population. Samples multiple KG-individuals from the population, each searches its own sub-graph with its own Poisson-sampled traversal, responses are ranked, and fitness updates. Returns the ranked candidates. Use this when the population is large enough and you want diverse, ranked perspectives on the same query.

## Inputs

- `query` (default: `''`)
- `population_id` (default: `''`)
- `top_k` (default: `'15'`)

## Steps

- `sememolution_query` → [`sememolution_query.py`](./sememolution_query.py)

## Usage

```
/run_jinx jinx_ref=kg_search_sememolution input_values={"query": "", "population_id": "", "top_k": "15"}
```
