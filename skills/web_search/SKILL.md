---
name: web_search
description: Search the web for information. Returns a list of results with titles,
  URLs, and snippets. Use when you need current info, facts, or answers from the internet.
---

# web_search

Search the web for information. Returns a list of results with titles, URLs, and snippets. Use when you need current info, facts, or answers from the internet.

## Inputs

- `query` (default: `''`)
- `provider` (default: `''`)
- `num_results` (default: `'10'`)

## Steps

- `search_web` → [`search_web.py`](./search_web.py)

## Usage

```
/run_jinx jinx_ref=web_search input_values={"query": "", "provider": "", "num_results": "10"}
```
