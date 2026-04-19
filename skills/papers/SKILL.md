---
name: papers
description: Multi-platform research paper browser with tabs for arXiv, Semantic Scholar,
  OpenReview, and Wikipedia
---

# papers

Multi-platform research paper browser with tabs for arXiv, Semantic Scholar, OpenReview, and Wikipedia

## Inputs

- `query` (default: `''`)
- `limit` (default: `10`)
- `text` (default: `'false'`)

## Steps

- `search_and_browse` → [`search_and_browse.py`](./search_and_browse.py)

## Usage

```
/run_jinx jinx_ref=papers input_values={"query": "", "limit": 10, "text": "false"}
```
