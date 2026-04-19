---
name: arxiv
description: Interactive arXiv paper browser
---

# arxiv

Interactive arXiv paper browser

## Inputs

- `query` (default: `''`)
- `author` (default: `''`)
- `category` (default: `''`)
- `title` (default: `''`)
- `abstract` (default: `''`)
- `limit` (default: `10`)
- `text` (default: `'false'`)

## Steps

- `search_and_browse` → [`search_and_browse.py`](./search_and_browse.py)

## Usage

```
/run_jinx jinx_ref=arxiv input_values={"query": "", "author": "", "category": "", "title": "", "abstract": "", "limit": 10, "text": "false"}
```
