---
name: db_search
description: Search conversation history database with interactive TUI
---

# db_search

Search conversation history database with interactive TUI

## Inputs

- `query` (default: `''`)
- `path` (default: `''`)
- `limit` (default: `'100'`)
- `text` (default: `'false'`)

## Steps

- `search_db` → [`search_db.py`](./search_db.py)

## Usage

```
/run_jinx jinx_ref=db_search input_values={"query": "", "path": "", "limit": "100", "text": "false"}
```
