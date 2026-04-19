---
name: file_search
description: Find and browse files with interactive TUI
---

# file_search

Find and browse files with interactive TUI

## Inputs

- `pattern` (default: `''`)
- `path` (default: `'.'`)
- `recursive` (default: `'true'`)
- `text` (default: `'false'`)

## Steps

- `search_files` → [`search_files.py`](./search_files.py)

## Usage

```
/run_jinx jinx_ref=file_search input_values={"pattern": "", "path": ".", "recursive": "true", "text": "false"}
```
