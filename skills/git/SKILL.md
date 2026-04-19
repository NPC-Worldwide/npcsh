---
name: git
description: Interactive terminal UI for git status, staging, diffs, log, and commits
---

# git

Interactive terminal UI for git status, staging, diffs, log, and commits

## Inputs

- `path` (default: `''`)

## Steps

- `git_tui` → [`git_tui.py`](./git_tui.py)

## Usage

```
/run_jinx jinx_ref=git input_values={"path": ""}
```
