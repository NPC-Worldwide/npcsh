---
name: explore
description: Thoroughly explore a codebase or directory. Discovers structure, reads
  key files iteratively using LLM-guided exploration, and produces a comprehensive
  analysis. Use when you need deep understanding of a project.
---

# explore

Thoroughly explore a codebase or directory. Discovers structure, reads key files iteratively using LLM-guided exploration, and produces a comprehensive analysis. Use when you need deep understanding of a project.

## Inputs

- `path` (default: `'.'`)
- `task` (default: `None`)
- `max_iterations` (default: `'15'`)
- `max_file_lines` (default: `'300'`)
- `model` (default: `None`)
- `provider` (default: `None`)

## Steps

- `explore_codebase` → [`explore_codebase.py`](./explore_codebase.py)

## Usage

```
/run_jinx jinx_ref=explore input_values={"path": ".", "task": null, "max_iterations": "15", "max_file_lines": "300", "model": null, "provider": null}
```
