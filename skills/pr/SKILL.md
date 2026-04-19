---
name: pr
description: Create a GitHub pull request with an LLM-generated title and description.
  Requires gh CLI.
---

# pr

Create a GitHub pull request with an LLM-generated title and description. Requires gh CLI.

## Inputs

- `task` (default: `''`)
- `base` (default: `''`)

## Steps

- `create_pr` → [`create_pr.py`](./create_pr.py)

## Usage

```
/run_jinx jinx_ref=pr input_values={"task": "", "base": ""}
```
