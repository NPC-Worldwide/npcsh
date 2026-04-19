---
name: commit
description: Generate a commit message from staged changes using the LLM. Stages all
  if nothing is staged. Shows the proposed message for confirmation.
---

# commit

Generate a commit message from staged changes using the LLM. Stages all if nothing is staged. Shows the proposed message for confirmation.

## Inputs

- `hint` (default: `''`)

## Steps

- `smart_commit` → [`smart_commit.py`](./smart_commit.py)

## Usage

```
/run_jinx jinx_ref=commit input_values={"hint": ""}
```
