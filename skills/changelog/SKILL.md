---
name: changelog
description: Generate a formatted changelog between two git refs. Groups commits by
  category (features, fixes, etc.) using the LLM.
---

# changelog

Generate a formatted changelog between two git refs. Groups commits by category (features, fixes, etc.) using the LLM.

## Inputs

- `from_ref` (default: `''`)
- `to_ref` (default: `'HEAD'`)

## Steps

- `gen_changelog` → [`gen_changelog.py`](./gen_changelog.py)

## Usage

```
/run_jinx jinx_ref=changelog input_values={"from_ref": "", "to_ref": "HEAD"}
```
