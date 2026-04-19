---
name: diff_summary
description: Get a plain-English summary of current git changes. Summarizes what changed
  and why using the LLM.
---

# diff_summary

Get a plain-English summary of current git changes. Summarizes what changed and why using the LLM.

## Inputs

- `ref` (default: `''`)

## Steps

- `summarize_diff` → [`summarize_diff.py`](./summarize_diff.py)

## Usage

```
/run_jinx jinx_ref=diff_summary input_values={"ref": ""}
```
