---
name: sample
description: Send a prompt directly to the LLM.
---

# sample

Send a prompt directly to the LLM.

## Inputs

- `prompt` (default: `''`)
- `model` (default: `''`)
- `provider` (default: `''`)

## Steps

- `send_prompt_to_llm` → [`send_prompt_to_llm.py`](./send_prompt_to_llm.py)

## Usage

```
/run_jinx jinx_ref=sample input_values={"prompt": "", "model": "", "provider": ""}
```
