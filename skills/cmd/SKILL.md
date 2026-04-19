---
name: cmd
description: Command mode - LLM generates and executes shell commands
---

# cmd

Command mode - LLM generates and executes shell commands

## Inputs

- `query` (default: `None`)
- `model` (default: `None`)
- `provider` (default: `None`)
- `stream` (default: `True`)

## Steps

- `cmd_execute` → [`cmd_execute.py`](./cmd_execute.py)

## Usage

```
/run_jinx jinx_ref=cmd input_values={"query": null, "model": null, "provider": null, "stream": true}
```
