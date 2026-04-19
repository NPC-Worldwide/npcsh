---
name: trigger
description: Creates a persistent listener (--listen) or a scheduled task (--cron).
---

# trigger

Creates a persistent listener (--listen) or a scheduled task (--cron).

## Inputs

- `listen` (default: `''`)
- `cron` (default: `''`)

## Steps

- `execute_command` → [`execute_command.py`](./execute_command.py)

## Usage

```
/run_jinx jinx_ref=trigger input_values={"listen": "", "cron": ""}
```
