---
name: init
description: Interactive wizard for initializing NPC projects with model detection
  and configuration
---

# init

Interactive wizard for initializing NPC projects with model detection and configuration

## Inputs

- `directory` (default: `''`)
- `templates` (default: `''`)
- `team_ctx` (default: `''`)
- `model` (default: `''`)
- `provider` (default: `''`)

## Steps

- `init_wizard` → [`init_wizard.py`](./init_wizard.py)

## Usage

```
/run_jinx jinx_ref=init input_values={"directory": "", "templates": "", "team_ctx": "", "model": "", "provider": ""}
```
