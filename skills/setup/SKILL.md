---
name: setup
description: Interactive setup wizard for npcsh - detect local models, configure defaults
---

# setup

Interactive setup wizard for npcsh - detect local models, configure defaults

## Inputs

- `skip_detection` (default: `''`)

## Steps

- `setup_wizard` → [`setup_wizard.py`](./setup_wizard.py)

## Usage

```
/run_jinx jinx_ref=setup input_values={"skip_detection": ""}
```
