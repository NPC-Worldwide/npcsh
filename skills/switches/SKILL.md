---
name: switches
description: List all switches from .ctx files
---

# switches

List all switches from .ctx files

## Inputs

- `scope` (default: `'all'`)

## Steps

- `list_switches` → [`list_switches.py`](./list_switches.py)

## Usage

```
/run_jinx jinx_ref=switches input_values={"scope": "all"}
```
