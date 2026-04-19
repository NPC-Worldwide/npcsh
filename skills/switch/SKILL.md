---
name: switch
description: Get or set a switch in the .ctx file
---

# switch

Get or set a switch in the .ctx file

## Inputs

- `name` (default: `''`)
- `value` (default: `None`)
- `scope` (default: `'workspace'`)

## Steps

- `manage_switch` → [`manage_switch.py`](./manage_switch.py)

## Usage

```
/run_jinx jinx_ref=switch input_values={"name": "", "value": null, "scope": "workspace"}
```
