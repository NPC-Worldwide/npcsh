---
name: set
description: Set configuration values
---

# set

Set configuration values

## Inputs

- `key` (default: `''`)
- `value` (default: `''`)

## Steps

- `set_config_value` → [`set_config_value.py`](./set_config_value.py)

## Usage

```
/run_jinx jinx_ref=set input_values={"key": "", "value": ""}
```
