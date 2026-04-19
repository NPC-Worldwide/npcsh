---
name: launch_app
description: Launch an application on the system
---

# launch_app

Launch an application on the system

## Inputs

- `command` (default: `''`)

## Steps

- `perform_launch` → [`perform_launch.py`](./perform_launch.py)

## Usage

```
/run_jinx jinx_ref=launch_app input_values={"command": ""}
```
