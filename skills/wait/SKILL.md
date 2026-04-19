---
name: wait
description: Wait/pause for a specified duration in seconds
---

# wait

Wait/pause for a specified duration in seconds

## Inputs

- `duration` (default: `1`)

## Steps

- `perform_wait` → [`perform_wait.py`](./perform_wait.py)

## Usage

```
/run_jinx jinx_ref=wait input_values={"duration": 1}
```
