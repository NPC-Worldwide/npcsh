---
name: wander
description: Interactive wandering mode - creative exploration with live TUI dashboard
---

# wander

Interactive wandering mode - creative exploration with live TUI dashboard

## Inputs

- `problem` (default: `None`)
- `environment` (default: `None`)
- `low_temp` (default: `0.5`)
- `high_temp` (default: `1.9`)
- `n_min` (default: `30`)
- `n_max` (default: `150`)
- `interruption_likelihood` (default: `0.1`)
- `sample_rate` (default: `0.5`)
- `n_high_temp_streams` (default: `5`)
- `include_events` (default: `True`)
- `num_events` (default: `3`)
- `model` (default: `None`)
- `provider` (default: `None`)

## Steps

- `wander_interactive` → [`wander_interactive.py`](./wander_interactive.py)

## Usage

```
/run_jinx jinx_ref=wander input_values={"problem": null, "environment": null, "low_temp": 0.5, "high_temp": 1.9, "n_min": 30, "n_max": 150, "interruption_likelihood": 0.1, "sample_rate": 0.5, "n_high_temp_streams": 5, "include_events": true, "num_events": 3, "model": null, "provider": null}
```
