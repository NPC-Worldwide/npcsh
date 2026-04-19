---
name: plan
description: Create, view, revise, or advance a structured execution plan for a task.
---

# plan

Create, view, revise, or advance a structured execution plan for a task.

## Inputs

- `goal` (default: `None`)
- `task` (default: `None`)
- `action` (default: `'create'`)
- `steps` (default: `None`)

## Steps

- `run_plan` → [`run_plan.py`](./run_plan.py)

## Usage

```
/run_jinx jinx_ref=plan input_values={"goal": null, "task": null, "action": "create", "steps": null}
```
