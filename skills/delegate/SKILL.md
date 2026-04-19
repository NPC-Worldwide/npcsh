---
name: delegate
description: Send a task to a specialist NPC who works on it until done. Only for
  complex tasks that need a specialist.
---

# delegate

Send a task to a specialist NPC who works on it until done. Only for complex tasks that need a specialist.

## Inputs

- `npc_name` (default: `{'description': 'Name of the NPC to delegate to'}`)
- `task` (default: `{'description': 'The task or request to delegate to the NPC'}`)
- `max_iterations` (default: `'10'`)

## Steps

- `delegate_with_review` → [`delegate_with_review.py`](./delegate_with_review.py)

## Usage

```
/run_jinx jinx_ref=delegate input_values={"npc_name": {"description": "Name of the NPC to delegate to"}, "task": {"description": "The task or request to delegate to the NPC"}, "max_iterations": "10"}
```
