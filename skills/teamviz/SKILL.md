---
name: teamviz
description: Visualize NPC team structure - NPCs, jinxes, and their relationships
---

# teamviz

Visualize NPC team structure - NPCs, jinxes, and their relationships

## Inputs

- `team_path` (default: `''`)
- `save` (default: `''`)

## Steps

- `visualize_team` → [`visualize_team.py`](./visualize_team.py)

## Usage

```
/run_jinx jinx_ref=teamviz input_values={"team_path": "", "save": ""}
```
