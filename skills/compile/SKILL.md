---
name: compile
description: Compile NPC profiles
---

# compile

Compile NPC profiles

## Inputs

- `npc_file_path` (default: `''`)
- `npc_team_dir` (default: `'./npc_team'`)

## Steps

- `compile_npcs` → [`compile_npcs.py`](./compile_npcs.py)

## Usage

```
/run_jinx jinx_ref=compile input_values={"npc_file_path": "", "npc_team_dir": "./npc_team"}
```
