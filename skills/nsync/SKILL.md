---
name: nsync
description: Sync npc_team files from the npcsh repo to ~/.npcsh/npc_team. Detects
  local modifications before overwriting.
---

# nsync

Sync npc_team files from the npcsh repo to ~/.npcsh/npc_team. Detects local modifications before overwriting.

## Inputs

- `force` (default: `''`)
- `dry_run` (default: `''`)
- `jinxes` (default: `''`)
- `npcs` (default: `''`)
- `ctx` (default: `''`)
- `images` (default: `''`)

## Steps

- `sync_npc_team` → [`sync_npc_team.py`](./sync_npc_team.py)

## Usage

```
/run_jinx jinx_ref=nsync input_values={"force": "", "dry_run": "", "jinxes": "", "npcs": "", "ctx": "", "images": ""}
```
