---
name: edit
description: Edit an NPC, jinx, context, or file. Usage - /edit npc, /edit jinx, /edit
  ctx, /edit file
---

# edit

Edit an NPC, jinx, context, or file. Usage - /edit npc, /edit jinx, /edit ctx, /edit file

## Inputs

- `type`
- `name` (default: `''`)
- `field` (default: `''`)
- `value` (default: `''`)
- `file_path` (default: `''`)
- `edit_instructions` (default: `''`)
- `backup` (default: `'false'`)

## Steps

- `edit_resource` → [`edit_resource.py`](./edit_resource.py)

## Usage

```
/run_jinx jinx_ref=edit input_values={"type": "<value>", "name": "", "field": "", "value": "", "file_path": "", "edit_instructions": "", "backup": "false"}
```
