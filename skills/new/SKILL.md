---
name: new
description: Create a new NPC, jinx, or file. Usage - /new npc, /new jinx, /new file
---

# new

Create a new NPC, jinx, or file. Usage - /new npc, /new jinx, /new file

## Inputs

- `type`
- `name` (default: `''`)
- `primary_directive` (default: `''`)
- `description` (default: `''`)
- `code` (default: `''`)
- `model` (default: `''`)
- `provider` (default: `''`)
- `jinxes` (default: `''`)
- `path` (default: `'lib/core'`)
- `inputs` (default: `''`)
- `file_path` (default: `''`)
- `edit_instructions` (default: `''`)

## Steps

- `create_resource` → [`create_resource.py`](./create_resource.py)

## Usage

```
/run_jinx jinx_ref=new input_values={"type": "<value>", "name": "", "primary_directive": "", "description": "", "code": "", "model": "", "provider": "", "jinxes": "", "path": "lib/core", "inputs": "", "file_path": "", "edit_instructions": ""}
```
