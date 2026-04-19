---
name: nql
description: Interactive DB viewer/manager or run NPC-SQL models
---

# nql

Interactive DB viewer/manager or run NPC-SQL models

## Inputs

- `models_dir` (default: `'~/.npcsh/npc_team/models'`)
- `db` (default: `''`)
- `model` (default: `''`)
- `schema` (default: `''`)
- `show` (default: `''`)
- `query` (default: `''`)
- `cron` (default: `''`)
- `install_cron` (default: `''`)

## Steps

- `run_nql` → [`run_nql.py`](./run_nql.py)

## Usage

```
/run_jinx jinx_ref=nql input_values={"models_dir": "~/.npcsh/npc_team/models", "db": "", "model": "", "schema": "", "show": "", "query": "", "cron": "", "install_cron": ""}
```
