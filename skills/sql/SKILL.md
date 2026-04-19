---
name: sql
description: Execute queries on the ~/npcsh_history.db to pull data. The database
  contains only information about conversations and other user-provided data. It does
  not store any information about individual files.
---

# sql

Execute queries on the ~/npcsh_history.db to pull data. The database contains only information about conversations and other user-provided data. It does not store any information about individual files.

## Inputs

- `sql_query` (default: `''`)

## Steps

- `execute_sql` → [`execute_sql.py`](./execute_sql.py)

## Usage

```
/run_jinx jinx_ref=sql input_values={"sql_query": ""}
```
