---
name: reattach
description: Interactive viewer to browse and reattach to previous conversations
---

# reattach

Interactive viewer to browse and reattach to previous conversations

## Inputs

- `path` (default: `''`)
- `all` (default: `'false'`)

## Steps

- `launch_viewer` → [`launch_viewer.py`](./launch_viewer.py)

## Usage

```
/run_jinx jinx_ref=reattach input_values={"path": "", "all": "false"}
```
