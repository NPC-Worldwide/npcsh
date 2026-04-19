---
name: serve
description: NPC Server Dashboard - start, stop, and monitor the API server
---

# serve

NPC Server Dashboard - start, stop, and monitor the API server

## Inputs

- `port` (default: `5337`)
- `cors` (default: `''`)

## Steps

- `serve_tui` → [`serve_tui.py`](./serve_tui.py)

## Usage

```
/run_jinx jinx_ref=serve input_values={"port": 5337, "cors": ""}
```
