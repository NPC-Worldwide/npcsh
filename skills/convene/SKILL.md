---
name: convene
description: Start a group discussion between multiple NPCs to get diverse perspectives
  on a topic.
---

# convene

Start a group discussion between multiple NPCs to get diverse perspectives on a topic.

## Inputs

- `topic` (default: `''`)
- `npcs` (default: `'alicanto,corca,guac'`)
- `rounds` (default: `3`)
- `model` (default: `None`)
- `provider` (default: `None`)

## Steps

- `convene_tui` → [`convene_tui.py`](./convene_tui.py)

## Usage

```
/run_jinx jinx_ref=convene input_values={"topic": "", "npcs": "alicanto,corca,guac", "rounds": 3, "model": null, "provider": null}
```
