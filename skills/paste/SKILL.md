---
name: paste
description: Grabs content from clipboard (images or text) and saves/displays it.
  Use this when Ctrl+V paste doesn't work properly.
---

# paste

Grabs content from clipboard (images or text) and saves/displays it. Use this when Ctrl+V paste doesn't work properly.

## Inputs

- `output_path` (default: `{'default': '', 'description': 'Optional path to save image to. If empty, saves to temp file.'}`)

## Steps

- `paste_clipboard` → [`paste_clipboard.py`](./paste_clipboard.py)

## Usage

```
/run_jinx jinx_ref=paste input_values={"output_path": {"default": "", "description": "Optional path to save image to. If empty, saves to temp file."}}
```
