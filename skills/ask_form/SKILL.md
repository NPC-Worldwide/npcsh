---
name: ask_form
description: Display a structured form for the user to fill out. Returns JSON dict
  of field values. Used by agents to gather structured input.
---

# ask_form

Display a structured form for the user to fill out. Returns JSON dict of field values. Used by agents to gather structured input.

## Inputs

- `title` (default: `'Input Required'`)
- `fields` (default: `'[]'`)

## Steps

- `render_form` → [`render_form.py`](./render_form.py)

## Usage

```
/run_jinx jinx_ref=ask_form input_values={"title": "Input Required", "fields": "[]"}
```
