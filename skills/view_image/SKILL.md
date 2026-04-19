---
name: view_image
description: Load an image file so the model can see it. Compresses if needed, attaches
  to conversation.
---

# view_image

Load an image file so the model can see it. Compresses if needed, attaches to conversation.

## Inputs

- `filepath` (default: `None`)
- `path` (default: `None`)
- `show_inline` (default: `True`)

## Steps

- `load_and_attach` → [`load_and_attach.py`](./load_and_attach.py)

## Usage

```
/run_jinx jinx_ref=view_image input_values={"filepath": null, "path": null, "show_inline": true}
```
