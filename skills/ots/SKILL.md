---
name: ots
description: 'Take screenshot and analyze with vision model. Usage: /ots <prompt>'
---

# ots

Take screenshot and analyze with vision model. Usage: /ots <prompt>

## Inputs

- `prompt`
- `image_paths_args` (default: `''`)
- `vmodel` (default: `''`)
- `vprovider` (default: `''`)

## Steps

- `analyze_screenshot_or_image` → [`analyze_screenshot_or_image.py`](./analyze_screenshot_or_image.py)

## Usage

```
/run_jinx jinx_ref=ots input_values={"prompt": "<value>", "image_paths_args": "", "vmodel": "", "vprovider": ""}
```
