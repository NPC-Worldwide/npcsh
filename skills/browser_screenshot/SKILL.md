---
name: browser_screenshot
description: Take a screenshot of the current browser page.
---

# browser_screenshot

Take a screenshot of the current browser page.

## Inputs

- `filename` (default: `{'description': 'Optional filename for screenshot', 'default': ''}`)

## Steps

- `browser_screenshot` → [`browser_screenshot.py`](./browser_screenshot.py)

## Usage

```
/run_jinx jinx_ref=browser_screenshot input_values={"filename": {"description": "Optional filename for screenshot", "default": ""}}
```
