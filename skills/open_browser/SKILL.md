---
name: open_browser
description: 'Open a browser and navigate to a URL. The browser stays open for follow-up
  commands.

  Use this to start browser automation.'
---

# open_browser

Open a browser and navigate to a URL. The browser stays open for follow-up commands.
Use this to start browser automation.

## Inputs

- `url` (default: `{'description': 'URL to navigate to'}`)
- `browser` (default: `'firefox'`)

## Steps

- `open_browser` → [`open_browser.py`](./open_browser.py)

## Usage

```
/run_jinx jinx_ref=open_browser input_values={"url": {"description": "URL to navigate to"}, "browser": "firefox"}
```
