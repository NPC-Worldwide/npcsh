---
name: browser_action
description: 'Perform an action in the browser. Actions:

  - click: Click element

  - type: Type text into element (clears first)

  - type_and_enter: Type text and press Enter

  - set_value: Force set value via JS (bypasses date pickers/validation)

  - select: Select dropdown option by visible text

  - wait: Wait for element to appear

  - scroll: Scroll page (up/down/to element)

  - get_text: Get text from element

  - get_page: Get page title, URL, and visible text

  - get_elements: Get interactive elements with their selectors

  - press_key: Press a key (enter, tab, escape, etc)

  Selectors: CSS ('
---

# browser_action

Perform an action in the browser. Actions:
- click: Click element
- type: Type text into element (clears first)
- type_and_enter: Type text and press Enter
- set_value: Force set value via JS (bypasses date pickers/validation)
- select: Select dropdown option by visible text
- wait: Wait for element to appear
- scroll: Scroll page (up/down/to element)
- get_text: Get text from element
- get_page: Get page title, URL, and visible text
- get_elements: Get interactive elements with their selectors
- press_key: Press a key (enter, tab, escape, etc)
Selectors: CSS (

## Inputs

- `action` (default: `{'description': 'Action: click, type, type_and_enter, set_value, select, wait, scroll, get_text, get_page, get_elements, press_key'}`)
- `selector` (default: `{'description': 'CSS selector or XPath (prefix xpath: for XPath)', 'default': ''}`)
- `value` (default: `{'description': 'Value for type/select, or scroll direction, or key name', 'default': ''}`)

## Steps

- `browser_action` → [`browser_action.py`](./browser_action.py)

## Usage

```
/run_jinx jinx_ref=browser_action input_values={"action": {"description": "Action: click, type, type_and_enter, set_value, select, wait, scroll, get_text, get_page, get_elements, press_key"}, "selector": {"description": "CSS selector or XPath (prefix xpath: for XPath)", "default": ""}, "value": {"description": "Value for type/select, or scroll direction, or key name", "default": ""}}
```
