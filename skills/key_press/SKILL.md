---
name: key_press
description: 'Press a keyboard key or key combination.

  Valid keys: enter, tab, escape, backspace, delete, space, up, down, left, right,

  home, end, pageup, pagedown, f1-f12, ctrl, alt, shift, command.

  For combinations use + like: ctrl+a, ctrl+c, ctrl+v, alt+tab, ctrl+shift+t

  For regular letters/numbers, use type_text instead.'
---

# key_press

Press a keyboard key or key combination.
Valid keys: enter, tab, escape, backspace, delete, space, up, down, left, right,
home, end, pageup, pagedown, f1-f12, ctrl, alt, shift, command.
For combinations use + like: ctrl+a, ctrl+c, ctrl+v, alt+tab, ctrl+shift+t
For regular letters/numbers, use type_text instead.

## Inputs

- `key` (default: `'enter'`)

## Steps

- `perform_key` → [`perform_key.py`](./perform_key.py)

## Usage

```
/run_jinx jinx_ref=key_press input_values={"key": "enter"}
```
