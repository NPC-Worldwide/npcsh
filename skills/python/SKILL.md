---
name: python
description: "Execute Python code directly. Code runs immediately \u2014 do NOT wrap\
  \ in function definitions unless you also call them. To create a file, use open('/path',\
  \ 'w') and write to it. To create a script file, write the script content to the\
  \ file with open(). Set the \"output\" variable to a result string. Do not write\
  \ print statements in your code. do not ever include bare backslashes preceding\
  \ strings and do not include guides for usage in python scripts unless explicitly\
  \ requested."
---

# python

Execute Python code directly. Code runs immediately — do NOT wrap in function definitions unless you also call them. To create a file, use open('/path', 'w') and write to it. To create a script file, write the script content to the file with open(). Set the "output" variable to a result string. Do not write print statements in your code. do not ever include bare backslashes preceding strings and do not include guides for usage in python scripts unless explicitly requested.

## Inputs

- `code`

## Steps

- `step_1` → [`step_1.py`](./step_1.py)

## Usage

```
/run_jinx jinx_ref=python input_values={"code": "<value>"}
```
