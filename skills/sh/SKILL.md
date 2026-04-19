---
name: sh
description: Run a shell command and return stdout. Use this to create files, run
  scripts, install packages, and any other shell operation.
---

# sh

Run a shell command and return stdout. Use this to create files, run scripts, install packages, and any other shell operation.

## Inputs

- `bash_command`

## Steps

- `execute_bash` → [`execute_bash.py`](./execute_bash.py)

## Usage

```
/run_jinx jinx_ref=sh input_values={"bash_command": "<value>"}
```
