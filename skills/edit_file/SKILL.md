---
name: edit_file
description: Creates or edits a file. If the file does not exist, creates it with
  the specified content. If the file exists, examines it and applies changes.
---

# edit_file

Creates or edits a file. If the file does not exist, creates it with the specified content. If the file exists, examines it and applies changes.

## Inputs

- `file_path`
- `edit_instructions`
- `backup` (default: `False`)

## Steps

- `edit_file` → [`edit_file.py`](./edit_file.py)

## Usage

```
/run_jinx jinx_ref=edit_file input_values={"file_path": "<value>", "edit_instructions": "<value>", "backup": false}
```
