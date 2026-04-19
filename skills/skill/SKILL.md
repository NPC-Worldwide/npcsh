---
name: skill
description: Constructs and serves a structured skill. Receives the full skill definition
  and returns the requested part. Do not call directly.
---

# skill

Constructs and serves a structured skill. Receives the full skill definition and returns the requested part. Do not call directly.

## Inputs

- `skill_name`
- `skill_description`
- `sections`
- `scripts_json`
- `references_json`
- `assets_json`
- `section`

## Steps

- `handle_skill` → [`handle_skill.py`](./handle_skill.py)

## Usage

```
/run_jinx jinx_ref=skill input_values={"skill_name": "<value>", "skill_description": "<value>", "sections": "<value>", "scripts_json": "<value>", "references_json": "<value>", "assets_json": "<value>", "section": "<value>"}
```
