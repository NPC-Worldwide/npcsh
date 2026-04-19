---
name: translate_tools
description: Map foreign tool names (e.g. Claude Code's Read/Grep/Edit from a CLAUDE.md
  or AGENTS.md import) onto npcsh jinxes via an ask_form prompt. Returns a JSON dict
  {foreign_name - jinx_name} for mappings the user approved; foreign names set to
  'skip' are omitted from the result.
---

# translate_tools

Map foreign tool names (e.g. Claude Code's Read/Grep/Edit from a CLAUDE.md or AGENTS.md import) onto npcsh jinxes via an ask_form prompt. Returns a JSON dict {foreign_name - jinx_name} for mappings the user approved; foreign names set to 'skip' are omitted from the result.

## Inputs

- `foreign_tools` (default: `'[]'`)
- `title` (default: `'Map imported tools to npcsh jinxes'`)

## Steps

- `prompt_mapping` → [`prompt_mapping.py`](./prompt_mapping.py)

## Usage

```
/run_jinx jinx_ref=translate_tools input_values={"foreign_tools": "[]", "title": "Map imported tools to npcsh jinxes"}
```
