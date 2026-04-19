---
name: release_notes
description: Generate release notes from commits since the last release. Pass a GitHub
  repo (e.g. npc-worldwide/npcpy) or run in a local repo.
---

# release_notes

Generate release notes from commits since the last release. Pass a GitHub repo (e.g. npc-worldwide/npcpy) or run in a local repo.

## Inputs

- `repo` (default: `''`)
- `from_tag` (default: `''`)
- `to_ref` (default: `''`)

## Steps

- `gen_release_notes` → [`gen_release_notes.py`](./gen_release_notes.py)

## Usage

```
/run_jinx jinx_ref=release_notes input_values={"repo": "", "from_tag": "", "to_ref": ""}
```
