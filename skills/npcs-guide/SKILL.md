---
name: npcs-guide
description: Reference guide for NPCs: definition, fields, primary_directive, model/provider, and jinx assignments.
source_jinx: npcsh/npc_team/jinxes/skills/npcs-guide.jinx
engine: skill
---

# npcs-guide

Reference guide for NPCs: definition, fields, primary_directive, model/provider, and jinx assignments.

## What-Is-An-Npc

An NPC is an AI agent defined by a YAML `.npc` file loaded by `npcpy.npc_compiler.NPC`.
NPCs have a name, primary directive, optional model/provider, and a list of jinxes they can use.


## File-Format

```yaml
#!/usr/bin/env npc
name: corca
primary_directive: |
  You are corca, the software development and shell specialist.
model: qwen3.5:2b
provider: ollama
jinxes:
  - {{ Jinx('read') }}
  - {{ Jinx('write') }}
  - {{ Jinx('shell') }}
```


## Required-Fields

- `name`: NPC identifier. Used for `@name` addressing and process labels.
- `primary_directive`: system prompt describing the NPC's role, skills, and constraints.
- `jinxes`: list of jinx specs (names or `{{ Jinx('relative/path') }}` references).
Optional but common: `model`, `provider`, `api_url`, `api_key`.


## Primary-Directive

The directive is injected as the system message for every LLM call the NPC makes.
It should be concise, specify when to act vs delegate, and list allowed tools.
It is editable in the team TUI NPCs tab by pressing Enter on the selected NPC.


## Model-And-Provider

- `model`: the LLM identifier (e.g., `qwen3.5:2b`, `claude-sonnet-4-6`).
- `provider`: the backend adapter (`ollama`, `anthropic`, `openai`, `openai-like`, `opencode`).
Resolution order: NPC field → team context `model`/`provider` → `~/.npcshrc` env vars → defaults.
Named provider configs can be defined in `npcsh.ctx` under `providers:`.


## Jinx-Assignment

The `jinxes:` list resolves against the team's loaded jinx catalog.
Use `{{ Jinx('name') }}` to resolve a jinx name to its relative path for portability.
Use `{{ jinxes_list('lib/browser/*') }}` to bulk-load a pattern.
The special spec `*` loads all team jinxes.


## Team-Role

A team file (`npcsh.ctx`) declares `forenpc:` which is the default NPC at startup.
Other NPCs can be addressed as `@name` in the shell or delegated to via the `delegate` jinx.
NPCs share the team context but each maintains its own directive and model settings.


## Runtime-Access

Inside jinx code the active NPC is available as `context.get('npc')` or the bare `npc` variable.
Useful attributes: `name`, `model`, `provider`, `primary_directive`, `jinxes_dict`.
There is no `db_conn` on the NPC; history is handled by the shell/server state.


## Available-Npcs

The default npcsh team includes:
corca (shell/code orchestrator), alicanto (deep research), kadiefa (exploratory), frederic (creative/math), plonk (browser/desktop automation).
