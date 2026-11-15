---
name: ctx-guide
description: Reference guide for .ctx team context files: structure, reserved keys, user-defined keys, and how they are preserved.
source_jinx: npcsh/npc_team/jinxes/skills/ctx-guide.jinx
engine: skill
---

# ctx-guide

Reference guide for .ctx team context files: structure, reserved keys, user-defined keys, and how they are preserved.

## What-Is-Ctx

A `.ctx` file is a YAML document that defines team-level context shared by all NPCs.
`npcsh.ctx` is the default context file loaded from the team directory.
It is read by `Team._load_team_context_file()` in `npcpy.npc_compiler`.


## Reserved-Keys

These keys are managed by the framework and updated from source on `/sync`:
- `name`: team name.
- `context`: shared directive text injected into all NPC system prompts.
- `databases`: list of database paths the team uses.
- `forenpc`: default NPC name at startup.
- `mcp_servers`: list of configured MCP servers.
- `providers`: named provider configurations.
- `available_executables`: allow-lists for read/write/execute shell commands.
- `SKILLS_DIRECTORY`: optional extra directory to load skill jinxes from.


## User-Defined-Keys

Any key not in the reserved list is preserved across `/sync` updates.
Examples: `COMPRESSION_INSTRUCTIONS`, custom instruction hooks, project-specific variables.
The shell merges these preserved values back into the newly copied `npcsh.ctx`.


## Custom-Instructions

Blank user-customizable fields can be added to `npcsh.ctx`:
  COMPRESSION_INSTRUCTIONS: ""
Jinxes read them at runtime by scanning `.ctx` files in the team directory.
This lets users tune behavior without editing framework jinxes.


## File-Example

```yaml
context: |
  The npcsh NPC team is devoted to providing a safe and helpful environment...
databases:
- ~/npcsh_history.db
forenpc: corca
mcp_servers: []
providers: []
available_executables:
  read: [cat, ls, head, tail, wc, git]
  write: [cp, echo, mkdir, mv, rm, touch]
  execute: [bash, python, python3, sh]
COMPRESSION_INSTRUCTIONS: ""
```


## Loading-Order

1. Team context file(s) are loaded first.
2. Jinxes are loaded and first-pass rendered.
3. NPCs are loaded; their files can reference jinxes by name via the Jinx() helper because the jinx path map already exists.


## Sync-Preservation

`team_sync::sync_team()` in `npcsh` deletes all non-`usr` team files and recopies them,
but before overwriting `npcsh.ctx` it records every user-defined key and writes them back.
This keeps user tweaks alive across framework updates.


## Access-In-Jinxes

To read a `.ctx` value inside a Python step:
  import yaml
  from pathlib import Path
  team_dir = os.path.expanduser("~/.npcsh/npc_team")
  for ctx_file in Path(team_dir).glob("*.ctx"):
      ctx = yaml.safe_load(open(ctx_file))
      value = ctx.get("KEY", "")
`context.get('team')` and `npc.shared_context` also expose context data.
