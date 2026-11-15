# Slash Commands vs. Jinxes

`npcsh` exposes two different kinds of `/`-prefixed and team-defined commands. They serve different purposes and work differently.

## Slash commands — built into the shell

Slash commands live in `npcsh` itself and are always available, no matter which team is loaded or whether a team is loaded at all. They control the shell session, open TUIs, or perform one-off actions (reload, exit, etc).

Type `/help` in npcsh to see the full list. They're grouped into:

| Category | Commands | Examples |
|----------|----------|----------|
| **Modes** | `/agent`, `/chat`, `/cmd` | Switch how input is interpreted (agentic / chat-only / bash-first) |
| **Terminal UIs** | `/ctx`, `/git`, `/jinxes`, `/kg`, `/loop`, `/loops`, `/memories`, `/model`, `/ps`, `/stats`, `/team` | Full-screen TUIs for the git graph, knowledge graph, processes, team state, etc |
| **System / Config** | `/clear`, `/config`, `/flush`, `/history`, `/reattach` | Session housekeeping |
| **System Commands** | `/init`, `/reload`, `/sync`, `/update`, `/verbose` | Initialize/reload the team from disk, update npcsh, toggle verbose mode |
| **Info** | `/exit`, `/quit`, `/tutorial`, `/help` | |

These are hardcoded in npcsh's source — they exist from the moment the shell boots, before any team is loaded.

## Jinxes — defined by the team, not the shell

Jinxes are reusable tools and workflows declared as **`.jinx` files in your team's `jinxes/` directory** (or in project-local `.jinx` folders). They are:

- **Team-scoped** — each team defines its own jinxes. Switch to another team and a different set appears.
- **Loaded at startup / on `/reload`** — the shell compiles your team's jinxes only after a team is present.
- **Invoked by name** — you type the jinx name directly (optionally prefixed with `/` if there's no collision), or `@npc jinx_name` to run it under a specific NPC.

`/jinxes` lists what's currently available for this session. If a team doesn't load any jinxes, the list is empty — the shell itself is still fully functional via the built-in slash commands.

In short: slash commands belong to *npcsh*, jinxes belong to *your team*.
