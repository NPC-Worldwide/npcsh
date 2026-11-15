# npcsh Documentation

`npcsh` is a shell for portable, composable multi-agent teams. Team context, agents, and tools are defined as plain files; the shell compiles them into a live multi-agent system you can chat with, schedule, or extend with custom tools.

## Quick Links

- [Installation Guide](installation.md)
- [Skills](skills.md) — knowledge-content jinxes with progressive section disclosure

## The Team

The bundled team ships with one default agent and several specialists. The default agent is `corca`, who acts as both the coding/shell specialist and the team orchestrator. Switch to an agent inside the shell with `/<agent>` or ask a one-off with `@<agent>`:

```bash
npcsh> @corca refactor the auth module and add tests
npcsh> @alicanto summarize the last three papers on transformers
```

## Commands

`npcsh` separates **slash commands** from **team jinxes**:

- **Slash commands** are built into the shell itself (mode switches like `/agent`, TUIs like `/git`, housekeeping like `/clear`, `/help`, `/exit`, ...). They exist regardless of which — or any — team is loaded.
- **Jinxes** are reusable tools your *team* defines (`.jinx` files under `npc_team/jinxes/`). You run them by name, they appear or disappear as you switch teams, and `/jinxes` lists whatever this session's team provides.

See [Commands](commands.md) for the full built-in slash-command table.

The CLI also exposes agents via `npc`:

```bash
npc ./npc_team/corca.npc "what is the biggest file on my computer?"
npc chat -n corca
```

## Agent Coders

Run external AI coding agents as a team NPC, directly from the shell. Type `claude`, `codex`, `gemini`, `opencode`, ... inside npcsh, or via `npc <coder>`:

- npcsh shows an agent picker (or honors `--npc <name>`),
- registers the NPC MCP server with the coder, and
- injects the persona through the coder's own CLI args — no files written to your project.

See [Agent Coders](agent-coders.md) for the full flow.

## Ctrl+R History Search

At the npcsh prompt, `Ctrl+R` opens a reverse-i-search over `~/.npcsh_history` (deduped, most-recent-first). Type to narrow, `Ctrl+R`/Down cycles, Enter accepts into the input buffer, Esc or Ctrl+G aborts.

## The CAT Data Layer

Everything customizable in `npcsh` lives as simple files across three layers:

| Layer | Files | Purpose |
|-------|-------|---------|
| **Context** | `.ctx` / `team.ctx` / `npc_team/*.ctx` | Shared team context: default model/provider, env vars, MCP servers |
| **Agents** | `.npc`, `agents.md`, `agents/` | Agent definitions: name, persona, directive, model/provider, and jinxes |
| **Tools** | `.jinx`, `skills/` | Reusable tools and workflows that agents invoke by name |

Files can live inside `npc_team/` or at the project root. The agent layer can use `.npc` files, a single `agents.md`, or an `agents/` directory — these are alternatives, not a required combination.

Because these are ordinary files, you can version them in git, share them across projects, and drop in agent definitions from other ecosystems.

## Contributing

Contributions are welcome! Submit issues and pull requests on the [GitHub repository](https://github.com/npc-worldwide/npcsh).

## License

This project is licensed under the MIT License.
