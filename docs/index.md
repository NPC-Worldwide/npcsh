# npcsh Documentation

`npcsh` is a shell for portable, composable multi-agent teams. Team context, agents, and tools are defined as plain files; the shell compiles them into a live multi-agent system you can chat with, schedule, or extend with custom tools.

## Quick Links

- [Installation Guide](installation.md)
- [Full Guide](guide.md) — agents, jinxes, orchestration, and the CAT data layer
- [Benchmarks](benchmarks.md) — pass/fail results across agentic-shell tasks
- [NPC Shell Commands](npcsh.md)
- [NPC CLI](npc_cli.md)
- [Skills](skills.md) — knowledge-content jinxes with progressive section disclosure

## Program Guides

- [NPC Shell](npcsh.md)
- [NPC CLI](npc_cli.md)
- [Alicanto](alicanto.md)
- [PTI](pti.md)
- [Spool](spool.md)
- [Wander](wander.md)
- [Yap](yap.md)
- [TLDR Cheat Sheet](TLDR_Cheat_sheet.md)

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
