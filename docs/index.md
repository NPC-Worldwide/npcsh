<p align="center">
  <a href= "https://github.com/cagostino/npcpy/blob/main/docs/npcpy.md">
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc-python.png" alt="npc-python logo" width=250></a>
</p>

# npcsh Documentation

`npcsh` is an agentic shell built around a declarative data layer for AI teams. Team context, agents, and tools are defined as plain files; the shell compiles them into a live multi-agent system you can chat with, schedule, or serve over an API.

## Quick Links

- [Installation Guide](installation.md)
- [Full Guide](guide.md) — agents, jinxes, orchestration, knowledge graphs, and more
- [Benchmarks](benchmarks.md) — pass/fail results across 125 agentic-shell tasks
- [NPC CLI](npc_cli.md)
- [Skills](skills.md) — knowledge-content jinxes with progressive section disclosure

## Program Guides

- [NPC Shell](npcsh.md)
- [Alicanto](alicanto.md)
- [Guac](guac.md)
- [PTI](pti.md)
- [Spool](spool.md)
- [Wander](wander.md)
- [Yap](yap.md)
- [TLDR Cheat Sheet](TLDR_Cheat_sheet.md)

## The NPC Data Layer

Everything customizable in `npcsh` lives as simple files across three layers:

| Layer | Files | Purpose |
|-------|-------|---------|
| **Team** | `.ctx` / `team.ctx` / `npc_team/*.ctx` | Shared context: default model/provider, forenpc, MCP servers, env vars |
| **Agents** | `.npc`, `agents.md`, `agents/` | Agent definitions: name, persona, directive, model/provider, and jinxes they can use |
| **Tools** | `.jinx`, `skills/` | Reusable tools and workflows that agents invoke by name |

Files can live inside `npc_team/` or at the project root. The agent layer can use `.npc` files, a single `agents.md`, or an `agents/` directory — these are alternatives, not a required combination.

Because these are ordinary files, you can version them in git, share them across projects, and drop in agent definitions from other ecosystems.

## Contributing

Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## Support

If you appreciate the work here, [consider supporting NPC Worldwide](https://buymeacoffee.com/npcworldwide). If you'd like to explore how to use `npcpy` to help your business, please reach out to info@npcworldwi.de.

## License

This project is licensed under the MIT License.
