# Agent Coders

npcsh can drive external AI coding agents (Claude Code, Codex, Gemini CLI, OpenCode, Amp, Aider) with your team's persona and jinxes loaded, straight from the npcsh prompt.

## How it works

When you run a known agent coder from inside npcsh (type `claude`, `codex`, ...) or through the `npc` CLI (`npc claude`), npcsh intercepts it and:

1. Resolves the team (`./npc_team` in your project, or `~/.npcsh/npc_team` as a global default).
2. Shows an agent picker for the team (or honors `--npc <name>`).
3. Registers the **NPC MCP server** (`npcsh mcp-server <team> --npc <name>`) with the coder so team jinxes show up as MCP tools.
4. Injects the persona through the coder's own CLI arguments/env, with **no project files written**.
5. Launches the coder.

The coder then runs as that persona, with your team's jinxes available as tools.

### MCP server

`npcsh mcp-server <team> [--npc <name>]` is a minimal stdio JSON-RPC MCP server exposing the team's jinxes (`tools/list`, `tools/call`) alongside the usual `initialize` handshake. It is what the coder spawns behind the scenes.

### Persona injection

Persona is delivered through the coder's native CLI flags — nothing is written into your repository:

- **Claude Code** — `--append-system-prompt "<persona>" --agents "<team-roster-json>"`
- **Codex CLI** — `-c instructions="<TOML-escaped persona>"`
- **Gemini CLI** — `GEMINI_SYSTEM_MD=<session tempfile>` (temp file is removed when the session ends)
- **OpenCode / Amp / Aider** — no known system-prompt flag today; they consume the persona via the `npc` MCP server

### Tracking attaches

Attach metadata is logged to `~/.npcsh/agent_coders.json`:

```json
{
  "codex": {
    "npc": "corca",
    "team_path": "/home/you/.npcsh/npc_team",
    "attached_at": 1793503615,
    "mcp_server": "npcsh mcp-server"
  }
}
```

## Examples

```bash
npcsh> claude                     # pick an agent, then Claude Code launches as that persona
npcsh> codex --npc corca          # Codex with the corca persona
npc codex --npc corca <args>      # same from the npc CLI
```

## Supported coders

`claude`, `codex`, `gemini`, `opencode`, `amp`, `aider`

`agent_coders::is_agent_coder` gates interception; each coder must be on your PATH to launch.
