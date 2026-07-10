# NPC Shell Commands

This is a reference for the built-in slash commands available inside `npcsh`. Because `npcsh` is a data-layer runtime, most of its power comes from the **jinxes** (tools) and **NPCs** defined in your team directory. The canonical, up-to-date list is always available by running:

```npcsh
npcsh> /help
```

To browse the jinxes available to the current team, run:

```npcsh
npcsh> /jinxes
```

---

## Modes

| Command | Description |
|---------|-------------|
| `/agent` | Full agent mode: the NPC can call jinxes, run bash, and use the LLM. |
| `/chat` | Chat-only mode: LLM responses without tool use. |
| `/cmd` | Command mode: input is run as bash first; if it fails, fall back to the LLM. |

---

## NPCs

| Command | Description |
|---------|-------------|
| `/<npc>` | Switch the current session to the named NPC (e.g., `/corca`). |
| `@<npc>` | Switch to the named NPC, or ask a one-off question: `@corca review this function`. |
| `/kill` | Kill the current process. |

---

## System / Config

| Command | Description |
|---------|-------------|
| `/clear` | Clear the current conversation. |
| `/config` | Open the configuration TUI for `~/.npcshrc`. |
| `/ctx` | Browse and edit team context fields. |
| `/history` | Show the conversation history for the current process. |
| `/memories` | Open the memory lifecycle TUI. |
| `/model` | Open the model-selection TUI. |
| `/reattach` | Reattach to previous conversation sessions. |
| `/set` | Set model, provider, or mode (e.g., `/set model=gpt-4o`). |
| `/setup` | Run the first-time setup TUI. |
| `/stats` | Show kernel stats. |
| `/team` | Open the team-management TUI. |

---

## Tools

| Command | Description |
|---------|-------------|
| `/commit` | Commit-helper TUI. |
| `/gitt` | Git TUI (status, stage, commit, history). |

---

## Loops / Cron

| Command | Description |
|---------|-------------|
| `/cron` | Cron management. |
| `/loop` | Create a new loop. |
| `/loop_demo` | Add a demo heartbeat loop. |
| `/loopoff` | Disable a loop. |
| `/loopon` | Enable a loop. |
| `/looprm` | Remove a loop. |
| `/loops` | List active loops. |

---

## System Commands

| Command | Description |
|---------|-------------|
| `/doctor` | Diagnose and auto-fix common issues. |
| `/init` | Initialize or reinitialize `npcsh`. |
| `/nsync` | Sync `npcsh` state. |
| `/refresh` | Refresh `npcsh` (alias of `/reload`). |
| `/reload` | Reload `npcsh` state. |
| `/shh` | Toggle quiet mode. |
| `/update` | Update `npcsh`. |
| `/usage` | Show usage info. |
| `/verbose` | Toggle verbose mode. |

---

## Info

| Command | Description |
|---------|-------------|
| `/exit` | Exit `npcsh`. |
| `/help` | Show this help. |
| `/jinxes` | List available jinxes. |
| `/ps` | List running processes. |
| `/quit` | Exit `npcsh`. |
| `/tutorial` | Run the interactive tutorial. |

---

## Jinxes

Jinxes are **tools that agents use**, not slash commands for users. They are defined as `.jinx` files in your team directory. Agents invoke jinxes by name; you can ask an NPC to use one, or the NPC may call one automatically when it fits the task.

Common jinxes in the bundled team include:

- `delegate`, `convene`, `deep_research`, `teamviz` — orchestration
- `py`, `shell`, `edit_file`, `load_file`, `compile`, `build` — code work
- `web_search`, `file_search`, `db_search` — search
- `computer_use`, `screenshot` — GUI / browser automation
- `gen_image` — image generation
- `skill`, `git-workflow`, `pr_review` — skills and workflows
- `memory_extractor` — memory extraction
- `crond`, `loop_plan` — scheduling helpers

The exact set depends on the team you have loaded. Use `/jinxes` inside `npcsh` to see the current list.

---

## `npc` CLI

The `npc` binary runs one-off instructions from the shell:

```bash
npc "what process is listening on port 5337?"
npc --model qwen3.5:9b --provider ollama "explain this file"
npc --npc corca --path rust/src/main.rs "review this code"
```

Common flags:

| Flag | Description |
|------|-------------|
| `-mo`, `--model` | Model ID |
| `-pr`, `--provider` | Provider (ollama, openai, anthropic, gemini, ...) |
| `-np`, `--npc` | NPC to use |
| `-t`, `--temperature` | Sampling temperature |
| `-p`, `--path` | File or directory to attach |
| `-h`, `--help` | Show help |

Run `npc --help` for the full list.
