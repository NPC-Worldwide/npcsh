<p align="center">
  <a href="https://github.com/npc-worldwide/npcsh/blob/main/docs/npcsh.md">
  <img src="https://raw.githubusercontent.com/NPC-Worldwide/npcsh/main/npcsh/npcsh.png" alt="npcsh logo" width=600></a>
</p>

<h1 align="center">npcsh</h1>

<p align="center">
  <strong>The agentic shell for building and running AI teams from the command line.</strong>
</p>

<p align="center">
  <a href="https://github.com/npc-worldwide/npcsh/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/npcsh/"><img src="https://img.shields.io/pypi/v/npcsh.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/npcsh/"><img src="https://img.shields.io/pypi/pyversions/npcsh.svg" alt="Python"></a>
  <a href="https://npc-shell.readthedocs.io/"><img src="https://img.shields.io/badge/docs-readthedocs-brightgreen.svg" alt="Docs"></a>
</p>

---

`npcsh` is an agentic shell for building, orchestrating, and interacting with teams of AI agents from the terminal. Instead of treating AI as a single chat window, `npcsh` gives you a **declarative data layer** for defining agents, tools, context, and workflows as plain files in a project directory. The shell then compiles that data into a live team you can chat with, delegate to, schedule, or serve over an API.

The fastest way to try it:

```bash
pip install 'npcsh[lite]'
npcsh
```

Then ask for help, edit files, search the web, or just chat:

```bash
npcsh> can you help me identify what process is listening on port 5337?
npcsh> please read through the markdown files in the docs folder and suggest changes
```

---

## The NPC Data Layer

Everything in `npcsh` is built around a small set of file types in an `npc_team/` directory. This lets you treat agents, tools, and context as data that can be versioned, shared, and composed across projects.

| File | Purpose |
|------|---------|
| **`.npc`** | Agent definitions (persona, directive, model, provider, available tools). Executable with a shebang. |
| **`.jinx`** | Jinja execution templates — reusable tools/workflows that agents invoke like functions. |
| **`.ctx`** | Team context: default model/provider, forenpc (orchestrator), MCP servers, env vars, shared memory. |
| **`.sql`** | NQL models — SQL with embedded AI functions, runnable locally or on Snowflake/Databricks/BigQuery. |
| **`.pipe`** | Assembly lines — multi-step workflow pipelines. |

A minimal project looks like this:

```
npc_team/
├── team.ctx            # team config + forenpc
├── sibiji.npc          # orchestrator
├── corca.npc           # coding specialist
├── agents.md           # optional bulk agent definitions
├── agents/             # optional one-agent-per-file markdown
├── jinxes/
│   ├── skills/
│   │   └── debugging/
│   │       └── SKILL.md
│   └── my_tool.jinx
└── models/
    └── daily_summary.sql
```

Because these are ordinary files, you can:

- Check an entire agent team into git.
- Share reusable jinxes/skills across projects.
- Drop in `agents.md` or `agents/` folders from other tools (Claude Code, Codex, etc.) and `npcsh` picks them up.
- Switch models, providers, or whole team configurations without touching code.

## Build Your Own Tools

Jinxes are the main extension point. A jinx is a YAML file that describes inputs, a prompt template, and one or more execution steps. They are invoked as slash commands and can call other jinxes, run Python or shell, query the local DB, or call LLMs.

```yaml
# jinxes/hello.jinx
jinx_name: hello
description: Greet someone by name.
inputs:
  - name
steps:
  - engine: llm
    prompt: |
      Say a warm, personalized hello to {{ name }}.
```

Use it inside `npcsh`:

```bash
/hello name=world
```

Or make it available to agents by adding it to an NPC:

```yaml
# corca.npc
name: corca
primary_directive: You are a coding specialist.
jinxes:
  - lib/core/python
  - lib/core/sh
  - lib/core/edit_file
  - hello
```

Skills are a special kind of jinx that serve instructional content progressively. A skill like `debugging` can expose sections (`reproduce`, `isolate`, `fix`) so agents only load the methodology they need, keeping token usage low.

## Capabilities

`npcsh` is not a command catalog — it is a runtime for capabilities you define and compose:

- **Agentic shell** — Chat with individual NPCs or the team orchestrator. Switch agents with `/npc` or invoke one directly with `/@corca`.
- **Custom tools** — Author jinxes and skills for your domain; agents use them automatically.
- **Multi-agent orchestration** — The forenpc delegates tasks, convenes discussions, and runs review loops across specialized NPCs.
- **Memory & knowledge graphs** — Conversations feed a memory lifecycle; approved memories can be synthesized into a queryable knowledge graph.
- **NQL** — SQL models with embedded AI functions that run on SQLite locally and translate to native AI functions on Snowflake, Databricks, and BigQuery.
- **Computer use** — GUI automation via vision, browser automation, screenshot analysis.
- **API server** — Serve any NPC team via OpenAI-compatible endpoints (`/serve`).
- **Scheduling** — Cron jobs, daemons, and triggered workflows.
- **Model portability** — Switch between Ollama, OpenAI, Anthropic, Gemini, DeepSeek, and any LiteLLM-compatible provider.

---

## Benchmark Results

The benchmark suite runs 135 tasks across 15 categories — shell commands, file operations, Python scripting, debugging, git, multi-step workflows, tool chaining, and more — and scores pass/fail. The table below reflects scores from the **current Rust-based `npcsh` runtime**. Additional models are being evaluated continuously; see `docs/benchmarks.md` for the full per-category breakdown.

<table>
<tr><th>Family</th><th>Model</th><th>Score</th></tr>
<tr><td><b>Kimi</b></td><td>k2.5</td><td><b>121/125 (97%)</b></td></tr>
<tr><td rowspan="6"><b>Qwen3.5</b></td><td>0.8b</td><td>31/125 (24%)</td></tr>
<tr><td>2b</td><td>81/125 (65%)</td></tr>
<tr><td>4b</td><td>77/125 (62%)</td></tr>
<tr><td>9b</td><td><b>100/125 (80%)</b></td></tr>
<tr><td>35b</td><td><b>111/125 (88%)</b></td></tr>
<tr><td>397b</td><td><b>120/125 (96%)</b></td></tr>
<tr><td rowspan="5"><b>Qwen3</b></td><td>0.6b</td><td>—</td></tr>
<tr><td>1.7b</td><td>42/125 (34%)</td></tr>
<tr><td>4b</td><td><b>94/125 (75%)</b></td></tr>
<tr><td>8b</td><td>85/125 (68%)</td></tr>
<tr><td>30b</td><td><b>103/125 (82%)</b></td></tr>
<tr><td rowspan="2"><b>Gemma4</b></td><td>e4b</td><td>34/125 (27%)</td></tr>
<tr><td>31b</td><td><b>105/125 (84%)</b></td></tr>
<tr><td rowspan="4"><b>Gemma3</b></td><td>1b</td><td>—</td></tr>
<tr><td>4b</td><td>37/125 (30%)</td></tr>
<tr><td>12b</td><td>77/125 (62%)</td></tr>
<tr><td>27b</td><td>73/125 (58%)</td></tr>
<tr><td rowspan="3"><b>Llama</b></td><td>3.2:1b</td><td>—</td></tr>
<tr><td>3.2:3b</td><td>26/125 (20%)</td></tr>
<tr><td>3.1:8b</td><td>60/125 (48%)</td></tr>
<tr><td rowspan="3"><b>Mistral</b></td><td>small3.2</td><td>72/125 (57%)</td></tr>
<tr><td>ministral-3</td><td>51/125 (40%)</td></tr>
<tr><td>large-3</td><td>59/125 (47%)</td></tr>
<tr><td><b>Devstral</b></td><td>2</td><td>60/125 (48%)</td></tr>
<tr><td><b>MiniMax</b></td><td>M2.7</td><td><b>120/125 (96%)</b></td></tr>
<tr><td><b>Phi</b></td><td>phi4</td><td>58/125 (46%)</td></tr>
<tr><td><b>GPT-OSS</b></td><td>20b</td><td>94/125 (75%)</td></tr>
<tr><td rowspan="2"><b>OLMo2</b></td><td>7b</td><td>13/125 (10%)</td></tr>
<tr><td>13b</td><td>47/125 (38%)</td></tr>
<tr><td><b>Cogito</b></td><td>3b</td><td>10/125 (8%)</td></tr>
<tr><td rowspan="2"><b>GLM</b></td><td>4.7-flash</td><td><b>102/125 (82%)</b></td></tr>
<tr><td>5</td><td><b>120/125 (96%)</b></td></tr>
<tr><td><b>Nemotron</b></td><td>3-super</td><td>49/125 (39%)</td></tr>
<tr><td rowspan="3"><b>Gemini</b></td><td>2.5-flash</td><td>—</td></tr>
<tr><td>3.1-flash</td><td>—</td></tr>
<tr><td>3.1-pro</td><td>—</td></tr>
<tr><td rowspan="2"><b>Claude</b></td><td>4.6-sonnet</td><td>—</td></tr>
<tr><td>4.5-haiku</td><td>—</td></tr>
<tr><td><b>GPT</b></td><td>5-mini</td><td>—</td></tr>
<tr><td rowspan="3"><b>DeepSeek</b></td><td>v4-flash</td><td><b>99/125 (79%)</b></td></tr>
<tr><td>chat</td><td>—</td></tr>
<tr><td>reasoner</td><td>—</td></tr>
</table>

Run the benchmark yourself against a local or remote model:

```bash
python -m npcsh.benchmark.rust_runner --model qwen3.5:9b --provider ollama
python -m npcsh.benchmark.rust_runner --model gemini-2.5-flash --provider gemini
```

For a more comprehensive view of npcsh's capabilities and the advantages of the NPC Context-Agent-Tool data layer, see [ALARA for Agents: Least-Privilege Context Engineering Through Portable Composable Multi-Agent Teams](https://arxiv.org/abs/2603.20380).

---

## Installation

### PyPI (recommended)

```bash
pip install 'npcsh[lite]'        # API + Ollama providers
pip install 'npcsh[local]'       # + local diffusers/transformers/torch
pip install 'npcsh[yap]'         # + voice mode
pip install 'npcsh[all]'          # everything
```

What you get:

- The `npcsh` Python launcher, which starts the NPCSH server and execs the Rust shell.
- The `npc` and `npcsh-bench` CLI entry points.
- A default global team in `~/.npcsh/npc_team/`.

### macOS system dependencies

```bash
brew install portaudio ffmpeg pygobject3 ollama
brew services start ollama
ollama pull qwen3.5:2b
```

### Linux system dependencies

```bash
sudo apt-get install espeak portaudio19-dev python3-pyaudio ffmpeg \
                     libcairo2-dev libgirepository1.0-dev inotify-tools
# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b
```

### Windows

Install [Ollama](https://ollama.com) and [ffmpeg](https://ffmpeg.org), then:

```powershell
ollama pull qwen3.5:2b
pip install 'npcsh[lite]'
```

### Rust edition (development / latest)

`npcsh` ships as a Python launcher that starts the NPCSH server and execs the Rust shell binary (`npcrsh`). The launcher looks for `npcrsh` at `~/.npcsh/bin/npcrsh` first, then falls back to PATH.

For normal use, install the pre-built release via pip or brew. The Rust source build is for development only.

Both editions share `~/npcsh_history.db` and `~/.npcsh/npc_team/`.

### Configuration

On first run, `npcsh` creates `~/.npcshrc`:

```bash
export NPCSH_CHAT_MODEL=qwen3.5:2b
export NPCSH_CHAT_PROVIDER=ollama
export NPCSH_DEFAULT_MODE=agent
export NPCSH_EMBEDDING_MODEL=nomic-embed-text
export NPCSH_EMBEDDING_PROVIDER=ollama
```

API keys can go in `~/.npcshrc`, `~/.bashrc`, or a project `.env`:

```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
export DEEPSEEK_API_KEY="your_key"
```

---

## Agent Formats

`npcsh` supports three ways to define agents inside `npc_team/`. They can be mixed; `.npc` files take precedence if names collide.

**`.npc` files** — full-featured YAML agent definitions:

```yaml
#!/usr/bin/env npc
name: analyst
primary_directive: You analyze data and provide insights.
model: qwen3:8b
provider: ollama
jinxes:
  - skills/data-analysis
```

**`agents.md`** — multiple agents in one markdown file:

```markdown
## summarizer
You summarize long documents into concise bullet points.

## fact_checker
You verify claims against reliable sources and flag inaccuracies.
```

**`agents/` directory** — one `.md` file per agent:

```markdown
---
model: gemini-2.5-flash
provider: gemini
---
You translate content between languages while preserving tone and idiom.
```

All formats inherit the team's default model/provider from `team.ctx` when not specified.

---

## Launching External Coding Tools with NPC Teams

Your `npc_team/` works beyond `npcsh`. The `npcpy` CLI launchers let you start Claude Code, Codex, Gemini CLI, Aider, Amp, and others as an NPC from your team, injecting that NPC's persona and team awareness.

```bash
pip install npcpy

npc-claude
npc-claude --npc corca
npc-codex --npc researcher
npc-gemini --npc analyst
npc-opencode --npc coder
npc-aider --npc reviewer
npc-amp --npc writer
```

For deeper integration (jinxes as MCP tools, team switching), register the NPC plugin:

```bash
npc-plugin claude
npc-plugin codex
```

---

## Read the Docs

Full guides and API reference at [npc-shell.readthedocs.io](https://npc-shell.readthedocs.io/en/latest/).

## Links

- **[npcpy](https://github.com/cagostino/npcpy)** — Python framework for building AI agents and teams
- **[Incognide](https://github.com/npc-worldwide/incognide)** — Desktop workspace for the NPC Toolkit ([download](https://enpisi.com/incognide))
- **[Newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)** — Stay in the loop

## Research

- Quantum-like nature of natural language interpretation: [arxiv](https://arxiv.org/abs/2506.10077), accepted at [QNLP 2025](https://qnlp.ai)
- Simulating hormonal cycles for AI: [arxiv](https://arxiv.org/abs/2508.11829)
- ALARA for Agents: [arxiv](https://arxiv.org/abs/2603.20380)

## Community & Support

[Discord](https://discord.gg/XrjTFmDAna) | [Monthly donation](https://buymeacoffee.com/npcworldwide) | [Merch](https://enpisi.com/shop) | Consulting: info@npcworldwi.de

## Contributing

Contributions welcome! Submit issues and pull requests on the [GitHub repository](https://github.com/npc-worldwide/npcsh).

## License

MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=npc-worldwide/npcsh&type=Date)](https://star-history.com/#npc-worldwide/npcsh&Date)
