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
  <a href="https://crates.io/crates/npcsh"><img src="https://img.shields.io/crates/v/npcsh.svg" alt="Crates.io"></a>
  <a href="https://npc-shell.readthedocs.io/"><img src="https://img.shields.io/badge/docs-readthedocs-brightgreen.svg" alt="Docs"></a>
</p>

---

`npcsh` makes the most of LLMs and agents through an interactive shell. Build teams of agents, schedule them on jobs, engineer context, and design custom Jinja Execution templates (Jinxes) for your agents to invoke for tool-use, prompts, and skills. 

Install `npcsh`:

```bash
curl -fsSL https://enpisi.com/install-npcsh.sh | sh
npcsh
```

Ask a question:
```bash
npcsh> what process is listening on port 5337?
```

Delegate to a coding agent:
```bash
npcsh> @corca refactor the auth module and add tests
```

Open the Git TUI after changes:
```bash
npcsh> /gitt
```

---

## Benchmark Results

The benchmark suite measures how well a model can drive `npcsh` as an agentic shell. It covers 100 tasks across 10 categories, from basic shell commands and file operations to multi-step workflows, debugging, git, and scripting. Each task is scored pass/fail by an automated verifier.

The table below shows scores (100 tasks).

### Scores (100 tasks)

<table>
<tr><th>Family</th><th>Model</th><th>Version</th><th>Score</th></tr>
<tr><td rowspan="3"><b>Qwen3.5</b></td><td>35b</td><td>v2.1.5</td><td>97/100 (97%)</td></tr>
<tr><td>9b</td><td>v2.1.5</td><td>95/100 (95%)</td></tr>
<tr><td>397b</td><td>v2.1.7</td><td>96/100 (96%)</td></tr>
<tr><td rowspan="2"><b>Ornith</b></td><td>35b</td><td>v2.1.7</td><td>97/100 (97%)</td></tr>
<tr><td>9b</td><td>v2.1.5</td><td>57/100 (57%)</td></tr>
<tr><td><b>Kimi</b></td><td>k2.7-code</td><td>v2.1.7</td><td>97/100 (97%)</td></tr>
<tr><td><b>Minimax</b></td><td>m3 428b (23b active)</td><td>v2.1.7</td><td>96/100 (96%)</td></tr>
<tr><td><b>DeepSeek</b></td><td>v4-flash 284b (13b active)</td><td>v2.1.7</td><td>95/100 (95%)</td></tr>
<tr><td rowspan="3"><b>Mistral</b></td><td>devstral-2 123b</td><td>v2.1.7</td><td>94/100 (94%)</td></tr>
<tr><td>devstral-small-2 24b</td><td>v2.1.7</td><td>94/100 (94%)</td></tr>
<tr><td>large-3 675b</td><td>v2.1.7</td><td>89/100 (89%)</td></tr>
<tr><td><b>North Mini Code 1.0</b></td><td>30b (3b active)</td><td>v2.1.5</td><td>93/100 (93%)</td></tr>
<tr><td><b>Gemma 4</b></td><td>31b</td><td>v2.1.7</td><td>92/100 (92%)</td></tr>
<tr><td><b>Qwen3.6</b></td><td>35b</td><td>v2.1.5</td><td>91/100 (91%)</td></tr>
<tr><td><b>Nemotron</b></td><td>3-super 120b (12b active)</td><td>v2.1.7</td><td>88/100 (88%)</td></tr>
<tr><td><b>Laguna XS 2.1</b></td><td>33b (3b active)</td><td>v2.1.5</td><td>24/100 (24%)</td></tr>
</table>

For a more comprehensive view of npcsh's capabilities and the advantages of the NPC Context-Agent-Tool data layer, see [ALARA for Agents: Least-Privilege Context Engineering Through Portable Composable Multi-Agent Teams](https://arxiv.org/abs/2603.20380).

---

## Installation

### Install script (recommended)

```bash
curl -fsSL https://enpisi.com/install-npcsh.sh | sh
```

This downloads the latest `npcsh` and `npc` Rust binaries for your platform into `~/.npcsh/bin`, adds that directory to your PATH in your shell rc file, and walks you through installing the `npcpy` Python backend — offering any existing virtualenvs it finds, or a fresh one via `uv`, `pyenv`, or `python3 -m venv`. Then run `npcsh`.

See [Python backend (npcpy)](#python-backend-npcpy) below for details on that requirement.

### Cargo

```bash
cargo install npcsh
```

### Python backend (npcpy)

> **Note:** this is a temporary requirement. The `npcpy` server will be replaced by a Rust-native runner for the AI parsing once [npcrs](https://github.com/npc-worldwide/npcrs) reaches greater stability.

`npcsh` drives its agent loop through a local `npcpy` server, which it spawns automatically on startup — there is nothing to run manually, but the Python interpreter `npcsh` uses must have `npcpy` importable:

```bash
pip install npcpy
```

Requires Python 3.10 or newer.

By default `npcsh` runs `python3 -m npcpy.serve` on `127.0.0.1:5237`. If `npcpy` lives in a non-default Python (a venv, conda env, or pyenv version), point `npcsh` at that interpreter:

```bash
export BACKEND_PYTHON_PATH=/path/to/python   # PYTHON_PATH also works
```

The server bind address can be changed with `NPCSH_SERVER_HOST` and `NPCSH_SERVER_PORT`.

If startup fails with `failed to spawn npcpy.serve` or `npcpy server did not become reachable after spawn`, the selected Python does not have `npcpy` installed, or the port is already taken by a stale server process.

### macOS system dependencies

```bash
brew install ollama
brew services start ollama
ollama pull qwen3.5:2b
```

### Linux system dependencies

```bash
# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b
```

### Windows

Install [Ollama](https://ollama.com), then use the install script from PowerShell via WSL, or install with cargo.

### Rust build (development)

```bash
cd rust
cargo build --release
cp target/release/npcsh ~/.npcsh/bin/npcsh
cp target/release/npc ~/.npcsh/bin/npc
```

For normal use, install the pre-built release via the install script or cargo. The source build is for development only.

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

The agent layer can be written in three formats. They can be mixed; `.npc` files take precedence if names collide.

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

## Read the Docs

Full guides and API reference at [npc-shell.readthedocs.io](https://npc-shell.readthedocs.io/en/latest/).

## Community & Support

[Discord](https://discord.gg/XrjTFmDAna) | [Monthly donation](https://buymeacoffee.com/npcworldwide) | [Merch](https://enpisi.com/shop) | Consulting: info@npcworldwi.de

## Contributing

Contributions welcome! Submit issues and pull requests on the [GitHub repository](https://github.com/npc-worldwide/npcsh).

## License

MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=npc-worldwide/npcsh&type=Date)](https://star-history.com/#npc-worldwide/npcsh&Date)
