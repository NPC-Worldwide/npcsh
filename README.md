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

`npcsh` makes the most of LLMs and agents through an interactive shell and one-off CLI. Build teams of agents, schedule them on jobs, engineer context, and design custom Jinja Execution templates (Jinxes) for your agents to invoke.

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

The benchmark suite measures how well a model can drive `npcsh` as an agentic shell. It covers 135 tasks across 15 categories, from basic shell commands and file operations to multi-step workflows, debugging, git, tool chaining, delegation, web search, and media generation. Each task is scored pass/fail by an automated verifier.

The table below shows scores for the new Rust runtime (100 tasks) and historical scores from the original Python runtime (125 tasks). For per-category breakdowns, see `docs/benchmarks.md`.

### Rust runtime scores (100 tasks)

Local Ollama runs executed with the Rust `npcsh` binary at `v2.1.2-33-g6b056cf`.

<table>
<tr><th>Family</th><th>Model</th><th>Version</th><th>Score</th></tr>
<tr><td rowspan="2"><b>Qwen3.5</b></td><td>35b</td><td>v2.1.2-33-g6b056cf</td><td><b>97/100 (97%)</b></td></tr>
<tr><td>9b</td><td>v2.1.2-33-g6b056cf</td><td>in progress</td></tr>
<tr><td><b>North</b></td><td>mini-code-1.0 latest</td><td>v2.1.2-33-g6b056cf</td><td>93/100 (93%)</td></tr>
<tr><td rowspan="2"><b>Ornith</b></td><td>9b</td><td>v2.1.2-33-g6b056cf</td><td>57/100 (57%)</td></tr>
<tr><td>35b</td><td>v2.1.2-33-g6b056cf</td><td>38/100 (38%)</td></tr>
<tr><td><b>Laguna</b></td><td>xs-2.1 latest</td><td>v2.1.2-33-g6b056cf</td><td>24/100 (24%)</td></tr>
</table>

### Historical Python runtime scores (125 tasks)

<table>
<tr><th>Family</th><th>Model</th><th>Score</th></tr>
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
<tr><td>v4-pro</td><td>—</td></tr>
<tr><td>reasoner</td><td>—</td></tr>
</table>

Run the benchmark yourself against a local or remote model:

```bash
python -m npcsh.benchmark.rust_runner --model qwen3.5:9b --provider ollama
python -m npcsh.benchmark.rust_runner --model deepseek-v4-pro --provider deepseek
```

For a more comprehensive view of npcsh's capabilities and the advantages of the NPC Context-Agent-Tool data layer, see [ALARA for Agents: Least-Privilege Context Engineering Through Portable Composable Multi-Agent Teams](https://arxiv.org/abs/2603.20380).

---

## Installation

### Install script (recommended)

```bash
curl -fsSL https://enpisi.com/install-npcsh.sh | sh
```

This downloads the latest `npcsh` and `npc` Rust binaries for your platform into `~/.npcsh/bin`. Add that directory to your PATH, then run `npcsh`.

### Cargo

```bash
cargo install npcsh
```

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
