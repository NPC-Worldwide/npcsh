## Installation

`npcsh` is distributed as pre-built Rust binaries and as a Rust crate. Pick the option that fits your workflow.

## Install script (recommended)

```bash
curl -fsSL https://enpisi.com/install-npcsh.sh | sh
```

The script downloads the latest `npcsh` and `npc` binaries for your platform into `~/.npcsh/bin`. Make sure that directory is on your PATH:

```bash
export PATH="$HOME/.npcsh/bin:$PATH"
```

Then run:

```bash
npcsh
```

## Cargo

```bash
cargo install npcsh
```

This installs the `npcsh` and `npc` binaries via crates.io.

## System dependencies

### Linux

```bash
# Ollama (optional, for local models)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b
ollama pull llava:7b
ollama pull nomic-embed-text
```

### macOS

```bash
brew install ollama
brew services start ollama
ollama pull qwen3.5:2b
ollama pull llava:7b
ollama pull nomic-embed-text
```

### Windows

Download and install [Ollama](https://ollama.com), then use the install script from PowerShell via WSL or install with cargo.

## Rust build (development / latest)

To build the Rust binaries from source:

```bash
cd npcsh/rust
cargo build --release
cp target/release/npcsh ~/.npcsh/bin/npcsh
cp target/release/npc ~/.npcsh/bin/npc
```

For normal use, install the pre-built release via the install script or cargo. The source build is for development only.

## Startup and configuration

Start the shell by typing:

```bash
npcsh
```

When initialized, `npcsh` generates a `.npcshrc` file in your home directory that stores your settings — default chat model/provider, image generation model/provider, embedding model/provider, database path, etc.

```bash
export NPCSH_CHAT_MODEL=qwen3.5:2b
export NPCSH_CHAT_PROVIDER=ollama
export NPCSH_DEFAULT_MODE=agent
export NPCSH_EMBEDDING_MODEL=nomic-embed-text
export NPCSH_EMBEDDING_PROVIDER=ollama
export NPCSH_STREAM_OUTPUT=1
```

The installer tries to source this file from your shell config automatically. If it does not (for example, you use an alternative rc file), add this to `.bashrc` or `.zshrc`:

```bash
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

`npcsh` supports inference via all major providers through LiteLLM, including but not limited to `openai`, `anthropic`, `ollama`, `gemini`, `deepseek`, and `openai-like` APIs. The `openai-like` provider is intended for custom or locally hosted servers (LM Studio, Llama CPP, etc.).

API keys can be placed in a project `.env` file, in `~/.npcshrc`, or in your existing shell config. `npcsh` always checks the current folder's `.env` first, so you can use per-project keys without manually switching them.

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GEMINI_API_KEY="your_gemini_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

Individual NPCs can override the default model/provider by setting `model` and `provider` in their `.npc` files.

## Project structure

On startup, `npcsh` will generate a folder at `~/.npcsh/` that contains the default global NPCs and jinxes if there is no `npc_team` in the current directory. It also records interactions in a local SQLite database at the path specified by `NPCSH_DB_PATH` (default `~/npcsh_history.db`).

```
~/.npcsh/
├── images/              # images created or uploaded during conversations
├── jobs/                # scheduled jobs
├── logs/                # logs for triggers and jobs
├── npc_team/            # global NPC team
│   ├── jinxes/          # global jinxes
│   └── assembly_lines/  # workflow pipelines
├── screenshots/         # taken with screenshot jinx or /ots
└── triggers/            # condition-triggered jobs
```

For project-specific teams, add an `npc_team/` directory to your project:

```
./npc_team/
├── team.ctx            # team config
├── jinxes/             # project jinxes
│   └── example.jinx
├── assembly_lines/     # project workflows
│   └── example.pipe
├── models/             # NQL SQL models
│   └── example.sql
├── example1.npc        # agent definition
└── example2.npc
```

`npcsh` automatically detects the local `npc_team/` and overlays it on the global team.
