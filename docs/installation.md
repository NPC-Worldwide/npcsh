:## Installation

`npcsh` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

## PyPI install (recommended)

```bash
pip install 'npcsh[lite]'        # API + Ollama providers
pip install 'npcsh[local]'       # + local diffusers/transformers/torch
pip install 'npcsh[yap]'         # + voice mode TTS/STT
pip install 'npcsh[all]'         # everything
```

What you get:

- The `npcsh` Python launcher, which starts the NPCSH server and execs the Rust shell.
- The `npc` and `npcsh-bench` CLI entry points.
- A default global team in `~/.npcsh/npc_team/`.

## System dependencies

### Linux

```bash
# Audio / TTS / STT
sudo apt-get install espeak portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils libcairo2-dev libgirepository1.0-dev ffmpeg

# File-system triggers
sudo apt install inotify-tools

# Ollama (optional, for local models)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b
ollama pull llava:7b
ollama pull nomic-embed-text
```

### macOS

```bash
brew install portaudio ffmpeg pygobject3 ollama
brew services start ollama
ollama pull qwen3.5:2b
ollama pull llava:7b
ollama pull nomic-embed-text
```

### Windows

Download and install [Ollama](https://ollama.com) and [ffmpeg](https://ffmpeg.org), then in PowerShell:

```powershell
ollama pull qwen3.5:2b
ollama pull llava:7b
ollama pull nomic-embed-text
pip install 'npcsh[lite]'
```

### Fedora notes

- `python3-dev` — fixes hnswlib issues with ChromaDB
- `xhost +` — required for pyautogui
- `python-tkinter` — required for pyautogui

## Rust edition (development / latest)

`npcsh` ships as a Python launcher that starts the NPCSH server and then execs the Rust shell binary (`npcrsh`). By default the launcher looks for `npcrsh` at `~/.npcsh/bin/npcrsh`, then falls back to PATH.

To build the Rust binary from source:

```bash
cd npcsh/rust
cargo build --release
cp target/release/npcrsh ~/.npcsh/bin/npcrsh

# macOS only: the copied binary must be ad-hoc re-signed or Gatekeeper will kill it
codesign -s - -f ~/.npcsh/bin/npcrsh
```

Both editions share `~/npcsh_history.db` and `~/.npcsh/npc_team/` and can be used interchangeably.

## Startup and configuration

After it has been pip installed, start the shell by typing:

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
