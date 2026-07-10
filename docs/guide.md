# npcsh Guide

`npcsh` is a runtime for AI teams defined as data. This guide covers the core concepts — the **NPC Data Layer**, agents, jinxes, team orchestration, memory/knowledge graphs — and how to build your own tools on top of them.

For a complete reference of individual slash commands, see [NPC Shell Commands](npcsh.md).

## NPC Data Layer

The core of npcsh's capabilities is powered by the NPC Data Layer. Upon initialization, a user will be prompted to make a team in the current directory or to use a global team stored in `~/.npcsh/` which houses the NPC team with its jinxes, models, contexts, assembly lines. By implementing these components as simple data structures, users can focus on tweaking the relevant parts of their multi-agent systems.

### Creating Custom Components

The data layer has three levels. Each is written as plain files and loaded by `npcsh` at startup:

- **Team** (`.ctx`): context shared by the whole team — default model/provider, forenpc (orchestrator), MCP server paths, env vars, and shared memory. The team context is loaded first and used as defaults for the agents under it.
- **Agents** (`.npc`, `agents.md`, `agents/`): agent definitions — name, persona, directive, model/provider overrides, and the jinxes they can use. `.npc` is one agent per YAML file, `agents.md` is many agents in one markdown file, and `agents/` is one agent per `.md` file. NPC files are executable: add `#!/usr/bin/env npc` as the first line and run them directly: `./myagent.npc "what's the weather?"`
- **Tools** (`.jinx`, `skills/`): jinxes are Jinja execution templates that provide function-like capabilities to agents. They can call other jinxes, run shell or Python, query the local DB, or call an LLM. Skills are a special kind of jinx that expose instructional content progressively.

The NPC Shell system integrates the capabilities of `npcpy` to maintain conversation history, track command execution, and provide intelligent autocomplete through an extensible command routing system. State is preserved between sessions, allowing for continuous knowledge building over time.

This architecture enables users to build complex AI workflows while maintaining a simple, declarative syntax that abstracts away implementation complexity. By organizing AI capabilities in composable data structures rather than code, `npcsh` creates a more accessible and adaptable framework for AI automation that can scale more intentionally. Within teams can be subteams, and these sub-teams may be called upon for orchestration, but importantly, when the orchestrator is deciding between using one of its own team's NPCs versus yielding to a sub-team, they see only the descriptions of the subteams rather than the full persona descriptions for each of the sub-team's agents, making it easier for the orchestrator to better delineate and keep their attention focused by restricting the number of options in each decisino step. Thus, they may yield to the sub-team's orchestrator, letting them decide which sub-team NPC to use based on their own team's agents.

Importantly, users can switch easily between the NPCs they are chatting with by typing `/<npc_name>` or `@<npc_name>` within the NPC shell. Likewise, NPCs use jinxes as tools when they need to take action.

## Team Orchestration

NPCs work together through orchestration patterns. The **forenpc** (specified in your team's `.ctx` file) acts as the coordinator, delegating tasks to specialized NPCs and convening group discussions.

### How NPCs and Jinxes Relate

Each NPC has a set of **jinxes** they can use, defined in their `.npc` file:

```yaml
# corca.npc
name: corca
primary_directive: "You are a coding specialist..."
model: claude-sonnet-4-20250514
provider: anthropic
jinxes:
  - lib/core/python
  - lib/core/sh
  - lib/core/edit_file
  - lib/core/load_file
```

When an NPC is invoked, they can only use the jinxes assigned to them. This creates **specialization**:
- `corca` has coding tools (python, sh, edit_file, load_file)
- `plonk` has browser automation (browser_action, screenshot, click)
- `alicanto` has research tools (python, sh, load_file)
- `frederic` has generation tools (vixynt, roll, sample)

<p align="center">
    <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/gh_images/team_npc.png" alt="NPC team browser", width=700>
</p>

The forenpc (orchestrator) can delegate to any team member based on their specialization.

### Skills — Knowledge Content for Agents

Skills are jinxes that serve instructional content instead of executing code. They use the `skill.jinx` sub-jinx (just like code jinxes use `python.jinx` or `sh.jinx`) and return sections of methodology on demand.

Because skills are jinxes, they're assigned to agents the same way — through the `jinxes:` list in `.npc` files:

```yaml
# reviewer.npc
name: reviewer
primary_directive: "You review code and provide feedback."
jinxes:
  - lib/core/sh
  - lib/core/python
  - skills/code-review
  - skills/debugging
```

The agent sees `code-review` and `debugging` in its tool catalog alongside `sh` and `python`. When it encounters a review task, it calls the skill to get methodology, then uses `sh` or `python` to do the actual work.

#### Two Authoring Formats

**SKILL.md folder** — a folder with a `SKILL.md` file (folder name = skill name):

```
jinxes/skills/debugging/
  SKILL.md             # YAML frontmatter + ## sections
  scripts/             # Optional
  references/          # Optional
```

```markdown
---
description: Debugging methodology. Use when asked to debug or troubleshoot.
---
# Debugging

## reproduce
First, reproduce the bug consistently.
Find the minimal reproduction case.

## isolate
Binary search through the codebase (git bisect).
Comment out components to isolate the cause.

## fix
Fix the root cause, not the symptom.
Add a test that fails without the fix.
```

**`.jinx` file** — a regular jinx with `engine: skill` steps:

```yaml
jinx_name: git-workflow
description: "Git workflow best practices. [Sections: branching, commits, merging]"
inputs:
- section: all
steps:
  - engine: skill
    skill_name: git-workflow
    skill_description: Git workflow best practices.
    sections:
      branching: |
        Use feature branches off main/develop.
        Name branches: feature/, fix/, chore/
      commits: |
        Imperative summary under 72 chars.
        One logical change per commit.
      merging: |
        Prefer squash merges for feature branches.
        Delete branches after merging.
    scripts_json: '[]'
    references_json: '[]'
    assets_json: '[]'
    section: '{{section}}'
```

#### Using Skills

In npcsh, skills are jinxes that provide instructional content. Agents invoke them by name when relevant and can request specific sections to minimize token usage (progressive disclosure). For example, an agent might call the `debugging` skill and ask for only the `reproduce` section.

#### Importing External Skills

Add `SKILLS_DIRECTORY` to your `.ctx` file to load skills from an external directory:

```yaml
model: qwen3.5:2b
provider: ollama
forenpc: lead-dev
SKILLS_DIRECTORY: ~/shared-skills
```

All `SKILL.md` folders and `.jinx` skill files in that directory are loaded alongside the team's own jinxes. This lets you maintain a single skills library shared across multiple teams.

### Delegation with Review Loop

The orchestrator can delegate a task to another NPC using the `delegate` jinx, which runs a review/feedback loop until the task is complete:

1. The orchestrator sends the task to the target NPC (e.g., `corca`).
2. The target NPC works on the task using its available jinxes.
3. The orchestrator reviews the output and decides: COMPLETE or needs more work.
4. If incomplete, the orchestrator provides feedback and the target NPC iterates.
5. This continues until complete or max iterations reached.

### Deep Research

The `deep_research` jinx runs multi-agent research — generates hypotheses, assigns persona-based sub-agents, runs iterative tool-calling loops, and synthesizes findings.

<p align="center">
    <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/gh_images/alicanto.png" alt="Alicanto deep research mode", width=500>
    <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/gh_images/alicanto_2.png" alt="Alicanto execution phase", width=500>
</p>

### Convening Multi-NPC Discussions

The `convene` jinx brings multiple NPCs together for a structured discussion. Each NPC contributes based on their persona, responds to each other, and the orchestrator synthesizes the result.

### Visualizing Team Structure

The `teamviz` jinx generates a view of how your NPCs and jinxes are connected:
- **Network view**: Organic layout showing NPC-jinx relationships.
- **Ordered view**: NPCs on left, jinxes grouped by category on right.

## Working with NPCs (Agents)

NPCs are AI agents with distinct personas, models, and tool sets. You can interact with them in two ways:

### Switching to an NPC

Use `/<npc_name>` or `@<npc_name>` to switch your session to a different NPC. All subsequent messages will be handled by that NPC until you switch again:

```bash
/corca              # Switch to corca for coding tasks
@frederic           # Switch to frederic for math/music
```

### One-Time Questions with @

Use `@<npc_name>` to ask a specific NPC a one-time question without switching your session:

```bash
@corca can you review this function for bugs?
@frederic what's the derivative of x^3 * sin(x)?
@alicanto search for recent papers on transformer architectures
```

The NPC responds using their persona and available jinxes, then control returns to your current NPC.

### Available NPCs

| NPC | Specialty | Key Jinxes |
|-----|-----------|-----------|
| `sibiji` | Orchestrator/coordinator | delegate, convene, python, sh |
| `corca` | Coding and development | python, sh, edit_file, load_file |
| `plonk` | Browser/GUI automation | browser_action, screenshot, click, key_press |
| `alicanto` | Research and analysis | python, sh, load_file |
| `frederic` | Math, physics, music | python, vixynt, roll, sample |
| `guac` | General assistant | python, sh, edit_file, load_file |
| `kadiefa` | Creative generation | vixynt |

## Capability Areas

Rather than memorizing a long command list, think of `npcsh` as a set of capability areas that you extend through the data layer. Each area is exposed through slash commands and TUIs; see [NPC Shell Commands](npcsh.md) for the full command reference.

| Area | What it does |
|------|--------------|
| **Agent chat** | Talk to NPCs, switch between them, ask one-off questions. |
| **Custom tools** | Author `.jinx` files and skills that agents use as tools. |
| **Orchestration** | Delegate tasks with review loops, convene multi-NPC discussions, visualize team structure. |
| **Memory** | Extract and review memories. |
| **Computer use** | GUI automation, browser automation, screenshot analysis. |
| **Search** | Web, database, and file search across the local data layer. |
| **Media** | Image generation via the `gen_image` jinx. |
| **Scheduling** | Cron jobs and loops. |
| **System / config** | Config editor, model browser, team browser, setup. |
| **Git** | Git TUI and commit helper. |

Most built-in commands launch full-screen TUIs. For CLI usage with `npc`, common flags include `--model (-mo)`, `--provider (-pr)`, `--npc (-np)`, and `--temperature (-t)`. Run `npc --help` for the full list.

### `/chat`, `/agent`, `/cmd` — Modes

- `/chat` — Chat-only mode (LLM, no tools).
- `/agent` — Full agent mode (tools + bash + LLM).
- `/cmd` — Bash first; if the command fails or is not understood, fall back to the LLM.

### `/<npc>` and `@<npc>` — Switching NPCs

Use `/<npc_name>` or `@<npc_name>` to switch your session to a different NPC. All subsequent messages go to that NPC until you switch again. Use `@<npc_name> <question>` to ask a one-off question without switching.

### `/config` — Configuration Editor

Interactive TUI for editing `~/.npcshrc` settings — models, providers, modes, and toggles. Navigate with j/k, edit text fields, toggle booleans, and cycle choices.

<p align="center">
    <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/gh_images/config.png" alt="Config editor TUI" width=500>
</p>

### `/memories` — Memory Browser

Open an interactive TUI for browsing, approving, and managing memories extracted from conversations.

### `/cron` — Scheduling

Manage cron jobs and loops from the shell. Use `/loop <npc> <interval> <task>` to create a recurring task, `/loops` to list them, and `/loopon`, `/loopoff`, `/looprm` to enable, disable, or remove them.

### `/gitt` — Git TUI

Open the Git TUI to view status, stage changes, commit, and browse history.

### `/commit` — Commit Helper

Launch the commit-helper TUI to write a conventional commit message from staged changes.

## Memory

`npcsh` maintains a memory lifecycle system that allows agents to learn from past interactions. Memories are extracted by the `memory_extractor` jinx and follow these stages:

1. **pending_approval** - New memories awaiting review
2. **human-approved** - Approved and ready for use
3. **human-rejected** - Rejected (used as negative examples)
4. **human-edited** - Modified by user before approval
5. **skipped** - Deferred for later review

### Memories

The `/memories` command opens an interactive TUI for browsing, reviewing, and managing memories:

```bash
/memories
```

The TUI provides:
- **Tab-based filtering** — switch between All, Pending, Approved, Rejected, etc.
- **Approve/Reject** — press `a` to approve, `x` to reject
- **Preview** — press Enter to see full memory content
- **Session stats** — tracks approvals/rejections during session

<p align="center">
    <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/gh_images/Screenshot%20from%202026-01-29%2016-03-08.png" alt="Memory Browser TUI", width=700>
</p>

## Installation

`npcsh` is distributed as pre-built Rust binaries, via crates.io, or from source.

### Install script (recommended)

```bash
curl -fsSL https://enpisi.com/install-npcsh.sh | sh
```

This downloads the latest `npcsh` and `npc` binaries for your platform into `~/.npcsh/bin`. Add that directory to your PATH:

```bash
export PATH="$HOME/.npcsh/bin:$PATH"
```

### Cargo

```bash
cargo install npcsh
```

### System dependencies

### Linux
```bash
# Ollama (for local models)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b
```

### macOS
```bash
brew install ollama
brew services start ollama
ollama pull qwen3.5:2b
```

### Windows
Download and install [Ollama](https://ollama.com), then use the install script from PowerShell via WSL or install with cargo.

### Rust build (development / latest)

```bash
cd rust
cargo build --release
cp target/release/npcsh ~/.npcsh/bin/npcsh
cp target/release/npc ~/.npcsh/bin/npc
```

## Configuration

When initialized, `npcsh` generates a `.npcshrc` file in your home directory:

```bash
export NPCSH_CHAT_MODEL=qwen3.5:2b
export NPCSH_CHAT_PROVIDER=ollama
export NPCSH_DEFAULT_MODE=agent
export NPCSH_EMBEDDING_MODEL=nomic-embed-text
export NPCSH_EMBEDDING_PROVIDER=ollama
export NPCSH_STREAM_OUTPUT=1
```

API keys go in a `.env` file or your shell config:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GEMINI_API_KEY="your_gemini_key"
```

Individual NPCs can use different models and providers by setting `model` and `provider` in their `.npc` files.

## Project Structure

A project is team + agents + tools. The three layers can live at the project root or inside an `npc_team/` directory. The agent layer can be either `.npc` files, a single `agents.md`, or an `agents/` directory — those are alternatives, not requirements to use all at once.

If both `npc_team/*.npc` and `agents.md`/`agents/` are present, npcsh asks which agent layout to use on first run and saves the choice in `.NPCSH_PREFERRED_TEAM_NAME`. Later runs use the preferred layout automatically.

**Layout A: everything under `npc_team/`**

```
myproject/
└── npc_team/
    ├── npcsh.ctx           # team-level context
    ├── sibiji.npc          # orchestrator
    ├── corca.npc           # coding specialist
    └── jinxes/             # tools
        ├── skills/
        │   └── debugging/
        │       └── SKILL.md
        ├── lib/
        │   └── core/
        │       ├── python.jinx
        │       └── sh.jinx
        └── my_tool.jinx
```

**Layout B: flat at project root**

```
myproject/
├── team.ctx                # team-level context
├── agents.md               # many agents in one file
└── jinxes/
    └── my_tool.jinx
```

**Layout C: flat agents directory**

```
myproject/
├── team.ctx                # team-level context
├── agents/                 # one agent per .md file
│   ├── translator.md
│   └── custom.md
└── jinxes/
    └── my_tool.jinx
```

## Environment Variables

```bash
export NPCSH_DB_PATH=~/npcsh_history.db  # Database path
```
