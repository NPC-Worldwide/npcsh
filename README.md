<p align="center">
  <a href= "https://github.com/npc-worldwide/npcsh/blob/main/docs/npcsh.md"> 
  <img src="https://raw.githubusercontent.com/NPC-Worldwide/npcsh/main/npcsh/npcsh.png" alt="npcsh logo" width=600></a>
</p> 

# npcsh

The NPC shell (`npcsh`) makes the most of multi-modal LLMs and agents through a powerful set of simple slash commands and novel interactive modes, all from the comfort of the command line. Build teams of agents and schedule them on jobs, engineer context, and design custom interaction modes and Jinja Execution templates (Jinxs) for you and your agents to invoke, all managed scalably for organizations of any size through the NPC data layer.

To get started:
```bash
# API providers (ollama, gemini, kimi, grok, deepseek, anthropic, openai, mistral, etc.)
pip install 'npcsh[lite]'

# Local models (diffusers/transformers/torch — large install)
pip install 'npcsh[local]'

# Voice mode (see OS-specific audio library instructions below)
pip install 'npcsh[yap]'

# Everything
pip install 'npcsh[all]'
```

Once installed, run `npcsh` to enter the NPC shell. The pip installation also provides the CLI tools `npc`, `wander`, `spool`, `yap`, and `nql` in your shell. Bin jinxs in `npc_team/jinxs/bin/` are automatically registered as additional CLI commands.


# Usage
  - Get help with a task:
      ```bash
      npcsh>can you help me identify what process is listening on port 5337?
      ```

  - Edit files
      ```bash
      npcsh>please read through the markdown files in the docs folder and suggest changes based on the current implementation in the src folder
      ```


  - **Search & Knowledge**
    ```bash
    /web_search "cerulean city"            # Web search
    /db_search "query"                     # Database search
    /file_search "pattern"                 # File search
    /memories                              # Interactive memory browser TUI
    /kg                                    # Interactive knowledge graph TUI
    /kg sleep                              # Evolve the knowledge graph
    /kg dream                              # Creative synthesis across domains
    /nql                                   # Database query TUI
    ```
    <p align="center">
        <img src="gh_images/Screenshot%20from%202026-01-29%2015-02-52.png" alt="Web search results", width=600>
    </p>
    <p align="center">
        <img src="gh_images/kg_facts_viewer.png" alt="Knowledge Graph TUI", width=500>
    </p>
    <p align="center">
        <img src="gh_images/nql_menu.png" alt="NQL data browser", width=500>
        <img src="gh_images/nql_table.png" alt="NQL table viewer", width=500>
    </p>

  - **Computer Use**

    ```bash
    /computer_use
    ```
    <p align="center">
        <img src="gh_images/plonk.png" alt="Plonk GUI automation TUI — task entry" width=500>
        <img src="gh_images/plonk_task.png" alt="Plonk GUI automation — completed flight search task" width=500>
    </p>

  - **Generate Images with Vixynt**

    ```bash
    /vixynt
    ```
    <p align="center">
        <img src="gh_images/vixynt.png" alt="Vixynt Image Creation Studio", width=500>
    </p>

  - Generate images directly
    ```bash
    /vixynt 'generate an image of a rabbit eating ham in the brink of dawn' model='gpt-image-1' provider='openai'
    ```
      <p align="center">
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/rabbit.PNG" alt="a rabbit eating ham in the bring of dawn", width=350>
      </p>
  - **Generate Videos with Roll**

    ```bash
    /roll
    ```
    <p align="center">
        <img src="gh_images/roll.png" alt="Roll Video Creation Studio" width=500>
    </p>

  - Generate videos directly

    ```bash
    /roll 'generate a video of a hat riding a dog' veo-3.1-fast-generate-preview  gemini
    ```

      <p align="center">
        <img src="https://raw.githubusercontent.com/NPC-Worldwide/npcsh/main/test_data/hatridingdog.gif" alt="video of a hat riding a dog", width=350>
      </p> 

  - **Multi-Agent Discussions**

    ```bash
    /convene "Is the universe a simulation?" npcs=alicanto,corca,guac rounds=3
    ```
    <p align="center">
        <img src="gh_images/convene.png" alt="Convene — multi-NPC structured discussion" width=500>
        <img src="gh_images/convene_convene.png" alt="Convene — live discussion view" width=500>
    </p>

  - **Serve an NPC Team**
    ```bash
    /serve --port 5337 --cors='http://localhost:5137/'
    ```
    This exposes your NPC team as a full agentic server with:
    - **OpenAI-compatible endpoints** for drop-in LLM replacement
      - `POST /v1/chat/completions` - Chat with NPCs (use `agent` param to select NPC)
      - `GET /v1/models` - List available NPCs as models
    - **NPC management**
      - `GET /npcs` - List team NPCs with their capabilities
      - `POST /chat` - Direct chat with NPC selection
    - **Jinx controls** - Execute jinxs remotely via API
    - **Team orchestration** - Delegate tasks and convene discussions programmatically
  - **Screenshot Analysis**: select an area on your screen and then send your question to the LLM
    ```bash
    /ots
    ```
  - **MCP-powered agentic shell**: full tabbed TUI with chat, tool management, and server controls.
    ```bash
    /mcp_shell
    /mcp_shell mcp_server_path=/path/to/server.py
    ```

  - **Build an NPC Team**: Generate deployment artifacts for your team.
    ```bash
    /build target=flask outdir=./dist port=5337
    /build target=docker outdir=./deploy
    /build target=cli outdir=./bin
    /build target=static outdir=./site
    ```

# NPC Data Layer

The core of npcsh's capabilities is powered by the NPC Data Layer. Upon initialization, a user will be prompted to make a team in the current directory or to use a global team stored in `~/.npcsh/` which houses the NPC team with its jinxs, models, contexts, assembly lines. By implementing these components as simple data structures, users can focus on tweaking the relevant parts of their multi-agent systems.

## Creating Custom Components

Users can extend NPC capabilities through simple YAML files:

- **NPCs** (.npc): are defined with a name, primary directive, and optional model specifications
- **Jinxs** (.jinx): Jinja execution templates that provide function-like capabilities and scaleable extensibility through Jinja references to call other jinxs to build upon. Jinxs are executed through prompt-based flows, allowing them to be used by models regardless of their tool-calling capabilities, making it possible then to enable agents at the edge of computing through this simple methodology.
- **Context** (.ctx): Specify contextual information, team preferences, MCP server paths, database connections, and other environment variables that are loaded for the team or for specific agents (e.g. `GUAC_FORENPC`). Teams are specified by their path and the team name in the `<team>.ctx` file. Teams organize collections of NPCs with shared context and specify a coordinator within the team context who is used whenever the team is called upon for orchestration.
- **SQL Models** (.sql): NQL (NPC Query Language) models combine SQL with AI-powered transformations. Place `.sql` files in `npc_team/models/` to create data pipelines with embedded LLM calls.

The NPC Shell system integrates the capabilities of `npcpy` to maintain conversation history, track command execution, and provide intelligent autocomplete through an extensible command routing system. State is preserved between sessions, allowing for continuous knowledge building over time.

This architecture enables users to build complex AI workflows while maintaining a simple, declarative syntax that abstracts away implementation complexity. By organizing AI capabilities in composable data structures rather than code, `npcsh` creates a more accessible and adaptable framework for AI automation that can scale more intentionally. Within teams can be subteams, and these sub-teams may be called upon for orchestration, but importantly, when the orchestrator is deciding between using one of its own team's NPCs versus yielding to a sub-team, they see only the descriptions of the subteams rather than the full persona descriptions for each of the sub-team's agents, making it easier for the orchestrator to better delineate and keep their attention focused by restricting the number of options in each decisino step. Thus, they may yield to the sub-team's orchestrator, letting them decide which sub-team NPC to use based on their own team's agents.

Importantly, users can switch easily between the NPCs they are chatting with by typing `/n npc_name` within the NPC shell. Likewise, they can create Jinxs and then use them from within the NPC shell by invoking the jinx name and the arguments required for the Jinx;  `/<jinx_name> arg1 arg2`

# Team Orchestration

NPCs work together through orchestration patterns. The **forenpc** (specified in your team's `.ctx` file) acts as the coordinator, delegating tasks to specialized NPCs and convening group discussions.

## How NPCs and Jinxs Relate

Each NPC has a set of **jinxs** they can use, defined in their `.npc` file:

```yaml
# corca.npc
name: corca
primary_directive: "You are a coding specialist..."
model: claude-sonnet-4-20250514
provider: anthropic
jinxs:
  - lib/core/python
  - lib/core/sh
  - lib/core/edit_file
  - lib/core/load_file
```

When an NPC is invoked, they can only use the jinxs assigned to them. This creates **specialization**:
- `corca` has coding tools (python, sh, edit_file, load_file)
- `plonk` has browser automation (browser_action, screenshot, click)
- `alicanto` has research tools (python, sh, load_file)
- `frederic` has generation tools (vixynt, roll, sample)

<p align="center">
    <img src="gh_images/team_npc.png" alt="NPC team browser", width=700>
</p>

The forenpc (orchestrator) can delegate to any team member based on their specialization.

## Skills — Knowledge Content for Agents

Skills are jinxs that serve instructional content instead of executing code. They use the `skill.jinx` sub-jinx (just like code jinxs use `python.jinx` or `sh.jinx`) and return sections of methodology on demand.

Because skills are jinxs, they're assigned to agents the same way — through the `jinxs:` list in `.npc` files:

```yaml
# reviewer.npc
name: reviewer
primary_directive: "You review code and provide feedback."
jinxs:
  - lib/core/sh
  - lib/core/python
  - skills/code-review
  - skills/debugging
```

The agent sees `code-review` and `debugging` in its tool catalog alongside `sh` and `python`. When it encounters a review task, it calls the skill to get methodology, then uses `sh` or `python` to do the actual work.

### Two Authoring Formats

**SKILL.md folder** — a folder with a `SKILL.md` file (folder name = skill name):

```
jinxs/skills/debugging/
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

### Using Skills

In npcsh, skills work as slash commands like any jinx:

```bash
/debugging                       # All sections
/debugging -s reproduce          # Just the reproduce section
/debugging -s list               # Available section names
/code-review -s correctness      # Just the correctness section
```

In the agent loop, the agent calls skills automatically when relevant — requesting specific sections to minimize token usage (progressive disclosure).

### Importing External Skills

Add `SKILLS_DIRECTORY` to your `.ctx` file to load skills from an external directory:

```yaml
model: llama3.2
provider: ollama
forenpc: lead-dev
SKILLS_DIRECTORY: ~/shared-skills
```

All `SKILL.md` folders and `.jinx` skill files in that directory are loaded alongside the team's own jinxs. This lets you maintain a single skills library shared across multiple teams.

## Delegation with Review Loop

The `/delegate` jinx sends a task to another NPC with automatic review and feedback:

```bash
/delegate npc_name=corca task="Write a Python function to parse JSON files" max_iterations=5
```

**How it works:**
1. The orchestrator sends the task to the target NPC (e.g., `corca`)
2. The target NPC works on the task using their available jinxs
3. The orchestrator **reviews** the output and decides: COMPLETE or needs more work
4. If incomplete, the orchestrator provides feedback and the target NPC iterates
5. This continues until complete or max iterations reached

```
┌─────────────────┐     task      ┌─────────────────┐
│   Orchestrator  │ ────────────▶ │   Target NPC    │
│    (sibiji)     │               │    (corca)      │
│                 │ ◀──────────── │                 │
│   Reviews work  │    output     │  Uses jinxs:    │
│   Gives feedback│               │  - python       │
└─────────────────┘               │  - sh           │
        │                         │  - edit_file    │
        │ feedback                └─────────────────┘
        ▼
   Iterate until
   task complete
```

## Deep Research

The `/deep_research` mode runs multi-agent deep research — generates hypotheses, assigns persona-based sub-agents, runs iterative tool-calling loops, and synthesizes findings.

<p align="center">
    <img src="gh_images/alicanto.png" alt="Alicanto deep research mode", width=500>
    <img src="gh_images/alicanto_2.png" alt="Alicanto execution phase", width=500>
</p>

## Convening Multi-NPC Discussions

The `/convene` jinx brings multiple NPCs together for a structured discussion:

```bash
/convene "How should we architect the new API?" --npcs corca,guac,frederic --rounds 3
```

**How it works:**
1. Each NPC contributes their perspective based on their persona
2. NPCs respond to each other, building on or challenging ideas
3. Random follow-ups create organic discussion flow
4. After all rounds, the orchestrator synthesizes key points

```
Round 1:
  [corca]: "From a code structure perspective..."
    [guac responds]: "I agree, but we should also consider..."
    [frederic]: "The mathematical elegance here suggests..."

Round 2:
  [guac]: "Building on what corca said..."
    [corca responds]: "That's a good point about..."

SYNTHESIS:
  - Key agreements: ...
  - Areas of disagreement: ...
  - Recommended next steps: ...
```

## Visualizing Team Structure

Use `/teamviz` to see how your NPCs and jinxs are connected:

```bash
/teamviz save=team_structure.png
```

This generates two views:
- **Network view**: Organic layout showing NPC-jinx relationships
- **Ordered view**: NPCs on left, jinxs grouped by category on right

Shared jinxs (like `python` used by 7 NPCs) appear with thicker connection bundles, helping you identify common capabilities and potential consolidation opportunities.

<p align="center">
    <img src="gh_images/teamviz.png" alt="Team structure visualization", width=700>
</p>

# NQL - SQL Models with AI Functions

NQL (NPC Query Language) enables AI-powered data transformations directly in SQL, similar to dbt but with embedded LLM calls. Create `.sql` files in `npc_team/models/` that combine standard SQL with `nql.*` AI function calls, then run them on a schedule to build analytical tables enriched with AI insights.

## How It Works

NQL models are SQL files with embedded AI function calls. When executed:

1. **Model Discovery**: The compiler finds all `.sql` files in your `models/` directory
2. **Dependency Resolution**: Models referencing other models via `{{ ref('model_name') }}` are sorted topologically
3. **Jinja Processing**: Template expressions (`{% %}`) are evaluated with access to NPC/team/jinx context
4. **Execution Path**:
   - **Native AI databases** (Snowflake, Databricks, BigQuery): NQL calls are translated to native AI functions (e.g., `SNOWFLAKE.CORTEX.COMPLETE()`)
   - **Standard databases** (SQLite, PostgreSQL, etc.): SQL executes first, then Python-based AI functions process each row
5. **Materialization**: Results are written back to the database as tables or views

## Example Model

```sql
{{ config(materialized='table') }}

SELECT
    command,
    count(*) as exec_count,
    nql.synthesize(
        'Analyze "{command}" usage pattern with {exec_count} executions',
        'sibiji',
        'pattern_insight'
    ) as insight
FROM command_history
GROUP BY command
```

The `nql.synthesize()` call:
- Takes a prompt template with `{column}` placeholders filled from each row
- Uses the specified NPC (`sibiji`) for context and model/provider settings
- Returns the AI response as a new column (`insight`)

## Enterprise Database Support

NQL **automatically translates** your `nql.*` function calls to native database AI functions under the hood. You write portable NQL syntax once, and the compiler handles the translation:

| Database | Auto-Translation | Your Code → Native SQL |
|----------|------------------|------------------------|
| **Snowflake** | Cortex AI | `nql.synthesize(...)` → `SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b', ...)` |
| **Databricks** | ML Serving | `nql.generate_text(...)` → `ai_query('databricks-meta-llama...', ...)` |
| **BigQuery** | Vertex AI | `nql.summarize(...)` → `ML.GENERATE_TEXT(MODEL 'gemini-pro', ...)` |
| **SQLite/PostgreSQL** | Python Fallback | SQL executes first, then AI applied row-by-row via `npcpy` |

Write models locally with SQLite, deploy to Snowflake/Databricks/BigQuery with zero code changes—the NQL compiler rewrites your AI calls to use native accelerated functions automatically.

## NQL Functions

**Built-in LLM functions** (from `npcpy.llm_funcs`):
- `nql.synthesize(prompt, npc, alias)` - Synthesize insights from multiple perspectives
- `nql.summarize(text, npc, alias)` - Summarize text content
- `nql.criticize(text, npc, alias)` - Provide critical analysis
- `nql.extract_entities(text, npc, alias)` - Extract named entities
- `nql.generate_text(prompt, npc, alias)` - General text generation
- `nql.translate(text, npc, alias)` - Translate between languages

**Team jinxs as functions**: Any jinx in your team can be called as `nql.<jinx_name>(...)`:
```sql
nql.sample('Generate variations of: {text}', 'frederic', 'variations')
```

**Model references**: Use `{{ ref('other_model') }}` to create dependencies between models. The compiler ensures models run in the correct order.

## Jinja Templating

NQL models support Jinja expressions (using `{% %}` delimiters) for dynamic access to NPC and team properties:

```sql
-- Use the team's forenpc dynamically
nql.synthesize('Analyze this...', '{% team.forenpc %}', 'result')

-- Access NPC properties
-- Model: {% npc('sibiji').model %}
-- Provider: {% npc('corca').provider %}
-- Directive: {% npc('frederic').directive %}

-- Access jinx metadata
-- Description: {% jinx('sample').description %}

-- Environment variables with defaults
-- API URL: {% env('NPCSH_API_URL', 'http://localhost:5337') %}
```

## Running Models

```bash
# List available models (shows [NQL] tag for models with AI functions)
nql show=1

# Run all models in dependency order
nql

# Run a specific model
nql model=daily_summary

# Use a different database
nql db=~/analytics.db

# Specify output schema
nql schema=analytics

# Schedule with cron (runs daily at 6am)
nql install_cron="0 6 * * *"
```

## Example: Analytics Pipeline

```
models/
├── base/
│   ├── command_stats.sql      # Pure SQL aggregations
│   └── daily_activity.sql     # Time-series breakdown
└── insights/
    ├── command_patterns.sql   # Uses nql.synthesize() on base stats
    └── weekly_summary.sql     # References command_patterns via {{ ref() }}
```

Run `nql` to execute the entire pipeline—base models first, then insights that depend on them.

# Working with NPCs (Agents)

NPCs are AI agents with distinct personas, models, and tool sets. You can interact with them in two ways:

## Switching to an NPC

Use `/npc <name>` or `/n <name>` to switch your session to a different NPC. All subsequent messages will be handled by that NPC until you switch again:

```bash
/npc corca          # Switch to corca for coding tasks
/n frederic         # Switch to frederic for math/music
```

You can also invoke an NPC directly as a slash command to switch to them:
```bash
/corca              # Same as /npc corca
/guac               # Same as /npc guac
```

## One-Time Questions with @

Use `@<npc_name>` to ask a specific NPC a one-time question without switching your session:

```bash
@corca can you review this function for bugs?
@frederic what's the derivative of x^3 * sin(x)?
@alicanto search for recent papers on transformer architectures
```

The NPC responds using their persona and available jinxs, then control returns to your current NPC.

## Available NPCs

| NPC | Specialty | Key Jinxs |
|-----|-----------|-----------|
| `sibiji` | Orchestrator/coordinator | delegate, convene, python, sh |
| `corca` | Coding and development | python, sh, edit_file, load_file |
| `plonk` | Browser/GUI automation | browser_action, screenshot, click, key_press |
| `alicanto` | Research and analysis | python, sh, sql, load_file |
| `frederic` | Math, physics, music | python, vixynt, roll, sample |
| `guac` | General assistant | python, sh, edit_file, load_file |
| `kadiefa` | Creative generation | vixynt |

<p align="center">
    <img src="gh_images/npc_menu.png" alt="NPC menu", width=700>
</p>

# Jinxs (Macros/Tools)

Jinxs are reusable tools that users and agents can invoke. They're activated with `/<jinx_name> ...` in npcsh or via the `npc` CLI in bash. For converting any `/<command>` in npcsh to bash, replace `/` with `npc `:

```bash
# In npcsh:
/sample "tell me a story about a sunset over the mountains"

# In bash:
npc sample "a sunset over mountains"
```

## All Commands

| Command | Description |
|---------|-------------|
| `/deep_research` | Multi-agent deep research — hypotheses, persona sub-agents, paper writing |
| `/mcp_shell` | MCP-powered agentic shell — chat, tool management, server controls |
| `/convene` | Multi-NPC structured discussion with live trains of thought |
| `/spool` | Chat session with fresh context, file attachments, and RAG |
| `/pti` | Pardon-the-interruption reasoning mode |
| `/computer_use` | GUI automation with vision |
| `/wander` | Exploratory thinking with temperature shifts |
| `/yap` | Voice chat — continuous VAD listening, auto-transcribe, TTS |
| `/guac` | Interactive Python REPL with LLM code generation |
| `/kg` | Knowledge graph browser — facts, concepts, links, search, graph |
| `/kg sleep` | Evolve knowledge graph through consolidation |
| `/kg dream` | Creative synthesis across KG domains |
| `/memories` | Memory browser — browse, approve, reject, filter by status |
| `/nql` | Database browser and NQL SQL model runner |
| `/papers` | Multi-platform research paper browser |
| `/arxiv` | ArXiv paper browser |
| `/git` | Git integration TUI — status, log, branches, stash, cherry-pick file picker |
| `/build` | Build team to deployable format (flask, docker, cli, static) |
| `/team` | Team config browser — context, NPCs, jinxs |
| `/config` | Interactive config editor |
| `/reattach` | Resume previous conversation sessions |
| `/delegate` | Delegate task to NPC with review loop |
| `/web_search` | Web search |
| `/db_search` | Database search |
| `/file_search` | File search |
| `/vixynt` | Generate/edit images |
| `/roll` | Video creation studio |
| `/crond` | System task manager (cron, daemons, processes) |
| `/sample` | Context-free LLM prompt |
| `/serve` | Serve NPC team as API with OpenAI-compatible endpoints |
| `/compile` | Compile NPC profiles |
| `/set` | Set config values — `/set model gemma3:4b`, `/set provider ollama` |
| `/teamviz` | Visualize team structure |
| `/ots` | Screenshot analysis |
| `/models` | Browse available models |
| `/chat` | Switch to chat mode |
| `/cmd` | Switch to command mode |
| `/switch` | Switch NPC |
| `/edit` | Edit NPC, jinx, context, or file — `/edit npc`, `/edit jinx`, `/edit ctx` |
| `/new` | Create new NPC, jinx, or file — `/new npc`, `/new jinx`, `/new file` |
| `/ask_form` | Structured form input for agents to gather user data |
| `/extract_memories` | Extract memories from recent conversations |
| `/reload` | Reload team jinxes and NPCs — `/reload`, `/reload npc name` |
| `/repo_issues` | Fetch GitHub issues and run LLM analysis on each |
| `/sync` | Sync npc_team files from repo to home |

Most commands launch full-screen TUIs — just type and interact. For CLI usage with `npc`, common flags include `--model (-mo)`, `--provider (-pr)`, `--npc (-np)`, and `--temperature (-t)`. Run `npc --help` for the full list.

### `/wander` — Creative Exploration
Wander mode shifts the model's temperature up and down as it explores a problem, producing divergent ideas followed by convergent synthesis. The live TUI dashboard shows the current temperature, accumulated thoughts, and a running summary.

<p align="center">
    <img src="gh_images/wander.png" alt="Wander TUI", width=500>
</p>

### `/guac` — Interactive Python REPL
Guac is an LLM-powered Python REPL with a live variable inspector, DataFrame viewer, and inline code execution. Describe what you want in natural language and the model writes and runs the code. Variables persist across turns. Drop file paths or type `run file.py` to load and execute scripts. Keys: Tab toggles natural language mode, Ctrl+P cycles panels.

<p align="center">
    <img src="gh_images/guac_session.png" alt="Guac Python REPL", width=500>
</p>

### `/arxiv` — Paper Browser
Browse, search, and read arXiv papers from the terminal. The TUI shows search results, full paper metadata, and rendered abstracts with j/k navigation and Enter to drill in.

<p align="center">
    <img src="gh_images/arxiv_search.png" alt="ArXiv search", width=500>
    <img src="gh_images/arxiv_paper.png" alt="ArXiv paper view", width=500>
</p>
<p align="center">
    <img src="gh_images/arxiv_abs.png" alt="ArXiv abstract view", width=700>
</p>

### `/reattach` — Session Browser
Resume previous conversation sessions. The TUI lists past sessions with timestamps and previews — select one to pick up where you left off.

<p align="center">
    <img src="gh_images/Screenshot%20from%202026-01-29%2014-43-20.png" alt="Reattach session browser", width=500>
</p>

### `/models` — Model Browser
Browse all available models across providers (Ollama, OpenAI, Anthropic, etc.), see which are currently active, and set new defaults interactively.

<p align="center">
    <img src="gh_images/models.png" alt="Models browser", width=500>
</p>

### `/roll` — Video Creation Studio
Interactive TUI for generating videos with parameter controls. Edit prompt, model, provider, dimensions, and frame count, then generate. Includes a gallery browser for previously generated videos.

```bash
/roll                    # Open interactive TUI
/roll "a sunset"         # Generate video directly (one-shot mode)
```

<p align="center">
    <img src="gh_images/roll.png" alt="Roll Video Creation Studio" width=500>
</p>

### `/config` — Configuration Editor
Interactive TUI for editing `~/.npcshrc` settings — models, providers, modes, and toggles. Navigate with j/k, edit text fields, toggle booleans, and cycle choices.

```bash
/config
```

<p align="center">
    <img src="gh_images/config.png" alt="Config editor TUI" width=500>
</p>

### `/crond` — System Task Manager
Multi-tab TUI for managing cron jobs, systemd user daemons, and system processes. Create new cron jobs and daemons using natural language, start/stop/restart services, kill processes, and monitor resource usage.

```bash
/crond
```

<p align="center">
    <img src="gh_images/crond.png" alt="Crond Cron tab" width=500>
    <img src="gh_images/crondaemon.png" alt="Crond Daemons tab" width=500>
</p>
<p align="center">
    <img src="gh_images/cron_processes.png" alt="Crond Processes tab" width=500>
</p>

# Memory & Knowledge Graph

`npcsh` maintains a memory lifecycle system that allows agents to learn and grow from past interactions. Memories progress through stages and can be incorporated into a knowledge graph for advanced retrieval.

### Memory Lifecycle

Memories are extracted from conversations via the `/extract_memories` jinx and follow this lifecycle:

1. **pending_approval** - New memories awaiting review
2. **human-approved** - Approved and ready for KG integration
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
    <img src="gh_images/Screenshot%20from%202026-01-29%2016-03-08.png" alt="Memory Browser TUI", width=700>
</p>

### Knowledge Graph

The `/kg` command opens an interactive browser for exploring the knowledge graph:

```bash
/kg                     # Browse facts, concepts, links, search, graph view
/kg sleep               # Evolve KG through consolidation
/kg dream               # Creative synthesis across domains
/kg evolve              # Alias for sleep
/kg sleep backfill=true # Import approved memories first, then evolve
/kg sleep ops=prune,deepen,abstract  # Specific operations
```

The TUI browser has 5 tabs: **Facts**, **Concepts**, **Links**, **Search**, and **Graph** — navigate with Tab, j/k, and Enter to drill into details.

<p align="center">
    <img src="gh_images/kg_facts_viewer.png" alt="Knowledge Graph Facts", width=500>
    <img src="gh_images/kg_links.png" alt="Knowledge Graph Links", width=500>
</p>
<p align="center">
    <img src="gh_images/kg_viewer.png" alt="Knowledge Graph Viewer", width=700>
</p>

**Evolution operations** (via `/kg sleep` or `/sleep`):
- **prune** — Remove redundant or low-value facts
- **deepen** — Add detail to existing facts
- **abstract** — Create higher-level generalizations
- **link** — Connect related facts and concepts

### Environment Variables

```bash
# Enable/disable automatic KG building (default: enabled)
export NPCSH_BUILD_KG=1

# Database path
export NPCSH_DB_PATH=~/npcsh_history.db
```

Full documentation: [npc-shell.readthedocs.io](https://npc-shell.readthedocs.io/en/latest/)

`npcsh` works with local and enterprise LLM providers through LiteLLM — Ollama, LMStudio, vLLM, MLX, OpenAI, Anthropic, Gemini, Deepseek, and more.

## Incognide

[Incognide](https://github.com/npc-worldwide/incognide) is a desktop workspace environment for integrating LLMs into your workflows. [Download executables](https://enpisi.com/downloads) or run `/incognide` in npcsh to install and serve it locally (requires `npm` and `node`).

## Community & Support

- [Newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A) — latest updates on `npcpy`, `npcsh`, and Incognide
- [Discord](https://discord.gg/VvYVT5YC) — discuss ideas for NPC tools
- [Buy us a coffee](https://buymeacoffee.com/npcworldwide) | [Merch](https://enpisi.com/shop) | [Lavanzaro](https://lavanzaro.com)
- For consulting: info@npcworldwi.de

## Installation
`npcsh` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. Below are the instructions for installing such dependencies on Linux, Mac, and Windows. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

### Linux install
<details>  <summary> Toggle </summary>
  
```bash

# these are for audio primarily, skip if you dont need tts
sudo apt-get install espeak
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils
sudo apt-get install libcairo2-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install ffmpeg

#And if you don't have ollama installed, use this:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install 'npcsh[lite]'
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install 'npcsh[local]'
# if you want to use tts/stt
pip install 'npcsh[yap]'
# if you want everything:
pip install 'npcsh[all]'

```

</details>


### Mac install

<details>  <summary> Toggle </summary>

```bash
#mainly for audio
brew install portaudio
brew install ffmpeg
brew install pygobject3

brew install ollama
brew services start ollama
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install npcsh[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcsh[local]
# if you want to use tts/stt
pip install npcsh[yap]

# if you want everything:
pip install npcsh[all]
```
</details>

### Windows Install

<details>  <summary> Toggle </summary>
Download and install ollama exe.

Then, in a powershell. Download and install ffmpeg.

```powershell
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install 'npcsh[lite]'
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install 'npcsh[local]'
# if you want to use tts/stt
pip install 'npcsh[yap]'

# if you want everything:
pip install 'npcsh[all]'
```
As of now, npcsh appears to work well with some of the core functionalities like /ots and /yap.

</details>

### Fedora Install (under construction)

<details>  <summary> Toggle </summary>
  
```bash
python3-dev #(fixes hnswlib issues with chroma db)
xhost +  (pyautogui)
python-tkinter (pyautogui)
```

</details>

## Startup Configuration and Project Structure
To initialize the NPC shell environment parameters correctly, first start the NPC shell:
```bash
npcsh
```
When initialized, `npcsh` will generate a `.npcshrc` file in your home directory that stores your npcsh settings.
Here is an example of what the `.npcshrc` file might look like after this has been run.
```bash
# NPCSH Configuration File
export NPCSH_INITIALIZED=1
export NPCSH_DB_PATH='~/npcsh_history.db'
export NPCSH_CHAT_MODEL=gemma3:4b
export NPCSH_CHAT_PROVIDER=ollama
export NPCSH_DEFAULT_MODE=agent
export NPCSH_EMBEDDING_MODEL=nomic-embed-text
export NPCSH_EMBEDDING_PROVIDER=ollama
export NPCSH_IMAGE_GEN_MODEL=gpt-image-1
export NPCSH_IMAGE_GEN_PROVIDER=openai
export NPCSH_INITIALIZED=1
export NPCSH_REASONING_MODEL=deepseek-r1
export NPCSH_REASONING_PROVIDER=deepseek
export NPCSH_SEARCH_PROVIDER=duckduckgo
export NPCSH_STREAM_OUTPUT=1
export NPCSH_VECTOR_DB_PATH=~/npcsh_chroma.db
export NPCSH_VIDEO_GEN_MODEL=runwayml/stable-diffusion-v1-5
export NPCSH_VIDEO_GEN_PROVIDER=diffusers
export NPCSH_VISION_MODEL=gpt-4o-mini
export NPCSH_VISION_PROVIDER=openai
```

`npcsh` also comes with a set of jinxs and NPCs that are used in processing. It will generate a folder at `~/.npcsh/` that contains the tools and NPCs that are used in the shell and these will be used in the absence of other project-specific ones. Additionally, `npcsh` records interactions and compiled information about npcs within a local SQLite database at the path specified in the `.npcshrc `file. This will default to `~/npcsh_history.db` if not specified. When the data mode is used to load or analyze data in CSVs or PDFs, these data will be stored in the same database for future reference.

The installer will automatically add this file to your shell config, but if it does not do so successfully for whatever reason you can add the following to your `.bashrc` or `.zshrc`:

```bash
# Source NPCSH configuration
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

We support inference via all providers supported by litellm. For openai-compatible providers that are not explicitly named in litellm, use simply `openai-like` as the provider. The default provider must be one of `['openai','anthropic','ollama', 'gemini', 'deepseek', 'openai-like']` and the model must be one available from those providers.

To use tools that require API keys, create an `.env` file in the folder where you are working or place relevant API keys as env variables in your `~/.npcshrc`. If you already have these API keys set in a `~/.bashrc` or a `~/.zshrc` or similar files, you need not additionally add them to `~/.npcshrc` or to an `.env` file. Here is an example of what an `.env` file might look like:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY='your_deepseek_key'
export GEMINI_API_KEY='your_gemini_key'
export PERPLEXITY_API_KEY='your_perplexity_key'
```


 Individual npcs can also be set to use different models and providers by setting the `model` and `provider` keys in the npc files.

 Once initialized, the global team lives in `~/.npcsh/npc_team/`. You can also create a project-specific team by adding an `npc_team/` directory to any project — npcsh picks it up automatically and overlays it on the global team.

```
npc_team/
├── jinxs/
│   ├── modes/            # TUI modes (deep_research, mcp_shell, computer_use, guac, kg, yap, etc.)
│   ├── skills/           # Skills — knowledge-content jinxs
│   │   ├── code-review/  # SKILL.md folder format
│   │   │   └── SKILL.md
│   │   ├── debugging/
│   │   │   └── SKILL.md
│   │   └── git-workflow.jinx  # .jinx format
│   ├── lib/
│   │   ├── core/         # Core tools (python, sh, sql, skill, edit_file, delegate, etc.)
│   │   │   └── search/   # Search tools (web_search, db_search, file_search)
│   │   ├── utils/        # Utility jinxs (set, compile, serve, teamviz, etc.)
│   │   ├── browser/      # Browser automation (browser_action, screenshot, etc.)
│   │   └── computer_use/ # Computer use (click, key_press, screenshot, etc.)
│   └── incognide/        # Incognide desktop workspace jinxs
├── models/               # NQL SQL models
│   ├── base/             # Base statistics models
│   └── insights/         # Models with nql.* AI functions
├── assembly_lines/       # Workflow pipelines
├── sibiji.npc            # Orchestrator NPC
├── corca.npc             # Coding specialist
├── ...                   # Other NPCs
├── mcp_server.py         # MCP server for tool integration
└── npcsh.ctx             # Team context (sets forenpc, team name)
```

<p align="center">
    <img src="gh_images/team_ui.png" alt="Team config browser", width=500>
    <img src="gh_images/jinx_menu.png" alt="Jinx browser", width=500>
</p>
<p align="center">
    <img src="gh_images/jinx_folder_viewer.png" alt="Jinx folder viewer", width=500>
    <img src="gh_images/jinx_ui.png" alt="Jinx detail view", width=500>
</p>

# Benchmark Results

How well can a model drive `npcsh` as an agentic shell? 125 tasks across 15 categories — from basic shell commands to multi-step workflows, code debugging, and tool chaining — scored pass/fail.

<table>
<tr><th>Family</th><th>Model</th><th>Score</th><th>Version</th><th>Results</th><th>Notes</th></tr>
<tr><td rowspan="5"><b>Qwen3</b></td><td>0.6b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>1.7b</td><td>69/125 (55%)</td><td>1.1.27</td><td></td><td></td></tr>
<tr><td>4b</td><td><b>94/125 (75%)</b></td><td>1.1.28</td><td></td><td></td></tr>
<tr><td>8b</td><td><b>104/125 (83%)</b></td><td>1.1.28</td><td></td><td></td></tr>
<tr><td>30b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td rowspan="4"><b>Gemma3</b></td><td>1b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>4b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>12b</td><td>64/117</td><td>1.1.27</td><td></td><td>partial</td></tr>
<tr><td>27b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td rowspan="3"><b>Llama</b></td><td>3.2:1b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>3.2:3b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>3.1:8b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td rowspan="2"><b>Mistral</b></td><td>small3.2</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>ministral</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td><b>Phi</b></td><td>phi4</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td><b>GPT-OSS</b></td><td>20b</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td><b>OLMo</b></td><td>—</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td><b>Cogito</b></td><td>—</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td rowspan="3"><b>Gemini</b></td><td>2.5-flash</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>3.1-flash</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>3.1-pro</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td rowspan="2"><b>Claude</b></td><td>4.6-sonnet</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>4.5-haiku</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td><b>GPT</b></td><td>5-mini</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td rowspan="2"><b>DeepSeek</b></td><td>chat</td><td>—</td><td>—</td><td></td><td></td></tr>
<tr><td>reasoner</td><td>—</td><td>—</td><td></td><td></td></tr>
</table>

<details><summary><b>Category breakdown</b></summary>

<table>
<tr>
<th rowspan="2">Category</th>
<th colspan="5">Qwen3</th>
<th colspan="4">Gemma3</th>
<th colspan="3">Llama</th>
<th colspan="2">Mistral</th>
<th>Phi</th>
<th>GPT-OSS</th>
<th>OLMo</th>
<th>Cogito</th>
<th colspan="3">Gemini</th>
<th colspan="2">Claude</th>
<th>GPT</th>
<th colspan="2">DeepSeek</th>
</tr>
<tr>
<th>0.6b</th><th>1.7b</th><th>4b</th><th>8b</th><th>30b</th>
<th>1b</th><th>4b</th><th>12b</th><th>27b</th>
<th>3.2:1b</th><th>3.2:3b</th><th>3.1:8b</th>
<th>small3.2</th><th>ministral</th>
<th>phi4</th>
<th>20b</th>
<th>—</th>
<th>—</th>
<th>2.5-flash</th><th>3.1-flash</th><th>3.1-pro</th>
<th>4.6-sonnet</th><th>4.5-haiku</th>
<th>5-mini</th>
<th>chat</th><th>reasoner</th>
</tr>
<tr><td>shell (10)</td><td>—</td><td>—</td><td>8</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>file-ops (10)</td><td>—</td><td>—</td><td>10</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>python (10)</td><td>—</td><td>—</td><td>5</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>data (10)</td><td>—</td><td>—</td><td>4</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>system (10)</td><td>—</td><td>—</td><td>9</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>text (10)</td><td>—</td><td>—</td><td>10</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>debug (10)</td><td>—</td><td>—</td><td>4</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>git (10)</td><td>—</td><td>—</td><td>9</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>multi-step (10)</td><td>—</td><td>—</td><td>6</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>scripting (10)</td><td>—</td><td>—</td><td>7</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>image-gen (5)</td><td>—</td><td>—</td><td>5</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>audio-gen (5)</td><td>—</td><td>—</td><td>5</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>web-search (5)</td><td>—</td><td>—</td><td>5</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>delegation (5)</td><td>—</td><td>—</td><td>2</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>tool-chain (5)</td><td>—</td><td>—</td><td>5</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td><b>Total (125)</b></td><td>—</td><td>—</td><td><b>94</b></td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</table>

</details>

```bash
python -m npcsh.benchmark.local_runner --model qwen3:4b --provider ollama
python -m npcsh.benchmark.local_runner --compare  # run all models
```

## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## License
This project is licensed under the MIT License.
