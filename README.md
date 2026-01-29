<p align="center">
  <a href= "https://github.com/npc-worldwide/npcsh/blob/main/docs/npcsh.md"> 
  <img src="https://raw.githubusercontent.com/NPC-Worldwide/npcsh/main/npcsh/npcsh.png" alt="npcsh logo" width=600></a>
</p> 

# npcsh

The NPC shell (`npcsh`) makes the most of multi-modal LLMs and agents through a powerful set of simple slash  commands and novel interactive modes, all from the comfort of the command line. Build teams of agents and schedule them on jobs, engineer context, and design custom interaction modes and Jinja Execution templates (Jinxs for you and your agents to invoke, all managed scalably for organizations of any size through the NPC  data layer.

To get started:
For users who want to mainly use models through APIs (`ollama`, `gemini`, `kimi`, `grok`, `deepseek`, `anthropic`, `openai`, `mistral`, or any others provided by litellm )
```bash
pip install 'npcsh[lite]' 
```
For users who want to use and fine-tune local models (this installs `diffusers`/`transformers`/`torch` stack so it is big):

```bash
pip install 'npcsh[local]'
```


For users who want to use the voice mode `yap` (see also the OS-specific installation instructions for installing needed system audio  libraries)
```bash
pip install 'npcsh[yap]'
```

Once installed: run
```bash
npcsh
```
and you will enter the NPC shell. 

If you do not have any local models



Additionally, the pip installation includes the following CLI tools available in bash: `npc` cli, `wander`, `spool`, `yap`, and `nql`. Bin jinxs in `npc_team/jinxs/bin/` are automatically registered as CLI commands. 


# Usage
  - Get help with a task: 
      ```bash
      npcsh>can you help me identify what process is listening on port 5337? 
      ```
      <p align="center"> 
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/port5337.png" alt="example of running npcsh to check what processes are listening on port 5337", width=600>
      </p>

  - Edit files
      ```bash
      npcsh>please read through the markdown files in the docs folder and suggest changes based on the current implementation in the src folder
      ```


  - **Search**
    - search the web
    ```bash
    /search "cerulean city" perplexity

    ```
    <p align="center">
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/search.gif" alt="example of search results", width=600>
    </p>
    
    - search approved memories
    ```bash
    /search query="how to deploy python apps" memory=true
    ```
        
    - search the knowledge graph

    ```bash
    /search query="user preferences for database" kg=true
    ```
        
    - execute a RAG search across files

    ```bash
    /search --rag -f ~/docs/api.md,~/docs/config.yaml "authentication setup"
    ```
        
    - brainblast search (searches many keyword combinations)

    ```bash
    /search query="git commands" brainblast=true
    ```
        
    - web search with specific provider

    ```bash
    /search query="family vacations" sprovider="perplexity"
    ```     

  - **Computer Use**

    ```bash
    /plonk 'find out the latest news on cnn' gemma3:12b ollama
    ```

  - **Generate Image**
    ```bash
    /vixynt 'generate an image of a rabbit eating ham in the brink of dawn' model='gpt-image-1' provider='openai'
    ```
      <p align="center">
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/rabbit.PNG" alt="a rabbit eating ham in the bring of dawn", width=250>
      </p>
  - **Generate Video**
    ```bash
    /roll 'generate a video of a hat riding a dog' veo-3.1-fast-generate-preview  gemini
    ```

      <p align="center">
        <img src="https://raw.githubusercontent.com/NPC-Worldwide/npcsh/main/test_data/hatridingdog.gif" alt="video of a hat riding a dog", width=250>
      </p> 

  - **Serve an NPC Team** (Agentic API Server)
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
  - **Use an mcp server**: make use of NPCs with MCP servers.
    ```bash
    /corca --mcp-server-path /path.to.server.py
    ```

  - **Build an NPC Team**:

    ``` bash
    npc build flask --output ./dist --port 5337
    npc build docker --output ./deploy
    npc build cli --output ./bin
    npc build static --api_url https://api.example.com
    ```

  - **NQL - AI-Powered SQL Models**:
    Run SQL models with embedded AI transformations using NPC agents:
    ```bash
    # List available models
    nql show=1

    # Run all models in dependency order
    nql

    # Run a specific model
    nql model=daily_summary

    # Schedule with cron (runs daily at 6am)
    nql install_cron="0 6 * * *"
    ```

  - **Visualize Team Structure**:
    ```bash
    npc teamviz save=team_structure.png
    ```
    Generates network and ordered views showing NPCs, jinxs, and their relationships.

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
  - lib/core/search
```

When an NPC is invoked, they can only use the jinxs assigned to them. This creates **specialization**:
- `corca` has coding tools (python, sh, edit_file)
- `plonk` has browser automation (browser_action, screenshot)
- `alicanto` has research tools (arxiv, semantic_scholar, paper_search)
- `frederic` has generation tools (vixynt, roll, sample)

The forenpc (orchestrator) can delegate to any team member based on their specialization.

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
| `sibiji` | Orchestrator/coordinator | delegate, convene, search |
| `corca` | Coding and development | python, sh, edit_file, search |
| `plonk` | Browser/GUI automation | browser_action, screenshot, click |
| `alicanto` | Research and analysis | arxiv, semantic_scholar, paper_search |
| `frederic` | Math, physics, music | python, vixynt, roll, sample, wander |
| `guac` | General assistant | python, sh, search |
| `kadiefa` | Creative generation | vixynt, roll, wander |

# Jinxs (Macros/Tools)

Jinxs are reusable tools that NPCs can invoke. They're activated with `/<jinx_name> ...` in npcsh or via the `npc` CLI in bash. For converting any `/<command>` in npcsh to bash, replace `/` with `npc `:

```bash
# In npcsh:
/vixynt "a sunset over mountains"

# In bash:
npc vixynt "a sunset over mountains"
```

## Jinx Commands

### Orchestration & Research
| Command | Description |
|---------|-------------|
| `/alicanto` | Deep research with multiple perspectives. Usage: `/alicanto 'query' --num-npcs 3 --depth 2` |
| `/convene` | Multi-NPC structured discussion. Usage: `/convene "topic" --npcs corca,guac --rounds 3` |
| `/delegate` | Delegate task to NPC with review loop. Usage: `/delegate npc_name=corca task="..." max_iterations=5` |
| `/search` | Web/memory/knowledge graph search. Usage: `/search 'query' --sprovider perplexity` |

### Generation
| Command | Description |
|---------|-------------|
| `/vixynt` | Generate/edit images. Usage: `/vixynt 'description' --igmodel gpt-image-1 --igprovider openai` |
| `/roll` | Generate videos. Usage: `/roll 'description' --vgmodel veo-3.1-fast-generate-preview --vgprovider gemini` |
| `/sample` | Context-free LLM prompt. Usage: `/sample 'question' -m gpt-4o-mini --temp 0.7` |

### Modes & Sessions
| Command | Description |
|---------|-------------|
| `/spool` | Interactive chat with fresh context/RAG. Usage: `/spool --attachments 'file1,file2' -n corca` |
| `/yap` | Voice chat mode. Usage: `/yap -n frederic` |
| `/pti` | Pardon-the-interruption reasoning mode. Usage: `/pti` |
| `/plonk` | GUI automation with vision. Usage: `/plonk 'click the submit button'` |
| `/wander` | Exploratory thinking with temperature shifts. Usage: `/wander 'query' --model deepseek-r1:32b` |

### Data & Analytics
| Command | Description |
|---------|-------------|
| `/nql` | Run NQL SQL models. Usage: `/nql show=1`, `/nql model=daily_summary` |
| `/teamviz` | Visualize team structure. Usage: `/teamviz save=output.png` |
| `/ots` | Screenshot analysis. Usage: `/ots` then select area |
| `/sleep` | Evolve knowledge graph. Usage: `/sleep --ops link_facts,deepen` |
| `/kg_search` | Search knowledge graph with multiple modes. Usage: `/kg_search query mode=hybrid depth=2` |
| `/mem_search` | Search approved memories. Usage: `/mem_search query status=approved top_k=10` |
| `/mem_review` | Review pending memories interactively. Usage: `/mem_review limit=50` |

### System & Config
| Command | Description |
|---------|-------------|
| `/build` | Build team to deployable format. Usage: `/build docker --output ./deploy` |
| `/serve` | Full agentic server with NPC management, jinx controls, and OpenAI-compatible endpoints. Usage: `/serve --port 5337` |
| `/compile` | Compile NPC profiles. Usage: `/compile path/to/npc` |
| `/init` | Initialize NPC project. Usage: `/init` |
| `/set` | Set config values. Usage: `/set model gemma3:4b`, `/set provider ollama` |
| `/help` | Show help. Usage: `/help` |
| `/jinxs` | List available jinxs. Usage: `/jinxs` |
| `/incognide` | Launch Incognide GUI. Usage: `/incognide` |
| `/trigger` | Set up system triggers. Usage: `/trigger 'description' -m gemma3:27b` |
    
    ## Common Command-Line Flags:
    
    ```
    Flag              Shorthand    | Flag              Shorthand    | Flag              Shorthand    | Flag              Shorthand   
    ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------
    --attachments     (-a)         | --height          (-h)         | --num_npcs        (-num_n)     | --team            (-tea)      
    --config_dir      (-con)       | --igmodel         (-igm)       | --output_file     (-o)         | --temperature     (-t)      
    --cors            (-cor)       | --igprovider      (-igp)       | --plots_dir       (-pl)        | --top_k                       
    --creativity      (-cr)        | --lang            (-l)         | --port            (-po)        | --top_p                       
    --depth           (-d)         | --max_tokens      (-ma)        | --provider        (-pr)        | --vmodel          (-vm)       
    --emodel          (-em)        | --messages        (-me)        | --refresh_period  (-re)        | --vprovider       (-vp)       
    --eprovider       (-ep)        | --model           (-mo)        | --rmodel          (-rm)        | --width           (-w)        
    --exploration     (-ex)        | --npc             (-np)        | --rprovider       (-rp)        |                               
    --format          (-f)         | --num_frames      (-num_f)     | --sprovider       (-s)         |                               
    ```

## Memory & Knowledge Graph

`npcsh` maintains a memory lifecycle system that allows agents to learn and grow from past interactions. Memories progress through stages and can be incorporated into a knowledge graph for advanced retrieval.

### Memory Lifecycle

Memories are extracted from conversations and follow this lifecycle:

1. **pending_approval** - New memories awaiting review
2. **human-approved** - Approved and ready for KG integration
3. **human-rejected** - Rejected (used as negative examples)
4. **human-edited** - Modified by user before approval
5. **skipped** - Deferred for later review

### Memory Commands

```bash
# Search through approved memories
/mem_search python                      # Keyword search
/mem_search python status=approved      # Filter by status
/mem_search python top_k=20             # Limit results

# Review pending memories interactively
/mem_review                             # Review with default limit
/mem_review limit=50                    # Review more at once
```

### Knowledge Graph

The knowledge graph stores facts and concepts extracted from approved memories, enabling semantic search and reasoning. Facts are linked to concepts, allowing traversal-based discovery.

```bash
# Keyword search
/kg_search python                       # Simple keyword match

# Semantic similarity search
/kg_search python mode=embedding        # Find semantically similar facts

# Graph traversal search
/kg_search python mode=link depth=3     # Traverse graph links

# Hybrid search (combines methods)
/kg_search python mode=all              # All methods combined

# Explore concepts
/kg_search type=concepts                # List all concepts
/kg_search concept="Machine Learning"   # Explore a specific concept
```

### Knowledge Graph Evolution

The `/sleep` command evolves the knowledge graph through consolidation, abstraction, and creative synthesis:

```bash
# Basic sleep (consolidation)
/sleep

# Import approved memories first, then evolve
/sleep backfill=true

# Dream mode - creative synthesis across domains
/sleep dream=true

# Combined backfill and dream
/sleep backfill=true dream=true

# Specific operations
/sleep ops=prune,deepen,abstract
```

**Operations:**
- **prune** - Remove redundant or low-value facts
- **deepen** - Add detail to existing facts
- **abstract** - Create higher-level generalizations
- **link** - Connect related facts and concepts

### Environment Variables

```bash
# Enable/disable automatic KG building (default: enabled)
export NPCSH_BUILD_KG=1

# Database path
export NPCSH_DB_PATH=~/npcsh_history.db
```

## Read the Docs
To see more about how to use the jinxs and modes in the NPC Shell, read the docs at [npc-shell.readthedocs.io](https://npc-shell.readthedocs.io/en/latest/)


## Inference Capabilities
- `npcsh` works with local and enterprise LLM providers through its LiteLLM integration, allowing users to run inference from Ollama, LMStudio, vLLM, MLX, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks. 

## Incognide
Incognide is a desktop workspace environment for integrating LLMs into your workflows in an organized and seamless manner. See the source code for Incognide [here](https://github.com/npc-worldwide/incognide). Download the executables at [our website](https://enpisi.com/downloads). For the most up to date development version, you can use Incognide by invoking it in npcsh 

```
/incognide
```
which will download and set up and serve the Incognide application within your `~/.npcsh` folder. It requires `npm` and `node` to work, and of course npcpy !

## Mailing List and Community
Interested to stay in the loop and to hear the latest and greatest about `npcpy`, `npcsh`, and Incognide? Be sure to sign up for the [newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!

[Join the discord to discuss ideas for npc tools](https://discord.gg/VvYVT5YC)
## Support
If you appreciate the work here, [consider supporting NPC Worldwide with a monthly donation](https://buymeacoffee.com/npcworldwide), [buying NPC-WW themed merch](https://enpisi.com/shop), [using and subscribing to Lavanzaro](lavanzaro.com),s or hiring us to help you explore how to use the NPC Toolkit and AI tools to help your business or research team, please reach out to info@npcworldwi.de .


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

# for triggers
sudo apt install inotify-tools


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

# for triggers
brew install inotify-tools


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

 Once initialized and set up, you will find the following in your `~/.npcsh` directory:
```bash
~/.npcsh/
├── npc_team/              # Global NPC team
│   ├── jinxs/
│   │   ├── bin/           # CLI commands (wander, spool, yap, nql, vixynt, roll, sample)
│   │   └── lib/           # Library jinxs by category
│   │       ├── core/      # python, sh, sql, search, edit_file, load_file
│   │       ├── browser/   # browser_action, screenshot
│   │       ├── orchestration/  # delegate, convene
│   │       └── research/  # arxiv, semantic_scholar, paper_search
│   ├── models/            # NQL SQL models
│   ├── assembly_lines/    # Workflow pipelines
│   ├── sibiji.npc         # Orchestrator NPC
│   ├── corca.npc          # Coding specialist
│   ├── plonk.npc          # Browser automation
│   ├── ...                # Other NPCs
│   └── npcsh.ctx          # Team context (sets forenpc, team name)
```
For cases where you wish to set up a project specific set of NPCs, jinxs, and assembly lines, add a `npc_team` directory to your project and `npcsh` should be able to pick up on its presence, like so:
```bash
./npc_team/            # Project-specific NPCs
├── jinxs/             # Project jinxs
│   ├── bin/           # Standalone CLI commands (unique names, auto-registered)
│   │   └── wander, spool, yap, nql, vixynt, roll, sample
│   └── lib/           # Library jinxs organized by category
│       ├── core/      # Core utilities (python, sh, sql, search, etc.)
│       ├── browser/   # Browser automation
│       ├── orchestration/  # delegate, convene
│       └── research/  # Research tools (arxiv, semantic_scholar)
├── models/            # NQL SQL models for AI-powered data pipelines
│   ├── base/          # Base statistics models
│   └── insights/      # Models with nql.* AI functions
├── assembly_lines/    # Agentic Workflows
│   └── example.line
├── example1.npc       # Example NPC
├── example2.npc       # Example NPC
└── team.ctx           # Team context file


```

## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.


## License
This project is licensed under the MIT License.
