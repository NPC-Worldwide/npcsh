
## NPC CLI

Installing `npcsh` installs two related command-line tools:

- **`npcsh`** — the interactive multi-agent shell (REPL).
- **`npc`** — the dedicated NPC/jinx executor.

They share the same `npcrs` Rust core and the same `npcpy` Python server, but they are separate binaries with separate responsibilities. `npcsh` is for interactive sessions; `npc` is for running `.npc` files, `.jinx` files, and `npc init`.

### How `npc` works

`npc` is shipped as a Rust binary (`npc`) built from `npcsh/rust/src/bin/npc.rs`. When you run `npc`, a Python launcher (`npcsh/npc_launcher.py`) first looks for the compiled Rust binary and execs it. If the Rust binary is missing, the launcher falls back to a pure-Python executor (`npcsh.npc`).

The Rust binary talks to the NPCSH server (`npcpy.serve`) over HTTP for LLM calls, and uses the `npcrs` kernel locally for jinx execution. The launcher starts the server on `127.0.0.1:5237` if it is not already running and sets `NPCSH_SERVER_URL` for the Rust binary.

### Running `.npc` files

```bash
npc ./npc_team/sibiji.npc "what is the biggest file on my computer?"
```

If you omit the prompt, `npc` drops into a minimal REPL for that NPC:

```bash
npc ./npc_team/sibiji.npc
```

### Running `.jinx` files

```bash
npc ./npc_team/jinxes/bin/render.jinx url=https://example.com
```

Jinx inputs can be passed positionally or as `key=value` pairs.

### Initializing a project

```bash
npc init          # create ./npc_team
npc init ./myteam # create ./myteam/npc_team
```

### Relationship to `npcsh`

- `npcsh` uses `launcher.py` to find the Rust `npcsh` binary, start the server, and run the shell.
- `npc` uses `npc_launcher.py` to find the Rust `npc` binary, start the same server, and run the NPC/jinx executor.
- Both launchers exist because the shell and the NPC CLI are separate tools; they share server-starting logic but target different Rust binaries.

### Legacy examples

Older versions of `npc` accepted free-form prompts like `npc 'what is the weather in tokyo?'`. That mode is handled by the Python fallback when the Rust binary is not installed. New installations should use the Rust-first binary and run `.npc` files directly.

## Serving
To serve an NPC project, first install redis-server and start it

on Ubuntu:
```bash
sudo apt update && sudo apt install redis-server
redis-server
```

on macOS:
```bash
brew install redis
redis-server
```
Then navigate to the project directory and run:

```bash
npc serve
```
If you want to specify a certain port, you can do so with the `-p` flag:
```bash
npc serve -p 5337
```
or with the `--port` flag:
```bash
npc serve --port 5337

```
If you want to initialize a project based on templates, and then make it available for serving, you can do so like this:
```bash
npc serve -t 'sales, marketing' -ctx 'im developing a team that will focus on sales and marketing within the logging industry. I need a team that can help me with the following: - generate leads - create marketing campaigns - build a sales funnel - close deals - manage customer relationships - manage sales pipeline - manage marketing campaigns - manage marketing budget' -m qwen3.5:2b -pr ollama
```
This will use the specified model and provider to generate a team of NPCs to fit the templates and context provided..


Once the server is up and running, you can access the API endpoints at `http://localhost:5337/api/`. Here are some example curl commands to test the endpoints:

```bash
echo "Testing health endpoint..."
curl -s http://localhost:5337/api/health | jq '.'

echo -e "\nTesting execute endpoint..."
curl -s -X POST http://localhost:5337/api/execute \
  -H "Content-Type: application/json" \
  -d '{"commandstr": "hello world", "currentPath": "~/", "conversationId": "test124"}' | jq '.'

echo -e "\nTesting conversations endpoint..."
curl -s "http://localhost:5337/api/conversations?path=/tmp" | jq '.'

echo -e "\nTesting conversation messages endpoint..."
curl -s http://localhost:5337/api/conversation/test123/messages | jq '.'
```

## Planned CLI Features


*   **Scripting (Future):** `npc scripts` (details to be defined)
    *   `npc run select +sql_model` (run migrations up)
    *   `npc run select +sql_model+` (run migrations up and down)
    *   `npc run select sql_model+` (run migrations down)
*   **Assembly Line (Future):** `npc run line <assembly_line>` (execute a predefined sequence of operations)
*   **Conjure (Future):** `npc conjure fabrication_plan.fab` (generate content based on a fabrication plan)



# Macros

While npcsh can decide the best option to use based on the user's input, the user can also execute certain actions with a macro. Macros are commands within the NPC shell that start with a forward slash (/) and are followed (in some cases) by the relevant arguments for those macros. Each macro is also available as a sub-program within the NPC CLI. In the following examples we demonstrate how to carry out the same operations from within npcsh and from a regular shell.


To learn about the available macros from within the shell, type:
```npcsh
npcsh> /help
```

or from bash
```bash
npc --help
#alternatively
npc -h
```
