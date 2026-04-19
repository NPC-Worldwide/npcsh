---
name: mcp_shell
description: MCP-powered agentic shell with tabbed TUI
---

# mcp_shell

MCP-powered agentic shell with tabbed TUI

## Inputs

- `mcp_server_path` (default: `None`)
- `initial_command` (default: `None`)
- `model` (default: `None`)
- `provider` (default: `None`)

## Steps

- `corca_tui` → [`corca_tui.py`](./corca_tui.py)

## Usage

```
/run_jinx jinx_ref=mcp_shell input_values={"mcp_server_path": null, "initial_command": null, "model": null, "provider": null}
```
