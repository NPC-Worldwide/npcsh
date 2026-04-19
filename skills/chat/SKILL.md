---
name: chat
description: Simple chat mode - LLM conversation without tool execution
---

# chat

Simple chat mode - LLM conversation without tool execution

## Inputs

- `query` (default: `None`)
- `model` (default: `None`)
- `provider` (default: `None`)
- `stream` (default: `True`)

## Steps

- `chat_response` → [`chat_response.py`](./chat_response.py)

## Usage

```
/run_jinx jinx_ref=chat input_values={"query": null, "model": null, "provider": null, "stream": true}
```
