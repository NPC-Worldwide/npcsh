---
name: npcpy-prompting
description: npcpy LLM prompting and JSON formatting patterns.
source_jinx: npcsh/npc_team/jinxes/skills/npcpy-prompting.jinx
engine: skill
---

# npcpy-prompting

npcpy LLM prompting and JSON formatting patterns.

## Imports

Always import at module level:
- `from npcpy.llm_funcs import get_llm_response`
- `from npcpy.npc_compiler import NPC`
- `from npcpy.gen.response import get_litellm_response` (only for streaming)


## Basic-Call

`get_llm_response(prompt, model, provider, **kwargs)` — first arg is POSITIONAL.
Do NOT write `prompt=prompt`. Do NOT use `NPC.call()`. The function is module-level.


## Json-Mode

Pass `format="json"` for structured output.
npcpy parses internally. Access via `response["response"]`.
Never call `json.loads()` manually.


## Npc-Object

Create an `NPC` to hold model, provider, and primary_directive.
Pass it as `npc=npc_instance` so npcpy reads those values:
```python
npc = NPC(name="...", primary_directive="...", model="...", provider="...")
response = get_llm_response(prompt, npc=npc, format="json", temperature=0.7)
data = response["response"]
```


## Messages

Pass conversation history as `messages=[{"role": "system", "content": msg}]`.
This is a kwarg like any other. It does not persist between calls.


## Parameters

Sampling kwargs to `get_llm_response`:
- `temperature`, `top_p`, `top_k`, `max_tokens`
- `stream=True` returns a generator in `response["response"]`


## Streaming

For token-level streaming use `get_litellm_response` with `stream=True`.
For segment-level use `get_llm_response(..., stream=True)` and iterate `response["response"]`.


## Anti-Patterns

- Do NOT use `json.loads(response["response"])`.
- Do NOT call `response.get("response")` and then parse it again.
- Do NOT assume `response` is a string when `format="json"` is used.


## Prompt-Formatting

When constructing prompt strings in Python:
- Use `f"""..."""` for all multiline prompts. Do NOT use implicit string concatenation.
- Do NOT put multiline strings directly in a `return` statement. Assign to a variable first, then return it.
- The closing `"""` must be at the same indentation as the variable assignment.
- Do NOT escape braces as `{{` inside f-strings. If you need literal curly braces in the prompt output, use explicit string concatenation:
  ```python
  prompt = f"""Write a JSON response like this:""" + """\n{'key': 'value'}\n"""
  ```
- Never use `f"..." f"..."` on adjacent lines or in parentheses expecting the parser to concatenate them.
