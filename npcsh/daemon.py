#!/usr/bin/env python3
"""
npcsh_llm_daemon.py — Python LLM service for the Rust npcsh runner.

Spawned by the Rust binary as a long-lived subprocess. Reads JSON requests
from stdin and writes JSON responses to stdout.

Request format:
    {
        "type": "llm",
        "messages": [...],          # list of dicts with role/content/tool_calls/etc
        "model": "qwen3.5:9b",
        "provider": "ollama",
        "prompt": "user text here",
        "context": "cwd/files/platform info",
        "tools": [...],               # tool schema definitions (optional)
        "tool_choice": "auto",        # optional
        "api_url": "...",             # optional
        "api_key": "...",             # optional
        "think": true,                # optional (true/false)
        "attachments": [...],         # optional
    }

Response format:
    {
        "ok": true,
        "response": "assistant text",
        "tool_calls": [...],          # list of tool call dicts (optional)
        "usage": {"input_tokens": 123, "output_tokens": 45},
        "raw": {...},                 # raw_response dict (optional, for thinking extraction)
    }

Or on error:
    {
        "ok": false,
        "error": "error message"
    }

Also supports a "setup" type for initialization (not needed currently
because setup is done before the REPL loop in Rust).
"""
import json
import os
import sys
import traceback

# Ensure npcsh/npcrs packages are importable
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Suppress stdout during heavy npcsh imports/setup (they print to stdout which
# corrupts our JSON wire protocol)
import io
_old_stdout = sys.stdout
sys.stdout = io.StringIO()

try:
    from npcsh._state import setup_shell, process_pipeline_command, ShellState
    from npcsh.routes import router

    # ── Setup (once at daemon startup) ──
    command_history, team, npc = setup_shell()
    from npcsh._state import initialize_router_with_jinxes
    initialize_router_with_jinxes(team, router)

    # Build a reusable ShellState for LLM calls
    initial_state = __import__("npcsh._state", fromlist=["initial_state"]).initial_state
    state = initial_state
    state.team = team
    state.npc = npc
    state.command_history = command_history
finally:
    _setup_output = sys.stdout.getvalue()
    sys.stdout = _old_stdout
    # Setup output suppressed — only log critical errors
    if _setup_output:
        pass

sys.stderr.write("npcsh-llm-daemon: ready\n")
sys.stderr.flush()


def _serialize_tool_calls(tool_calls):
    """Convert npcpy/OpenAI tool-call objects to plain JSON dicts."""
    if not tool_calls:
        return None
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            result.append(tc)
        elif hasattr(tc, "to_dict"):
            result.append(tc.to_dict())
        elif hasattr(tc, "__dict__"):
            d = vars(tc).copy()
            # flatten nested function object if present
            if hasattr(tc, "function"):
                func = tc.function
                if hasattr(func, "__dict__"):
                    d["function"] = vars(func).copy()
                elif hasattr(func, "to_dict"):
                    d["function"] = func.to_dict()
                else:
                    d["function"] = str(func)
            result.append(d)
        else:
            result.append(str(tc))
    return result


def _extract_thinking(raw):
    """Extract thinking/reasoning content from raw response for display."""
    thinking_text = reasoning_text = None
    if raw is None:
        return None, None
    try:
        # Anthropic reasoning blocks
        if hasattr(raw, "content") and isinstance(getattr(raw, "content", None), list):
            for block in raw.content:
                if getattr(block, "type", None) == "thinking":
                    thinking_text = getattr(block, "thinking", None)
                elif getattr(block, "type", None) == "reasoning":
                    reasoning_text = getattr(block, "reasoning", None)
        if isinstance(raw, dict):
            thinking_text = raw.get("thinking") or raw.get("reasoning_content") or thinking_text
            reasoning_text = raw.get("reasoning") or reasoning_text
        if hasattr(raw, "choices") and raw.choices:
            first = raw.choices[0]
            if hasattr(first, "message") and first.message:
                msg = first.message
                if hasattr(msg, "reasoning_content"):
                    reasoning_text = msg.reasoning_content
                if hasattr(msg, "thinking"):
                    thinking_text = msg.thinking
    except Exception:
        pass
    return thinking_text, reasoning_text


def _extract_thinking_from_content(content):
    """Parse <think>...</think>, <thinking>...</thinking>, or [thinking]...[/thinking] tags from response content."""
    if not content:
        return None
    import re
    # <think>...</think>
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    # <thinking>...</thinking> (deepseek style)
    m = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    # [thinking]...[/thinking] (kimi-k2.6 via ollama style)
    m = re.search(r"\[thinking\](.*?)\[/thinking\]", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

def _consume_stream(llm_result, provider=""):
    """Consume a streaming response iterator, print chunks to stderr, return final dict.
    
    Mirrors npcpy.streaming.parse_stream_chunk logic for all provider formats.
    """
    stream = llm_result.get("response")
    if stream is None or isinstance(stream, str):
        return llm_result, False

    full_content = ""
    full_reasoning = ""
    # Track tool calls by ID to avoid concatenating complete objects
    tool_calls_by_id = {}
    usage = None

    try:
        for chunk in stream:
            content = ""
            reasoning = ""

            # --- Ollama / HF style (message-based) ---
            if provider == "ollama" or provider == "transformers":
                msg = getattr(chunk, "message", None)
                if msg is None and hasattr(chunk, "get"):
                    msg = chunk.get("message", {})
                if msg is None:
                    msg = {}

                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "") or ""
                reasoning = getattr(msg, "thinking", None) or (msg.get("thinking") if isinstance(msg, dict) else None) or ""

                tcs = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
                if tcs:
                    for tc in tcs:
                        tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
                        tc_func = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
                        if tc_func:
                            tc_name = getattr(tc_func, "name", None) or (tc_func.get("name") if isinstance(tc_func, dict) else None)
                            tc_args = getattr(tc_func, "arguments", None) or (tc_func.get("arguments") if isinstance(tc_func, dict) else None)
                            if tc_name:
                                # Use name as key if no ID (ollama often lacks IDs)
                                key = tc_id or tc_name
                                if isinstance(tc_args, dict):
                                    tc_args = json.dumps(tc_args)
                                elif tc_args is None:
                                    tc_args = "{}"
                                else:
                                    tc_args = str(tc_args)
                                # For incremental streaming, append; for complete objects, replace
                                existing = tool_calls_by_id.get(key, {"arguments": "", "name": tc_name, "id": tc_id or key})
                                existing["arguments"] += tc_args
                                existing["name"] = tc_name
                                tool_calls_by_id[key] = existing

                # Ollama done chunk
                done = getattr(chunk, "done", None) or (chunk.get("done") if isinstance(chunk, dict) else None)
                if done:
                    input_tok = getattr(chunk, "prompt_eval_count", None) or (chunk.get("prompt_eval_count") if isinstance(chunk, dict) else None)
                    output_tok = getattr(chunk, "eval_count", None) or (chunk.get("eval_count") if isinstance(chunk, dict) else None)
                    if input_tok is not None and output_tok is not None:
                        usage = {"prompt_tokens": input_tok, "completion_tokens": output_tok, "total_tokens": input_tok + output_tok}

            # --- LiteLLM / OpenAI style (choices-based) ---
            elif hasattr(chunk, "choices") and chunk.choices:
                for c in chunk.choices:
                    delta = getattr(c, "delta", None)
                    if delta is None:
                        continue
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning += delta.reasoning_content
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            tc_id = getattr(tc, "id", None) or ""
                            tc_idx = getattr(tc, "index", 0)
                            key = f"{tc_id}_{tc_idx}"
                            tc_name = ""
                            tc_args = ""
                            if hasattr(tc, "function") and tc.function:
                                if hasattr(tc.function, "name") and tc.function.name:
                                    tc_name = tc.function.name
                                if hasattr(tc.function, "arguments") and tc.function.arguments:
                                    tc_args = tc.function.arguments
                            existing = tool_calls_by_id.get(key, {"arguments": "", "name": tc_name, "id": tc_id or key})
                            existing["arguments"] += tc_args
                            existing["name"] = tc_name or existing["name"]
                            tool_calls_by_id[key] = existing
                if hasattr(chunk, "usage") and chunk.usage:
                    pt = getattr(chunk.usage, "prompt_tokens", 0) or 0
                    ct = getattr(chunk.usage, "completion_tokens", 0) or 0
                    usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

            # --- Plain dict fallback ---
            elif isinstance(chunk, dict):
                content = chunk.get("content", "") or chunk.get("response", "") or ""

            # Extract thinking tags from content (kimi-k2.6 via ollama outputs [thinking]...[/thinking])
            content_thinking = None
            if content:
                content_thinking = _extract_thinking_from_content(content)
                if content_thinking:
                    import re as _re
                    content = _re.sub(r"\[thinking\].*?\[/thinking\]", "", content, flags=_re.DOTALL).strip()

            # Stream to stderr — thinking via [THINK], content via [STREAM]
            # Newlines are escaped with \x01 so Rust read_line() can process each chunk as a single line.
            NL_ESC = "\x01"
            if reasoning:
                safe = reasoning.replace("\n", NL_ESC)
                sys.stderr.write(f"[THINK]{safe}\n")
                sys.stderr.flush()
                full_reasoning += reasoning
            if content_thinking:
                safe = content_thinking.replace("\n", NL_ESC)
                sys.stderr.write(f"[THINK]{safe}\n")
                sys.stderr.flush()
                full_reasoning += content_thinking
            if content:
                safe = content.replace("\n", NL_ESC)
                sys.stderr.write(f"[STREAM]{safe}\n")
                sys.stderr.flush()
                full_content += content

    except Exception as e:
        sys.stderr.write(f"[daemon-stream-error] {e}\n")
        sys.stderr.flush()

    llm_result["response"] = full_content
    if full_reasoning:
        llm_result["thinking"] = full_reasoning
    # Build tool_calls list from accumulated dicts
    if tool_calls_by_id:
        tcs = []
        for key, acc in tool_calls_by_id.items():
            args_str = acc["arguments"]
            # If accumulated args is a valid JSON dict, keep it; else wrap as string
            try:
                parsed = json.loads(args_str)
                if isinstance(parsed, dict):
                    args_str = json.dumps(parsed)
                else:
                    # Got concatenated objects — try to extract the last complete one
                    # or keep the first valid JSON object
                    args_str = json.dumps(parsed[0]) if isinstance(parsed, list) and parsed else json.dumps(parsed)
            except json.JSONDecodeError:
                # Try to find a complete JSON object in the string
                import re
                matches = re.findall(r'\{[^}]+\}', args_str)
                if matches:
                    try:
                        args_str = json.dumps(json.loads(matches[0]))
                    except Exception:
                        pass
                if not args_str:
                    args_str = "{}"
            tcs.append({
                "id": acc["id"],
                "type": "function",
                "function": {
                    "name": acc["name"],
                    "arguments": args_str,
                }
            })
        llm_result["tool_calls"] = tcs
    if usage is not None:
        llm_result["usage"] = usage
    else:
        # Normalize any existing usage from non-streaming paths
        existing = llm_result.get("usage") if isinstance(llm_result, dict) else None
        if existing:
            normalized = {}
            if "input_tokens" in existing:
                normalized["prompt_tokens"] = existing["input_tokens"]
            if "output_tokens" in existing:
                normalized["completion_tokens"] = existing["output_tokens"]
            if "prompt_tokens" in existing:
                normalized["prompt_tokens"] = existing["prompt_tokens"]
            if "completion_tokens" in existing:
                normalized["completion_tokens"] = existing["completion_tokens"]
            if "total_tokens" in existing:
                normalized["total_tokens"] = existing["total_tokens"]
            else:
                pt = normalized.get("prompt_tokens", 0)
                ct = normalized.get("completion_tokens", 0)
                normalized["total_tokens"] = pt + ct
            llm_result["usage"] = normalized
    llm_result["streamed"] = True
    return llm_result, True



# Keep a pristine handle to the real stdout for JSON responses
_real_stdout = sys.stdout

def _send_json(obj):
    """Serialize obj to JSON and write it to the real stdout, guarding against any accidental prints."""
    try:
        payload = json.dumps(obj, default=str) + "\n"
        _real_stdout.write(payload)
        _real_stdout.flush()
    except Exception:
        fallback = json.dumps({"ok": False, "error": "failed to serialize response"}) + "\n"
        _real_stdout.write(fallback)
        _real_stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    # Capture anything that mistakenly prints to stdout during processing
    buf = io.StringIO()
    sys.stdout = buf
    resp = {"ok": False, "error": "unhandled"}
    try:
        req = json.loads(line)
        req_type = req.get("type", "llm")

        if req_type == "llm":
            # Sync state from request
            state.messages = req.get("messages", state.messages)
            state.chat_model = req.get("model", state.chat_model)
            state.chat_provider = req.get("provider", state.chat_provider)
            if req.get("api_url"):
                state.api_url = req["api_url"]
            if req.get("api_key"):
                setattr(state, "api_key", req["api_key"])
            if req.get("think") is not None:
                state.think = req["think"]

            prompt = req.get("prompt", "")
            context = req.get("context", "")
            tools = req.get("tools")
            tool_choice = req.get("tool_choice", "auto")
            attachments = req.get("attachments")

            # Import at call time to avoid circular import issues
            from npcpy.llm_funcs import get_llm_response
            from npcpy.gen.response import get_model_context_window

            think_kwargs = {}
            if state.think is not None:
                think_kwargs["think"] = state.think

            llm_result = get_llm_response(
                prompt,
                model=state.chat_model,
                provider=state.chat_provider,
                npc=state.npc,
                team=state.team,
                messages=state.messages,
                stream=True,
                attachments=attachments,
                context=context if context else None,
                tools=tools,
                tool_choice=tool_choice,
                **think_kwargs,
            )

            # Consume streamed response if needed
            llm_result, _ = _consume_stream(llm_result, provider=state.chat_provider)

            # Extract thinking for display
            raw = llm_result.get("raw_response") if isinstance(llm_result, dict) else None
            thinking, reasoning = _extract_thinking(raw)
            response_text = llm_result.get("response", "") if isinstance(llm_result, dict) else ""
            if not thinking:
                # preserve thinking accumulated during streaming
                thinking = llm_result.get("thinking")
            if not thinking:
                thinking = _extract_thinking_from_content(response_text)

            usage = llm_result.get("usage") if isinstance(llm_result, dict) else None
            tool_calls = llm_result.get("tool_calls") if isinstance(llm_result, dict) else None
            tool_calls = _serialize_tool_calls(tool_calls)

            resp = {
                "ok": True,
                "response": response_text,
                "tool_calls": tool_calls,
                "usage": usage,
                "thinking": thinking,
                "reasoning": reasoning,
                "streamed": llm_result.get("streamed", False) if isinstance(llm_result, dict) else False,
            }
        elif req_type == "setup":
            # Return basic setup info (team name, npc names, etc.)
            resp = {
                "ok": True,
                "team_name": team.name if team else "npcsh",
                "npcs": list(team.npcs.keys()) if team else [],
                "jinxes": list(team.jinxes_dict.keys()) if team and hasattr(team, "jinxes_dict") else [],
            }
        else:
            resp = {"ok": False, "error": f"Unknown request type: {req_type}"}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        leaked = buf.getvalue()
        sys.stdout = _real_stdout
        if leaked:
            sys.stderr.write("[daemon-stdout-leak] " + leaked)
            sys.stderr.flush()

    _send_json(resp)
