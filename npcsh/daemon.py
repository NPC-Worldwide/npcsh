#!/usr/bin/env python3
"""
npcsh_llm_daemon.py — Python LLM service for the Rust npcsh runner.

Modes:
  1. Subprocess mode (default): reads JSON requests from stdin, writes JSON
     responses to stdout.  Spawned by the Rust binary on each shell session.
  2. Socket mode (--socket PATH): listens on a Unix domain socket and handles
     multiple concurrent client connections via threads.

Request format:
    {
        "type": "llm",
        "messages": [...],
        "model": "qwen3.5:9b",
        "provider": "ollama",
        "prompt": "user text here",
        "context": "cwd/files/platform info",
        "tools": [...],
        "tool_choice": "auto",
        "api_url": "...",
        "api_key": "...",
        "think": true,
        "attachments": [...],
    }

Response format (stdout in stdio mode, socket wfile in socket mode):
    {
        "ok": true,
        "response": "assistant text",
        "tool_calls": [...],
        "usage": {"input_tokens": 123, "output_tokens": 45},
        "raw": {...},
    }

Or on error:
    {
        "ok": false,
        "error": "error message"
    }

Also supports a "setup" type for initialization (not needed currently
because setup is done before the REPL loop in Rust).
"""
import argparse
import json
import os
import socketserver
import sys
import threading
import traceback

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import io
_old_stdout = sys.stdout
sys.stdout = io.StringIO()

try:
    from npcsh._state import setup_shell
    from npcsh.routes import router

    command_history, team, npc = setup_shell()
    from npcsh._state import initialize_router_with_jinxes
    initialize_router_with_jinxes(team, router)

    initial_state = __import__("npcsh._state", fromlist=["initial_state"]).initial_state
    state = initial_state
    state.team = team
    state.npc = npc
    state.command_history = command_history
finally:
    _setup_output = sys.stdout.getvalue()
    sys.stdout = _old_stdout
    if _setup_output:
        pass


_state_lock = threading.Lock()


def _get_state_snapshot(req):
    """Return a dict with all mutable fields needed for an LLM call.

    This avoids mutating the shared `state` object directly — instead we
    read from it (and from the request) under a lock, then pass the
    snapshot into get_llm_response()."""
    with _state_lock:
        messages = req.get("messages", getattr(state, "messages", []))
        chat_model = req.get("model", getattr(state, "chat_model", None))
        chat_provider = req.get("provider", getattr(state, "chat_provider", None))
        api_url = req.get("api_url", getattr(state, "api_url", None))
        api_key = req.get("api_key", getattr(state, "api_key", None))
        think = req.get("think", getattr(state, "think", None))
        return {
            "messages": messages,
            "chat_model": chat_model,
            "chat_provider": chat_provider,
            "api_url": api_url,
            "api_key": api_key,
            "think": think,
            "npc": getattr(state, "npc", None),
            "team": getattr(state, "team", None),
        }


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
    """Parse  ..., <thinking>...</thinking>, or [thinking]...[/thinking] tags from response content."""
    if not content:
        return None
    import re
    m = re.search(r"  (.*?) ", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\[thinking\](.*?)\[/thinking\]", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _consume_stream(llm_result, provider="", writer=None):
    """Consume a streaming response iterator, print chunks to writer, return final dict.

    Mirrors npcpy.streaming.parse_stream_chunk logic for all provider formats.

    `writer` is a file-like object with .write() and .flush().  Defaults to
    sys.stderr for backward compat with subprocess mode.
    """
    if writer is None:
        writer = sys.stderr

    stream = llm_result.get("response")
    if stream is None or isinstance(stream, str):
        return llm_result, False

    full_content = ""
    full_reasoning = ""
    tool_calls_by_id = {}
    usage = None

    try:
        for chunk in stream:
            content = ""
            reasoning = ""

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
                                key = tc_id or tc_name
                                if isinstance(tc_args, dict):
                                    tc_args = json.dumps(tc_args)
                                elif tc_args is None:
                                    tc_args = "{}"
                                else:
                                    tc_args = str(tc_args)
                                existing = tool_calls_by_id.get(key, {"arguments": "", "name": tc_name, "id": tc_id or key})
                                existing["arguments"] += tc_args
                                existing["name"] = tc_name
                                tool_calls_by_id[key] = existing

                done = getattr(chunk, "done", None) or (chunk.get("done") if isinstance(chunk, dict) else None)
                if done:
                    input_tok = getattr(chunk, "prompt_eval_count", None) or (chunk.get("prompt_eval_count") if isinstance(chunk, dict) else None)
                    output_tok = getattr(chunk, "eval_count", None) or (chunk.get("eval_count") if isinstance(chunk, dict) else None)
                    if input_tok is not None and output_tok is not None:
                        usage = {"prompt_tokens": input_tok, "completion_tokens": output_tok, "total_tokens": input_tok + output_tok}

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

            elif isinstance(chunk, dict):
                content = chunk.get("content", "") or chunk.get("response", "") or ""

            content_thinking = None
            if content:
                content_thinking = _extract_thinking_from_content(content)
                if content_thinking:
                    import re as _re
                    content = _re.sub(r"\[thinking\].*?\[/thinking\]", "", content, flags=_re.DOTALL).strip()

            NL_ESC = "\x01"
            if reasoning:
                safe = reasoning.replace("\n", NL_ESC)
                writer.write(f"[THINK]{safe}\n")
                writer.flush()
                full_reasoning += reasoning
            if content_thinking:
                safe = content_thinking.replace("\n", NL_ESC)
                writer.write(f"[THINK]{safe}\n")
                writer.flush()
                full_reasoning += content_thinking
            if content:
                safe = content.replace("\n", NL_ESC)
                writer.write(f"[STREAM]{safe}\n")
                writer.flush()
                full_content += content

    except Exception as e:
        writer.write(f"[daemon-stream-error] {e}\n")
        writer.flush()

    llm_result["response"] = full_content
    if full_reasoning:
        llm_result["thinking"] = full_reasoning
    if tool_calls_by_id:
        tcs = []
        for key, acc in tool_calls_by_id.items():
            args_str = acc["arguments"]
            try:
                parsed = json.loads(args_str)
                if isinstance(parsed, dict):
                    args_str = json.dumps(parsed)
                else:
                    args_str = json.dumps(parsed[0]) if isinstance(parsed, list) and parsed else json.dumps(parsed)
            except json.JSONDecodeError:
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


def process_request(req, writer):
    """Process a single JSON request and write the response (and any streaming
    chunks) to `writer`.

    `writer` must be a file-like object with `.write(str)` and `.flush()`.
    In stdio mode this is sys.stderr for streaming + _real_stdout for the
    final JSON.  In socket mode it is the socket's makefile object for
    everything.
    """
    req_type = req.get("type", "llm")

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    resp = {"ok": False, "error": "unhandled"}

    try:
        if req_type == "llm":
            snap = _get_state_snapshot(req)

            prompt = req.get("prompt", "")
            context = req.get("context", "")
            tools = req.get("tools")
            tool_choice = req.get("tool_choice", "auto")
            attachments = req.get("attachments")

            from npcpy.llm_funcs import get_llm_response

            think_kwargs = {}
            if snap["think"] is not None:
                think_kwargs["think"] = snap["think"]

            llm_result = get_llm_response(
                prompt,
                model=snap["chat_model"],
                provider=snap["chat_provider"],
                npc=snap["npc"],
                team=snap["team"],
                messages=snap["messages"],
                stream=True,
                attachments=attachments,
                context=context if context else None,
                tools=tools,
                tool_choice=tool_choice,
                **think_kwargs,
            )

            llm_result, _ = _consume_stream(
                llm_result, provider=snap["chat_provider"], writer=writer
            )

            raw = llm_result.get("raw_response") if isinstance(llm_result, dict) else None
            thinking, reasoning = _extract_thinking(raw)
            response_text = llm_result.get("response", "") if isinstance(llm_result, dict) else ""
            if not thinking:
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
        sys.stdout = old_stdout
        if leaked:
            writer.write("[daemon-stdout-leak] " + leaked)
            writer.flush()

    try:
        payload = json.dumps(resp, default=str) + "\n"
        writer.write(payload)
        writer.flush()
    except Exception:
        fallback = json.dumps({"ok": False, "error": "failed to serialize response"}) + "\n"
        writer.write(fallback)
        writer.flush()


def run_stdio():
    """Read JSON lines from stdin, write JSON responses to stdout.
    This is the original subprocess mode used when Rust spawns the daemon."""
    global _real_stdout
    _real_stdout = sys.stdout
    real_stdout = sys.stdout

    sys.stderr.write("npcsh-llm-daemon: ready\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            payload = json.dumps({"ok": False, "error": f"JSON decode: {e}"}) + "\n"
            real_stdout.write(payload)
            real_stdout.flush()
            continue

        class StdioWriter:
            def write(self, text):
                sys.stderr.write(text)
            def flush(self):
                sys.stderr.flush()

        process_request(req, StdioWriter())


class DaemonHandler(socketserver.StreamRequestHandler):
    """One instance per client connection.  Handles multiple requests on the
    same connection (persistent socket)."""

    def handle(self):
        rfile = self.rfile
        wfile = self.wfile

        wfile.write(b"npcsh-llm-daemon: ready\n")
        wfile.flush()

        for line in rfile:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                payload = json.dumps({"ok": False, "error": f"JSON decode: {e}"}) + "\n"
                wfile.write(payload.encode("utf-8"))
                wfile.flush()
                continue

            class SocketWriter:
                def __init__(self, wf):
                    self._wf = wf
                def write(self, text):
                    self._wf.write(text.encode("utf-8"))
                def flush(self):
                    self._wf.flush()

            process_request(req, SocketWriter(wfile))


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """One thread per client connection."""
    daemon_threads = True
    allow_reuse_address = True


def run_socket(socket_path):
    """Listen on a Unix domain socket and handle concurrent connections."""
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    parent = os.path.dirname(socket_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    server = ThreadedUnixServer(socket_path, DaemonHandler)

    os.chmod(socket_path, 0o660)

    sys.stderr.write(f"npcsh-llm-daemon: listening on {socket_path}\n")
    sys.stderr.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


def main():
    parser = argparse.ArgumentParser(description="npcsh LLM daemon")
    parser.add_argument(
        "--socket",
        metavar="PATH",
        help="Run in persistent socket mode listening on a Unix domain socket",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Alias for --socket (used by service managers)",
    )
    args = parser.parse_args()

    socket_path = args.socket
    if args.daemon and not socket_path:
        socket_path = os.environ.get(
            "NPCSH_DAEMON_SOCKET",
            os.path.expanduser("~/.npcsh/daemon.sock"),
        )

    if socket_path:
        run_socket(socket_path)
    else:
        run_stdio()


if __name__ == "__main__":
    main()
