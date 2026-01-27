import pytest

import npcsh._state as state_module
from npcsh._state import ShellState, process_pipeline_command


@pytest.fixture(autouse=True)
def stub_available_models(monkeypatch):
    monkeypatch.setattr(state_module, "get_locally_available_models", lambda path: {})


def test_model_supports_tool_calls_uses_heuristics(monkeypatch):
    # Force ollama inspection to be inconclusive so heuristics are used.
    monkeypatch.setattr(state_module, "_ollama_supports_tools", lambda model: None)
    assert not state_module.model_supports_tool_calls("gemma3:4b", "ollama")
    assert state_module.model_supports_tool_calls("qwen3:0.6b", "ollama")


def test_react_flow_used_for_non_tool_models(tmp_path, monkeypatch):
    calls = {"check": 0, "llm": 0}

    def fake_check_llm_command(*args, **kwargs):
        calls["check"] += 1
        # Simulate the real check_llm_command which appends user message
        msgs = list(kwargs.get("messages", []))
        command = args[0] if args else kwargs.get("command", "")
        if command:
            msgs.append({"role": "user", "content": command})
        return {"output": "react-path", "messages": msgs}

    monkeypatch.setattr(state_module, "model_supports_tool_calls", lambda m, p: False)
    monkeypatch.setattr(state_module, "check_llm_command", fake_check_llm_command)
    monkeypatch.setattr(state_module, "get_llm_response", lambda *a, **k: None)

    shell_state = ShellState(current_path=str(tmp_path))
    updated_state, output = process_pipeline_command(
        "explain something", None, shell_state, stream_final=False, router=None
    )

    assert calls["check"] == 1
    # Output can be a string or dict with 'output' key
    if isinstance(output, dict):
        assert output.get("output") == "react-path"
    else:
        assert output == "react-path"
    # user message should be appended
    assert updated_state.messages and updated_state.messages[0]["role"] == "user"


def test_tool_call_flow_for_tool_capable_models(tmp_path, monkeypatch):
    calls = {"check": 0, "llm": 0}

    def fake_get_llm_response(*args, **kwargs):
        calls["llm"] += 1
        msgs = kwargs.get("messages", [])
        return {"response": "tool-route", "messages": msgs + [{"role": "assistant", "content": "tool-route"}]}

    monkeypatch.setattr(state_module, "model_supports_tool_calls", lambda m, p: True)
    monkeypatch.setattr(state_module, "get_llm_response", fake_get_llm_response)
    monkeypatch.setattr(
        state_module,
        "collect_llm_tools",
        lambda _state: (
            [
                {
                    "type": "function",
                    "function": {"name": "dummy", "description": "d", "parameters": {"type": "object", "properties": {}}},
                }
            ],
            {"dummy": lambda **kwargs: "ok"},
        ),
    )
    monkeypatch.setattr(
        state_module,
        "check_llm_command",
        lambda *args, **kwargs: (
            calls.__setitem__("check", calls["check"] + 1)
            or {"output": "should-not-run", "messages": kwargs.get("messages", [])}
        ),
    )

    shell_state = ShellState(current_path=str(tmp_path))
    updated_state, output = process_pipeline_command(
        "explain something", None, shell_state, stream_final=False, router=None
    )

    assert calls["llm"] == 1
    assert calls["check"] == 0
    # Output can be a string or dict with 'output' key
    if isinstance(output, dict):
        assert output.get("output") == "tool-route"
    else:
        assert output == "tool-route"
    assert updated_state.messages and updated_state.messages[-1]["role"] == "assistant"
