import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from npcsh._state import initial_state, execute_command
from npcsh.routes import router
from npcpy.npc_compiler import NPC

def test_chat_preserves_history():
    state = initial_state
    state.current_mode = 'chat'
    state.npc = NPC(name="test-npc", model="gemma", provider="ollama")
    state.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the date?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "call_123", "function": {"name": "date", "arguments": "{}"}}]},
        {"role": "tool", "content": "2026-05-18", "tool_call_id": "call_123"},
        {"role": "assistant", "content": "The date is 2026-05-18."}
    ]

    print(f"Initial history size: {len(state.messages)}")

    jinx_path = os.path.abspath("npcsh/npc_team/jinxes/lib/core/chat.jinx")
    router.load_jinx_routes(os.path.dirname(jinx_path))

    import npcsh._state
    original_get_llm = npcsh._state.get_llm_response

    def mock_get_llm_response(query, **kwargs):
        msgs = kwargs.get('messages', [])
        for m in msgs:
            if m['role'] == 'tool':
                print("❌ FAIL: Tool message found in clean_msgs passed to LLM!")
            if m['role'] == 'assistant' and 'tool_calls' in m:
                print("❌ FAIL: Tool call found in clean_msgs passed to LLM!")

        return {
            "response": "Hello from chat!",
            "messages": msgs + [{"role": "assistant", "content": "Hello from chat!"}]
        }

    npcsh._state.get_llm_response = mock_get_llm_response

    try:
        state, output = execute_command("/chat hello", state, router=router)
        print(f"Final history size: {len(state.messages)}")

        tool_msgs = [m for m in state.messages if m.get('role') == 'tool']
        if not tool_msgs:
            raise AssertionError("Tool messages were deleted from history")
        else:
            print("✅ SUCCESS: Tool messages preserved in history.")

        contents = [str(m.get('content', '')) for m in state.messages]
        if not any("Hello from chat!" in c for c in contents):
            raise AssertionError(f"New chat response not found. Last contents: {contents[-3:]}")
        else:
            print("✅ SUCCESS: New response found in messages.")

    finally:
        npcsh._state.get_llm_response = original_get_llm

if __name__ == "__main__":
    test_chat_preserves_history()
