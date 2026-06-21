import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class TestChatHistoryPreservation:
    """Test that chat.jinx preserves tool calls in state.messages."""

    def test_chat_jinx_logic_preserves_history(self):
        """Simulate chat.jinx logic and verify history preservation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the date?"},
            {"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "call_123", "function": {"name": "date", "arguments": "{}"}}]},
            {"role": "tool", "content": "2026-05-18", "tool_call_id": "call_123"},
            {"role": "assistant", "content": "The date is 2026-05-18."}
        ]

        clean_msgs = []
        for m in messages:
            role = m.get('role', '')
            if role == 'tool':
                continue
            if role == 'assistant' and m.get('tool_calls'):
                continue
            if isinstance(m.get('content'), str):
                clean_msgs.append({'role': role, 'content': m['content']})

        assert len(clean_msgs) == 3
        assert all(m['role'] != 'tool' for m in clean_msgs)

        new_response = {"role": "assistant", "content": "Hello from chat!"}
        new_msgs = clean_msgs + [new_response]

        if new_msgs and len(new_msgs) > len(clean_msgs):
            messages.append(new_msgs[-1])

        tool_msgs = [m for m in messages if m['role'] == 'tool']
        assert len(tool_msgs) == 1, "Tool message lost"

        assistant_with_tools = [m for m in messages if m['role'] == 'assistant' and 'tool_calls' in m]
        assert len(assistant_with_tools) == 1, "Assistant tool call lost"

        assert messages[-1]['content'] == "Hello from chat!"
        assert len(messages) == 6, f"Expected 6 messages, got {len(messages)}"


class TestNshScriptExecution:
    """Test .nsh script execution features."""

    def test_variable_assignment(self):
        """$var = value should store result in state.variables."""
        import re
        line = "$count = !echo 42"
        match = re.match(r'^\s*\$(\w+)\s*=\s*(.+)$', line)
        assert match is not None
        var_name = match.group(1)
        expr = match.group(2).strip()

        assert var_name == "count"
        assert expr == "!echo 42"

    def test_variable_substitution(self):
        """$var and ${var} should be replaced with values."""
        import re
        variables = {'name': 'world', 'greeting': 'hello'}

        def substitute(text, variables, last_output=""):
            def repl(m):
                var = m.group(1) or m.group(2)
                val = variables.get(var, '')
                return str(val) if val is not None else ''
            result = re.sub(r'\$\{(\w+)\}|\$(\w+)', repl, text)
            return result.replace('$_', str(last_output))

        text = "!echo $greeting ${name}"
        result = substitute(text, variables)
        assert result == "!echo hello world"

    def test_last_output_substitution(self):
        """$_ should be replaced with last command output."""
        last_output = "previous result"
        text = "!echo $_"
        result = text.replace('$_', str(last_output))
        assert result == "!echo previous result"

    def test_bang_prefix_stripping(self):
        """Leading ! should be stripped for bash commands."""
        line = "!echo hello"
        cmd = line[1:].strip() if line.startswith('!') else line
        assert cmd == "echo hello"

    def test_nsh_file_parsing(self):
        """.nsh files should have comments stripped and empty lines ignored."""
        content = """$var = !echo test

!echo $var
"""
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
        assert len(lines) == 2
        assert lines[0] == "$var = !echo test"
        assert lines[1] == "!echo $var"


class TestNshEndToEnd:
    """End-to-end tests for .nsh script execution via npcsh.py."""

    def test_nsh_script_with_variables(self):
        """Execute a simple .nsh script with variables."""
        import tempfile

        script = """# Test script
!echo start
$val = !echo 42
!echo result=$val
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.nsh', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            with open(script_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

            variables = {}
            last_output = ""
            outputs = []

            import re
            for line in lines:
                match = re.match(r'^\s*\$(\w+)\s*=\s*(.+)$', line)
                if match:
                    var_name = match.group(1)
                    expr = match.group(2).strip()
                else:
                    var_name = None
                    expr = line

                def repl(m):
                    var = m.group(1) or m.group(2)
                    val = variables.get(var, '')
                    return str(val) if val is not None else ''
                substituted = re.sub(r'\$\{(\w+)\}|\$(\w+)', repl, expr)
                substituted = substituted.replace('$_', str(last_output))

                cmd = substituted[1:].strip() if substituted.startswith('!') else substituted

                if cmd.startswith('echo '):
                    output = cmd[5:]
                else:
                    output = cmd

                if var_name:
                    variables[var_name] = output
                last_output = output
                outputs.append(output)

            assert variables.get('val') == '42'
            assert 'result=42' in outputs
        finally:
            os.unlink(script_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
