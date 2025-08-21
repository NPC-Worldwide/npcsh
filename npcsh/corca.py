import os
import sys
import asyncio
import shlex
import argparse
from contextlib import AsyncExitStack
from typing import Optional, Callable, Dict, Any, Tuple, List

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("FATAL: 'mcp-client' package not found. Please run 'pip install mcp-client'.", file=sys.stderr)
    sys.exit(1)

from termcolor import colored, cprint
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_compiler import NPC
from npcpy.npc_sysenv import render_markdown

from npcsh._state import (
    ShellState,
    CommandHistory,
    execute_command as core_execute_command,
    process_result,
    get_multiline_input,
    readline_safe_prompt,
    setup_shell
)

class MCPClientNPC:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.session: Optional[ClientSession] = None
        self._exit_stack = asyncio.new_event_loop().run_until_complete(self._init_stack())
        self.available_tools_llm: List[Dict[str, Any]] = []
        self.tool_map: Dict[str, Callable] = {}
        self.server_script_path: Optional[str] = None

    async def _init_stack(self):
        return AsyncExitStack()

    def _log(self, message: str, color: str = "cyan") -> None:
        if self.debug:
            cprint(f"[MCP Client] {message}", color, file=sys.stderr)

    async def _connect_async(self, server_script_path: str) -> None:
        self._log(f"Attempting to connect to MCP server: {server_script_path}")
        self.server_script_path = server_script_path
        abs_path = os.path.abspath(server_script_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"MCP server script not found: {abs_path}")

        if abs_path.endswith('.py'):
            cmd_parts = [sys.executable, abs_path]
        elif os.access(abs_path, os.X_OK):
            cmd_parts = [abs_path]
        else:
            raise ValueError(f"Unsupported MCP server script type or not executable: {abs_path}")

        server_params = StdioServerParameters(command=cmd_parts[0], args=cmd_parts[1:], env=os.environ.copy())

        if self.session:
            await self._exit_stack.aclose()
        
        self._exit_stack = AsyncExitStack()

        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(*stdio_transport))
        await self.session.initialize()

        response = await self.session.list_tools()
        self.available_tools_llm = []
        self.tool_map = {}

        if response.tools:
            for mcp_tool in response.tools:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                        "parameters": getattr(mcp_tool, "inputSchema", {"type": "object", "properties": {}})
                    }
                }
                self.available_tools_llm.append(tool_def)
                
                async def execute_tool(tool_name: str, args: dict):
                    if not self.session:
                        return {"error": "No MCP session"}
                    return await self.session.call_tool(tool_name, args)
                
                self.tool_map[mcp_tool.name] = (lambda name=mcp_tool.name: lambda **kwargs: asyncio.run(execute_tool(name, kwargs)))()

        tool_names = list(self.tool_map.keys())
        self._log(f"Connection successful. Tools: {', '.join(tool_names) if tool_names else 'None'}")

    def connect_sync(self, server_script_path: str) -> bool:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_async(server_script_path))
            return True
        except Exception as e:
            cprint(f"MCP connection failed: {e}", "red", file=sys.stderr)
            return False
            
    def disconnect_sync(self):
        if self.session:
            self._log("Disconnecting MCP session.")
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if not loop.is_closed():
                try:
                    async def close_session():
                        await self.session.close()
                    loop.run_until_complete(close_session())
                except RuntimeError:
                    pass
            self.session = None

def execute_command_corca(command: str, state: ShellState, command_history) -> Tuple[ShellState, Any]:
    mcp_tools = []
    mcp_tool_map = {}
    
    if hasattr(state, 'mcp_client') and state.mcp_client and state.mcp_client.session:
        mcp_tools = state.mcp_client.available_tools_llm
        mcp_tool_map = state.mcp_client.tool_map
    else:
        cprint("Warning: Corca agent has no tools. No MCP server connected.", "yellow", file=sys.stderr)

    active_npc = state.npc
    if not isinstance(state.npc, NPC):
        active_npc = NPC(name="default")

    response_dict = get_llm_response(
        prompt=command,
        model=active_npc.model or state.chat_model,
        provider=active_npc.provider or state.chat_provider,
        npc=state.npc,
        messages=state.messages,
        tools=mcp_tools,
        tool_map=mcp_tool_map,
        auto_process_tool_calls=True,
        stream=state.stream_output
    )

    state.messages = response_dict.get("messages", state.messages)
    
    final_output = response_dict.get('response', '')
    tool_results = response_dict.get('tool_results', [])

    if tool_results:
        summary_lines = ["\n**Tool Activity:**"]
        for call_result in tool_results:
            func_name = call_result.get('function_name', 'unknown_tool')
            args = call_result.get('arguments', {})
            result = call_result.get('result', 'No result')
            arg_str = ", ".join([f"{k}={v}" for k, v in args.items()])
            summary_lines.append(f"- **Call:** `{func_name}({arg_str})`")
            summary_lines.append(f"  - **Result:** `{str(result)[:200]}`")
        
        final_output = f"{final_output}\n\n" + "\n".join(summary_lines)

    return state, {"output": final_output}

def print_corca_welcome_message():
    turq = "\033[38;2;64;224;208m"
    chrome = "\033[38;2;211;211;211m"
    reset = "\033[0m"
    
    print(
        f"""
Welcome to {turq}C{chrome}o{turq}r{chrome}c{turq}a{reset}!
{turq}       {turq}       {turq}      {chrome}      {chrome}     
{turq}   ____ {turq}  ___  {turq} ____ {chrome}  ____  {chrome} __ _ 
{turq}  /  __|{turq} / _ \\ {turq}|  __\\{chrome} /  __| {chrome}/ _` |
{turq} |  |__ {turq}| (_) |{turq}| |   {chrome}|  |__{chrome} | (_| |
{turq}  \\____| {turq}\\___/ {turq}| |    {chrome}\\____| {chrome}\\__,_|
{turq}       {turq}            {turq}        {chrome}      {chrome}                      
{reset}
An MCP-powered shell for advanced agentic workflows.
        """
    )
    

def enter_corca_mode(command: str, **kwargs):
    state: ShellState = kwargs.get('shell_state')
    command_history: CommandHistory = kwargs.get('command_history')

    if not state or not command_history:
        return {"output": "Error: Corca mode requires shell state and history.", "messages": kwargs.get('messages', [])}

    all_command_parts = shlex.split(command)
    parsed_args = all_command_parts[1:]
    
    parser = argparse.ArgumentParser(prog="/corca", description="Enter Corca MCP-powered mode.")
    parser.add_argument("--mcp-server-path", type=str, help="Path to an MCP server script.")
    
    try:
        args = parser.parse_args(parsed_args)
    except SystemExit:
         return {"output": "Invalid arguments for /corca. See /help corca.", "messages": state.messages}

    print_corca_welcome_message()
    
    mcp_client = MCPClientNPC()
    server_path = args.mcp_server_path
    if not server_path and state.team and hasattr(state.team, 'team_ctx'):
        server_path = state.team.team_ctx.get('mcp_server')
    
    if server_path:
        if mcp_client.connect_sync(server_path):
            state.mcp_client = mcp_client
    else:
        cprint("No MCP server path provided. Corca mode will have limited agent functionality.", "yellow")
        state.mcp_client = None

    while True:
        try:
            prompt_npc_name = "npc"
            if state.npc:
                prompt_npc_name = state.npc.name
            
            prompt_str = f"{colored(os.path.basename(state.current_path), 'blue')}:corca:{prompt_npc_name}> "
            prompt = readline_safe_prompt(prompt_str)
            
            user_input = get_multiline_input(prompt).strip()
            
            if user_input.lower() in ["exit", "quit", "done"]:
                break
            
            if not user_input:
                continue

            state, output = execute_command_corca(user_input, state, command_history)
            process_result(user_input, state, output, command_history)

        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print("\nExiting Corca Mode.")
            break
            
    if state.mcp_client:
        state.mcp_client.disconnect_sync()
        state.mcp_client = None
    
    render_markdown("\n# Exiting Corca Mode")
    return {"output": "", "messages": state.messages}

def main():
    parser = argparse.ArgumentParser(description="Corca - An MCP-powered npcsh shell.")
    parser.add_argument("--mcp-server-path", type=str, help="Path to an MCP server script to connect to.")
    args = parser.parse_args()

    command_history, team, default_npc = setup_shell()

    from npcsh._state import initial_state
    initial_shell_state = initial_state
    initial_shell_state.team = team
    initial_shell_state.npc = default_npc
    
    fake_command_str = "/corca"
    if args.mcp_server_path:
        fake_command_str = f'/corca --mcp-server-path "{args.mcp_server_path}"'
        
    kwargs = {
        'command': fake_command_str,
        'shell_state': initial_shell_state,
        'command_history': command_history
    }
    
    enter_corca_mode(**kwargs)

if __name__ == "__main__":
    main()