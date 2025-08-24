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

        server_params = StdioServerParameters(
            command=cmd_parts[0], 
            args=['-c', f'import sys; sys.path.pop(0) if sys.path[0] == "{os.path.dirname(abs_path)}" else None; exec(open("{abs_path}").read())'], 
            env=os.environ.copy(),
            cwd=os.path.dirname(os.path.dirname(abs_path))  # Run from project root
        )
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
                    
                    print(f"DEBUG: About to call MCP tool {tool_name}")
                    try:
                        # Add a timeout
                        result = await asyncio.wait_for(
                            self.session.call_tool(tool_name, args), 
                            timeout=30.0
                        )
                        print(f"DEBUG: MCP tool {tool_name} returned: {type(result)}")
                        return result
                    except asyncio.TimeoutError:
                        print(f"DEBUG: Tool {tool_name} timed out after 30 seconds")
                        return {"error": f"Tool {tool_name} timed out"}
                    except Exception as e:
                        print(f"DEBUG: Tool {tool_name} error: {e}")
                        return {"error": str(e)}
                
                def make_tool_func(tool_name):
                    async def tool_func(**kwargs):
                        print(f"DEBUG: Tool wrapper called for {tool_name} with {kwargs}")
                        # Clean up None string values
                        cleaned_kwargs = {}
                        for k, v in kwargs.items():
                            if v == 'None':
                                cleaned_kwargs[k] = None
                            else:
                                cleaned_kwargs[k] = v
                        result = await execute_tool(tool_name, cleaned_kwargs)
                        print(f"DEBUG: Tool wrapper got result: {type(result)}")
                        return result
                    
                    def sync_wrapper(**kwargs):
                        print(f"DEBUG: Sync wrapper called for {tool_name}")
                        return asyncio.run(tool_func(**kwargs))
                    
                    return sync_wrapper
                self.tool_map[mcp_tool.name] = make_tool_func(mcp_tool.name)
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
        auto_process_tool_calls=False,
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


            # this is process_result
            team_name = result_state.team.name if result_state.team else "__none__"
            npc_name = result_state.npc.name if isinstance(result_state.npc, NPC) else "__none__"
            
            # Determine the actual NPC object to use for this turn's operations
            active_npc = result_state.npc if isinstance(result_state.npc, NPC) else NPC(
                name="default", 
                model=result_state.chat_model, 
                provider=result_state.chat_provider, 
                db_conn=command_history.engine)
            save_conversation_message(
                command_history,
                result_state.conversation_id,
                "user",
                user_input,
                wd=result_state.current_path,
                model=active_npc.model,
                provider=active_npc.provider,
                npc=npc_name,
                team=team_name,
                attachments=result_state.attachments,
            )
            result_state.attachments = None

            final_output_str = None
            output_content = output.get('output') if isinstance(output, dict) else output
            model_for_stream = output.get('model', active_npc.model) if isinstance(output, dict) else active_npc.model
            provider_for_stream = output.get('provider', active_npc.provider) if isinstance(output, dict) else active_npc.provider

            print('\n')
            if user_input =='/help':
                render_markdown(output.get('output'))
            elif result_state.stream_output:


                final_output_str = print_and_process_stream_with_markdown(output_content, 
                                                                        model_for_stream, 
                                                                        provider_for_stream, 
                                                                        show=True)
            elif output_content is not None:
                final_output_str = str(output_content)
                render_markdown(final_output_str)

            if final_output_str:
                if result_state.messages:
                    if result_state.messages[-1].get("role") != "assistant":
                        result_state.messages.append({"role": "assistant", 
                                                "content": final_output_str})
                save_conversation_message(
                    command_history,
                    result_state.conversation_id,
                    "assistant",
                    final_output_str,
                    wd=result_state.current_path,
                    model=active_npc.model,
                    provider=active_npc.provider,
                    npc=npc_name,
                    team=team_name,
                )

                conversation_turn_text = f"User: {user_input}\nAssistant: {final_output_str}"
                engine = command_history.engine


                if result_state.build_kg:
                    import pdb 
                    pdb.set_trace()
                    try:
                        if not should_skip_kg_processing(user_input, final_output_str):

                            npc_kg = load_kg_from_db(engine, team_name, npc_name, result_state.current_path)
                            evolved_npc_kg, _ = kg_evolve_incremental(
                                existing_kg=npc_kg, 
                                new_content_text=conversation_turn_text,
                                model=active_npc.model, 
                                provider=active_npc.provider, 
                                get_concepts=True,
                                link_concepts_facts = False, 
                                link_concepts_concepts = False, 
                                link_facts_facts = False, 

                                
                            )
                            save_kg_to_db(engine,
                                        evolved_npc_kg, 
                                        team_name, 
                                        npc_name, 
                                        result_state.current_path)
                    except Exception as e:
                        print(colored(f"Error during real-time KG evolution: {e}", "red"))

                # --- Part 3: Periodic Team Context Suggestions ---
                result_state.turn_count += 1

                if result_state.turn_count > 0 and result_state.turn_count % 10 == 0:
                    print(colored("\nChecking for potential team improvements...", "cyan"))
                    try:
                        summary = breathe(messages=result_state.messages[-20:], 
                                        npc=active_npc)
                        characterization = summary.get('output')

                        if characterization and result_state.team:
                            team_ctx_path = os.path.join(result_state.team.team_path, "team.ctx")
                            ctx_data = {}
                            if os.path.exists(team_ctx_path):
                                with open(team_ctx_path, 'r') as f:
                                    ctx_data = yaml.safe_load(f) or {}
                            current_context = ctx_data.get('context', '')

                            prompt = f"""Based on this characterization: {characterization},

                            suggest changes (additions, deletions, edits) to the team's context. 
                            Additions need not be fully formed sentences and can simply be equations, relationships, or other plain clear items.
                            
                            Current Context: "{current_context}". 
                            
                            Respond with JSON: {{"suggestion": "Your sentence."
                            }}"""
                            response = get_llm_response(prompt, npc=active_npc, format="json")
                            suggestion = response.get("response", {}).get("suggestion")

                            if suggestion:
                                new_context = (current_context + " " + suggestion).strip()
                                print(colored(f"{result_state.npc.name} suggests updating team context:", "yellow"))
                                print(f"  - OLD: {current_context}\n  + NEW: {new_context}")
                                if input("Apply? [y/N]: ").strip().lower() == 'y':
                                    ctx_data['context'] = new_context
                                    with open(team_ctx_path, 'w') as f:
                                        yaml.dump(ctx_data, f)
                                    print(colored("Team context updated.", "green"))
                                else:
                                    print("Suggestion declined.")
                    except Exception as e:
                        import traceback
                        print(colored(f"Could not generate team suggestions: {e}", "yellow"))
                        traceback.print_exc()
                        



                    import pdb 
                    pdb.set_trace()
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