import os
import sys
import asyncio
import shlex
import argparse
from contextlib import AsyncExitStack
from typing import Optional, Callable, Dict, Any, Tuple, List
import shutil
import traceback
from litellm.exceptions import Timeout, ContextWindowExceededError, RateLimitError, BadRequestError

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("FATAL: 'mcp-client' package not found. Please run 'pip install mcp-client'.", file=sys.stderr)
    sys.exit(1)

from termcolor import colored, cprint
import json
from npcpy.llm_funcs import get_llm_response, breathe
from npcpy.npc_compiler import NPC
from npcpy.npc_sysenv import render_markdown, print_and_process_stream_with_markdown
from npcpy.memory.command_history import load_kg_from_db, save_conversation_message, save_kg_to_db
from npcpy.memory.knowledge_graph import kg_evolve_incremental, kg_dream_process, kg_initial, kg_sleep_process
from npcsh._state import (
    ShellState,
    CommandHistory,
    execute_command as core_execute_command,
    process_result,
    get_multiline_input,
    readline_safe_prompt,
    setup_shell, 
    should_skip_kg_processing, 
    NPCSH_CHAT_PROVIDER, 
    NPCSH_CHAT_MODEL,
    get_team_ctx_path
)
import yaml 
from pathlib import Path

class MCPClientNPC:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.session: Optional[ClientSession] = None
        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
        self._exit_stack = self._loop.run_until_complete(self._init_stack())
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
            args=[abs_path],
            env=os.environ.copy(),
            cwd=Path(abs_path).parent
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
                
                def make_tool_func(tool_name_closure):
                    async def tool_func(**kwargs):
                        if not self.session:
                            return {"error": "No MCP session"}
                        
                        self._log(f"About to call MCP tool {tool_name_closure}")
                        try:
                            cleaned_kwargs = {}
                            for k, v in kwargs.items():
                                if v == 'None':
                                    cleaned_kwargs[k] = None
                                else:
                                    cleaned_kwargs[k] = v
                            result = await asyncio.wait_for(
                                self.session.call_tool(tool_name_closure, cleaned_kwargs), 
                                timeout=30.0
                            )
                            self._log(f"MCP tool {tool_name_closure} returned: {type(result)}")
                            return result
                        except asyncio.TimeoutError:
                            self._log(f"Tool {tool_name_closure} timed out after 30 seconds", "red")
                            return {"error": f"Tool {tool_name_closure} timed out"}
                        except Exception as e:
                            self._log(f"Tool {tool_name_closure} error: {e}", "red")
                            return {"error": str(e)}
                    
                    def sync_wrapper(**kwargs):
                        self._log(f"Sync wrapper called for {tool_name_closure}")
                        return self._loop.run_until_complete(tool_func(**kwargs))
                    
                    return sync_wrapper
                self.tool_map[mcp_tool.name] = make_tool_func(mcp_tool.name)
        tool_names = list(self.tool_map.keys())
        self._log(f"Connection successful. Tools: {', '.join(tool_names) if tool_names else 'None'}")

    def connect_sync(self, server_script_path: str) -> bool:
        loop = self._loop
        if loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            loop = self._loop

        try:
            loop.run_until_complete(self._connect_async(server_script_path))
            return True
        except Exception as e:
            cprint(f"MCP connection failed: {e}", "red", file=sys.stderr)
            return False
            
    def disconnect_sync(self):
        if self.session:
            self._log("Disconnecting MCP session.")
            loop = self._loop
            if not loop.is_closed():
                try:
                    async def close_session():
                        await self.session.close()
                        await self._exit_stack.aclose()
                    loop.run_until_complete(close_session())
                except RuntimeError:
                    pass
                except Exception as e:
                    print(f"Error during MCP client disconnect: {e}", file=sys.stderr)
            self.session = None
            self._exit_stack = None


def process_mcp_stream(stream_response, active_npc):
    collected_content = ""
    tool_calls = []
    
    interrupted = False
    sys.stdout.write('\033[s')
    sys.stdout.flush()
    
    try:
        for chunk in stream_response:        
            if hasattr(active_npc, 'provider') and active_npc.provider == "ollama" and 'gpt-oss' not in active_npc.model:
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'tool_calls') and chunk.message.tool_calls:
                    for tool_call in chunk.message.tool_calls:
                        tool_call_data = {'id': getattr(tool_call, 'id', ''),
                            'type': 'function',
                            'function': {
                                'name': getattr(tool_call.function, 'name', '') if hasattr(tool_call, 'function') else '',
                                'arguments': getattr(tool_call.function, 'arguments', {}) if hasattr(tool_call, 'function') else {}
                            }
                        }
                        if isinstance(tool_call_data['function']['arguments'], str):
                            try:
                                tool_call_data['function']['arguments'] = json.loads(tool_call_data['function']['arguments'])
                            except json.JSONDecodeError:
                                tool_call_data['function']['arguments'] = {'raw': tool_call_data['function']['arguments']}
                        
                        tool_calls.append(tool_call_data)
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content') and chunk.message.content:
                    collected_content += chunk.message.content
                    print(chunk.message.content, end='', flush=True)
                    
            else:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    if hasattr(delta, 'content') and delta.content:
                        collected_content += delta.content
                        print(delta.content, end='', flush=True)
                    
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if hasattr(tool_call_delta, 'index'):
                                idx = tool_call_delta.index
                                
                                while len(tool_calls) <= idx:
                                    tool_calls.append({
                                        'id': '',
                                        'type': 'function',
                                        'function': {'name': '', 'arguments': ''}
                                    })
                                
                                if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                                    tool_calls[idx]['id'] = tool_call_delta.id
                                if hasattr(tool_call_delta, 'function'):
                                    if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                        tool_calls[idx]['function']['name'] = tool_call_delta.function.name
                                    
                                    if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                                        tool_calls[idx]['function']['arguments'] += tool_call_delta.function.arguments
    except KeyboardInterrupt:
        interrupted = True
        print('\n⚠️ Stream interrupted by user')
    
    sys.stdout.write('\033[u')
    sys.stdout.write('\033[0J')
    sys.stdout.flush()
    
    if collected_content:
        render_markdown(collected_content)
    
    return collected_content, tool_calls

def clean_orphaned_tool_calls(messages):
    cleaned_messages = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if msg.get("role") == "tool":
            # Check if there's a preceding assistant message with tool_calls
            found_preceding_assistant = False
            for j in range(i-1, -1, -1):
                prev_msg = messages[j]
                if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                    # Check if this tool response matches any tool call
                    tool_call_ids = {tc["id"] for tc in prev_msg["tool_calls"]}
                    if msg.get("tool_call_id") in tool_call_ids:
                        found_preceding_assistant = True
                        break
                elif prev_msg.get("role") in ["user", "assistant"]:
                    break
            
            if found_preceding_assistant:
                cleaned_messages.append(msg)
            # Skip orphaned tool responses
            
        elif (msg.get("role") == "assistant" and msg.get("tool_calls")):
            tool_call_ids = {tc["id"] for tc in msg["tool_calls"]}
            j = i + 1
            found_responses = set()
            
            while j < len(messages):
                next_msg = messages[j]
                if next_msg.get("role") == "tool":
                    if next_msg.get("tool_call_id") in tool_call_ids:
                        found_responses.add(next_msg.get("tool_call_id"))
                elif next_msg.get("role") in ["user", "assistant"]:
                    break
                j += 1
            
            missing_responses = tool_call_ids - found_responses
            if missing_responses:
                assistant_msg = msg.copy()
                assistant_msg["tool_calls"] = [
                    tc for tc in msg["tool_calls"] 
                    if tc["id"] not in missing_responses
                ]
                if not assistant_msg["tool_calls"]:
                    del assistant_msg["tool_calls"]
                cleaned_messages.append(assistant_msg)
            else:
                cleaned_messages.append(msg)
        else:
            cleaned_messages.append(msg)
        i += 1
    
    return cleaned_messages


def get_llm_response_with_handling(prompt, npc, messages, tools, stream, team, context=None):
    """Unified LLM response with exception handling."""
    messages = clean_orphaned_tool_calls(messages)
    
    try:
        return get_llm_response(
            prompt=prompt,
            npc=npc,
            messages=messages,
            tools=tools,
            auto_process_tool_calls=False,
            stream=stream,
            team=team,
            context=context
        )
    except Timeout:
        return get_llm_response(
            prompt=prompt,
            npc=npc,
            messages=messages,
            tools=tools,
            auto_process_tool_calls=False,
            stream=stream,
            team=team
        )
    except ContextWindowExceededError:
        print('compressing..... ')
        compressed_state = npc.compress_planning_state(messages)
        compressed_messages = [{"role": "system", "content": compressed_state}]
        return get_llm_response(
            prompt=prompt,
            npc=npc,
            messages=compressed_messages,
            tools=tools,
            auto_process_tool_calls=False,
            stream=stream,
            team=team
        )
    except RateLimitError:
        import time
        print('rate limit hit... waiting 60 seconds')
        time.sleep(60)
        print('compressing..... ')
        compressed_state = npc.compress_planning_state(messages)
        compressed_messages = [{"role": "system", "content": compressed_state}]
        return get_llm_response(
            prompt=prompt,
            npc=npc,
            messages=compressed_messages,
            tools=tools,
            auto_process_tool_calls=False,
            stream=stream,
            team=team
        )
    except BadRequestError as e:
        if "tool_call_id" in str(e).lower():
            cleaned_messages = clean_orphaned_tool_calls(messages)
            return get_llm_response(
                prompt=prompt,
                npc=npc,
                messages=cleaned_messages,
                tools=tools,
                auto_process_tool_calls=False,
                stream=stream,
                team=team,
                context=context
            )
        else:
            raise e



def execute_command_corca(command: str, state: ShellState, command_history, selected_mcp_tools_names: Optional[List[str]] = None) -> Tuple[ShellState, Any]:
    mcp_tools_for_llm = []
    
    if hasattr(state, 'mcp_client') and state.mcp_client and state.mcp_client.session:
        all_available_mcp_tools = state.mcp_client.available_tools_llm
        
        if selected_mcp_tools_names and len(selected_mcp_tools_names) > 0:
            mcp_tools_for_llm = [
                tool_def for tool_def in all_available_mcp_tools
                if tool_def['function']['name'] in selected_mcp_tools_names
            ]
            if not mcp_tools_for_llm:
                cprint("Warning: No selected MCP tools found or matched. Corca will proceed without tools.", "yellow", file=sys.stderr)
        else:
            mcp_tools_for_llm = all_available_mcp_tools
    else:
        cprint("Warning: Corca agent has no tools. No MCP server connected.", "yellow", file=sys.stderr)

    if len(state.messages) > 20:
        compressed_state = state.npc.compress_planning_state(state.messages)
        state.messages = [{"role": "system", "content": state.npc.get_system_prompt() + f' Your current task: {compressed_state}'}]
        print("Compressed messages during tool execution.")
    
    response_dict = get_llm_response_with_handling(
        prompt=command,
        npc=state.npc,
        messages=state.messages,
        tools=mcp_tools_for_llm,
        stream=state.stream_output,
        team=state.team,
        context=f' The users working directory is {state.current_path}'
    )
         
    stream_response = response_dict.get('response')
    messages = response_dict.get('messages', state.messages)
    tool_calls = response_dict.get('tool_calls', [])
    
    collected_content, stream_tool_calls = process_mcp_stream(stream_response, state.npc)
    
    if stream_tool_calls:
        tool_calls = stream_tool_calls

    state.messages = messages
    
    if tool_calls and hasattr(state, 'mcp_client') and state.mcp_client:
        final_content, state.messages = execute_mcp_tool_calls(
            tool_calls, 
            state.mcp_client, 
            state.messages, 
            state.npc, 
            state.stream_output
        )
        if final_content:
            collected_content = final_content
    
    return state, {
        "output": collected_content,
        "tool_calls": tool_calls,
        "messages": state.messages
    }


def execute_mcp_tool_calls(tool_calls, mcp_client, messages, npc, stream_output):
    if not tool_calls or not mcp_client:
        return None, messages

    messages = clean_orphaned_tool_calls(messages)
    
    print(colored("\n🔧 Executing MCP tools...", "cyan"))
    
    while tool_calls:
        tool_responses = []

        if len(messages) > 20:
            compressed_state = npc.compress_planning_state(messages)
            messages = [{"role": "system", "content": npc.get_system_prompt() + f' Your current task: {compressed_state}'}]
            print("Compressed messages during tool execution.")
        
        
        for tool_call in tool_calls:
            tool_name = tool_call['function']['name']
            tool_args = tool_call['function']['arguments']
            tool_call_id = tool_call['id']
            
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args) if tool_args.strip() else {}
                except json.JSONDecodeError:
                    tool_args = {}
            
            try:
                print(f"  Calling MCP tool: {tool_name} with args: {tool_args}")
                
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                mcp_result = loop.run_until_complete(
                    mcp_client.session.call_tool(tool_name, tool_args)
                )
                
                tool_content = ""
                if hasattr(mcp_result, 'content') and mcp_result.content:
                    for content_item in mcp_result.content:
                        if hasattr(content_item, 'text'):
                            tool_content += content_item.text
                        elif hasattr(content_item, 'data'):
                            tool_content += str(content_item.data)
                        else:
                            tool_content += str(content_item)
                else:
                    tool_content = str(mcp_result)
                
                tool_responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_content
                })
                
                print(colored(f"  ✓ {tool_name} completed", "green"))
                
            except KeyboardInterrupt:
                print(colored(f"\n  ⚠️ Tool execution interrupted", "yellow"))
                return None, messages
            except Exception as e:
                print(colored(f"  ✗ {tool_name} failed: {e}", "red"))
                tool_responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": f"Error: {str(e)}"
                })
        
        current_messages = messages + tool_responses
        
        try:
            follow_up_response = get_llm_response_with_handling(
                prompt="",
                npc=npc,
                messages=current_messages,
                tools=mcp_client.available_tools_llm,
                stream=stream_output,
                team=None
            )
        except KeyboardInterrupt:
            print(colored(f"\n  ⚠️ Follow-up response interrupted", "yellow"))
            return None, messages
        
        follow_up_messages = follow_up_response.get('messages', current_messages)
        follow_up_content = follow_up_response.get('response', '')
        follow_up_tool_calls = []
        
        if stream_output:
            if hasattr(follow_up_content, '__iter__'):
                collected_content, follow_up_tool_calls = process_mcp_stream(follow_up_content, npc)
            else:
                collected_content = str(follow_up_content)
            follow_up_content = collected_content
        else:
            if follow_up_messages:
                last_message = follow_up_messages[-1]
                if last_message.get("role") == "assistant" and "tool_calls" in last_message:
                    follow_up_tool_calls = last_message["tool_calls"]
        
        messages = follow_up_messages
        
        if not follow_up_tool_calls:
            if not stream_output:
                print('\n')
                render_markdown(follow_up_content)
            return follow_up_content, messages
        else:
            if follow_up_content or follow_up_tool_calls:
                assistant_message = {"role": "assistant", "content": follow_up_content}
                if follow_up_tool_calls:
                    assistant_message["tool_calls"] = follow_up_tool_calls
                messages.append(assistant_message)
        
        tool_calls = follow_up_tool_calls
        print(colored("\n🔧 Executing follow-up MCP tools...", "cyan"))
    
    return None, messages

def _resolve_and_copy_mcp_server_path(
    explicit_path: Optional[str],
    current_path: Optional[str],
    team_ctx_mcp_servers: Optional[List[Dict[str, str]]],
    interactive: bool = False,
    auto_copy_bypass: bool = False
) -> Optional[str]:
    default_mcp_server_name = "mcp_server.py"
    npcsh_default_template_path = Path(__file__).parent / default_mcp_server_name

    def _copy_template_if_missing(destination_dir: Path, description: str) -> Optional[Path]:
        destination_file = destination_dir / default_mcp_server_name
        if not npcsh_default_template_path.exists():
            cprint(f"Error: Default {default_mcp_server_name} template not found at {npcsh_default_template_path}", "red")
            return None
        
        if not destination_file.exists():
            if auto_copy_bypass or not interactive:
                destination_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(npcsh_default_template_path, destination_file)
                print(colored(f"Automatically copied default {default_mcp_server_name} to {destination_file}", "green"))
                return destination_file
            else: 
                choice = input(colored(f"No {default_mcp_server_name} found in {description}. Copy default template to {destination_file}? (y/N): ", "yellow")).strip().lower()
                if choice == 'y':
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(npcsh_default_template_path, destination_file)
                    print(colored(f"Copied default {default_mcp_server_name} to {destination_file}", "green"))
                    return destination_file
                else:
                    print(colored("Skipping copy.", "yellow"))
                    return None
        return destination_file

    if explicit_path:
        abs_explicit_path = Path(explicit_path).expanduser().resolve()
        if abs_explicit_path.exists():
            print(f"Using explicit MCP server path: {abs_explicit_path}")
            return str(abs_explicit_path)
        else:
            cprint(f"Warning: Explicit MCP server path not found: {abs_explicit_path}", "yellow")

    if team_ctx_mcp_servers:
        for server_entry in team_ctx_mcp_servers:
            server_path_from_ctx = server_entry.get("value")
            if server_path_from_ctx:
                abs_ctx_path = Path(server_path_from_ctx).expanduser().resolve()
                if abs_ctx_path.exists():
                    print(f"Using MCP server path from team context: {abs_ctx_path}")
                    return str(abs_ctx_path)
                else:
                    cprint(f"Warning: MCP server path from team context not found: {abs_ctx_path}", "yellow")

    if current_path:
        project_npc_team_dir = Path(current_path).resolve() / "npc_team"
        project_mcp_server_file = project_npc_team_dir / default_mcp_server_name
        
        if project_mcp_server_file.exists():
            print(f"Using project-specific MCP server path: {project_mcp_server_file}")
            return str(project_mcp_server_file)
        else:
            copied_path = _copy_template_if_missing(project_npc_team_dir, "project's npc_team directory")
            if copied_path:
                return str(copied_path)

    global_npc_team_dir = Path.home() / ".npcsh" / "npc_team"
    global_mcp_server_file = global_npc_team_dir / default_mcp_server_name
    
    if global_mcp_server_file.exists():
        print(f"Using global MCP server path: {global_mcp_server_file}")
        return str(global_mcp_server_file)
    else:
        copied_path = _copy_template_if_missing(global_npc_team_dir, "global npc_team directory")
        if copied_path:
            return str(copied_path)
            
    cprint("No MCP server script found in any expected location.", "yellow")
    return None
def print_corca_welcome_message():
    turq = "\033[38;2;64;224;208m"
    chrome = "\033[38;2;211;211;211m"
    orange = "\033[38;2;255;165;0m"
    reset = "\033[0m"
    
    print(
        f"""
{turq} ██████    ██████    ██████    ██████    ██████{reset}
{turq}██    ██  ██    ██  ██    ██  ██    ██  ██🦌🦌██{reset}
{turq}██        ██    ██  ██    ██  ██        ██🦌🦌██{reset}
{chrome}██        ██    ██  ████████  ██        ████████{reset}
{chrome}██        ██    ██  ██  ███   ██        ██    ██{reset}
{chrome}██    ██  ██    ██  ██   ███  ██    ██  ██    ██{reset}
{orange} ██████    ██████   ██    ███  ███████  ██    ██{reset}

{chrome}                🦌 C O R C A 🦌{reset}
                        
    {turq}MCP-powered shell for agentic workflows{reset}
        """
    )    
    
def create_corca_state_and_mcp_client(conversation_id, command_history, npc=None, team=None,
                                     current_path=None, mcp_server_path_from_request: Optional[str] = None):
    from npcsh._state import ShellState
    
    state = ShellState(
        conversation_id=conversation_id,
        stream_output=True,
        current_mode="corca",
        chat_model=os.environ.get("NPCSH_CHAT_MODEL", "gemma3:4b"),
        chat_provider=os.environ.get("NPCSH_CHAT_PROVIDER", "ollama"),
        current_path=current_path or os.getcwd(),
        npc=npc,
        team=team
    )
    state.command_history = command_history
    
    team_ctx_mcp_servers = None
    if team and hasattr(team, 'team_path'):
        team_ctx = _load_team_context(team.team_path)
        team_ctx_mcp_servers = team_ctx.get('mcp_servers', [])
        
        if npc and isinstance(npc, NPC):
            if not npc.model and team_ctx.get('model'):
                npc.model = team_ctx['model']
            if not npc.provider and team_ctx.get('provider'):
                npc.provider = team_ctx['provider']
        
        if not state.chat_model and team_ctx.get('model'):
            state.chat_model = team_ctx['model']
        if not state.chat_provider and team_ctx.get('provider'):
            state.chat_provider = team_ctx['provider']
    
    auto_copy_bypass = os.getenv("NPCSH_CORCA_AUTO_COPY_MCP_SERVER", "false").lower() == "true"

    resolved_server_path = _resolve_and_copy_mcp_server_path(
        explicit_path=mcp_server_path_from_request,
        current_path=current_path,
        team_ctx_mcp_servers=team_ctx_mcp_servers,
        interactive=False,
        auto_copy_bypass=auto_copy_bypass,
        force_global=False
    )

    state.mcp_client = None
    if resolved_server_path:
        try:
            client_instance = MCPClientNPC()
            if client_instance.connect_sync(resolved_server_path):
                state.mcp_client = client_instance
                print(f"Successfully connected MCP client for {conversation_id} to {resolved_server_path}")
            else:
                print(f"Failed to connect MCP client for {conversation_id} to {resolved_server_path}. Tools will be unavailable.")
        except ImportError:
            print("WARNING: npcsh.corca or MCPClientNPC not found. Cannot initialize MCP client.", file=sys.stderr)
        except FileNotFoundError as e:
            print(f"MCP Client Error: {e}")
        except ValueError as e:
            print(f"MCP Client Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during MCP client initialization: {e}")
            traceback.print_exc()

    return state

                
def process_corca_result(
    user_input: str,
    result_state: ShellState,
    output: Any,
    command_history: CommandHistory,
):
    from npcpy.llm_funcs import get_facts
    from npcpy.memory.memory_processor import memory_approval_ui
    from npcsh._state import format_memory_context
    
    team_name = result_state.team.name if result_state.team else "__none__"
    npc_name = result_state.npc.name if isinstance(result_state.npc, NPC) else "__none__"
    
    active_npc = result_state.npc if isinstance(result_state.npc, NPC) else NPC(
        name="default", 
        model=result_state.chat_model, 
        provider=result_state.chat_provider, 
        db_conn=command_history.engine
    )
    
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

    output_content = output.get('output') if isinstance(output, dict) else output
    tool_calls = output.get('tool_calls', []) if isinstance(output, dict) else []
    final_output_str = None
    
    if tool_calls and hasattr(result_state, 'mcp_client') and result_state.mcp_client:
        final_output_str, result_state.messages = execute_mcp_tool_calls(
            tool_calls, 
            result_state.mcp_client, 
            result_state.messages, 
            result_state.npc, 
            result_state.stream_output
        )
    else:
        print('\n')
        if result_state.stream_output:
            final_output_str = print_and_process_stream_with_markdown(
                output_content, 
                result_state.npc.model, 
                result_state.npc.provider, 
                show=True
            )
        else:
            final_output_str = str(output_content)
            render_markdown(final_output_str)

    if final_output_str:
        if not result_state.messages or result_state.messages[-1].get("role") != "assistant" or result_state.messages[-1].get("content") != final_output_str:
            result_state.messages.append({"role": "assistant", "content": final_output_str})
        
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

        memory_examples = command_history.get_memory_examples_for_context(
            npc=npc_name,
            team=team_name, 
            directory_path=result_state.current_path
        )
        
        memory_context = format_memory_context(memory_examples)
        
        approved_facts = []
        try:
            facts = get_facts(
                conversation_turn_text,
                model=active_npc.model,
                provider=active_npc.provider,
                npc=active_npc,
                context=memory_context
            )
            
            if facts:
                memories_for_approval = []
                for i, fact in enumerate(facts):
                    memories_for_approval.append({
                        "memory_id": f"temp_{i}",
                        "content": fact['statement'],
                        "context": f"Type: {fact.get('type', 'unknown')}, Source: {fact.get('source_text', '')}",
                        "npc": npc_name,
                        "fact_data": fact
                    })
                
                approvals = memory_approval_ui(memories_for_approval)
                
                for approval in approvals:
                    fact_data = next(m['fact_data'] for m in memories_for_approval 
                                   if m['memory_id'] == approval['memory_id'])
                    
                    command_history.add_memory_to_database(
                        message_id=f"{result_state.conversation_id}_{len(result_state.messages)}",
                        conversation_id=result_state.conversation_id,
                        npc=npc_name,
                        team=team_name,
                        directory_path=result_state.current_path,
                        initial_memory=fact_data['statement'],
                        status=approval['decision'],
                        model=active_npc.model,
                        provider=active_npc.provider,
                        final_memory=approval.get('final_memory')
                    )
                    
                    if approval['decision'] in ['human-approved', 'human-edited']:
                        approved_fact = {
                            'statement': approval.get('final_memory') or fact_data['statement'],
                            'source_text': fact_data.get('source_text', ''),
                            'type': fact_data.get('type', 'explicit'),
                            'generation': 0
                        }
                        approved_facts.append(approved_fact)
                
        except Exception as e:
            print(colored(f"Memory generation error: {e}", "yellow"))

        if result_state.build_kg and approved_facts:
            try:
                if not should_skip_kg_processing(user_input, final_output_str):
                    npc_kg = load_kg_from_db(engine, team_name, npc_name, result_state.current_path)
                    evolved_npc_kg, _ = kg_evolve_incremental(
                        existing_kg=npc_kg, 
                        new_facts=approved_facts,
                        model=active_npc.model, 
                        provider=active_npc.provider, 
                        npc=active_npc,
                        get_concepts=True,
                        link_concepts_facts=False, 
                        link_concepts_concepts=False, 
                        link_facts_facts=False,                         
                    )
                    save_kg_to_db(
                        engine,
                        evolved_npc_kg, 
                        team_name, 
                        npc_name, 
                        result_state.current_path
                    )
            except Exception as e:
                print(colored(f"Error during real-time KG evolution: {e}", "red"))

        result_state.turn_count += 1

        if result_state.turn_count > 0 and result_state.turn_count % 10 == 0:
            print(colored("\nChecking for potential team improvements...", "cyan"))
            try:
                summary = breathe(messages=result_state.messages[-20:], 
                                npc=active_npc)
                characterization = summary.get('output')

                if characterization and result_state.team:
                    team_ctx_path = get_team_ctx_path(result_state.team.team_path)
                    if not team_ctx_path:
                        team_ctx_path = os.path.join(result_state.team.team_path, "team.ctx")
                    
                    ctx_data = _load_team_context(result_state.team.team_path)
                    current_context = ctx_data.get('context', '')

                    prompt = f"""Based on this characterization: {characterization},

                    suggest changes (additions, deletions, edits) to the team's context. 
                    Additions need not be fully formed sentences and can simply be equations, relationships, or other plain clear items.
                    
                    Current Context: "{current_context}". 
                    
                    Respond with JSON: """ + """
                    {
                    "suggestion": "Your sentence.
                    }
                    """
                    response = get_llm_response(prompt, 
                                        npc=active_npc, 
                                        format="json",
                                        team=result_state.team)   
                    suggestion = response.get("response", {}).get("suggestion")

                    if suggestion:
                        new_context = (current_context + " " + suggestion).strip()
                        print(colored(f"{result_state.npc.name} suggests updating team context:", "yellow"))
                        print(f"  - OLD: {current_context}\n  + NEW: {new_context}")
                        
                        choice = input("Apply? [y/N/e(dit)]: ").strip().lower()
                        
                        if choice == 'y':
                            ctx_data['context'] = new_context
                            with open(team_ctx_path, 'w') as f:
                                yaml.dump(ctx_data, f)
                            print(colored("Team context updated.", "green"))
                        elif choice == 'e':
                            edited_context = input(f"Edit context [{new_context}]: ").strip()
                            if edited_context:
                                ctx_data['context'] = edited_context
                            else:
                                ctx_data['context'] = new_context
                            with open(team_ctx_path, 'w') as f:
                                yaml.dump(ctx_data, f)
                            print(colored("Team context updated with edits.", "green"))
                        else:
                            print("Suggestion declined.")        
            except Exception as e:
                import traceback
                print(colored(f"Could not generate team suggestions: {e}", "yellow"))
                traceback.print_exc()                
                
                
def _read_npcsh_global_env() -> Dict[str, str]:
    global_env_file = Path(".npcsh_global")
    env_vars = {}
    if global_env_file.exists():
        try:
            with open(global_env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        except Exception as e:
            print(f"Warning: Could not read .npcsh_global: {e}")
    return env_vars

def _load_team_context(team_path: str) -> Dict[str, Any]:
    """Load team context from any .ctx file in the team directory"""
    ctx_path = get_team_ctx_path(team_path)
    if not ctx_path or not os.path.exists(ctx_path):
        return {}
    
    try:
        with open(ctx_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load team context from {ctx_path}: {e}")
        return {}


def _write_to_npcsh_global(key: str, value: str) -> None:
    global_env_file = Path(".npcsh_global")
    env_vars = _read_npcsh_global_env()
    env_vars[key] = value
    
    try:
        with open(global_env_file, 'w') as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
    except Exception as e:
        print(f"Warning: Could not write to .npcsh_global: {e}")


def _resolve_and_copy_mcp_server_path(
    explicit_path: Optional[str],
    current_path: Optional[str],
    team_ctx_mcp_servers: Optional[List[Dict[str, str]]],
    interactive: bool = False,
    auto_copy_bypass: bool = False,
    force_global: bool = False
) -> Optional[str]:
    default_mcp_server_name = "mcp_server.py"
    npcsh_default_template_path = Path(__file__).parent / default_mcp_server_name
    
    global_env = _read_npcsh_global_env()
    prefer_global = global_env.get("NPCSH_PREFER_GLOBAL_MCP_SERVER", "false").lower() == "true"

    def _copy_template_if_missing(destination_dir: Path, description: str) -> Optional[Path]:
        destination_file = destination_dir / default_mcp_server_name
        if not npcsh_default_template_path.exists():
            cprint(f"Error: Default {default_mcp_server_name} template not found at {npcsh_default_template_path}", "red")
            return None
        
        if not destination_file.exists():
            if auto_copy_bypass or not interactive:
                destination_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(npcsh_default_template_path, destination_file)
                print(colored(f"Automatically copied default {default_mcp_server_name} to {destination_file}", "green"))
                return destination_file
            else:
                choice = input(colored(f"No {default_mcp_server_name} found in {description}. Copy default template to {destination_file}? (y/N/g for global): ", "yellow")).strip().lower()
                if choice == 'y':
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(npcsh_default_template_path, destination_file)
                    print(colored(f"Copied default {default_mcp_server_name} to {destination_file}", "green"))
                    return destination_file
                elif choice == 'g':
                    _write_to_npcsh_global("NPCSH_PREFER_GLOBAL_MCP_SERVER", "true")
                    print(colored("Set preference to use global MCP server.", "green"))
                    return None
                else:
                    print(colored("Skipping copy.", "yellow"))
                    return None
        return destination_file

    if explicit_path:
        abs_explicit_path = Path(explicit_path).expanduser().resolve()
        if abs_explicit_path.exists():
            print(f"Using explicit MCP server path: {abs_explicit_path}")
            return str(abs_explicit_path)
        else:
            cprint(f"Warning: Explicit MCP server path not found: {abs_explicit_path}", "yellow")

    if team_ctx_mcp_servers:
        for server_entry in team_ctx_mcp_servers:
            server_path_from_ctx = server_entry.get("value")
            if server_path_from_ctx:
                abs_ctx_path = Path(server_path_from_ctx).expanduser().resolve()
                if abs_ctx_path.exists():
                    print(f"Using MCP server path from team context: {abs_ctx_path}")
                    return str(abs_ctx_path)
                else:
                    cprint(f"Warning: MCP server path from team context not found: {abs_ctx_path}", "yellow")

    if not (force_global or prefer_global):
        if current_path:
            project_npc_team_dir = Path(current_path).resolve() / "npc_team"
            project_mcp_server_file = project_npc_team_dir / default_mcp_server_name
            
            if project_mcp_server_file.exists():
                print(f"Using project-specific MCP server path: {project_mcp_server_file}")
                return str(project_mcp_server_file)
            else:
                copied_path = _copy_template_if_missing(project_npc_team_dir, "project's npc_team directory")
                if copied_path:
                    return str(copied_path)

    global_npc_team_dir = Path.home() / ".npcsh" / "npc_team"
    global_mcp_server_file = global_npc_team_dir / default_mcp_server_name
    
    if global_mcp_server_file.exists():
        print(f"Using global MCP server path: {global_mcp_server_file}")
        return str(global_mcp_server_file)
    else:
        copied_path = _copy_template_if_missing(global_npc_team_dir, "global npc_team directory")
        if copied_path:
            return str(copied_path)
            
    cprint("No MCP server script found in any expected location.", "yellow")
    return None
def create_corca_state_and_mcp_client(conversation_id, 
                                      command_history, 
                                      npc=None, 
                                      team=None,
                                      current_path=None, 
                                      mcp_server_path: Optional[str] = None):
    from npcsh._state import ShellState
    
    state = ShellState(
        conversation_id=conversation_id,
        stream_output=True,
        current_mode="corca",
        chat_model=os.environ.get("NPCSH_CHAT_MODEL", "gemma3:4b"),
        chat_provider=os.environ.get("NPCSH_CHAT_PROVIDER", "ollama"),
        current_path=current_path or os.getcwd(),
        npc=npc,
        team=team
    )
    state.command_history = command_history
    
    auto_copy_bypass = os.getenv("NPCSH_CORCA_AUTO_COPY_MCP_SERVER", "false").lower() == "true"

    resolved_server_path = _resolve_and_copy_mcp_server_path(
        explicit_path=mcp_server_path,
        current_path=current_path,
        team_ctx_mcp_servers=team.team_ctx.get('mcp_servers', []) if team and hasattr(team, 'team_ctx') else None,
        interactive=False,
        auto_copy_bypass=auto_copy_bypass,
        force_global=False
    )

    state.mcp_client = None
    if resolved_server_path:
        try:
            client_instance = MCPClientNPC()
            if client_instance.connect_sync(resolved_server_path):
                state.mcp_client = client_instance
                print(f"Successfully connected MCP client for {conversation_id} to {resolved_server_path}")
            else:
                print(f"Failed to connect MCP client for {conversation_id} to {resolved_server_path}. Tools will be unavailable.")
        except ImportError:
            print("WARNING: npcsh.corca or MCPClientNPC not found. Cannot initialize MCP client.", file=sys.stderr)
        except FileNotFoundError as e:
            print(f"MCP Client Error: {e}")
        except ValueError as e:
            print(f"MCP Client Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during MCP client initialization: {e}")
            traceback.print_exc()

    return state

def enter_corca_mode(command: str, **kwargs):
    state: ShellState = kwargs.get('shell_state')
    command_history: CommandHistory = kwargs.get('command_history')

    if not state or not command_history:
        return {"output": "Error: Corca mode requires shell state and history.", "messages": kwargs.get('messages', [])}

    all_command_parts = shlex.split(command)
    parser = argparse.ArgumentParser(prog="/corca", description="Enter Corca MCP-powered mode.")
    parser.add_argument("--mcp-server-path", type=str, help="Path to an MCP server script.")
    parser.add_argument("-g", "--global", dest="force_global", action="store_true", help="Force use of global MCP server.")
    
    try:
        known_args, remaining_args = parser.parse_known_args(all_command_parts[1:])
    except SystemExit:
         return {"output": "Invalid arguments for /corca. See /help corca.", "messages": state.messages}

    print_corca_welcome_message()
    
    auto_copy_bypass = os.getenv("NPCSH_CORCA_AUTO_COPY_MCP_SERVER", "false").lower() == "true"

    resolved_server_path = _resolve_and_copy_mcp_server_path(
        explicit_path=known_args.mcp_server_path,
        current_path=state.current_path,
        team_ctx_mcp_servers=state.team.team_ctx.get('mcp_servers', []) if state.team and hasattr(state.team, 'team_ctx') else None,
        interactive=True,
        auto_copy_bypass=auto_copy_bypass,
        force_global=known_args.force_global
    )

    mcp_client = None
    if resolved_server_path:
        try:
            mcp_client = MCPClientNPC()
            if mcp_client.connect_sync(resolved_server_path):
                state.mcp_client = mcp_client
            else:
                cprint(f"Failed to connect to MCP server at {resolved_server_path}. Corca mode will have limited agent functionality.", "yellow")
                state.mcp_client = None
        except Exception as e:
            cprint(f"Error connecting to MCP server: {e}. Corca mode will have limited agent functionality.", "red")
            traceback.print_exc()
            state.mcp_client = None
    else:
        cprint("No MCP server path provided or found. Corca mode will have limited agent functionality.", "yellow")
        state.mcp_client = None

    while True:
        try:
            prompt_npc_name = "npc"
            if state.npc:
                prompt_npc_name = state.npc.name
            
            prompt_str = f"{colored(os.path.basename(state.current_path), 'blue')}:{prompt_npc_name}🦌> "
            prompt = readline_safe_prompt(prompt_str)
            
            if remaining_args:
                user_input = " ".join(remaining_args)
                remaining_args = []
            else:
                user_input = get_multiline_input(prompt).strip()
            
            if user_input.lower() in ["exit", "quit", "done"]:
                break
            
            if not user_input:
                continue
            try:
                state, output = execute_command_corca(user_input, state, command_history)
            
                process_corca_result(user_input, 
                            state, 
                            output, 
                            command_history, 
                                )
            except Exception as e:
                print(f'An Exception has occurred {e}')   
                         
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
    parser.add_argument("-g", "--global", dest="force_global", action="store_true", help="Force use of global MCP server.")
    args = parser.parse_args()

    command_history, team, default_npc = setup_shell()
    
    project_team_path = os.path.abspath('./npc_team/')
    global_team_path = os.path.expanduser('~/.npcsh/npc_team/')
    
    project_corca_path = os.path.join(project_team_path, "corca.npc")
    global_corca_path = os.path.join(global_team_path, "corca.npc")
    
    if os.path.exists(project_corca_path):
        default_npc = NPC(file=project_corca_path, 
                          db_conn=command_history.engine)
    elif os.path.exists(global_corca_path):
        default_npc = NPC(file=global_corca_path, 
                          db_conn=command_history.engine)

    if default_npc.model is None:
        if team.model is not None:
            default_npc.model = team.model
        else:
            default_npc.model = NPCSH_CHAT_MODEL

    if default_npc.provider is None:
        if team.provider is not None:
            default_npc.provider = team.provider
        else:
            default_npc.provider = NPCSH_CHAT_PROVIDER

    from npcsh._state import initial_state
    initial_shell_state = initial_state
    initial_shell_state.team = team
    initial_shell_state.npc = default_npc
    
    fake_command_str = "/corca"
    if args.mcp_server_path:
        fake_command_str = f'/corca --mcp-server-path "{args.mcp_server_path}"'
    elif args.force_global:
        fake_command_str = "/corca --global"
        
    kwargs = {
        'command': fake_command_str,
        'shell_state': initial_shell_state,
        'command_history': command_history
    }
    
    enter_corca_mode(**kwargs)

if __name__ == "__main__":
    main()
