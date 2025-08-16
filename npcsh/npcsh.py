import os
import sys
import atexit
import subprocess
import shlex
import re
from datetime import datetime
import argparse
import importlib.metadata
import textwrap
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import platform
try:
    from termcolor import colored
except: 
    pass

try:
    import chromadb
except ImportError:
    chromadb = None
import shutil
import json
import sqlite3
import copy
import yaml

# Local Application Imports
from npcsh._state import (
    setup_npcsh_config,
    initial_state, 
    is_npcsh_initialized,
    initialize_base_npcs_if_needed,
    orange,
    ShellState,
    interactive_commands,
    BASH_COMMANDS,
    start_interactive_session,
    validate_bash_command
    )

from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    render_markdown,
    get_locally_available_models,
    get_model_and_provider,
)
from npcsh.routes import router
from npcpy.data.image import capture_screenshot
from npcpy.memory.command_history import (
    CommandHistory,
    save_conversation_message,
    load_kg_from_db, 
    save_kg_to_db, 
)
from npcpy.npc_compiler import NPC, Team, load_jinxs_from_directory
from npcpy.llm_funcs import (
    check_llm_command,
    get_llm_response,
    execute_llm_command,
    breathe
)
from npcpy.memory.knowledge_graph import (
    kg_initial,
    kg_evolve_incremental
)
from npcpy.gen.embeddings import get_embeddings

try:
    import readline
except:
    print('no readline support, some features may not work as desired. ')
# --- Constants ---
try:
    VERSION = importlib.metadata.version("npcpy")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"

TERMINAL_EDITORS = ["vim", "emacs", "nano"]
EMBEDDINGS_DB_PATH = os.path.expanduser("~/npcsh_chroma.db")
HISTORY_DB_DEFAULT_PATH = os.path.expanduser("~/npcsh_history.db")
READLINE_HISTORY_FILE = os.path.expanduser("~/.npcsh_readline_history")
DEFAULT_NPC_TEAM_PATH = os.path.expanduser("~/.npcsh/npc_team/")
PROJECT_NPC_TEAM_PATH = "./npc_team/"

# --- Global Clients ---
try:
    chroma_client = chromadb.PersistentClient(path=EMBEDDINGS_DB_PATH) if chromadb else None
except Exception as e:
    print(f"Warning: Failed to initialize ChromaDB client at {EMBEDDINGS_DB_PATH}: {e}")
    chroma_client = None




def get_path_executables() -> List[str]:
    """Get executables from PATH (cached for performance)"""
    if not hasattr(get_path_executables, '_cache'):
        executables = set()
        path_dirs = os.environ.get('PATH', '').split(os.pathsep)
        for path_dir in path_dirs:
            if os.path.isdir(path_dir):
                try:
                    for item in os.listdir(path_dir):
                        item_path = os.path.join(path_dir, item)
                        if os.path.isfile(item_path) and os.access(item_path, os.X_OK):
                            executables.add(item)
                except (PermissionError, OSError):
                    continue
        get_path_executables._cache = sorted(list(executables))
    return get_path_executables._cache


import logging

# Set up completion logger
completion_logger = logging.getLogger('npcsh.completion')
completion_logger.setLevel(logging.WARNING)  # Default to WARNING (quiet)

# Add handler if not already present
if not completion_logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('[%(name)s] %(message)s')
    handler.setFormatter(formatter)
    completion_logger.addHandler(handler)

def make_completer(shell_state: ShellState):
    def complete(text: str, state_index: int) -> Optional[str]:
        """Main completion function"""
        try:
            buffer = readline.get_line_buffer()
            begidx = readline.get_begidx()
            endidx = readline.get_endidx()
            
            completion_logger.debug(f"text='{text}', buffer='{buffer}', begidx={begidx}, endidx={endidx}, state_index={state_index}")
            
            matches = []
            
            # Check if we're completing a slash command
            if begidx > 0 and buffer[begidx-1] == '/':
                completion_logger.debug(f"Slash command completion - text='{text}'")
                slash_commands = get_slash_commands(shell_state)
                completion_logger.debug(f"Available slash commands: {slash_commands}")
                
                if text == '':
                    matches = [cmd[1:] for cmd in slash_commands]
                else:
                    full_text = '/' + text
                    matching_commands = [cmd for cmd in slash_commands if cmd.startswith(full_text)]
                    matches = [cmd[1:] for cmd in matching_commands]
                
                completion_logger.debug(f"Slash command matches: {matches}")
                
            elif is_command_position(buffer, begidx):
                completion_logger.debug("Command position detected")
                bash_matches = [cmd for cmd in BASH_COMMANDS if cmd.startswith(text)]
                matches.extend(bash_matches)
                
                interactive_matches = [cmd for cmd in interactive_commands.keys() if cmd.startswith(text)]
                matches.extend(interactive_matches)
                
                if len(text) >= 1:
                    path_executables = get_path_executables()
                    exec_matches = [cmd for cmd in path_executables if cmd.startswith(text)]
                    matches.extend(exec_matches[:20])
            else:
                completion_logger.debug("File completion")
                matches = get_file_completions(text)
            
            matches = sorted(list(set(matches)))
            completion_logger.debug(f"Final matches: {matches}")
            
            if state_index < len(matches):
                result = matches[state_index]
                completion_logger.debug(f"Returning: '{result}'")
                return result
            else:
                completion_logger.debug(f"No match for state_index {state_index}")
            
        except Exception as e:
            completion_logger.error(f"Exception in completion: {e}")
            completion_logger.debug("Exception details:", exc_info=True)
        
        return None
    
    return complete

def get_slash_commands(state: ShellState) -> List[str]:
    """Get available slash commands from router and team"""
    commands = []
    
    completion_logger.debug("Getting slash commands...")
    
    # Router commands
    if router and hasattr(router, 'routes'):
        router_cmds = [f"/{cmd}" for cmd in router.routes.keys()]
        commands.extend(router_cmds)
        completion_logger.debug(f"Router commands: {router_cmds}")
    
    # Team jinxs
    if state.team and hasattr(state.team, 'jinxs_dict'):
        jinx_cmds = [f"/{jinx}" for jinx in state.team.jinxs_dict.keys()]
        commands.extend(jinx_cmds)
        completion_logger.debug(f"Jinx commands: {jinx_cmds}")
    
    # NPC names for switching
    if state.team and hasattr(state.team, 'npcs'):
        npc_cmds = [f"/{npc}" for npc in state.team.npcs.keys()]
        commands.extend(npc_cmds)
        completion_logger.debug(f"NPC commands: {npc_cmds}")
    
    # Mode switching commands
    mode_cmds = ['/cmd', '/agent', '/chat']
    commands.extend(mode_cmds)
    completion_logger.debug(f"Mode commands: {mode_cmds}")
    
    result = sorted(commands)
    completion_logger.debug(f"Final slash commands: {result}")
    return result
def get_file_completions(text: str) -> List[str]:
    """Get file/directory completions"""
    try:
        if text.startswith('/'):
            basedir = os.path.dirname(text) or '/'
            prefix = os.path.basename(text)
        elif text.startswith('./') or text.startswith('../'):
            basedir = os.path.dirname(text) or '.'
            prefix = os.path.basename(text)
        else:
            basedir = '.'
            prefix = text
        
        if not os.path.exists(basedir):
            return []
        
        matches = []
        try:
            for item in os.listdir(basedir):
                if item.startswith(prefix):
                    full_path = os.path.join(basedir, item)
                    if basedir == '.':
                        completion = item
                    else:
                        completion = os.path.join(basedir, item)
                    
                    # Just return the name, let readline handle spacing/slashes
                    matches.append(completion)
        except (PermissionError, OSError):
            pass
        
        return sorted(matches)
    except Exception:
        return []
def is_command_position(buffer: str, begidx: int) -> bool:
    """Determine if cursor is at a command position"""
    # Get the part of buffer before the current word
    before_word = buffer[:begidx]
    
    # Split by command separators 
    parts = re.split(r'[|;&]', before_word)
    current_command_part = parts[-1].strip()
    
    # If there's nothing before the current word in this command part,
    # or only whitespace, we're at command position
    return len(current_command_part) == 0


def readline_safe_prompt(prompt: str) -> str:
    ansi_escape = re.compile(r"(\033\[[0-9;]*[a-zA-Z])")
    return ansi_escape.sub(r"\001\1\002", prompt)

def print_jinxs(jinxs):
    output = "Available jinxs:\n"
    for jinx in jinxs:
        output += f"  {jinx.jinx_name}\n"
        output += f"   Description: {jinx.description}\n"
        output += f"   Inputs: {jinx.inputs}\n"
    return output

def open_terminal_editor(command: str) -> str:
    try:
        os.system(command)
        return 'Terminal editor closed.'
    except Exception as e:
        return f"Error opening terminal editor: {e}"

def get_multiline_input(prompt: str) -> str:
    lines = []
    current_prompt = prompt
    while True:
        try:
            line = input(current_prompt)
            if line.endswith("\\"):
                lines.append(line[:-1])
                current_prompt = readline_safe_prompt("> ")
            else:
                lines.append(line)
                break
        except EOFError:
            print("Goodbye!")
            sys.exit(0)
    return "\n".join(lines)

def split_by_pipes(command: str) -> List[str]:
    parts = []
    current = ""
    in_single_quote = False
    in_double_quote = False
    escape = False

    for char in command:
        if escape:
            current += char
            escape = False
        elif char == '\\':
            escape = True
            current += char
        elif char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current += char
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_single_quote
            current += char
        elif char == '|' and not in_single_quote and not in_double_quote:
            parts.append(current.strip())
            current = ""
        else:
            current += char

    if current:
        parts.append(current.strip())
    return parts

def parse_command_safely(cmd: str) -> List[str]:
    try:
        return shlex.split(cmd)
    except ValueError as e:
        if "No closing quotation" in str(e):
            if cmd.count('"') % 2 == 1:
                cmd += '"'
            elif cmd.count("'") % 2 == 1:
                cmd += "'"
            try:
                return shlex.split(cmd)
            except ValueError:
                return cmd.split()
        else:
            return cmd.split()

def get_file_color(filepath: str) -> tuple:
    if not os.path.exists(filepath):
         return "grey", []
    if os.path.isdir(filepath):
        return "blue", ["bold"]
    elif os.access(filepath, os.X_OK) and not os.path.isdir(filepath):
        return "green", ["bold"]
    elif filepath.endswith((".zip", ".tar", ".gz", ".bz2", ".xz", ".7z")):
        return "red", []
    elif filepath.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")):
        return "magenta", []
    elif filepath.endswith((".py", ".pyw")):
        return "yellow", []
    elif filepath.endswith((".sh", ".bash", ".zsh")):
        return "green", []
    elif filepath.endswith((".c", ".cpp", ".h", ".hpp")):
        return "cyan", []
    elif filepath.endswith((".js", ".ts", ".jsx", ".tsx")):
        return "yellow", []
    elif filepath.endswith((".html", ".css", ".scss", ".sass")):
        return "magenta", []
    elif filepath.endswith((".md", ".txt", ".log")):
        return "white", []
    elif os.path.basename(filepath).startswith("."):
        return "cyan", []
    else:
        return "white", []

def format_file_listing(output: str) -> str:
    colored_lines = []
    current_dir = os.getcwd()
    for line in output.strip().split("\n"):
        parts = line.split()
        if not parts:
            colored_lines.append(line)
            continue

        filepath_guess = parts[-1]
        potential_path = os.path.join(current_dir, filepath_guess)

        color, attrs = get_file_color(potential_path)
        colored_filepath = colored(filepath_guess, color, attrs=attrs)

        if len(parts) > 1 :
             # Handle cases like 'ls -l' where filename is last
             colored_line = " ".join(parts[:-1] + [colored_filepath])
        else:
             # Handle cases where line is just the filename
             colored_line = colored_filepath

        colored_lines.append(colored_line)

    return "\n".join(colored_lines)

def wrap_text(text: str, width: int = 80) -> str:
    lines = []
    for paragraph in text.split("\n"):
        if len(paragraph) > width:
             lines.extend(textwrap.wrap(paragraph, width=width, replace_whitespace=False, drop_whitespace=False))
        else:
             lines.append(paragraph)
    return "\n".join(lines)

# --- Readline Setup and Completion ---

def setup_readline() -> str:
    """Setup readline with history and completion"""
    try:
        readline.read_history_file(READLINE_HISTORY_FILE)
        readline.set_history_length(1000)
        
        # Don't set completer here - it will be set in run_repl with state
        readline.parse_and_bind("tab: complete")
        
        readline.parse_and_bind("set enable-bracketed-paste on")
        readline.parse_and_bind(r'"\C-r": reverse-search-history')
        readline.parse_and_bind(r'"\C-e": end-of-line')
        readline.parse_and_bind(r'"\C-a": beginning-of-line')
        
        return READLINE_HISTORY_FILE
        
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"Warning: Could not read readline history file {READLINE_HISTORY_FILE}: {e}")


def save_readline_history():
    try:
        readline.write_history_file(READLINE_HISTORY_FILE)
    except OSError as e:
        print(f"Warning: Could not write readline history file {READLINE_HISTORY_FILE}: {e}")




valid_commands_list = list(router.routes.keys()) + list(interactive_commands.keys()) + ["cd", "exit", "quit"] + BASH_COMMANDS




# --- Command Execution Logic ---

def store_command_embeddings(command: str, output: Any, state: ShellState):
    if not chroma_client or not state.embedding_model or not state.embedding_provider:
        if not chroma_client: print("Warning: ChromaDB client not available for embeddings.", file=sys.stderr)
        return
    if not command and not output:
        return

    try:
        output_str = str(output) if output else ""
        if not command and not output_str: return # Avoid empty embeddings

        texts_to_embed = [command, output_str]

        embeddings = get_embeddings(
            texts_to_embed,
            state.embedding_model,
            state.embedding_provider,
        )

        if not embeddings or len(embeddings) != 2:
             print(f"Warning: Failed to generate embeddings for command: {command[:50]}...", file=sys.stderr)
             return

        timestamp = datetime.now().isoformat()
        npc_name = state.npc.name if isinstance(state.npc, NPC) else state.npc

        metadata = [
            {
                "type": "command", "timestamp": timestamp, "path": state.current_path,
                "npc": npc_name, "conversation_id": state.conversation_id,
            },
            {
                "type": "response", "timestamp": timestamp, "path": state.current_path,
                "npc": npc_name, "conversation_id": state.conversation_id,
            },
        ]

        collection_name = f"{state.embedding_provider}_{state.embedding_model}_embeddings"
        try:
            collection = chroma_client.get_or_create_collection(collection_name)
            ids = [f"cmd_{timestamp}_{hash(command)}", f"resp_{timestamp}_{hash(output_str)}"]

            collection.add(
                embeddings=embeddings,
                documents=texts_to_embed,
                metadatas=metadata,
                ids=ids,
            )
        except Exception as e:
            print(f"Warning: Failed to add embeddings to collection '{collection_name}': {e}", file=sys.stderr)

    except Exception as e:
        print(f"Warning: Failed to store embeddings: {e}", file=sys.stderr)


def handle_interactive_command(cmd_parts: List[str], state: ShellState) -> Tuple[ShellState, str]:
    command_name = cmd_parts[0]
    print(f"Starting interactive {command_name} session...")
    try:
        return_code = start_interactive_session(
            interactive_commands[command_name], cmd_parts[1:]
        )
        output = f"Interactive {command_name} session ended with return code {return_code}"
    except Exception as e:
        output = f"Error starting interactive session {command_name}: {e}"
    return state, output

def handle_cd_command(cmd_parts: List[str], state: ShellState) -> Tuple[ShellState, str]:
    original_path = os.getcwd()
    target_path = cmd_parts[1] if len(cmd_parts) > 1 else os.path.expanduser("~")
    try:
        os.chdir(target_path)
        state.current_path = os.getcwd()
        output = f"Changed directory to {state.current_path}"
    except FileNotFoundError:
        output = colored(f"cd: no such file or directory: {target_path}", "red")
    except Exception as e:
        output = colored(f"cd: error changing directory: {e}", "red")
        os.chdir(original_path) # Revert if error

    return state, output


def handle_bash_command(
    cmd_parts: List[str],
    cmd_str: str,
    stdin_input: Optional[str],
    state: ShellState,
) -> Tuple[bool, str]:
    try:
        process = subprocess.Popen(
            cmd_parts,
            stdin=subprocess.PIPE if stdin_input is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=state.current_path
        )
        stdout, stderr = process.communicate(input=stdin_input)

        if process.returncode != 0:
            return False, stderr.strip() if stderr else f"Command '{cmd_str}' failed with return code {process.returncode}."

        if stderr.strip():
            print(colored(f"stderr: {stderr.strip()}", "yellow"), file=sys.stderr)
        
        if cmd_parts[0] in ["ls", "find", "dir"]:
            return True, format_file_listing(stdout.strip())

        return True, stdout.strip()

    except FileNotFoundError:
        return False, f"Command not found: {cmd_parts[0]}"
    except PermissionError:
        return False, f"Permission denied: {cmd_str}"

def execute_slash_command(command: str, stdin_input: Optional[str], state: ShellState, stream: bool) -> Tuple[ShellState, Any]:
    """Executes slash commands using the router or checking NPC/Team jinxs."""
    command_parts = command.split()
    command_name = command_parts[0].lstrip('/')

    if command_name in ['n', 'npc']:
        npc_to_switch_to = command_parts[1] if len(command_parts) > 1 else None
        if npc_to_switch_to and state.team and npc_to_switch_to in state.team.npcs:
            state.npc = state.team.npcs[npc_to_switch_to]
            return state, f"Switched to NPC: {npc_to_switch_to}"
        else:
            available_npcs = list(state.team.npcs.keys()) if state.team else []
            return state, colored(f"NPC '{npc_to_switch_to}' not found. Available NPCs: {', '.join(available_npcs)}", "red")
    handler = router.get_route(command_name)
    #print(handler)
    if handler:
        # Prepare kwargs for the handler
        handler_kwargs = {
            'stream': stream,
            'npc': state.npc, 
            'team': state.team,
            'messages': state.messages,
            'model': state.chat_model, 
            'provider': state.chat_provider,
            'api_url': state.api_url,
            'api_key': state.api_key,
        }
        #print(handler_kwargs, command)
        if stdin_input is not None:
            handler_kwargs['stdin_input'] = stdin_input

        try:
            result_dict = handler(command, **handler_kwargs)

            if isinstance(result_dict, dict):
                state.messages = result_dict.get("messages", state.messages)
                return state, result_dict
            else:
                return state, result_dict

        except Exception as e:
            import traceback
            print(f"Error executing slash command '{command_name}':", file=sys.stderr)
            traceback.print_exc()
            return state, colored(f"Error executing slash command '{command_name}': {e}", "red")

    active_npc = state.npc if isinstance(state.npc, NPC) else None
    jinx_to_execute = None
    executor = None
    if active_npc and command_name in active_npc.jinxs_dict:
        jinx_to_execute = active_npc.jinxs_dict[command_name]
        executor = active_npc
    elif state.team and command_name in state.team.jinxs_dict:
        jinx_to_execute = state.team.jinxs_dict[command_name]
        executor = state.team

    if jinx_to_execute:
        args = command_parts[1:]
        try:
            jinx_output = jinx_to_execute.run(
                *args,
                state=state,
                stdin_input=stdin_input,
                messages=state.messages # Pass messages explicitly if needed
            )
            return state, jinx_output
        except Exception as e:
            import traceback
            print(f"Error executing jinx '{command_name}':", file=sys.stderr)
            traceback.print_exc()
            return state, colored(f"Error executing jinx '{command_name}': {e}", "red")

    if state.team and command_name in state.team.npcs:
        new_npc = state.team.npcs[command_name]
        state.npc = new_npc # Update state directly
        return state, f"Switched to NPC: {new_npc.name}"

    return state, colored(f"Unknown slash command or jinx: {command_name}", "red")

def process_pipeline_command(
    cmd_segment: str,
    stdin_input: Optional[str],
    state: ShellState,
    stream_final: bool
    ) -> Tuple[ShellState, Any]:

    if not cmd_segment:
        return state, stdin_input

    available_models_all = get_locally_available_models(state.current_path)
    available_models_all_list = [item for key, item in available_models_all.items()]
    model_override, provider_override, cmd_cleaned = get_model_and_provider(
        cmd_segment, available_models_all_list
    )
    cmd_to_process = cmd_cleaned.strip()
    if not cmd_to_process:
         return state, stdin_input

    # Priority: 1. Inline Override, 2. NPC Model, 3. Global Model
    npc_model = state.npc.model if isinstance(state.npc, NPC) and state.npc.model else None
    npc_provider = state.npc.provider if isinstance(state.npc, NPC) and state.npc.provider else None

    exec_model = model_override or npc_model or state.chat_model
    exec_provider = provider_override or npc_provider or state.chat_provider

    if cmd_to_process.startswith("/"):
        return execute_slash_command(cmd_to_process, stdin_input, state, stream_final)
    
    cmd_parts = parse_command_safely(cmd_to_process)
    if not cmd_parts:
        return state, stdin_input

    if validate_bash_command(cmd_parts):
        command_name = cmd_parts[0]
        if command_name in interactive_commands:
            return handle_interactive_command(cmd_parts, state)
        if command_name == "cd":
            return handle_cd_command(cmd_parts, state)

        success, result = handle_bash_command(cmd_parts, cmd_to_process, stdin_input, state)
        if success:
            return state, result
        else:
            print(colored(f"Bash command failed. Asking LLM for a fix: {result}", "yellow"), file=sys.stderr)
            fixer_prompt = f"The command '{cmd_to_process}' failed with the error: '{result}'. Provide the correct command."
            response = execute_llm_command(
                fixer_prompt, 
                model=exec_model,      # Uses corrected model
                provider=exec_provider, # Uses corrected provider
                npc=state.npc, 
                stream=stream_final, 
                messages=state.messages
            )
            state.messages = response['messages']     
            return state, response['response']
    else:
        full_llm_cmd = f"{cmd_to_process} {stdin_input}" if stdin_input else cmd_to_process
        path_cmd = 'The current working directory is: ' + state.current_path
        ls_files = 'Files in the current directory (full paths):\n' + "\n".join([os.path.join(state.current_path, f) for f in os.listdir(state.current_path)]) if os.path.exists(state.current_path) else 'No files found in the current directory.'
        platform_info = f"Platform: {platform.system()} {platform.release()} ({platform.machine()})"
        info = path_cmd + '\n' + ls_files + '\n' + platform_info + '\n' 

        llm_result = check_llm_command(
            full_llm_cmd,
            model=exec_model,      
            provider=exec_provider, 
            api_url=state.api_url,
            api_key=state.api_key,
            npc=state.npc,
            team=state.team,
            messages=state.messages,
            images=state.attachments,
            stream=stream_final,
            context=info,

        )
        if isinstance(llm_result, dict):
            state.messages = llm_result.get("messages", state.messages)
            output = llm_result.get("output")
            return state, output
        else:
            return state, llm_result
        
def check_mode_switch(command:str , state: ShellState):
    if command in ['/cmd', '/agent', '/chat',]:
        state.current_mode = command[1:]
        return True, state     

    return False, state
def execute_command(
    command: str,
    state: ShellState,
    ) -> Tuple[ShellState, Any]:

    if not command.strip():
        return state, ""
    mode_change, state = check_mode_switch(command, state)
    if mode_change:
        return state, 'Mode changed.'

    original_command_for_embedding = command
    commands = split_by_pipes(command)
    stdin_for_next = None
    final_output = None
    current_state = state 
    npc_model = state.npc.model if isinstance(state.npc, NPC) and state.npc.model else None
    npc_provider = state.npc.provider if isinstance(state.npc, NPC) and state.npc.provider else None
    active_model = npc_model or state.chat_model
    active_provider = npc_provider or state.chat_provider

    if state.current_mode == 'agent':
        for i, cmd_segment in enumerate(commands):
            print(len(commands))
            is_last_command = (i == len(commands) - 1)
            stream_this_segment = is_last_command and state.stream_output # Use state's stream setting

            try:
                current_state, output = process_pipeline_command(
                    cmd_segment.strip(),
                    stdin_for_next,
                    current_state, 
                    stream_final=stream_this_segment
                )

                if is_last_command:
                    final_output = output # Capture the output of the last command

                if isinstance(output, str):
                    stdin_for_next = output
                elif not isinstance(output, str):
                    try:
                        full_stream_output = print_and_process_stream_with_markdown(output, 
                                                                                    state.npc.model, 
                                                                                    state.npc.provider)
                        stdin_for_next = full_stream_output
                        if is_last_command: 
                            final_output = full_stream_output
                    except:
                        if output is not None: # Try converting other types to string
                            try: 
                                stdin_for_next = str(output)
                            except Exception:
                                print(f"Warning: Cannot convert output to string for piping: {type(output)}", file=sys.stderr)
                                stdin_for_next = None
                        else: # Output was None
                            stdin_for_next = None


            except Exception as pipeline_error:
                import traceback
                traceback.print_exc()
                error_msg = colored(f"Error in pipeline stage {i+1} ('{cmd_segment[:50]}...'): {pipeline_error}", "red")
                # Return the state as it was when the error occurred, and the error message
                return current_state, error_msg

        # Store embeddings using the final state
        if final_output is not None and isinstance(final_output,str):
            store_command_embeddings(original_command_for_embedding, final_output, current_state)

        # Return the final state and the final output
        return current_state, final_output


    elif state.current_mode == 'chat':
        # Only treat as bash if it looks like a shell command (starts with known command or is a slash command)
        cmd_parts = parse_command_safely(command)
        is_probably_bash = (
            cmd_parts
            and (
                cmd_parts[0] in interactive_commands
                or cmd_parts[0] in BASH_COMMANDS
                or command.strip().startswith("./")
                or command.strip().startswith("/")
            )
        )
        if is_probably_bash:
            try:
                command_name = cmd_parts[0]
                if command_name in interactive_commands:
                    return handle_interactive_command(cmd_parts, state)
                elif command_name == "cd":
                    return handle_cd_command(cmd_parts, state)
                else:
                    try:
                        bash_state, bash_output = handle_bash_command(cmd_parts, command, None, state)
                        return bash_state, bash_output
                    except Exception as bash_err:
                        return state, colored(f"Bash execution failed: {bash_err}", "red")
            except Exception:
                pass  # Fall through to LLM

        # Otherwise, treat as chat (LLM)
        response = get_llm_response(
            command, 
            model=active_model,          
            provider=active_provider,    
            npc=state.npc,
            stream=state.stream_output,
            messages=state.messages
        )
        state.messages = response['messages']
        return state, response['response']

    elif state.current_mode == 'cmd':

        response = execute_llm_command(command, 
                                        model=active_model,          
                                        provider=active_provider,  
                                                 npc = state.npc, 
                                                 stream = state.stream_output, 
                                                 messages = state.messages) 
        state.messages = response['messages']     
        return state, response['response']

    """    
    # to be replaced with a standalone corca mode
    
    elif state.current_mode == 'ride':
        # Allow bash commands in /ride mode
        cmd_parts = parse_command_safely(command)
        is_probably_bash = (
            cmd_parts
            and (
                cmd_parts[0] in interactive_commands
                or cmd_parts[0] in BASH_COMMANDS
                or command.strip().startswith("./")
                or command.strip().startswith("/")
            )
        )
        if is_probably_bash:
            try:
                command_name = cmd_parts[0]
                if command_name in interactive_commands:
                    return handle_interactive_command(cmd_parts, state)
                elif command_name == "cd":
                    return handle_cd_command(cmd_parts, state)
                else:
                    try:
                        bash_state, bash_output = handle_bash_command(cmd_parts, command, None, state)
                        return bash_state, bash_output
                    except Exception as bash_err:
                        return state, colored(f"Bash execution failed: {bash_err}", "red")
            except Exception:
                return state, colored("Failed to parse or execute bash command.", "red")

        # Otherwise, run the agentic ride loop
        return agentic_ride_loop(command, state)
    """


def check_deprecation_warnings():
    if os.getenv("NPCSH_MODEL"):
        cprint(
            "Deprecation Warning: NPCSH_MODEL/PROVIDER deprecated. Use NPCSH_CHAT_MODEL/PROVIDER.",
            "yellow",
        )

def print_welcome_message():
    print(
            """
Welcome to \033[1;94mnpc\033[0m\033[1;38;5;202msh\033[0m!
\033[1;94m                    \033[0m\033[1;38;5;202m        _       \\\\
\033[1;94m _ __   _ __    ___ \033[0m\033[1;38;5;202m  ___  | |___    \\\\
\033[1;94m| '_ \\ | '  \\  / __|\033[0m\033[1;38;5;202m / __/ | |_ _|    \\\\
\033[1;94m| | | || |_) |( |__ \033[0m\033[1;38;5;202m \\_  \\ | | | |    //
\033[1;94m|_| |_|| .__/  \\___|\033[0m\033[1;38;5;202m |___/ |_| |_|   //
       \033[1;94m| |          \033[0m\033[1;38;5;202m                //
       \033[1;94m| |
       \033[1;94m|_|

Begin by asking a question, issuing a bash command, or typing '/help' for more information.

            """
        )

def setup_shell() -> Tuple[CommandHistory, Team, Optional[NPC]]:
    check_deprecation_warnings()
    setup_npcsh_config()

    db_path = os.getenv("NPCSH_DB_PATH", HISTORY_DB_DEFAULT_PATH)
    db_path = os.path.expanduser(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    command_history = CommandHistory(db_path)


    try:
        history_file = setup_readline()
        atexit.register(save_readline_history)
        atexit.register(command_history.close)
    except:
        pass

    project_team_path = os.path.abspath(PROJECT_NPC_TEAM_PATH)
    global_team_path = os.path.expanduser(DEFAULT_NPC_TEAM_PATH)
    team_dir = None
    default_forenpc_name = None

    if os.path.exists(project_team_path):
        team_dir = project_team_path
        default_forenpc_name = "forenpc"
    else:
        if not os.path.exists('.npcsh_global'):
            resp = input(f"No npc_team found in {os.getcwd()}. Create a new team here? [Y/n]: ").strip().lower()
            if resp in ("", "y", "yes"):
                team_dir = project_team_path
                os.makedirs(team_dir, exist_ok=True)
                default_forenpc_name = "forenpc"
                forenpc_directive = input(
                    f"Enter a primary directive for {default_forenpc_name} (default: 'You are the forenpc of the team...'): "
                ).strip() or "You are the forenpc of the team, coordinating activities between NPCs on the team, verifying that results from NPCs are high quality and can help to adequately answer user requests."
                forenpc_model = input("Enter a model for your forenpc (default: llama3.2): ").strip() or "llama3.2"
                forenpc_provider = input("Enter a provider for your forenpc (default: ollama): ").strip() or "ollama"
                
                with open(os.path.join(team_dir, f"{default_forenpc_name}.npc"), "w") as f:
                    yaml.dump({
                        "name": default_forenpc_name, "primary_directive": forenpc_directive,
                        "model": forenpc_model, "provider": forenpc_provider
                    }, f)
                
                ctx_path = os.path.join(team_dir, "team.ctx")
                folder_context = input("Enter a short description for this project/team (optional): ").strip()
                team_ctx_data = {
                    "forenpc": default_forenpc_name, "model": forenpc_model,
                    "provider": forenpc_provider, "api_key": None, "api_url": None,
                    "context": folder_context if folder_context else None
                }
                use_jinxs = input("Use global jinxs folder (g) or copy to this project (c)? [g/c, default: g]: ").strip().lower()
                if use_jinxs == "c":
                    global_jinxs_dir = os.path.expanduser("~/.npcsh/npc_team/jinxs")
                    if os.path.exists(global_jinxs_dir):
                        shutil.copytree(global_jinxs_dir, team_dir, dirs_exist_ok=True)
                else:
                    team_ctx_data["use_global_jinxs"] = True

                with open(ctx_path, "w") as f:
                    yaml.dump(team_ctx_data, f)
            else:
                render_markdown('From now on, npcsh will assume you will use the global team when activating from this folder. \n If you change your mind and want to initialize a team, use /init from within npcsh, `npc init` or `rm .npcsh_global` from the current working directory.')
                with open(".npcsh_global", "w") as f:
                    pass
                team_dir = global_team_path
                default_forenpc_name = "sibiji"  
        elif os.path.exists(global_team_path):
            team_dir = global_team_path
            default_forenpc_name = "sibiji"            
        

    team_ctx = {}
    for filename in os.listdir(team_dir):
        if filename.endswith(".ctx"):
            try:
                with open(os.path.join(team_dir, filename), "r") as f:
                    team_ctx = yaml.safe_load(f) or {}
                break
            except Exception as e:
                print(f"Warning: Could not load context file {filename}: {e}")

    forenpc_name = team_ctx.get("forenpc", default_forenpc_name)
    #render_markdown(f"- Using forenpc: {forenpc_name}")

    if team_ctx.get("use_global_jinxs", False):
        jinxs_dir = os.path.expanduser("~/.npcsh/npc_team/jinxs")
    else:
        jinxs_dir = os.path.join(team_dir, "jinxs")
        
    jinxs_list = load_jinxs_from_directory(jinxs_dir)
    jinxs_dict = {jinx.jinx_name: jinx for jinx in jinxs_list}

    forenpc_obj = None
    forenpc_path = os.path.join(team_dir, f"{forenpc_name}.npc")


    #render_markdown('- Loaded team context'+ json.dumps(team_ctx, indent=2))


    
    if os.path.exists(forenpc_path):
        forenpc_obj = NPC(file = forenpc_path, 
                          jinxs=jinxs_list)
        if forenpc_obj.model is None:
            forenpc_obj.model= team_ctx.get("model", initial_state.chat_model)
        if forenpc_obj.provider is None:
            forenpc_obj.provider=team_ctx.get('provider', initial_state.chat_provider)
            
    else:
        print(f"Warning: Forenpc file '{forenpc_name}.npc' not found in {team_dir}.")

    team = Team(team_path=team_dir, 
                forenpc=forenpc_obj, 
                jinxs=jinxs_dict)

    for npc_name, npc_obj in team.npcs.items():
        if not npc_obj.model:
            npc_obj.model = initial_state.chat_model
        if not npc_obj.provider:
            npc_obj.provider = initial_state.chat_provider

    # Also apply to the forenpc specifically
    if team.forenpc and isinstance(team.forenpc, NPC):
        if not team.forenpc.model:
            team.forenpc.model = initial_state.chat_model
        if not team.forenpc.provider:
            team.forenpc.provider = initial_state.chat_provider
    team_name_from_ctx = team_ctx.get("name")
    if team_name_from_ctx:
        team.name = team_name_from_ctx
    elif team_dir and os.path.basename(team_dir) != 'npc_team':
        team.name = os.path.basename(team_dir)
    else:
        team.name = "global_team" # fallback for ~/.npcsh/npc_team

    return command_history, team, forenpc_obj

# In your main npcsh.py file

def process_result(
    user_input: str,
    result_state: ShellState,
    output: Any,
    command_history: CommandHistory
):
    # --- Part 1: Save Conversation & Determine Output ---
    
    # Define team and NPC names early for consistent logging
    team_name = result_state.team.name if result_state.team else "__none__"
    npc_name = result_state.npc.name if isinstance(result_state.npc, NPC) else "__none__"
    
    # Determine the actual NPC object to use for this turn's operations
    active_npc = result_state.npc if isinstance(result_state.npc, NPC) else NPC(
        name="default", 
        model=result_state.chat_model, 
        provider=result_state.chat_provider
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

    final_output_str = None
    output_content = output.get('output') if isinstance(output, dict) else output
    

    if result_state.stream_output:
        final_output_str = print_and_process_stream_with_markdown(output_content, active_npc.model, active_npc.provider)
    elif output_content is not None:
        final_output_str = str(output_content)
        render_markdown(final_output_str)

    if final_output_str:
        # Append assistant message to state for context continuity
        if result_state.messages and (not result_state.messages or result_state.messages[-1].get("role") != "assistant"):
            result_state.messages.append({"role": "assistant", "content": final_output_str})

        # Save assistant message to the database
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

        # --- Hierarchical Knowledge Graph Evolution ---
        conversation_turn_text = f"User: {user_input}\nAssistant: {final_output_str}"
        conn = command_history.conn

        try:

            npc_kg = load_kg_from_db(conn, team_name, npc_name, "__npc_global__")
            evolved_npc_kg, _ = kg_evolve_incremental(
                existing_kg=npc_kg, new_content_text=conversation_turn_text,
                model=active_npc.model, provider=active_npc.provider
            )
            save_kg_to_db(conn, evolved_npc_kg, team_name, npc_name, result_state.current_path)
        except Exception as e:
            print(colored(f"Error during real-time KG evolution: {e}", "red"))

        # --- Part 3: Periodic Team Context Suggestions ---
        result_state.turn_count += 1
        if result_state.turn_count > 0 and result_state.turn_count % 10 == 0:
            print(colored("\nChecking for potential team improvements...", "cyan"))
            try:
                summary = breathe(messages=result_state.messages[-20:], npc=active_npc)
                key_facts = summary.get('output', {}).get('facts', [])

                if key_facts and result_state.team:
                    team_ctx_path = os.path.join(result_state.team.team_path, "team.ctx")
                    ctx_data = {}
                    if os.path.exists(team_ctx_path):
                        with open(team_ctx_path, 'r') as f:
                           ctx_data = yaml.safe_load(f) or {}
                    current_context = ctx_data.get('context', '')

                    prompt = f"""Based on these key topics: {key_facts},
                    suggest changes (additions, deletions, edits) to the team's context. 
                    Additions need not be fully formed sentences and can simply be equations, relationships, or other plain clear items.
                    
                    Current Context: "{current_context}". 
                    
                    Respond with JSON: {{"suggestion": "Your sentence."}}"""
                    response = get_llm_response(prompt, npc=active_npc, format="json")
                    suggestion = response.get("response", {}).get("suggestion")

                    if suggestion:
                        new_context = (current_context + " " + suggestion).strip()
                        print(colored("AI suggests updating team context:", "yellow"))
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
                


def run_repl(command_history: CommandHistory, initial_state: ShellState):
    state = initial_state
    print_welcome_message()


    render_markdown(f'- Using {state.current_mode} mode. Use /agent, /cmd, or /chat to switch to other modes')
    render_markdown(f'- To switch to a different NPC, type /npc <npc_name> or /n <npc_name> to switch to that NPC.')
    render_markdown('\n- Here are the current NPCs available in your team: ' + ', '.join([npc_name for npc_name in state.team.npcs.keys()]))


    is_windows = platform.system().lower().startswith("win")
    try:
        completer = make_completer(state)
        readline.set_completer(completer)
    except:
        pass
    session_scopes = set()


    def exit_shell(current_state: ShellState):
        """
        On exit, iterates through all active scopes from the session and
        creates/updates the specific knowledge graph for each one.
        """
        print("\nGoodbye!")
        print(colored("Processing and archiving all session knowledge...", "cyan"))
        
        conn = command_history.conn
        integrator_npc = NPC(name="integrator", model=current_state.chat_model, provider=current_state.chat_provider)

        # Process each unique scope that was active during the session
        for team_name, npc_name, path in session_scopes:
            try:
                print(f"  -> Archiving knowledge for: T='{team_name}', N='{npc_name}', P='{path}'")
                
                # Get all messages for the current conversation that happened in this specific path
                convo_id = current_state.conversation_id
                all_messages = command_history.get_conversations_by_id(convo_id)
                
                scope_messages = [
                    m for m in all_messages 
                    if m.get('directory_path') == path and m.get('team') == team_name and m.get('npc') == npc_name
                ]
                
                full_text = "\n".join([f"{m['role']}: {m['content']}" for m in scope_messages if m.get('content')])

                if not full_text.strip():
                    print("     ...No content for this scope, skipping.")
                    continue

                # Load the existing KG for this specific, real scope
                current_kg = load_kg_from_db(conn, team_name, npc_name, path)
                
                # Evolve it with the full text from the session for this scope
                evolved_kg, _ = kg_evolve_incremental(
                    existing_kg=current_kg,
                    new_content_text=full_text,
                    model=integrator_npc.model,
                    provider=integrator_npc.provider
                )
                
                # Save the updated KG back to the database under the same exact scope
                save_kg_to_db(conn, evolved_kg, team_name, npc_name, path)

            except Exception as e:
                import traceback
                print(colored(f"Failed to process KG for scope ({team_name}, {npc_name}, {path}): {e}", "red"))
                traceback.print_exc()

        sys.exit(0)



    while True:
        try:
            try:
                completer = make_completer(state)
                readline.set_completer(completer)
            except:
                pass

            display_model = state.chat_model
            if isinstance(state.npc, NPC) and state.npc.model:
                display_model = state.npc.model

            if is_windows:
                cwd_part = os.path.basename(state.current_path)
                if isinstance(state.npc, NPC):
                    prompt_end = f":{state.npc.name}:{display_model}> "
                else:
                    prompt_end = ":npcsh> "
                prompt = f"{cwd_part}{prompt_end}"
            else:
                cwd_colored = colored(os.path.basename(state.current_path), "blue")
                if isinstance(state.npc, NPC):
                    prompt_end = f":🤖{orange(state.npc.name)}:{display_model}> "
                else:
                    prompt_end = f":🤖{colored('npc', 'blue', attrs=['bold'])}{colored('sh', 'yellow')}> "
                prompt = readline_safe_prompt(f"{cwd_colored}{prompt_end}")

            user_input = get_multiline_input(prompt).strip()
            # Handle Ctrl+Z (ASCII SUB, '\x1a') as exit (Windows and Unix)
            if user_input == "\x1a":
                exit_shell(state)

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                if isinstance(state.npc, NPC):
                    print(f"Exiting {state.npc.name} mode.")
                    state.npc = None
                    continue
                else:
                    exit_shell(state)
            team_name = state.team.name if state.team else "__none__"
            npc_name = state.npc.name if isinstance(state.npc, NPC) else "__none__"
            session_scopes.add((team_name, npc_name, state.current_path))

            state, output = execute_command(user_input, state)
            process_result(user_input, state, output, command_history)
        
        except KeyboardInterrupt:
            if is_windows:
                # On Windows, Ctrl+C cancels the current input line, show prompt again
                print("^C")
                continue
            else:
                # On Unix, Ctrl+C exits the shell as before
                exit_shell(state)
        except EOFError:
            # Ctrl+D: exit shell cleanly
            exit_shell(state)
def main() -> None:
    parser = argparse.ArgumentParser(description="npcsh - An NPC-powered shell.")
    parser.add_argument(
        "-v", "--version", action="version", version=f"npcsh version {VERSION}"
    )
    parser.add_argument(
         "-c", "--command", type=str, help="Execute a single command and exit."
    )
    args = parser.parse_args()

    command_history, team, default_npc = setup_shell()

    initial_state.npc = default_npc 
    initial_state.team = team
    #import pdb 
    #pdb.set_trace()

    # add a -g global command to indicate if to use the global or project, otherwise go thru normal flow
    
    if args.command:
         state = initial_state
         state.current_path = os.getcwd()
         final_state, output = execute_command(args.command, state)
         if final_state.stream_output:
              for chunk in output: 
                  print(str(chunk), end='')
              print()
         elif output is not None:
              print(output)
    else:
        run_repl(command_history, initial_state)

if __name__ == "__main__":
    main()