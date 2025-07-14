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
try:
    from inspect import isgenerator
except: 
    pass
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

import yaml

# Local Application Imports
from npcsh._state import (
    setup_npcsh_config,
    is_npcsh_initialized,
    initialize_base_npcs_if_needed,
    orange,
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
)
from npcpy.memory.knowledge_graph import breathe
from npcpy.memory.sleep import sleep, forget
from npcpy.npc_compiler import NPC, Team, load_jinxs_from_directory
from npcpy.llm_funcs import check_llm_command, get_llm_response, execute_llm_command
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



from npcsh._state import initial_state, ShellState

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
    try:
        readline.read_history_file(READLINE_HISTORY_FILE)

        readline.set_history_length(1000)
        readline.parse_and_bind("set enable-bracketed-paste on")
        #readline.parse_and_bind('"\e[A": history-search-backward')
        #readline.parse_and_bind('"\e[B": history-search-forward')
        readline.parse_and_bind(r'"\C-r": reverse-search-history')
        readline.parse_and_bind(r'"\C-e": end-of-line')
        readline.parse_and_bind(r'"\C-a": beginning-of-line')
        #if sys.platform == "darwin":
        #    readline.parse_and_bind("bind ^I rl_complete")
        #else:
        #    readline.parse_and_bind("tab: complete")

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


# --- Placeholder for actual valid commands ---
# This should be populated dynamically based on router, builtins, and maybe PATH executables
valid_commands_list = list(router.routes.keys()) + list(interactive_commands.keys()) + ["cd", "exit", "quit"] + BASH_COMMANDS

def complete(text: str, state: int) -> Optional[str]:
    try:
        buffer = readline.get_line_buffer()
    except:
        print('couldnt get readline buffer')
    line_parts = parse_command_safely(buffer) # Use safer parsing
    word_before_cursor = ""
    if len(line_parts) > 0 and not buffer.endswith(' '):
         current_word = line_parts[-1]
    else:
         current_word = "" # Completing after a space

    try:
        # Command completion (start of line or after pipe/semicolon)
        # This needs refinement to detect context better
        is_command_start = not line_parts or (len(line_parts) == 1 and not buffer.endswith(' ')) # Basic check
        if is_command_start and not text.startswith('-'): # Don't complete options as commands
            cmd_matches = [cmd + ' ' for cmd in valid_commands_list if cmd.startswith(text)]
            # Add executables from PATH? (Can be slow)
            # path_executables = [f + ' ' for f in shutil.get_exec_path() if os.path.basename(f).startswith(text)]
            # cmd_matches.extend(path_executables)
            return cmd_matches[state]

        # File/Directory completion (basic)
        # Improve context awareness (e.g., after 'cd', 'ls', 'cat', etc.)
        if text and (not text.startswith('/') or os.path.exists(os.path.dirname(text))):
             basedir = os.path.dirname(text)
             prefix = os.path.basename(text)
             search_dir = basedir if basedir else '.'
             try:
                 matches = [os.path.join(basedir, f) + ('/' if os.path.isdir(os.path.join(search_dir, f)) else ' ')
                            for f in os.listdir(search_dir) if f.startswith(prefix)]
                 return matches[state]
             except OSError: # Handle permission denied etc.
                  return None

    except IndexError:
        return None
    except Exception: # Catch broad exceptions during completion
        return None

    return None


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
                #some respond with output, some with response, needs to be fixed upstream
                output = result_dict.get("output") or result_dict.get("response")
                state.messages = result_dict.get("messages", state.messages)
                return state, output
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

    exec_model = model_override or state.chat_model
    exec_provider = provider_override or state.chat_provider

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
                model=exec_model, 
                provider=exec_provider, 
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
            shell=True,
        )
        if isinstance(llm_result, dict):
            state.messages = llm_result.get("messages", state.messages)
            output = llm_result.get("output")
            return state, output
        else:
            return state, llm_result        
def check_mode_switch(command:str , state: ShellState):
    if command in ['/cmd', '/agent', '/chat', '/ride']:
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

    if state.current_mode == 'agent':
        for i, cmd_segment in enumerate(commands):
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
                elif isgenerator(output):
                    if not stream_this_segment: # If intermediate output is a stream, consume for piping
                        full_stream_output = "".join(map(str, output))
                        stdin_for_next = full_stream_output
                        if is_last_command: 
                            final_output = full_stream_output
                    else: # Final output is a stream, don't consume, can't pipe
                        stdin_for_next = None
                        final_output = output
                elif output is not None: # Try converting other types to string
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
        if final_output is not None and not (isgenerator(final_output) and current_state.stream_output):
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
            model=state.chat_model, 
            provider=state.chat_provider, 
            npc=state.npc,
            stream=state.stream_output,
            messages=state.messages
        )
        state.messages = response['messages']
        return state, response['response']

    elif state.current_mode == 'cmd':

        response = execute_llm_command(command, 
                                                 model = state.chat_model, 
                                                 provider = state.chat_provider, 
                                                 npc = state.npc, 
                                                 stream = state.stream_output, 
                                                 messages = state.messages) 
        state.messages = response['messages']     
        return state, response['response']

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
                    except CommandNotFoundError:
                        return state, colored(f"Command not found: {command_name}", "red")
                    except Exception as bash_err:
                        return state, colored(f"Bash execution failed: {bash_err}", "red")
            except Exception:
                return state, colored("Failed to parse or execute bash command.", "red")

        # Otherwise, run the agentic ride loop
        return agentic_ride_loop(command, state)
@dataclass
class RideState:
    """Lightweight state tracking for /ride mode"""
    todos: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    mistakes: List[str] = field(default_factory=list)
    successes: List[str] = field(default_factory=list)
    current_todo_index: int = 0
    current_subtodo_index: int = 0
    
    def get_context_summary(self) -> str:
        """Generate lightweight context for LLM prompts"""
        context = []
        if self.facts:
            context.append(f"Facts: {'; '.join(self.facts[:5])}")  # Limit to 5 most recent
        if self.mistakes:
            context.append(f"Recent mistakes: {'; '.join(self.mistakes[-3:])}")
        if self.successes:
            context.append(f"Recent successes: {'; '.join(self.successes[-3:])}")
        return "\n".join(context)

def interactive_edit_list(items: List[str], item_type: str) -> List[str]:
    """Interactive REPL for editing lists of items with regeneration options"""
    while True:
        print(f"\nCurrent {item_type}:")
        for i, item in enumerate(items, 1):
            print(f"{i}. {item}")
        
        choice = input(f"\nEdit {item_type} (e<num> to edit, d<num> to delete, a to add, r to regenerate, c to add context, ok to continue): ").strip()
        
        if choice.lower() == 'ok':
            break
        elif choice.lower() == 'r':
            print("Regenerating list...")
            return "REGENERATE"  # Special signal to regenerate
        elif choice.lower() == 'c':
            additional_context = input("Add more context: ").strip()
            if additional_context:
                return {"ADD_CONTEXT": additional_context, "items": items}
        elif choice.lower() == 'a':
            new_item = input(f"Enter new {item_type[:-1]}: ").strip()
            if new_item:
                items.append(new_item)
        elif choice.lower().startswith('e'):
            try:
                idx = int(choice[1:]) - 1
                if 0 <= idx < len(items):
                    print(f"Current: {items[idx]}")
                    new_item = input("New version: ").strip()
                    if new_item:
                        items[idx] = new_item
            except ValueError:
                print("Invalid format. Use e<number>")
        elif choice.lower().startswith('d'):
            try:
                idx = int(choice[1:]) - 1
                if 0 <= idx < len(items):
                    items.pop(idx)
            except ValueError:
                print("Invalid format. Use d<number>")
        else:
            print("Invalid choice. Use: e<num>, d<num>, a, r (regenerate), c (add context), or ok")
    
    return items
def generate_todos(user_goal: str, state: ShellState, additional_context: str = "") -> List[Dict[str, Any]]:
    """Generate high-level todos for the user's goal"""
    path_cmd = 'The current working directory is: ' + state.current_path
    ls_files = 'Files in the current directory (full paths):\n' + "\n".join([os.path.join(state.current_path, f) for f in os.listdir(state.current_path)]) if os.path.exists(state.current_path) else 'No files found in the current directory.'
    platform_info = f"Platform: {platform.system()} {platform.release()} ({platform.machine()})"
    info = path_cmd + '\n' + ls_files + '\n' + platform_info 


    
    high_level_planning_instruction = """
    You are a high-level project planner. When a user asks to work on a file or code,
    structure your plan using a simple, high-level software development lifecycle:
    1. First, understand the current state (e.g., read the relevant file).
    2. Second, make the required changes based on the user's goal.
    3. Third, verify the changes work as intended (e.g., test the code).
    Your generated todos should reflect this high-level thinking.


    
    """
    
    prompt = f"""
    {high_level_planning_instruction}

    User goal: {user_goal}
    
    {additional_context}
    
    Generate a list of 3 todos to accomplish this goal. Use specific actionable language based on the user request. 
    Do not make assumptions about user needs. 
    Every todo must be directly sourced from the user's request.     
    If users request specific files to be incorporated, you MUST include the full path to the file in the todo.
    Here is some relevant information for the current folder and working directory that may be relevant:
    {info}

    For example, if the user says "I need to add a new function to calculate the average of a list of numbers my research.py script" and the current working directory is /home/user/projects and one
     of the available files in the current directory is /home/user/projects/research.py then one of the todos should be:
    - "Add a new function to /home/user/projects/research.py to calculate the average of a list of numbers"
    Do not truncate paths. Do not additional paths. Use them exactly as they are provided here.
    
    Each todo should be:
    - Specific and actionable
    - Independent where possible
    - Focused on a single major component

    Remember, it is critical to provide as much relevant information as possible. Even if the user only refers to a file or something by a relative path, it is
    critical for operation that you provide the full path to the file in the todo item.

    Return JSON with format:
    {{
        "todos": [
            todo1, todo2, todo3,
                    ]
    }}
    """
    
    response = get_llm_response(
        prompt,
        model=state.chat_model,
        provider=state.chat_provider,
        npc=state.npc,
        format="json"
    )
    
    todos_data = response.get("response", {}).get("todos", [])
    return todos_data


def generate_constraints(todos: List[Dict[str, Any]], user_goal: str, state: ShellState) -> List[str]:
    """Generate constraints and requirements that define relationships between todos"""
    prompt = f"""
    User goal: {user_goal}
    
    Todos to accomplish:
    {chr(10).join([f"- {todo}" for todo in todos])}
    
    Based ONLY on what the user explicitly stated in their goal, identify any constraints or requirements they mentioned.
    Do NOT invent new constraints. Only extract constraints that are directly stated or clearly implied by the user's request.
    
    Examples of valid constraints:
    - If user says "without breaking existing functionality" -> "Maintain existing functionality"
    - If user says "must be fast" -> "Performance must be optimized"
    - If user says "should integrate with X" -> "Must integrate with X"
    
    If the user didn't specify any constraints, return an empty list.
    
    Return JSON with format:
    {{
        "constraints": ["constraint 1", "constraint 2", ...]
    }}
    """
    
    response = get_llm_response(
        prompt,
        model=state.chat_model,
        provider=state.chat_provider,
        npc=state.npc,
        format="json"
    )
    
    constraints_data = response.get("response", {})
    
    if isinstance(constraints_data, dict):
        constraints = constraints_data.get("constraints", [])
        # Make sure we're getting strings, not dicts
        cleaned_constraints = []
        for c in constraints:
            if isinstance(c, str):
                cleaned_constraints.append(c)
        return cleaned_constraints
    else:
        return []
def should_break_down_todo(todo, state: ShellState) -> bool:
    """Ask LLM if a todo needs breakdown, then ask user for confirmation"""
    prompt = f"""
    Todo: {todo}

        
    Does this todo need to be broken down into smaller, more atomic components?
    Consider:
    - Is it complex enough to warrant breakdown?
    - Would breaking it down make execution clearer?
    - Are there multiple distinct steps involved?
    
    Return JSON: {{"should_break_down": true/false, "reason": "explanation"}}
    """
    
    response = get_llm_response(
        prompt,
        model=state.chat_model,
        provider=state.chat_provider,
        npc=state.npc,
        format="json"
    )
    
    result = response.get("response", {})
    llm_suggests = result.get("should_break_down", False)
    reason = result.get("reason", "No reason provided")
    
    if llm_suggests:
        print(f"\nLLM suggests breaking down: '{todo}'")
        print(f"Reason: {reason}")
        user_choice = input("Break it down? [y/N]: ").strip().lower()
        return user_choice in ['y', 'yes']
    
    return False

def generate_subtodos(todo, state: ShellState) -> List[Dict[str, Any]]:
    """Generate atomic subtodos for a complex todo"""
    prompt = f"""
    Parent todo: {todo}
    
    Break this down into atomic, executable subtodos. Each subtodo should be:
    - A single, concrete action
    - Executable in one step
    - Clear and unambiguous
    
    Return JSON with format:
    {{
        "subtodos": [
             "subtodo description",
            ...
        ]
    }}
    """
    
    response = get_llm_response(
        prompt,
        model=state.chat_model,
        provider=state.chat_provider,
        npc=state.npc,
        format="json"
    )
    
    return response.get("response", {}).get("subtodos", [])
def execute_todo_item(todo: Dict[str, Any], ride_state: RideState, shell_state: ShellState) -> bool:
    """Execute a single todo item using the existing jinx system"""
    path_cmd = 'The current working directory is: ' + shell_state.current_path
    ls_files = 'Files in the current directory (full paths):\n' + "\n".join([os.path.join(shell_state.current_path, f) for f in os.listdir(shell_state.current_path)]) if os.path.exists(shell_state.current_path) else 'No files found in the current directory.'
    platform_info = f"Platform: {platform.system()} {platform.release()} ({platform.machine()})"
    info = path_cmd + '\n' + ls_files + '\n' + platform_info 

    command = f"""

    General information:
    {info}
    
    Execute this todo: {todo}
    
    Constraints to follow:
    {chr(10).join([f"- {c}" for c in ride_state.constraints])}
    """
    
    print(f"\nExecuting: {todo}")
    

    result = check_llm_command(
        command,
        model=shell_state.chat_model,
        provider=shell_state.chat_provider,
        npc=shell_state.npc,
        team=shell_state.team,
        messages=[],
        stream=shell_state.stream_output,
        shell=True,
    )
    
    output_payload = result.get("output", "")
    output_str = ""

    if isgenerator(output_payload):
        output_str = print_and_process_stream_with_markdown(output_payload, shell_state.chat_model, shell_state.chat_provider)
    elif isinstance(output_payload, dict):
        output_str = output_payload.get('output', str(output_payload))
        if 'output' in output_str:
            output_str = output_payload['output']
        elif 'response' in output_str:
            output_str = output_payload['response']
        render_markdown(output_str)
    elif output_payload:
        output_str = str(output_payload)
        render_markdown(output_str)

    user_feedback = input(f"\nTodo completed successfully? [y/N/notes]: ").strip()
    
    if user_feedback.lower() in ['y', 'yes']:
        return True, output_str
    elif user_feedback.lower() in ['n', 'no']:
        mistake = input("What went wrong? ").strip()
        ride_state.mistakes.append(f"Failed {todo}: {mistake}")
        return False, output_str
    else:
        ride_state.facts.append(f"Re: {todo}: {user_feedback}")
        success = input("Mark as completed? [y/N]: ").strip().lower() in ['y', 'yes']
        return success, output_str

def agentic_ride_loop(user_goal: str, state: ShellState) -> tuple:
    """
    New /ride mode: hierarchical planning with human-in-the-loop control
    """
    ride_state = RideState()
    
    # 1. Generate high-level todos
    print("🚀 Generating high-level todos...")
    todos = generate_todos(user_goal, state)
    
    # 2. User reviews/edits todos
    print("\n📋 Review and edit todos:")
    todo_descriptions = [todo for todo in todos]
    edited_descriptions = interactive_edit_list(todo_descriptions, "todos")
    

    ride_state.todos = edited_descriptions
    
    # 3. Generate constraints
    print("\n🔒 Generating constraints...")
    constraints = generate_constraints(edited_descriptions, user_goal, state)
    
    # 4. User reviews/edits constraints
    print("\n📐 Review and edit constraints:")
    edited_constraints = interactive_edit_list(constraints, "constraints")
    ride_state.constraints = edited_constraints
    
    # 5. Execution loop
    print("\n⚡ Starting execution...")
    
    for i, todo in enumerate(edited_descriptions):
        print(f"\n--- Todo {i+1}/{len(todos)}: {todo} ---")
        
        def attempt_execution(current_todo):
            # This inner function handles the execution and retry logic
            success, output_str = execute_todo_item(current_todo, ride_state, state)
            if not success:
                retry = input("Retry this todo? [y/N]: ").strip().lower()
                if retry in ['y', 'yes']:
                    success, output_str = execute_todo_item(current_todo, ride_state, state)
            return success, output_str

        if should_break_down_todo(todo, state):
            print("Breaking down todo...")
            subtodos = generate_subtodos(todo, state)
            subtodo_descriptions = [st for st in subtodos]
            edited_subtodos = interactive_edit_list(subtodo_descriptions, "subtodos")

            for j, subtodo_desc in enumerate(edited_subtodos):
                subtodo = {"description": subtodo_desc, "type": "atomic"}
                success, output = attempt_execution(subtodo)
                if success:
                    ride_state.successes.append({"description": subtodo_desc, "output": output})
                else:
                    print("Subtodo failed. Continuing to next...")
        else:
            success, output = attempt_execution(todo)
            if success:
                ride_state.successes.append({"description": todo, "output": output})
    # 6. Final summary
    print("\n🎯 Execution Summary:")
    print(f"Successes: {len(ride_state.successes)}")
    print(f"Mistakes: {len(ride_state.mistakes)}")
    print(f"Facts learned: {len(ride_state.facts)}")
    
    return state, {
        "todos_completed": len(ride_state.successes),
        "ride_state": ride_state,
        "final_context": ride_state.get_context_summary()
    }
# --- Main Application Logic ---

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
\033[1;94m                    \033[0m\033[1;38;5;202m                \\\\
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
        readline.set_completer(complete)
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
                        shutil.copytree(global_jinxs_dir, os.path.join(team_dir, "jinxs"), dirs_exist_ok=True)
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
    print(f"Using forenpc: {forenpc_name}")

    if team_ctx.get("use_global_jinxs", False):
        jinxs_dir = os.path.expanduser("~/.npcsh/npc_team/jinxs")
    else:
        jinxs_dir = os.path.join(team_dir, "jinxs")
        
    jinxs_list = load_jinxs_from_directory(jinxs_dir)
    jinxs_dict = {jinx.jinx_name: jinx for jinx in jinxs_list}

    forenpc_obj = None
    forenpc_path = os.path.join(team_dir, f"{forenpc_name}.npc")
    #print('forenpc_path', forenpc_path)
    #print('jinx list', jinxs_list)
    if os.path.exists(forenpc_path):

        forenpc_obj = NPC(file = forenpc_path, jinxs=jinxs_list)
    else:
        print(f"Warning: Forenpc file '{forenpc_name}.npc' not found in {team_dir}.")

    team = Team(team_path=team_dir, forenpc=forenpc_obj, jinxs=jinxs_dict)
    return command_history, team, forenpc_obj

def process_result(
    user_input: str,
    result_state: ShellState,
    output: Any,
    command_history: CommandHistory):

    npc_name = result_state.npc.name if isinstance(result_state.npc, NPC) else result_state.npc
    team_name = result_state.team.name if isinstance(result_state.team, Team) else result_state.team
    save_conversation_message(
        command_history,
        result_state.conversation_id,
        "user",
        user_input,
        wd=result_state.current_path,
        model=result_state.chat_model, # Log primary chat model? Or specific used one?
        provider=result_state.chat_provider,
        npc=npc_name,
        team=team_name,
        attachments=result_state.attachments,
    )
    
    result_state.attachments = None # Clear attachments after logging user message

    final_output_str = None
    if user_input =='/help':
        render_markdown(output)
    elif result_state.stream_output:

        try:
            final_output_str = print_and_process_stream_with_markdown(output, result_state.chat_model, result_state.chat_provider)
        except AttributeError as e:
            if isinstance(output, str):
                if len(output) > 0:
                    final_output_str = output
                    render_markdown(final_output_str) 
        except TypeError as e:

            if isinstance(output, str):
                if len(output) > 0:
                    final_output_str = output
                    render_markdown(final_output_str)
            elif isinstance(output, dict):
                if 'output' in output:
                    final_output_str = output['output']
                    render_markdown(final_output_str)
                        
    elif output is not None:
        final_output_str = str(output)
        render_markdown(final_output_str)
    if final_output_str and result_state.messages and result_state.messages[-1].get("role") != "assistant":
        result_state.messages.append({"role": "assistant", "content": final_output_str})

    #print(result_state.messages)

    print() # Add spacing after output

    if final_output_str:
        save_conversation_message(
            command_history,
            result_state.conversation_id,
            "assistant",
            final_output_str,
            wd=result_state.current_path,
            model=result_state.chat_model,
            provider=result_state.chat_provider,
            npc=npc_name,
            team=team_name,
        )

def run_repl(command_history: CommandHistory, initial_state: ShellState):
    state = initial_state
    print_welcome_message()
    print(f'Using {state.current_mode} mode. Use /agent, /cmd, /chat, or /ride to switch to other modes')
    print(f'To switch to a different NPC, type /<npc_name>')
    is_windows = platform.system().lower().startswith("win")

    def exit_shell(state):
        print("\nGoodbye!")
        # update the team ctx file  to update the context and the preferences 





        #print('beginning knowledge consolidation')
        #try:
        #    breathe_result = breathe(state.messages, state.chat_model, state.chat_provider, state.npc)
        #    print(breathe_result)
        #except KeyboardInterrupt:
        #    print("Knowledge consolidation interrupted. Exiting immediately.")
        sys.exit(0)

    while True:
        try:
            if is_windows:
                cwd_part = os.path.basename(state.current_path)
                if isinstance(state.npc, NPC):
                    prompt_end = f":{state.npc.name}> "
                else:
                    prompt_end = ":npcsh> "
                prompt = f"{cwd_part}{prompt_end}"
            else:
                cwd_colored = colored(os.path.basename(state.current_path), "blue")
                if isinstance(state.npc, NPC):
                    prompt_end = f":🤖{orange(state.npc.name)}:{state.chat_model}> "
                else:
                    prompt_end = f":🤖{colored('npc', 'blue', attrs=['bold'])}{colored('sh', 'yellow')}> "
                prompt = readline_safe_prompt(f"{cwd_colored}{prompt_end}")
            cwd_colored = colored(os.path.basename(state.current_path), "blue")
            if isinstance(state.npc, NPC):
                prompt_end = f":🤖{orange(state.npc.name)}> "
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

            state.current_path = os.getcwd()
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
         if final_state.stream_output and isgenerator(output):
              for chunk in output: print(str(chunk), end='')
              print()
         elif output is not None:
              print(output)
    else:
        run_repl(command_history, initial_state)

if __name__ == "__main__":
    main()