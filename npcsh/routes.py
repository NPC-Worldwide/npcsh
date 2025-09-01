# --- START OF FILE routes.py ---

from typing import Callable, Dict, Any, List, Optional, Union
import functools
import os
import subprocess
import sys
from pathlib import Path

import traceback
import shlex
import time
from datetime import datetime
from sqlalchemy import create_engine
import logging 
import json 
from npcpy.data.load import load_file_contents

from npcpy.llm_funcs import (
    get_llm_response,
    gen_image,
    gen_video,
    breathe,
)
from npcpy.npc_compiler import initialize_npc_project
from npcpy.npc_sysenv import render_markdown
from npcpy.work.plan import execute_plan_command
from npcpy.work.trigger import execute_trigger_command
from npcpy.work.desktop import perform_action
from npcpy.memory.search import execute_rag_command, execute_search_command, execute_brainblast_command
from npcpy.memory.command_history import CommandHistory, load_kg_from_db, save_kg_to_db
from npcpy.serve import start_flask_server
from npcpy.mix.debate import run_debate
from npcpy.data.image import capture_screenshot
from npcpy.npc_compiler import NPC, Team, Jinx,initialize_npc_project
from npcpy.data.web import search_web
from npcpy.memory.knowledge_graph import kg_sleep_process, kg_dream_process


from npcsh._state import (
    NPCSH_VISION_MODEL, 
    NPCSH_VISION_PROVIDER, 
    set_npcsh_config_value,
    NPCSH_API_URL,
    NPCSH_CHAT_MODEL, 
    NPCSH_CHAT_PROVIDER, 
    NPCSH_STREAM_OUTPUT,
    NPCSH_IMAGE_GEN_MODEL, 
    NPCSH_IMAGE_GEN_PROVIDER,
    NPCSH_VIDEO_GEN_MODEL,
    NPCSH_VIDEO_GEN_PROVIDER,
    NPCSH_EMBEDDING_MODEL,
    NPCSH_EMBEDDING_PROVIDER,
    NPCSH_REASONING_MODEL,
    NPCSH_REASONING_PROVIDER,
    NPCSH_SEARCH_PROVIDER,
    CANONICAL_ARGS, 
    normalize_and_expand_flags, 
    get_argument_help
)
from npcsh.corca import enter_corca_mode
from npcsh.guac import enter_guac_mode
from npcsh.plonk import execute_plonk_command, format_plonk_summary
from npcsh.alicanto import alicanto
from npcsh.pti import enter_pti_mode
from npcsh.spool import enter_spool_mode
from npcsh.wander import enter_wander_mode
from npcsh.yap import enter_yap_mode



NPC_STUDIO_DIR = Path.home() / ".npcsh" / "npc-studio"


class CommandRouter:
    def __init__(self):
        self.routes = {}
        self.help_info = {}

    def route(self, command: str, help_text: str = "") -> Callable:
        def wrapper(func):
            self.routes[command] = func
            self.help_info[command] = help_text

            @functools.wraps(func)
            def wrapped_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_func
        return wrapper

    def get_route(self, command: str) -> Optional[Callable]:
        return self.routes.get(command)

    def execute(self, command_str: str, **kwargs) -> Any:
        command_name = command_str.split()[0].lstrip('/')
        route_func = self.get_route(command_name)
        if route_func:
            return route_func(command=command_str, **kwargs)
        return None

    def get_commands(self) -> List[str]:
        return list(self.routes.keys())

    def get_help(self, command: str = None) -> Dict[str, str]:
        if command:
            if command in self.help_info:
                return {command: self.help_info[command]}
            return {}
        return self.help_info

router = CommandRouter()
def get_help_text():
    commands = router.get_commands()
    help_info = router.help_info

    commands.sort()
    output = "# Available Commands\n\n"
    for cmd in commands:
        help_text = help_info.get(cmd, "")
        output += f"/{cmd} - {help_text}\n\n"
    
    arg_help_map = get_argument_help()
    if arg_help_map:
        output += "## Common Command-Line Flags\n\n"
        output += "The shortest unambiguous prefix works (e.g., `-t` for `--temperature`).\n\n"
        

        output += "```\n"

        all_args_to_show = CANONICAL_ARGS[:]
        all_args_to_show.sort()


        NUM_COLUMNS = 4
        FLAG_WIDTH = 18   
        ALIAS_WIDTH = 12  
        COLUMN_SEPARATOR = " | "

        rows_per_column = (len(all_args_to_show) + NUM_COLUMNS - 1) // NUM_COLUMNS
        columns = [all_args_to_show[i:i + rows_per_column] for i in range(0, len(all_args_to_show), rows_per_column)]

        def get_shortest_alias(arg):
            if arg in arg_help_map and arg_help_map[arg]:
                return min(arg_help_map[arg], key=len)
            return ""

        header_parts = []
        for _ in range(NUM_COLUMNS):
            flag_header = "Flag".ljust(FLAG_WIDTH)
            alias_header = "Shorthand".ljust(ALIAS_WIDTH)
            header_parts.append(f"{flag_header}{alias_header}")
        output += COLUMN_SEPARATOR.join(header_parts) + "\n"

        divider_parts = []
        for _ in range(NUM_COLUMNS):

            divider_part = "-" * (FLAG_WIDTH + ALIAS_WIDTH)
            divider_parts.append(divider_part)
        output += COLUMN_SEPARATOR.join(divider_parts) + "\n"


        for i in range(rows_per_column):
            row_parts = []
            for col_idx in range(NUM_COLUMNS):
                if col_idx < len(columns) and i < len(columns[col_idx]):
                    arg = columns[col_idx][i]
                    alias = get_shortest_alias(arg)
                    alias_display = f"(-{alias})" if alias else ""
                    
                    flag_part = f"--{arg}".ljust(FLAG_WIDTH)
                    alias_part = alias_display.ljust(ALIAS_WIDTH)
                    row_parts.append(f"{flag_part}{alias_part}")
                else:

                    row_parts.append(" " * (FLAG_WIDTH + ALIAS_WIDTH))
            
            output += COLUMN_SEPARATOR.join(row_parts) + "\n"


        output += "```\n"

    output += """
\n## Note
- Bash commands and programs can be executed directly (try bash first, then LLM).
- Use '/exit' or '/quit' to exit the current NPC mode or the npcsh shell.
- Jinxs defined for the current NPC or Team can also be used like commands (e.g., /screenshot).
"""
    return output
def safe_get(kwargs, key, default=None):
    return kwargs.get(key, default)

@router.route("breathe", "Condense context on a regular cadence")
def breathe_handler(command: str, **kwargs):
    #try:
    result = breathe(**kwargs)
    if isinstance(result, dict): 
        return result
    #except NameError:
    #     return {"output": "Breathe function not available."}
    #except Exception as e:
    #    traceback.print_exc()
    #    return {"output": f"Error during breathe: {e}"}

@router.route("compile", "Compile NPC profiles")
def compile_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    npc_team_dir = safe_get(kwargs, 'current_path', './npc_team')
    parts = command.split()
    npc_file_path_arg = parts[1] if len(parts) > 1 else None
    output = ""
    try:
        if npc_file_path_arg:
            npc_full_path = os.path.abspath(npc_file_path_arg)
            if os.path.exists(npc_full_path):
                npc = NPC(npc_full_path)
                output = f"Compiled NPC: {npc_full_path}"
            else:
                output = f"Error: NPC file not found: {npc_full_path}"
        else:
            npc = NPC(npc_full_path)

            output = f"Compiled all NPCs in directory: {npc_team_dir}"
    except NameError:
        output = "Compile functions (compile_npc_file, compile_team_npcs) not available."
    except Exception as e:
        traceback.print_exc()
        output = f"Error compiling: {e}"
    return {"output": output, "messages": messages, "npc": npc}



@router.route("corca", "Enter the Corca MCP-powered agentic shell. Usage: /corca [--mcp-server-path path]")
def corca_handler(command: str, **kwargs):
    return enter_corca_mode(command=command, **kwargs)
    
@router.route("flush", "Flush the last N messages")
def flush_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    try:
        parts = command.split()
        n = int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError):
        return {"output": "Usage: /flush [number_of_messages_to_flush]", "messages": messages}

    if n <= 0:
        return {"output": "Error: Number of messages must be positive.", "messages": messages}

    new_messages = list(messages)
    original_len = len(new_messages)
    removed_count = 0

    if new_messages and new_messages[0].get("role") == "system":
        system_message = new_messages[0]
        working_messages = new_messages[1:]
        num_to_remove = min(n, len(working_messages))
        if num_to_remove > 0:
            final_messages = [system_message] + working_messages[:-num_to_remove]
            removed_count = num_to_remove
        else:
            final_messages = [system_message]
    else:
        num_to_remove = min(n, original_len)
        if num_to_remove > 0:
            final_messages = new_messages[:-num_to_remove]
            removed_count = num_to_remove
        else:
            final_messages = []

    output = f"Flushed {removed_count} message(s). Context is now {len(final_messages)} messages."
    return {"output": output, "messages": final_messages}

@router.route("guac", "Enter guac mode")
def guac_handler(command,  **kwargs):
    '''
    Guac ignores input npc and npc_team dirs and manually sets them to be at ~/.npcsh/guac/
    
    '''
    config_dir = safe_get(kwargs, 'config_dir', None)
    plots_dir = safe_get(kwargs, 'plots_dir', None)
    refresh_period = safe_get(kwargs, 'refresh_period', 100)
    lang = safe_get(kwargs, 'lang', None)
    messages = safe_get(kwargs, "messages", [])
    db_conn = safe_get(kwargs, 'db_conn', create_engine('sqlite:///'+os.path.expanduser('~/npcsh_history.db')))
    
    npc_file = '~/.npcsh/guac/npc_team/guac.npc'
    npc_team_dir = os.path.expanduser('~/.npcsh/guac/npc_team/')
    
    npc = NPC(file=npc_file, db_conn=db_conn)

    team = Team(npc_team_dir, db_conn=db_conn)

    
    enter_guac_mode(
                    npc=npc, 
                    team=team, 
                    config_dir=config_dir, 
                    plots_dir=plots_dir,
                    npc_team_dir=npc_team_dir,
                    refresh_period=refresh_period, lang=lang)
    
    return {"output": 'Exiting Guac Mode', "messages": safe_get(kwargs, "messages", [])}


@router.route("help", "Show help for commands, NPCs, or Jinxs. Usage: /help [topic]")
def help_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    parts = shlex.split(command)
    if len(parts) < 2:
        return {"output": get_help_text(), "messages": messages}
    target = parts[1].lstrip('/') # User might type /help /sample, so we clean it.
    output = ""



    if target in router.get_commands():
        help_text = router.get_help(target).get(target, "No description available.")
        output = f"## Help for Command: `/{target}`\n\n- **Description**: {help_text}"
        return {"output": output, "messages": messages}

    team = safe_get(kwargs, 'team')
    if team and target in team.npcs:
        npc_obj = team.npcs[target]
        output = f"## Help for NPC: `{target}`\n\n"
        output += f"- **Primary Directive**: {npc_obj.primary_directive}\n"
        output += f"- **Default Model**: `{npc_obj.model}`\n"
        output += f"- **Default Provider**: `{npc_obj.provider}`\n"
        if hasattr(npc_obj, 'jinxs_dict') and npc_obj.jinxs_dict:
            jinx_names = ", ".join([f"`{j}`" for j in npc_obj.jinxs_dict.keys()])
            output += f"- **Associated Jinxs**: {jinx_names}\n"
        return {"output": output, "messages": messages}

    # 3. Is it a Jinx?
    npc = safe_get(kwargs, 'npc')
    jinx_obj = None
    source = ""
    if npc and hasattr(npc, 'jinxs_dict') and target in npc.jinxs_dict:
        jinx_obj = npc.jinxs_dict[target]
        source = f" (from NPC: `{npc.name}`)"
    elif team and hasattr(team, 'jinxs_dict') and target in team.jinxs_dict:
        jinx_obj = team.jinxs_dict[target]
        source = f" (from Team: `{team.name}`)"

    if jinx_obj:
        output = f"## Help for Jinx: `/{target}`{source}\n\n"
        output += f"- **Description**: {jinx_obj.description}\n"
        if hasattr(jinx_obj, 'inputs') and jinx_obj.inputs:
            inputs_str = json.dumps(jinx_obj.inputs, indent=2)
            output += f"- **Inputs**:\n```json\n{inputs_str}\n```\n"
        return {"output": output, "messages": messages}


    return {"output": f"Sorry, no help topic found for `{target}`.", "messages": messages}




@router.route("init", "Initialize NPC project")
def init_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    try:
        parts = shlex.split(command)
        directory = "."
        templates = None
        context = None
        # Basic parsing example (needs improvement for robust flag handling)
        if len(parts) > 1 and not parts[1].startswith("-"):
            directory = parts[1]
        # Add logic here to parse -t, -ctx flags if needed

        initialize_npc_project(
            directory=directory,
            templates=templates,
            context=context,
            model=safe_get(kwargs, 'model'),
            provider=safe_get(kwargs, 'provider')
        )
        output = f"NPC project initialized in {os.path.abspath(directory)}."
    except NameError:
        output = "Init function (initialize_npc_project) not available."
    except Exception as e:
        traceback.print_exc()
        output = f"Error initializing project: {e}"
    return {"output": output, "messages": messages}

def ensure_repo():
    """Clone or update the npc-studio repo."""
    if not NPC_STUDIO_DIR.exists():
        os.makedirs(NPC_STUDIO_DIR.parent, exist_ok=True)
        subprocess.check_call([
            "git", "clone",
            "https://github.com/npc-worldwide/npc-studio.git",
            str(NPC_STUDIO_DIR)
        ])
    else:
        subprocess.check_call(
            ["git", "pull"],
            cwd=NPC_STUDIO_DIR
        )

def install_dependencies():
    """Install npm and pip dependencies."""
    # Install frontend deps
    subprocess.check_call(["npm", "install"], cwd=NPC_STUDIO_DIR)

    # Install backend deps
    req_file = NPC_STUDIO_DIR / "requirements.txt"
    if req_file.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
def launch_npc_studio(path_to_open: str = None):
    """
    Launch the NPC Studio backend + frontend.
    Returns PIDs for processes.
    """
    ensure_repo()
    install_dependencies()

    # Start backend (Flask server)
    backend = subprocess.Popen(
        [sys.executable, "npc_studio_serve.py"],
        cwd=NPC_STUDIO_DIR, 
        shell = False
    )

    # Start server (Electron)
    dev_server = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=NPC_STUDIO_DIR,
        shell=False
    )
    
    # Start frontend (Electron)
    frontend = subprocess.Popen(
        ["npm", "start"],
        cwd=NPC_STUDIO_DIR,
        shell=False
    )

    return backend, dev_server, frontend
# ========== Router handler ==========
@router.route("npc-studio", "Start npc studio")
def npc_studio_handler(command: str, **kwargs):
    messages = kwargs.get("messages", [])
    user_command = " ".join(command.split()[1:])

    try:
        backend, electron, frontend = launch_npc_studio(user_command or None)
        return {
            "output": f"NPC Studio started!\nBackend PID={backend.pid}, Electron PID={electron.pid} Frontend PID={frontend.pid}",
            "messages": messages
        }
    except Exception as e:
        return {
            "output": f"Failed to start NPC Studio: {e}",
            "messages": messages
        }
@router.route("ots", "Take screenshot and analyze with vision model")
def ots_handler(command: str, **kwargs):
    command_parts = command.split()
    image_paths = []
    npc = safe_get(kwargs, 'npc')
    vision_model = safe_get(kwargs, 
                            'vmodel',
                            NPCSH_VISION_MODEL)
    vision_provider = safe_get(kwargs, 
                               'vprovider', 
                               NPCSH_VISION_PROVIDER)
    messages = safe_get(kwargs, 
                        'messages', 
                        [])
    stream = safe_get(kwargs, 
                      'stream',
                        NPCSH_STREAM_OUTPUT)

    try:
        if len(command_parts) > 1:
            for img_path_arg in command_parts[1:]:
                full_path = os.path.abspath(img_path_arg)
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                else:
                    return {"output": f"Error: Image file not found at {full_path}", "messages": messages}
        else:
            screenshot_info = capture_screenshot(full=False)
            if screenshot_info and "file_path" in screenshot_info:
                image_paths.append(screenshot_info["file_path"])
                print(f"Screenshot captured: {screenshot_info.get('filename', os.path.basename(screenshot_info['file_path']))}")
            else:
                 return {"output": "Error: Failed to capture screenshot.", "messages": messages}

        if not image_paths:
            return {"output": "No valid images found or captured.", "messages": messages}

        user_prompt = safe_get(kwargs, 'stdin_input')
        if user_prompt is None:
            try:
                user_prompt = input(
                    "Enter a prompt for the LLM about these images (or press Enter to skip): "
                )
            except EOFError:
                 user_prompt = "Describe the image(s)."

        if not user_prompt or not user_prompt.strip():
            user_prompt = "Describe the image(s)."

        response_data = get_llm_response(
            prompt=user_prompt,
            model=vision_model,
            provider=vision_provider,
            messages=messages,
            images=image_paths,
            stream=stream,
            npc=npc,
            api_url=safe_get(kwargs, 'api_url'),
            api_key=safe_get(kwargs, 'api_key')
        )
        return {"output": response_data.get('response'), "messages": response_data.get('messages'), "model": vision_model, "provider": vision_provider}

    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error during /ots command: {e}", "messages": messages}




@router.route("plan", "Execute a plan command")
def plan_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    user_command = " ".join(command.split()[1:])
    if not user_command:
        return {"output": "Usage: /plan <description_of_plan>", "messages": messages}
    #try:
    return execute_plan_command(command=user_command, **kwargs)

    #return {"output": "Plan function (execute_plan_command) not available.", "messages": messages}
    #except Exception as e:
    #    traceback.print_exc()
    #    return {"output": f"Error executing plan: {e}", "messages": messages}

@router.route("pti", "Enter Pardon-The-Interruption mode for human-in-the-loop reasoning.")
def pti_handler(command: str, **kwargs):
    return enter_pti_mode(command=command, **kwargs)

@router.route("plonk", "Use vision model to interact with GUI. Usage: /plonk <task description>")
def plonk_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    
    # FIXED: Use the pre-parsed positional arguments for the request,
    # leaving flags to be handled by kwargs.
    positional_args = safe_get(kwargs, 'positional_args', [])
    request_str = " ".join(positional_args)

    if not request_str:
        return {"output": "Usage: /plonk <task_description> [--vmodel model_name] [--vprovider provider_name]", "messages": messages}

    try:
        plonk_context = safe_get(kwargs, 'plonk_context')
        
        # This part now works automatically with CLI flags because they are in kwargs
        summary_data = execute_plonk_command(
            request=request_str,
            model=safe_get(kwargs, 'vmodel', NPCSH_VISION_MODEL),
            provider=safe_get(kwargs, 'vprovider', NPCSH_VISION_PROVIDER),
            npc=safe_get(kwargs, 'npc'),
            plonk_context=plonk_context,
            debug=True # Or could be controlled by a flag: safe_get(kwargs, 'debug', False)
        )        
        
        if summary_data and isinstance(summary_data, list):
            output_report = format_plonk_summary(summary_data)
            return {"output": output_report, "messages": messages}
        else:
            return {"output": "Plonk command did not complete within the maximum number of iterations.", "messages": messages}

    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing plonk command: {e}", "messages": messages}


@router.route("brainblast", "Execute an advanced chunked search on command history")
def brainblast_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])    
    parts = shlex.split(command)
    search_query = " ".join(parts[1:]) if len(parts) > 1 else ""
    
    if not search_query:
        return {"output": "Usage: /brainblast <search_terms>", "messages": messages}
    
    # Get the command history instance
    command_history = kwargs.get('command_history')
    if not command_history:
        #print('no command history provided to brainblast')
        # Create a new one if not provided
        db_path = safe_get(kwargs, "history_db_path", os.path.expanduser('~/npcsh_history.db'))
        try:
            command_history = CommandHistory(db_path)
            kwargs['command_history'] = command_history
        except Exception as e:
            return {"output": f"Error connecting to command history: {e}", "messages": messages}
    
    try:
        # Remove messages from kwargs to avoid duplicate argument error
        if 'messages' in kwargs:
            del kwargs['messages']
            
        # Execute the brainblast command
        return execute_brainblast_command(
                                    command=search_query,
                                    **kwargs)   
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing brainblast command: {e}", "messages": messages}

@router.route("rag", "Execute a RAG command using ChromaDB embeddings with optional file input (-f/--file)")
def rag_handler(command: str, **kwargs):    
    parts = shlex.split(command)
    user_command = []
    file_paths = []
    
    i = 1  # Skip the first element which is "rag"
    while i < len(parts):
        if parts[i] == "-f" or parts[i] == "--file":
            # We found a file flag, get the file path
            if i + 1 < len(parts):
                file_paths.append(parts[i + 1])
                i += 2  # Skip both the flag and the path
            else:
                return {"output": "Error: -f/--file flag needs a file path", "messages": messages}
        else:
            # This is part of the user query
            user_command.append(parts[i])
            i += 1
    
    user_command = " ".join(user_command)
    
    vector_db_path = safe_get(kwargs, "vector_db_path", os.path.expanduser('~/npcsh_chroma.db'))
    embedding_model = safe_get(kwargs, "emodel", NPCSH_EMBEDDING_MODEL)
    embedding_provider = safe_get(kwargs, "eprovider", NPCSH_EMBEDDING_PROVIDER)
    
    if not user_command and not file_paths:
        return {"output": "Usage: /rag [-f file_path] <query>", "messages": kwargs.get('messages', [])}
    
    try:
        # Process files if provided
        file_contents = []
        for file_path in file_paths:
            try:
                chunks = load_file_contents(file_path)
                file_name = os.path.basename(file_path)
                file_contents.extend([f"[{file_name}] {chunk}" for chunk in chunks])
            except Exception as file_err:
                file_contents.append(f"Error processing file {file_path}: {str(file_err)}")
        exe_rag =  execute_rag_command(
            command=user_command,
            vector_db_path=vector_db_path,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            file_contents=file_contents if file_paths else None,
            **kwargs
        )
        return {'output':exe_rag.get('response'), 'messages': exe_rag.get('messages', kwargs.get('messages', []))}
    
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing RAG command: {e}", "messages": kwargs.get('messages', [])}
@router.route("roll", "generate a video")
def roll_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    prompt = " ".join(command.split()[1:])
    num_frames = safe_get(kwargs, 'num_frames', 125)
    width = safe_get(kwargs, 'width', 256)
    height = safe_get(kwargs, 'height', 256)
    output_path = safe_get(kwargs, 'output_path', "output.mp4")    
    if not prompt:
        return {"output": "Usage: /roll <your prompt>", "messages": messages}
    try:
        result = gen_video(
            prompt=prompt,
            model=safe_get(kwargs, 'vgmodel', NPCSH_VIDEO_GEN_MODEL),
            provider=safe_get(kwargs, 'vgprovider', NPCSH_VIDEO_GEN_PROVIDER),
            npc=safe_get(kwargs, 'npc'),
            num_frames = num_frames,
            width = width,
            height = height,
            output_path=output_path,
            
            **safe_get(kwargs, 'api_kwargs', {})
        )
        return result
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error generating video: {e}", "messages": messages}
    

@router.route("sample", "Send a prompt directly to the LLM")
def sample_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    

    positional_args = safe_get(kwargs, 'positional_args', [])
    prompt = " ".join(positional_args)

    if not prompt:
        return {"output": "Usage: /sample <your prompt> [-m --model] model  [-p --provider] provider", 
                "messages": messages}

    try:
        result = get_llm_response(
            prompt=prompt,
            **kwargs
        )
        if result and isinstance(result, dict):
            return {
                "output": result.get('response'), 
                "messages": result.get('messages', messages), 
                "model": kwargs.get('model'), 
                "provider":kwargs.get('provider'), 
                "npc":kwargs.get("npc"),
            }
        else:
            # Handle cases where get_llm_response might fail and return something unexpected
            return {"output": str(result), "messages": messages}

    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error sampling LLM: {e}", "messages": messages}
@router.route("search", "Execute a web search command")
def search_handler(command: str, **kwargs):
    """    
    Executes a search command.
    # search commands will bel ike :
    # '/search "search term" '
    # '/search -sp perplexity ..
    # '/search -sp google ..
    # extract provider if its there
    # check for either -p or --p        
    """
    messages = safe_get(kwargs, "messages", [])
    
    # The query is now in 'positional_args'
    positional_args = safe_get(kwargs, 'positional_args', [])
    query = " ".join(positional_args)
    
    if not query:
        return {"output": "Usage: /search [-sp name --sprovider name] query", 
                "messages": messages}
    search_provider = safe_get(kwargs, 'sprovider', NPCSH_SEARCH_PROVIDER)
    render_markdown(f'- Searching {search_provider} for "{query}"'    )


    
    if not query:
        return {"output": "Usage: /search <query>", "messages": messages}
    search_provider = safe_get(kwargs, 'search_provider', NPCSH_SEARCH_PROVIDER)
    try:
        search_results = search_web(query, provider=search_provider)
        output = "\n".join([f"- {res}" for res in search_results]) if search_results else "No results found."
    except Exception as e:
        traceback.print_exc()
        output = f"Error during web search: {e}"
    return {"output": output, "messages": messages}



@router.route("serve", "Serve an NPC Team")
def serve_handler(command: str, **kwargs):
    #print('calling serve handler')
    #print(kwargs)

    port   = safe_get(kwargs, "port", 5337)
    #print(port, type(port))
    messages = safe_get(kwargs, "messages", [])
    cors = safe_get(kwargs, "cors", None)
    if cors:
        cors_origins = [origin.strip() for origin in cors.split(",")]
    else:
        cors_origins = None

        start_flask_server(
            port=port, 
            cors_origins=cors_origins,
        )


    return {"output": None, "messages": messages}

@router.route("set", "Set configuration values")
def set_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    parts = command.split(maxsplit=1)
    if len(parts) < 2 or '=' not in parts[1]:
        return {"output": "Usage: /set <key>=<value>", "messages": messages}

    key_value = parts[1]
    key, value = key_value.split('=', 1)
    key = key.strip()
    value = value.strip().strip('"\'')

    try:
        set_npcsh_config_value(key, value)
        output = f"Configuration value '{key}' set."
    except NameError:
        output = "Set function (set_npcsh_config_value) not available."
    except Exception as e:
        traceback.print_exc()
        output = f"Error setting configuration '{key}': {e}"
    return {"output": output, "messages": messages}

@router.route("sleep", "Evolve knowledge graph. Use --dream to also run creative synthesis.")
def sleep_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    npc = safe_get(kwargs, 'npc')
    team = safe_get(kwargs, 'team')
    model = safe_get(kwargs, 'model')
    provider = safe_get(kwargs, 'provider')

    is_dreaming = safe_get(kwargs, 'dream', False)
    operations_str = safe_get(kwargs, 'ops')
    
    operations_config = None
    if operations_str and isinstance(operations_str, str):
        operations_config = [op.strip() for op in operations_str.split(',')]

    # Define the scope variables clearly at the start
    team_name = team.name if team else "__none__"
    npc_name = npc.name if isinstance(npc, NPC) else "__none__"
    current_path = os.getcwd()
    scope_str = f"Team: '{team_name}', NPC: '{npc_name}', Path: '{current_path}'"

    # ADDED: Log the scope being checked for clarity
    render_markdown(f"- Checking knowledge graph for scope: {scope_str}")

    try:
        db_path = os.getenv("NPCSH_DB_PATH", os.path.expanduser("~/npcsh_history.db"))
        command_history = CommandHistory(db_path)
        engine = command_history.engine
    except Exception as e:
        return {"output": f"Error connecting to history database for KG access: {e}", "messages": messages}

    try:
        current_kg = load_kg_from_db(engine, team_name, npc_name, current_path)

        # FIXED: Provide a detailed and helpful message when the KG is empty
        if not current_kg or not current_kg.get('facts'):
            output_msg = f"Knowledge graph for the current scope is empty. Nothing to process.\n"
            output_msg += f"  - Scope Checked: {scope_str}\n\n"
            output_msg += "**Hint:** Have a conversation or run some commands first to build up knowledge in this specific context. The KG is unique to each combination of Team, NPC, and directory."
            return {"output": output_msg, "messages": messages}

        # Store initial stats for the final report
        original_facts = len(current_kg.get('facts', []))
        original_concepts = len(current_kg.get('concepts', []))
        
        # --- SEQUENTIAL EXECUTION ---

        # 1. Always run the sleep process for maintenance first.
        process_type = "Sleep"
        ops_display = f"with operations: {operations_config}" if operations_config else "with random operations"
        render_markdown(f"- Initiating sleep process {ops_display}")
        
        evolved_kg, _ = kg_sleep_process(
            existing_kg=current_kg,
            model=model,
            provider=provider,
            npc=npc,
            operations_config=operations_config
        )

        # 2. If --dream is specified, run the dream process on the *result* of the sleep process.
        if is_dreaming:
            process_type += " & Dream"
            render_markdown(f"- Initiating dream process on the evolved KG...")
            evolved_kg, _ = kg_dream_process(
                existing_kg=evolved_kg,
                model=model,
                provider=provider,
                npc=npc
            )

        # 3. Save the final state of the KG back to the database
        save_kg_to_db(conn, evolved_kg, team_name, npc_name, current_path)

        # 4. Report the final, cumulative changes back to the user
        new_facts = len(evolved_kg.get('facts', []))
        new_concepts = len(evolved_kg.get('concepts', []))

        output = f"{process_type} process complete.\n"
        output += f"- Facts: {original_facts} -> {new_facts} ({new_facts - original_facts:+})\n"
        output += f"- Concepts: {original_concepts} -> {new_concepts} ({new_concepts - original_concepts:+})"
        
        print(evolved_kg.get('facts'))
        print(evolved_kg.get('concepts'))
        
        return {"output": output, "messages": messages}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"output": f"Error during KG evolution process: {e}", "messages": messages}
    finally:
        if 'command_history' in locals() and command_history:
            command_history.close()




@router.route("spool", "Enter interactive chat (spool) mode")
def spool_handler(command: str, **kwargs):
    try:
        npc = safe_get(kwargs, 'npc')
        team = safe_get(kwargs, 'team')
        
        if isinstance(npc, str) and team:
            npc_name = npc
            if npc_name in team.npcs:
                npc = team.npcs[npc_name]
            else:
                return {"output": f"Error: NPC '{npc_name}' not found in team. Available NPCs: {', '.join(team.npcs.keys())}", "messages": safe_get(kwargs, "messages", [])}
        kwargs['npc'] = npc
        return enter_spool_mode(            
                                **kwargs)
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error entering spool mode: {e}", "messages": safe_get(kwargs, "messages", [])}
    
@router.route("jinxs", "Show available jinxs for the current NPC/Team")
def jinxs_handler(command: str, **kwargs):
    npc = safe_get(kwargs, 'npc')
    team = safe_get(kwargs, 'team')
    output = "Available Jinxs:\n"
    jinxs_listed = set()

    def format_jinx(name, jinx_obj):
        desc = getattr(jinx_obj, 'description', 'No description available.')
        return f"- /{name}: {desc}\n"

    if npc and isinstance(npc, NPC) and hasattr(npc, 'jinxs_dict') and npc.jinxs_dict:
        output += f"\n--- Jinxs for NPC: {npc.name} ---\n"
        for name, jinx in sorted(npc.jinxs_dict.items()):
            output += format_jinx(name, jinx)
            jinxs_listed.add(name)

    if team and hasattr(team, 'jinxs_dict') and team.jinxs_dict:
         team_has_jinxs = False
         team_output = ""
         for name, jinx in sorted(team.jinxs_dict.items()):
             if name not in jinxs_listed:
                 team_output += format_jinx(name, jinx)
                 team_has_jinxs = True
         if team_has_jinxs:
             output += f"\n--- Jinxs for Team: {getattr(team, 'name', 'Unnamed Team')} ---\n"
             output += team_output

    if not jinxs_listed and not (team and hasattr(team, 'jinxs_dict') and team.jinxs_dict):
        output = "No jinxs available for the current context."

    return {"output": output.strip(), "messages": safe_get(kwargs, "messages", [])}

@router.route("trigger", "Execute a trigger command")
def trigger_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    user_command = " ".join(command.split()[1:])
    if not user_command:
        return {"output": "Usage: /trigger <trigger_description>", "messages": messages}
    try:
        return execute_trigger_command(command=user_command, **kwargs)
    except NameError:
        return {"output": "Trigger function (execute_trigger_command) not available.", "messages": messages}
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing trigger: {e}", "messages": messages}
@router.route("vixynt", "Generate images from text descriptions")
def vixynt_handler(command: str, **kwargs):
    npc = safe_get(kwargs, 'npc')
    model = safe_get(kwargs, 'igmodel', NPCSH_IMAGE_GEN_MODEL)
    provider = safe_get(kwargs, 'igprovider', NPCSH_IMAGE_GEN_PROVIDER)
    height = safe_get(kwargs, 'height', 1024)
    width = safe_get(kwargs, 'width', 1024)
    output_file_base = safe_get(kwargs, 'output_file')
    attachments = safe_get(kwargs, 'attachments')
    n_images = safe_get(kwargs, 'n_images', 1) # Get n_images from kwargs
    if isinstance(attachments, str):
        attachments = attachments.split(',')
    
    messages = safe_get(kwargs, 'messages', [])

    user_prompt = " ".join(safe_get(kwargs, 'positional_args', []))

    if not user_prompt:
        return {"output": "Usage: /vixynt <prompt> [--output_file path] [--attachments path] [--n_images num]", "messages": messages}
    
    try:
        # Call gen_image, passing n_images and expecting a list of images
        images_list = gen_image(
            prompt=user_prompt,
            model=model,
            provider=provider,
            npc=npc,
            height=height,
            width=width,
            n_images=n_images, # Pass n_images
            input_images=attachments
        )

        saved_files = []
        if not isinstance(images_list, list):
            images_list = [images_list] if images_list is not None else []

        for i, image in enumerate(images_list):
            if image is None:
                continue

            if output_file_base is None:
                os.makedirs(os.path.expanduser("~/.npcsh/images/"), exist_ok=True)
                current_output_file = (
                    os.path.expanduser("~/.npcsh/images/")
                    + f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                )
            else:
                base_name, ext = os.path.splitext(os.path.expanduser(output_file_base))
                current_output_file = f"{base_name}_{i}{ext}"
            
            image.save(current_output_file)
            image.show()
            saved_files.append(current_output_file)

        if saved_files:
            if attachments:
                output = f"Image(s) edited and saved to: {', '.join(saved_files)}"
            else:
                output = f"Image(s) generated and saved to: {', '.join(saved_files)}"
        else:
            output = f"No images {'edited' if attachments else 'generated'}."

    except Exception as e:
        traceback.print_exc()
        output = f"Error {'editing' if attachments else 'generating'} image: {e}"

    return {
        "output": output,
        "messages": messages,
        "model": model,
        "provider": provider
    }


@router.route("wander", "Enter wander mode (experimental)")
def wander_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    
    # General parser for key=value arguments
    try:
        parts = shlex.split(command)
        problem_parts = []
        wander_params = {}
        
        i = 1  # Start after the 'wander' command name
        while i < len(parts):
            part = parts[i]
            
            if '=' in part:
                # This is the start of a key=value pair
                key, initial_value = part.split('=', 1)
                
                # Consume all subsequent parts that do NOT contain '=' as part of this value
                value_parts = [initial_value]
                j = i + 1
                while j < len(parts) and '=' not in parts[j]:
                    value_parts.append(parts[j])
                    j += 1
                
                # Join the reconstructed value and store it
                wander_params[key] = " ".join(value_parts)
                # Advance the main loop index past the consumed parts
                i = j
            else:
                # This part belongs to the problem string
                problem_parts.append(part)
                i += 1
        
        problem = " ".join(problem_parts)
    except Exception as e:
        return {"output": f"Error parsing arguments: {e}", "messages": messages}
        
    if not problem:
        return {"output": "Usage: /wander <problem> [key=value...]", "messages": messages}

    try:
        # Build the argument list for enter_wander_mode
        mode_args = {
            'problem': problem,
            'npc': safe_get(kwargs, 'npc'),
            'model': safe_get(kwargs, 'model'),
            'provider': safe_get(kwargs, 'provider'),
            # Use parsed params with defaults
            'environment': wander_params.get('environment'),
            'low_temp': float(wander_params.get('low-temp', 0.5)),
            'high_temp': float(wander_params.get('high-temp', 1.9)),
            'interruption_likelihood': float(wander_params.get('interruption-likelihood', 1)),
            'sample_rate': float(wander_params.get('sample-rate', 0.4)),
            'n_high_temp_streams': int(wander_params.get('n-high-temp-streams', 5)),
            'include_events': bool(wander_params.get('include-events', False)),
            'num_events': int(wander_params.get('num-events', 3))
        }
        
        result = enter_wander_mode(**mode_args)
        
        if isinstance(result, list) and result:
            output = result[-1].get("insight", "Wander mode session complete.")
        else:
            output = str(result) if result else "Wander mode session complete."
            
        messages.append({"role": "assistant", "content": output})
        return {"output": output, "messages": messages}
        
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error during wander mode: {e}", "messages": messages}

@router.route("yap", "Enter voice chat (yap) mode")
def yap_handler(command: str, **kwargs):
    try:
        return enter_yap_mode(
            ** kwargs
            )
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error entering yap mode: {e}", "messages": safe_get(kwargs, "messages", [])}

@router.route("alicanto", "Conduct deep research with multiple perspectives, identifying gold insights and cliff warnings")
def alicanto_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    
    # Parse command with shlex to properly handle quoted strings
    parts = shlex.split(command)
    
    # Process arguments
    query = ""
    num_npcs = safe_get(kwargs, 'num_npcs', 5)
    depth = safe_get(kwargs, 'depth', 3)
    exploration_factor = safe_get(kwargs, 'exploration', 0.3)
    creativity_factor = safe_get(kwargs, 'creativity', 0.5)
    output_format = safe_get(kwargs, 'format', 'report')
    
    # Parse command-line arguments
    i = 1  # Skip "alicanto" command
    while i < len(parts):
        if parts[i].startswith('--'):
            option = parts[i][2:]  # Remove '--'
            if option in ['num-npcs', 'npcs']:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    num_npcs = int(parts[i + 1])
                    i += 2
                else:
                    i += 1
            elif option in ['depth', 'd']:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    depth = int(parts[i + 1])
                    i += 2
                else:
                    i += 1
            elif option in ['exploration', 'e']:
                if i + 1 < len(parts) and parts[i + 1].replace('.', '', 1).isdigit():
                    exploration_factor = float(parts[i + 1])
                    i += 2
                else:
                    i += 1
            elif option in ['creativity', 'c']:
                if i + 1 < len(parts) and parts[i + 1].replace('.', '', 1).isdigit():
                    creativity_factor = float(parts[i + 1])
                    i += 2
                else:
                    i += 1
            elif option in ['format', 'f']:
                if i + 1 < len(parts):
                    output_format = parts[i + 1]
                    i += 2
                else:
                    i += 1
            else:
                # Skip unknown option
                i += 1
        else:
            # This is part of the request
            query += parts[i] + " "
            i += 1
    
    query = query.strip()
    
    # Also apply any kwargs that were passed directly (these override command line args)
    if 'num_npcs' in kwargs:
        try:
            num_npcs = int(kwargs['num_npcs'])
        except ValueError:
            return {"output": "Error: num_npcs must be an integer", "messages": messages}
    
    if 'depth' in kwargs:
        try:
            depth = int(kwargs['depth'])
        except ValueError:
            return {"output": "Error: depth must be an integer", "messages": messages}
    
    if 'exploration' in kwargs:
        try:
            exploration_factor = float(kwargs['exploration'])
        except ValueError:
            return {"output": "Error: exploration must be a float", "messages": messages}
            
    if 'creativity' in kwargs:
        try:
            creativity_factor = float(kwargs['creativity'])
        except ValueError:
            return {"output": "Error: creativity must be a float", "messages": messages}
    
    if not query:
        return {"output": "Usage: /alicanto <research query> [--num-npcs N] [--depth N] [--exploration 0.3] [--creativity 0.5] [--format report|summary|full]", "messages": messages}
    
    try:
        logging.info(f"Starting Alicanto research on: {query}")
        result = alicanto(
            request=query,
            num_npcs=num_npcs,
            depth=depth,
            memory=3,
            context=None,
            model=safe_get(kwargs, 'model', NPCSH_CHAT_MODEL),
            provider=safe_get(kwargs, 'provider', NPCSH_CHAT_PROVIDER),
            exploration_factor=exploration_factor,
            creativity_factor=creativity_factor,
            output_format=output_format
        )
        
        # Format the output based on the result type
        if isinstance(result, dict):
            if "integration" in result:
                output = result["integration"]
            else:
                output = "Alicanto research completed. Full results available in returned data."
        else:
            output = result
            
        return {"output": output, "messages": messages, "alicanto_result": result}
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error during Alicanto research: {e}")
        return {"output": f"Error during Alicanto research: {e}", "messages": messages}