from chroptiks.plotting_utils import * 
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
import sys
import argparse
import importlib.metadata
import matplotlib.pyplot as plt 

import logging
plt.ioff()
import shlex
import platform
import yaml
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import traceback

try:
    from termcolor import colored
except ImportError:
    pass

import sys 

from npcpy.memory.command_history import CommandHistory, start_new_conversation
from npcpy.npc_compiler import Team, NPC
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_sysenv import render_markdown,print_and_process_stream


from npcsh._state import (
    ShellState,
    execute_command,
    make_completer,
    process_result,
    readline_safe_prompt,
    setup_shell,
    get_multiline_input,
    orange
)
import threading
import time
import ctypes
import ctypes.util

try:
    import readline
except ImportError:
    print('no readline support, some features may not work as desired.')

try:
    VERSION = importlib.metadata.version("npcsh")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"

GUAC_REFRESH_PERIOD = os.environ.get('GUAC_REFRESH_PERIOD', 100)
READLINE_HISTORY_FILE = os.path.expanduser("~/.guac_readline_history")
# File extension mapping for organization
EXTENSION_MAP = {
    "PNG": "images", "JPG": "images", "JPEG": "images", "GIF": "images", "SVG": "images",
    "MP4": "videos", "AVI": "videos", "MOV": "videos", "WMV": "videos", "MPG": "videos", "MPEG": "videos",
    "DOC": "documents", "DOCX": "documents", "PDF": "documents", "PPT": "documents", "PPTX": "documents",
    "XLS": "documents", "XLSX": "documents", "TXT": "documents", "CSV": "documents",
    "ZIP": "archives", "RAR": "archives", "7Z": "archives", "TAR": "archives", "GZ": "archives", "BZ2": "archives",
    "ISO": "archives", "NPY": "data", "NPZ": "data", "H5": "data", "HDF5": "data", "PKL": "data", "JOBLIB": "data"
}

_guac_monitor_thread = None
_guac_monitor_stop_event = None

def _clear_readline_buffer():
    """Clear the current readline input buffer and redisplay prompt."""
    try:
        # Preferred: use Python readline API if available
        if hasattr(readline, "replace_line") and hasattr(readline, "redisplay"):
            readline.replace_line("", 0)
            readline.redisplay()
            return True
    except Exception:
        pass

    # Fallback: call rl_replace_line and rl_redisplay from the linked readline/libedit
    try:
        libname = ctypes.util.find_library("readline") or ctypes.util.find_library("edit") or "readline"
        rl = ctypes.CDLL(libname)
        # rl_replace_line(char *text, int clear_undo)
        rl.rl_replace_line.argtypes = [ctypes.c_char_p, ctypes.c_int]
        rl.rl_redisplay.argtypes = []
        rl.rl_replace_line(b"", 0)
        rl.rl_redisplay()
        return True
    except Exception:
        return False

def _file_drop_monitor(npc_team_dir: Path, state: ShellState, locals_dict: Dict[str, Any], poll_interval: float = 0.2):
    """
    Background thread: poll readline.get_line_buffer() and process file drops immediately.
    """
    processed_bufs = set()
    stop_event = _guac_monitor_stop_event
    while stop_event is None or not stop_event.is_set():
        try:
            buf = ""
            try:
                buf = readline.get_line_buffer()
            except Exception:
                buf = ""
            if not buf:
                time.sleep(poll_interval)
                continue

            # Normalize buffer
            candidate = buf.strip()
            # If quoted, remove quotes
            if (candidate.startswith("'") and candidate.endswith("'")) or (candidate.startswith('"') and candidate.endswith('"')):
                inner = candidate[1:-1]
            else:
                inner = candidate

            # quick check: must be single token and existing file
            if " " not in inner and Path(inner.replace('~', str(Path.home()))).expanduser().exists() and Path(inner.replace('~', str(Path.home()))).expanduser().is_file():
                # Avoid double-processing same buffer
                if buf in processed_bufs:
                    time.sleep(poll_interval)
                    continue
                processed_bufs.add(buf)

                # Immediately process: copy and load
                try:
                    # Use your existing handler for multi-file copies to ensure directory structure
                    # But we want immediate execution for a single file: call _handle_file_drop first to copy
                    modified_input, processed_files = _handle_file_drop(buf, npc_team_dir)
                    if processed_files:
                        target_path = processed_files[0]
                        # Generate loading code based on original file (inner) and target_path
                        loading_code = _generate_file_analysis_code(inner, target_path)
                        # Execute via your normal execute_python_code so it records in history
                        print("\n[guac] Detected file drop — processing automatically...")
                        # Note: execute_python_code expects state and locals_dict
                        _state, exec_output = execute_python_code(loading_code, state, locals_dict)
                        # Print whatever result execute_python_code returned (it will already have been captured)
                        if exec_output:
                            print(exec_output)
                        # Clear the current readline buffer so user doesn't have to press Enter
                        _clear_readline_buffer()
                except Exception as e:
                    print(f"[guac][ERROR] file drop processing failed: {e}")
        except Exception:
            # Be resilient: don't let thread die
            pass
        time.sleep(poll_interval)


def is_python_code(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    try:
        compile(text, "<input>", "eval")
        return True
    except SyntaxError:
        try:
            compile(text, "<input>", "exec")
            return True
        except SyntaxError:
            return False
    except (OverflowError, ValueError):
        return False
def execute_python_code(code_str: str, state: ShellState, locals_dict: Dict[str, Any]) -> Tuple[ShellState, Any]:
    import io
    output_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    final_output_str = None
    is_expression = False

    try:
        sys.stdout = output_capture
        sys.stderr = output_capture

        if '\n' not in code_str.strip() and not re.match(r"^\s*(def|class|for|while|if|try|with|import|from|@)", code_str.strip()):
            try:
                compiled_expr = compile(code_str, "<input>", "eval")
                exec_result = eval(compiled_expr, locals_dict)
                if exec_result is not None and not output_capture.getvalue().strip():
                    print(repr(exec_result), file=sys.stdout)
                is_expression = True 
            except SyntaxError: 
                is_expression = False
            except Exception: 
                is_expression = False
                raise 
        
        if not is_expression: 
            compiled_code = compile(code_str, "<input>", "exec")
            exec(compiled_code, locals_dict)

    except SyntaxError: 
        exc_type, exc_value, _ = sys.exc_info()
        error_lines = traceback.format_exception_only(exc_type, exc_value)
        adjusted_error_lines = [line.replace('File "<input>"', 'Syntax error in input') for line in error_lines]
        print("".join(adjusted_error_lines), file=output_capture, end="")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb, file=output_capture)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        final_output_str = output_capture.getvalue().strip()
        output_capture.close()
    
    # ADD THIS LINE:
    _capture_plot_state(state.conversation_id, state.command_history.db_path, Path.cwd() / "npc_team")
    
    if state.command_history:
        state.command_history.add_command(code_str, [final_output_str if final_output_str else ""], "", state.current_path)
    return state, final_output_str

# Modify _generate_file_analysis_code - add the capture call to each code block:
def _generate_file_analysis_code(file_path: str, target_path: str) -> str:
    """Generate Python code to load and analyze the dropped file"""
    ext = Path(file_path).suffix.lower()
    file_var_name = f"file_{datetime.now().strftime('%H%M%S')}"
    
    capture_code = f"""
# Capture file analysis state
_capture_file_state('{state.conversation_id}', '{state.command_history.db_path}', r'{target_path}', '''AUTO_GENERATED_CODE''', locals())
"""
    
    if ext == '.pdf':
        return f"""
# Automatically loaded PDF file
import PyPDF2
import pandas as pd
try:
    with open(r'{target_path}', 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        {file_var_name}_text = ""
        for page_num in range(len(pdf_reader.pages)):
            {file_var_name}_text += pdf_reader.pages[page_num].extract_text()
    
    print(f"📄 Loaded PDF: {{len(pdf_reader.pages)}} pages, {{len({file_var_name}_text)}} characters")
    print("First 500 characters:")
    print({file_var_name}_text[:500])
    print("\\n--- PDF loaded as '{file_var_name}_text' variable ---")
    {capture_code}
except Exception as e:
    print(f"Error loading PDF: {{e}}")
    {file_var_name}_text = None
"""
    
    elif ext in ['.csv']:
        return f"""
# Automatically loaded CSV file
import pandas as pd
try:
    {file_var_name}_df = pd.read_csv(r'{target_path}')
    print(f"📊 Loaded CSV: {{len({file_var_name}_df)}} rows, {{len({file_var_name}_df.columns)}} columns")
    print("Columns:", list({file_var_name}_df.columns))
    print("\\nFirst 5 rows:")
    print({file_var_name}_df.head())
    print(f"\\n--- CSV loaded as '{file_var_name}_df' variable ---")
    {capture_code}
except Exception as e:
    print(f"Error loading CSV: {{e}}")
    {file_var_name}_df = None
"""
    
    elif ext in ['.xlsx', '.xls']:
        return f"""
# Automatically loaded Excel file
import pandas as pd
try:
    {file_var_name}_df = pd.read_excel(r'{target_path}')
    print(f"📊 Loaded Excel: {{len({file_var_name}_df)}} rows, {{len({file_var_name}_df.columns)}} columns")
    print("Columns:", list({file_var_name}_df.columns))
    print("\\nFirst 5 rows:")
    print({file_var_name}_df.head())
    print(f"\\n--- Excel loaded as '{file_var_name}_df' variable ---")
    {capture_code}
except Exception as e:
    print(f"Error loading Excel: {{e}}")
    {file_var_name}_df = None
"""
    
    elif ext in ['.json']:
        return f"""
# Automatically loaded JSON file
import json
try:
    with open(r'{target_path}', 'r') as file:
        {file_var_name}_data = json.load(file)
    print(f"📄 Loaded JSON: {{type({file_var_name}_data)}}")
    if isinstance({file_var_name}_data, dict):
        print("Keys:", list({file_var_name}_data.keys()))
    elif isinstance({file_var_name}_data, list):
        print(f"List with {{len({file_var_name}_data)}} items")
    print(f"\\n--- JSON loaded as '{file_var_name}_data' variable ---")
    {capture_code}
except Exception as e:
    print(f"Error loading JSON: {{e}}")
    {file_var_name}_data = None
"""
    
    elif ext in ['.txt', '.md']:
        return f"""
# Automatically loaded text file
try:
    with open(r'{target_path}', 'r', encoding='utf-8') as file:
        {file_var_name}_text = file.read()
    print(f"📄 Loaded text file: {{len({file_var_name}_text)}} characters")
    print("First 500 characters:")
    print({file_var_name}_text[:500])
    print(f"\\n--- Text loaded as '{file_var_name}_text' variable ---")
    {capture_code}
except Exception as e:
    print(f"Error loading text file: {{e}}")
    {file_var_name}_text = None
"""
    
    elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
        return f"""
# Automatically loaded image file
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
try:
    {file_var_name}_img = Image.open(r'{target_path}')
    {file_var_name}_array = np.array({file_var_name}_img)
    print(f"🖼️ Loaded image: {{({file_var_name}_img.size)}} pixels, mode: {{{file_var_name}_img.mode}}")
    print(f"Array shape: {{{file_var_name}_array.shape}}")
    
    plt.figure(figsize=(8, 6))
    plt.imshow({file_var_name}_img)
    plt.axis('off')
    plt.title('Loaded Image: {Path(file_path).name}')
    plt.show()
    print(f"\\n--- Image loaded as '{file_var_name}_img' and '{file_var_name}_array' variables ---")
    {capture_code}
except Exception as e:
    print(f"Error loading image: {{e}}")
    {file_var_name}_img = None
    {file_var_name}_array = None
"""
    
    else:
        return f"""
# Automatically loaded file (unknown type)
try:
    with open(r'{target_path}', 'rb') as file:
        {file_var_name}_data = file.read()
    print(f"📄 Loaded binary file: {{len({file_var_name}_data)}} bytes")
    print(f"File extension: {ext}")
    print(f"\\n--- Binary data loaded as '{file_var_name}_data' variable ---")
    {capture_code}
except Exception as e:
    print(f"Error loading file: {{e}}")
    {file_var_name}_data = None
"""




def _handle_guac_refresh(state: ShellState, project_name: str, src_dir: Path):
    if not state.command_history or not state.npc:
        print("Cannot refresh: command history or NPC not available.")
        return
    
    history_entries = state.command_history.get_all()
    if not history_entries:
        print("No command history to analyze for refresh.")
        return
    
    py_commands = []
    for entry in history_entries: 
        if len(entry) > 2 and isinstance(entry[2], str) and entry[2].strip() and not entry[2].startswith('/'):
            py_commands.append(entry[2]) 
    
    if not py_commands:
        print("No relevant commands in history to analyze for refresh.")
        return

    prompt_parts = [
        "Analyze the following Python commands or natural language queries that led to Python code execution by a user:",
        "\n```python",
        "\n".join(py_commands[-20:]),
        "```\n",
        "Based on these, suggest 1-3 useful Python helper functions that the user might find valuable.",
        "Provide only the Python code for these functions, wrapped in ```python ... ``` blocks.",
        "Do not include any other text or explanation outside the code blocks."
    ]
    prompt = "\n".join(prompt_parts)

    try:
        # Ensure state.npc is not None before accessing .model or .provider
        npc_model = state.npc.model if state.npc and state.npc.model else state.chat_model
        npc_provider = state.npc.provider if state.npc and state.npc.provider else state.chat_provider

        response = get_llm_response(prompt, 
                                    model=npc_model, 
                                    provider=npc_provider, 
                                    npc=state.npc, 
                                    stream=False)
        suggested_code_raw = response.get("response", "").strip()
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', suggested_code_raw, re.DOTALL)
        
        if not code_blocks:
            if "def " in suggested_code_raw:
                code_blocks = [suggested_code_raw]
            else:
                print("\nNo functions suggested by LLM or format not recognized.")
                return
        
        suggested_functions_code = "\n\n".join(block.strip() for block in code_blocks)
        if not suggested_functions_code.strip():
            print("\nLLM did not suggest any functions.")
            return
        
        print("\n=== Suggested Helper Functions ===\n")
        render_markdown(f"```python\n{suggested_functions_code}\n```")
        print("\n===============================\n")
        
        user_choice = input("Add these functions to your main.py? (y/n): ").strip().lower()
        if user_choice == 'y':
            main_py_path = src_dir / "main.py"
            with open(main_py_path, "a") as f:
                f.write("\n\n# --- Functions suggested by /refresh ---\n")
                f.write(suggested_functions_code)
                f.write("\n# --- End of suggested functions ---\n")
            print(f"Functions appended to {main_py_path}.")
            print(f"To use them in the current session: import importlib; importlib.reload({project_name}.src.main); from {project_name}.src.main import *")
        else:
            print("Suggested functions not added.")
    except Exception as e:
        print(f"Error during /refresh: {e}")
        traceback.print_exc()



def ensure_global_guac_team():
    """Ensure a global guac team exists at ~/.npcsh/guac/npc_team/."""
    base_dir = Path.home() / ".npcsh" / "guac"
    team_dir = base_dir / "npc_team"
    team_dir.mkdir(parents=True, exist_ok=True)

    guac_npc_path = team_dir / "guac.npc"
    if not guac_npc_path.exists():
        guac = {
            "name": "guac",
            "primary_directive": (
                "You are guac, the global coordinator NPC for Guac Mode. "
                "Always prioritize Python code, concise answers, and coordination."
            ),
            "model": os.environ.get("NPCSH_CHAT_MODEL", "gemma3:4b"),
            "provider": os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
        }
        with open(guac_npc_path, "w") as f:
            yaml.dump(guac, f, default_flow_style=False)
        print(f"✅ Created global guac NPC at {guac_npc_path}")

    ctx_path = team_dir / "team.ctx"
    if not ctx_path.exists():
        ctx = {
            "team_name": "guac_global_team",
            "forenpc": "guac",
            "model": os.environ.get("NPCSH_CHAT_MODEL", "gemma3:4b"),
            "provider": os.environ.get("NPCSH_CHAT_PROVIDER", "ollama"),
            "description": "Global guac team context"
        }
        with open(ctx_path, "w") as f:
            yaml.dump(ctx, f, default_flow_style=False)
        print(f"✅ Created global guac team.ctx at {ctx_path}")

    return team_dir
def setup_guac_mode(config_dir=None, plots_dir=None, npc_team_dir=None,
                    lang='python', default_mode_choice=None):
    base_dir = Path.cwd()                
    
    # Check if we should default to global without prompting
    if GUAC_GLOBAL_FLAG_FILE.exists():
        print("💡 Using global Guac team as default (previously set).")
        team_dir = ensure_global_guac_team()
        return {
            "language": lang, "package_root": team_dir, "plots_dir": plots_dir,
            "npc_team_dir": team_dir, "config_dir": config_dir, "default_mode": default_mode_choice or "agent",
            "project_description": "Global guac team for analysis.", "package_name": "guac"
        }

    # default: project npc_team_dir
    if npc_team_dir is None:
        npc_team_dir = base_dir / "npc_team"
    else:
        npc_team_dir = Path(npc_team_dir)
    npc_team_dir.mkdir(parents=True, exist_ok=True)
    workspace_dirs = _get_workspace_dirs(npc_team_dir)
    _ensure_workspace_dirs(workspace_dirs)

    team_ctx_path = npc_team_dir / "team.ctx"
    existing_ctx = {}
    if team_ctx_path.exists():
        try:
            with open(team_ctx_path, "r") as f:
                existing_ctx = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not read team.ctx: {e}")

    package_root = existing_ctx.get("GUAC_PACKAGE_ROOT")
    package_name = existing_ctx.get("GUAC_PACKAGE_NAME")

    if package_root is None or package_name is None:
        try:
            response = input("Enter package root (Enter for current dir): ").strip()
            package_root = response if response else str(base_dir)
            response = input("Enter package name (Enter for 'project'): ").strip()
            package_name = response if response else "project"
        except (KeyboardInterrupt, EOFError):
            print("⚠️ Project setup interrupted. Falling back to global guac team...")
            GUAC_GLOBAL_FLAG_FILE.touch() # Create the flag file to remember this choice
            team_dir = ensure_global_guac_team()
            return {
                "language": lang, "package_root": team_dir, "plots_dir": plots_dir,
                "npc_team_dir": team_dir, "config_dir": config_dir, "default_mode": default_mode_choice or "agent",
                "project_description": "Global guac team for analysis.", "package_name": "guac"
            }

    project_description = existing_ctx.get("GUAC_PROJECT_DESCRIPTION", "")
    if not project_description:
        try:
            project_description = input("Enter a project description: ").strip() or "No description."
        except (KeyboardInterrupt, EOFError):
            project_description = "No description provided."

    updated_ctx = {**existing_ctx}
    updated_ctx.update({
        "GUAC_TEAM_NAME": "guac_team",
        "GUAC_DESCRIPTION": f"A team for {lang} analysis for project {package_name}",
        "GUAC_FORENPC": "guac", "GUAC_PROJECT_DESCRIPTION": project_description,
        "GUAC_LANG": lang, "GUAC_PACKAGE_ROOT": package_root, "GUAC_PACKAGE_NAME": package_name,
        "GUAC_WORKSPACE_PATHS": {k: str(v) for k, v in workspace_dirs.items()},
    })

    pkg_root_path = Path(package_root)
    try:
        pkg_root_path.mkdir(parents=True, exist_ok=True)
        (pkg_root_path / package_name / "__init__.py").touch()
    except Exception as e:
        logging.warning("Could not ensure package root/dir: %s", e)

    with open(team_ctx_path, "w") as f:
        yaml.dump(updated_ctx, f, default_flow_style=False)
    print("Updated team.ctx with GUAC-specific information.")

    setup_py_path = pkg_root_path / "setup.py"
    desc = project_description.replace('"', '\\"')
    if not setup_py_path.exists():
        setup_content = f'''from setuptools import setup, find_packages
setup(name="{package_name}", version="0.0.1", description="{desc}", packages=find_packages())
'''
        setup_py_path.write_text(setup_content)
        logging.info("Created minimal setup.py at %s", setup_py_path)

    default_mode_val = default_mode_choice or "agent"
    setup_npc_team(npc_team_dir, lang)

    print(f"\nGuac mode configured for package: {package_name} at {package_root}")
    print(f"Workspace created at: {workspace_dirs['workspace']}")
    return {
        "language": lang, "package_root": Path(package_root), "plots_dir": plots_dir,
        "npc_team_dir": npc_team_dir, "config_dir": config_dir, "default_mode": default_mode_val,
        "project_description": project_description, "package_name": package_name
    }
def setup_npc_team(npc_team_dir, lang, is_subteam=False):
    # Create Guac-specific NPCs
    guac_npc = {
        "name": "guac", 
        "primary_directive": (
            f"You are guac, an AI assistant operating in a Python environment. "
            f"When asked to perform actions or generate code, prioritize Python. "
            f"For general queries, provide concise answers. "
            f"When routing tasks (agent mode), consider Python-based tools or direct Python code generation if appropriate. "
            f"If generating code directly (cmd mode), ensure it's Python."
        )
    }
    caug_npc = {
        "name": "caug",
        "primary_directive": f"You are caug, a specialist in big data statistical methods in {lang}. You never make scatter plots with discrete values unless asked. "
    }

    parsely_npc = {
        "name": "parsely",
        "primary_directive": f"You are parsely, a specialist in mathematical methods in {lang}."
    }

    toon_npc = {
        "name": "toon",
        "primary_directive": f"You are toon, a specialist in brute force methods in {lang}."
    }

    for npc_data in [guac_npc, caug_npc, parsely_npc, toon_npc]:
        npc_file = npc_team_dir / f"{npc_data['name']}.npc"
        if not npc_file.exists():  # Don't overwrite existing NPCs
            with open(npc_file, "w") as f:
                yaml.dump(npc_data, f, default_flow_style=False)
            print(f"Created NPC: {npc_data['name']}")
        else:
            print(f"NPC already exists: {npc_data['name']}")

    # Only create team.ctx for subteams, otherwise use the main one
    if is_subteam:
        team_ctx_model = os.environ.get("NPCSH_CHAT_MODEL", "gemma3:4b")
        team_ctx_provider = os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
        team_ctx = {
            "team_name": "guac_team", 
            "description": f"A subteam for {lang} analysis", 
            "forenpc": "guac",
            "model": team_ctx_model, 
            "provider": team_ctx_provider
        }
        with open(npc_team_dir / "team.ctx", "w") as f:
            yaml.dump(team_ctx, f, default_flow_style=False)

def _get_workspace_dirs(npc_team_dir: Path) -> Dict[str, Path]:
    """Get workspace directories from the npc_team directory"""
    workspace_dir = npc_team_dir / "guac_workspace"
    return {
        "workspace": workspace_dir,
        "plots": workspace_dir / "plots", 
        "data_inputs": workspace_dir / "data_inputs",
        "data_outputs": workspace_dir / "data_outputs"
    }

def _ensure_workspace_dirs(workspace_dirs: Dict[str, Path]):
    """Ensure all workspace directories exist"""
    for directory in workspace_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
import shutil

def _detect_file_drop(input_text: str) -> bool:
    """Detect if input is just a file path (drag and drop)"""
    
    stripped = input_text.strip()
    
    # Remove quotes if present
    if stripped.startswith("'") and stripped.endswith("'"):
        stripped = stripped[1:-1]
    elif stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1]
    
    # Must be a single token (no spaces) - this is key!
    if len(stripped.split()) != 1:
        return False
    
    # Must not contain Python operators or syntax
    python_indicators = ['(', ')', '[', ']', '{', '}', '=', '+', '-', '*', '/', '%', '&', '|', '^', '<', '>', '!', '?', ':', ';', ',']
    if any(indicator in stripped for indicator in python_indicators):
        return False
    
    # Must not start with common Python keywords or look like Python
    python_keywords = ['import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', 'with', 'lambda', 'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple']
    if any(stripped.startswith(keyword) for keyword in python_keywords):
        return False
    

import hashlib
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Add these classes after your imports
Base = declarative_base()

class PlotState(Base):
    __tablename__ = 'plot_states'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255))
    plot_hash = Column(String(32))
    plot_description = Column(Text)
    figure_path = Column(String(500))
    data_summary = Column(String(500))
    change_significance = Column(Float)
    timestamp = Column(DateTime, default=func.now())

class FileAnalysisState(Base):
    __tablename__ = 'file_analysis_states'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255))
    file_path = Column(String(1000))
    file_hash = Column(String(32))
    analysis_summary = Column(Text)
    variable_names = Column(Text)
    timestamp = Column(DateTime, default=func.now())

def _capture_plot_state(session_id: str, db_path: str, npc_team_dir: Path):
    """Capture plot state if significant change"""
    if not plt.get_fignums():
        return
    
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Get plot info
    fig = plt.gcf()
    axes = fig.get_axes()
    data_points = sum(len(line.get_xdata()) for ax in axes for line in ax.get_lines())
    
    # Create hash and check if different from last
    plot_hash = hashlib.md5(f"{len(axes)}{data_points}".encode()).hexdigest()
    
    last = session.query(PlotState).filter(PlotState.session_id == session_id).order_by(PlotState.timestamp.desc()).first()
    if last and last.plot_hash == plot_hash:
        session.close()
        return
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dirs = _get_workspace_dirs(npc_team_dir)
    plot_path = workspace_dirs["plots"] / f"state_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    # Save to DB
    plot_state = PlotState(
        session_id=session_id,
        plot_hash=plot_hash,
        plot_description=f"Plot with {len(axes)} axes, {data_points} points",
        figure_path=str(plot_path),
        data_summary=f"{data_points} data points",
        change_significance=1.0 if not last else 0.5
    )
    
    session.add(plot_state)
    session.commit()
    session.close()
    print(f"📊 Plot state captured -> {plot_path.name}")

def _capture_file_state(session_id: str, db_path: str, file_path: str, analysis_code: str, locals_dict: Dict):
    """Capture file analysis state"""
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Get file hash
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
    except:
        file_hash = "unknown"
    
    # Get variables created
    file_stem = Path(file_path).stem.lower()
    vars_created = [k for k in locals_dict.keys() if not k.startswith('_') and file_stem in k.lower()]
    
    file_state = FileAnalysisState(
        session_id=session_id,
        file_path=file_path,
        file_hash=file_hash,
        analysis_summary=f"Loaded {Path(file_path).name} -> {len(vars_created)} variables",
        variable_names=json.dumps(vars_created)
    )
    
    session.add(file_state)
    session.commit()
    session.close()
    print(f"📁 File state captured: {Path(file_path).name}")

def _get_plot_context(session_id: str, db_path: str) -> str:
    """Get plot context for LLM"""
    engine = create_engine(f'sqlite:///{db_path}')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    plots = session.query(PlotState).filter(PlotState.session_id == session_id).order_by(PlotState.timestamp.desc()).limit(3).all()
    session.close()
    
    if not plots:
        return "No plots in session."
    
    context = "Recent plots:\n"
    for i, plot in enumerate(plots):
        if i == 0:
            context += f"📊 CURRENT: {plot.plot_description}\n"
        else:
            context += f"📊 Previous: {plot.plot_description}\n"
    return context

def _get_file_context(session_id: str, db_path: str) -> str:
    """Get file context for LLM"""
    engine = create_engine(f'sqlite:///{db_path}')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    files = session.query(FileAnalysisState).filter(FileAnalysisState.session_id == session_id).order_by(FileAnalysisState.timestamp.desc()).all()
    session.close()
    
    if not files:
        return "No files analyzed."
    
    context = "Analyzed files:\n"
    for file in files:
        context += f"📁 {Path(file.file_path).name}: {file.analysis_summary}\n"
    return context
def _generate_file_analysis_code(file_path: str, target_path: str) -> str:
    """Generate Python code to load and analyze the dropped file"""
    ext = Path(file_path).suffix.lower()
    file_var_name = f"file_{datetime.now().strftime('%H%M%S')}"
    
    if ext == '.pdf':
        return f"""
# Automatically loaded PDF file
import PyPDF2
import pandas as pd
try:
    with open(r'{target_path}', 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        {file_var_name}_text = ""
        for page_num in range(len(pdf_reader.pages)):
            {file_var_name}_text += pdf_reader.pages[page_num].extract_text()
    
    print(f"📄 Loaded PDF: {{len(pdf_reader.pages)}} pages, {{len({file_var_name}_text)}} characters")
    print("First 500 characters:")
    print({file_var_name}_text[:500])
    print("\\n--- PDF loaded as '{file_var_name}_text' variable ---")
except Exception as e:
    print(f"Error loading PDF: {{e}}")
    {file_var_name}_text = None
"""
    
    elif ext in ['.csv']:
        return f"""
# Automatically loaded CSV file
import pandas as pd
try:
    {file_var_name}_df = pd.read_csv(r'{target_path}')
    print(f"📊 Loaded CSV: {{len({file_var_name}_df)}} rows, {{len({file_var_name}_df.columns)}} columns")
    print("Columns:", list({file_var_name}_df.columns))
    print("\\nFirst 5 rows:")
    print({file_var_name}_df.head())
    print(f"\\n--- CSV loaded as '{file_var_name}_df' variable ---")
except Exception as e:
    print(f"Error loading CSV: {{e}}")
    {file_var_name}_df = None
"""
    
    elif ext in ['.xlsx', '.xls']:
        return f"""
# Automatically loaded Excel file
import pandas as pd
try:
    {file_var_name}_df = pd.read_excel(r'{target_path}')
    print(f"📊 Loaded Excel: {{len({file_var_name}_df)}} rows, {{len({file_var_name}_df.columns)}} columns")
    print("Columns:", list({file_var_name}_df.columns))
    print("\\nFirst 5 rows:")
    print({file_var_name}_df.head())
    print(f"\\n--- Excel loaded as '{file_var_name}_df' variable ---")
except Exception as e:
    print(f"Error loading Excel: {{e}}")
    {file_var_name}_df = None
"""
    
    elif ext in ['.json']:
        return f"""
# Automatically loaded JSON file
import json
try:
    with open(r'{target_path}', 'r') as file:
        {file_var_name}_data = json.load(file)
    print(f"📄 Loaded JSON: {{type({file_var_name}_data)}}")
    if isinstance({file_var_name}_data, dict):
        print("Keys:", list({file_var_name}_data.keys()))
    elif isinstance({file_var_name}_data, list):
        print(f"List with {{len({file_var_name}_data)}} items")
    print(f"\\n--- JSON loaded as '{file_var_name}_data' variable ---")
except Exception as e:
    print(f"Error loading JSON: {{e}}")
    {file_var_name}_data = None
"""
    
    elif ext in ['.txt', '.md']:
        return f"""
# Automatically loaded text file
try:
    with open(r'{target_path}', 'r', encoding='utf-8') as file:
        {file_var_name}_text = file.read()
    print(f"📄 Loaded text file: {{len({file_var_name}_text)}} characters")
    print("First 500 characters:")
    print({file_var_name}_text[:500])
    print(f"\\n--- Text loaded as '{file_var_name}_text' variable ---")
except Exception as e:
    print(f"Error loading text file: {{e}}")
    {file_var_name}_text = None
"""
    
    elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
        return f"""
# Automatically loaded image file
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
try:
    {file_var_name}_img = Image.open(r'{target_path}')
    {file_var_name}_array = np.array({file_var_name}_img)
    print(f"🖼️ Loaded image: {{({file_var_name}_img.size)}} pixels, mode: {{{file_var_name}_img.mode}}")
    print(f"Array shape: {{{file_var_name}_array.shape}}")
    
    plt.figure(figsize=(8, 6))
    plt.imshow({file_var_name}_img)
    plt.axis('off')
    plt.title('Loaded Image: {Path(file_path).name}')
    plt.show()
    print(f"\\n--- Image loaded as '{file_var_name}_img' and '{file_var_name}_array' variables ---")
except Exception as e:
    print(f"Error loading image: {{e}}")
    {file_var_name}_img = None
    {file_var_name}_array = None
"""
    
    else:
        return f"""
# Automatically loaded file (unknown type)
try:
    with open(r'{target_path}', 'rb') as file:
        {file_var_name}_data = file.read()
    print(f"📄 Loaded binary file: {{len({file_var_name}_data)}} bytes")
    print(f"File extension: {ext}")
    print(f"\\n--- Binary data loaded as '{file_var_name}_data' variable ---")
except Exception as e:
    print(f"Error loading file: {{e}}")
    {file_var_name}_data = None
"""
def _handle_file_drop(input_text: str, npc_team_dir: Path) -> Tuple[str, List[str]]:
    """Handle file drops by copying files to appropriate workspace directories"""
    #print(f"[DEBUG] _handle_file_drop called with input: '{input_text}'")
    
    # Immediately check if this is a single file path
    stripped = input_text.strip("'\"")
    if os.path.exists(stripped) and os.path.isfile(stripped):
        print(f"[DEBUG] Direct file drop detected: {stripped}")
        
        workspace_dirs = _get_workspace_dirs(npc_team_dir)
        _ensure_workspace_dirs(workspace_dirs)
        
        expanded_path = Path(stripped).resolve()
        
        ext = expanded_path.suffix[1:].upper() if expanded_path.suffix else "OTHERS"
        category = EXTENSION_MAP.get(ext, "data_inputs")
        target_dir = workspace_dirs.get(category, workspace_dirs["data_inputs"])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{expanded_path.name}"
        target_path = target_dir / new_filename
        
        try:
            shutil.copy2(expanded_path, target_path)
            print(f"📁 Copied {expanded_path.name} to workspace: {target_path}")
            
            # Generate and execute loading code
            loading_code = _generate_file_analysis_code(str(expanded_path), str(target_path))
            print(f"\n# Auto-generated file loading code:\n---\n{loading_code}\n---\n")
            
            # Actually execute the loading code
            exec(loading_code)
            
            return "", [str(target_path)]
        except Exception as e:
            print(f"[ERROR] Failed to process file drop: {e}")
            return input_text, []
    
    # Existing multi-file handling logic
    processed_files = []
    file_paths = re.findall(r"'([^']+)'|\"([^\"]+)\"|(\S+)", input_text)
    file_paths = [path for group in file_paths for path in group if path]
    
    #print(f"[DEBUG] Found file paths: {file_paths}")
    
    if not file_paths:

        return input_text, processed_files, file_paths
    
    modified_input = input_text

    return modified_input, processed_files, file_paths


def _capture_plot_state(session_id: str, db_path: str, npc_team_dir: Path):
    """Capture plot state if significant change"""
    if not plt.get_fignums():
        return
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get plot info
        fig = plt.gcf()
        axes = fig.get_axes()
        data_points = sum(len(line.get_xdata()) for ax in axes for line in ax.get_lines())
        
        # Create hash and check if different from last
        plot_hash = hashlib.md5(f"{len(axes)}{data_points}".encode()).hexdigest()
        
        last = session.query(PlotState).filter(PlotState.session_id == session_id).order_by(PlotState.timestamp.desc()).first()
        if last and last.plot_hash == plot_hash:
            session.close()
            return
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_dirs = _get_workspace_dirs(npc_team_dir)
        plot_path = workspace_dirs["plots"] / f"state_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Save to DB
        plot_state = PlotState(
            session_id=session_id,
            plot_hash=plot_hash,
            plot_description=f"Plot with {len(axes)} axes, {data_points} points",
            figure_path=str(plot_path),
            data_summary=f"{data_points} data points",
            change_significance=1.0 if not last else 0.5
        )
    
        session.add(plot_state)
        session.commit()
        session.close()
        print(f"📊 Plot state captured -> {plot_path.name}")
        
    except Exception as e:
        print(f"Error capturing plot state: {e}")

def _capture_file_state(session_id: str, db_path: str, file_path: str, analysis_code: str, locals_dict: Dict):
    """Capture file analysis state"""
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get file hash
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except:
            file_hash = "unknown"
        
        # Get variables created
        file_stem = Path(file_path).stem.lower()
        vars_created = [k for k in locals_dict.keys() if not k.startswith('_') and file_stem in k.lower()]
        
        file_state = FileAnalysisState(
            session_id=session_id,
            file_path=file_path,
            file_hash=file_hash,
            analysis_summary=f"Loaded {Path(file_path).name} -> {len(vars_created)} variables",
            variable_names=json.dumps(vars_created)
        )
        
        session.add(file_state)
        session.commit()
        session.close()
        print(f"📁 File state captured: {Path(file_path).name}")
        
    except Exception as e:
        print(f"Error capturing file state: {e}")

def _get_plot_context(session_id: str, db_path: str) -> str:
    """Get plot context for LLM"""
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        plots = session.query(PlotState).filter(PlotState.session_id == session_id).order_by(PlotState.timestamp.desc()).limit(3).all()
        session.close()
        
        if not plots:
            return "No plots in session."
        
        context = "Recent plots:\n"
        for i, plot in enumerate(plots):
            if i == 0:
                context += f"📊 CURRENT: {plot.plot_description}\n"
            else:
                context += f"📊 Previous: {plot.plot_description}\n"
        return context
        
    except Exception as e:
        return f"Error retrieving plot context: {e}"

def _get_file_context(session_id: str, db_path: str) -> str:
    """Get file context for LLM"""
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        files = session.query(FileAnalysisState).filter(FileAnalysisState.session_id == session_id).order_by(FileAnalysisState.timestamp.desc()).all()
        session.close()
        
        if not files:
            return "No files analyzed."
        
        context = "Analyzed files:\n"
        for file in files:
            context += f"📁 {Path(file.file_path).name}: {file.analysis_summary}\n"
        return context
        
    except Exception as e:
        return f"Error retrieving file context: {e}"



def _save_matplotlib_figures(npc_team_dir: Path) -> List[str]:
    """Save all matplotlib figures to the plots directory and return paths"""
    workspace_dirs = _get_workspace_dirs(npc_team_dir)
    _ensure_workspace_dirs(workspace_dirs)
    
    saved_figures = []
    if plt.get_fignums():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)
            fig_path = workspace_dirs["plots"] / f"{timestamp}_figure_{i+1}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            saved_figures.append(str(fig_path))
            print(f"📊 Saved figure to: {fig_path}")
        
        plt.close('all')
    
    return saved_figures

import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# --- New Emoji Function for Agentic Mode ---
def _get_guac_agent_emoji(failures: int, max_fail: int = 3) -> str:
    """
    Returns an avocado emoji representing the state based on consecutive failures.
    Includes "puke" emoji for max_fail, and "skull" for exceeding max_fail + 20.
    """
    if failures == 0:
        return "🥑"   # Fresh
    elif failures == 1:
        return "🥑🔪" # Sliced, contemplating next steps
    elif failures == 2:
        return "🥑🥣" # In the bowl, getting mashed
    elif failures == max_fail:
        return "🥑🤢" # Going bad, critical issue (puke)
    elif failures > max_fail + 20: # Skull for 20+ over max
        return "🥑💀" # Rotten
    elif failures > max_fail:
        return "🥑🟤" # Bruised and bad
    else:
        return "🥑❓" # Unknown state

# --- New Helper for Persisting Global Choice ---
GUAC_GLOBAL_FLAG_FILE = Path.home() / ".npcsh" / ".guac_use_global"


def _run_agentic_mode(command: str,
                      state: ShellState,
                      locals_dict: Dict[str, Any],
                      npc_team_dir: Path) -> Tuple[ShellState, Any]:
    """Run agentic mode with continuous iteration based on progress"""
    max_iterations = 5  # Increased slightly for more complex tasks
    iteration = 0
    full_output = []
    current_command = command
    consecutive_failures = 0
    max_consecutive_failures = 3 # This is the limit before stopping

    # Build context of existing variables
    existing_vars_context = "EXISTING VARIABLES IN ENVIRONMENT:\n"
    for var_name, var_value in locals_dict.items():
        if not var_name.startswith('_') and var_name not in ['In', 'Out', 'exit', 'quit', 'get_ipython']:
            try:
                var_type = type(var_value).__name__
                var_repr = repr(var_value)
                if len(var_repr) > 100:
                    var_repr = var_repr[:97] + "..."
                existing_vars_context += f"- {var_name} ({var_type}): {var_repr}\n"
            except:
                existing_vars_context += f"- {var_name} ({type(var_value).__name__}): <unrepresentable>\n"
    previous_code = ''
    next_step = ''
    steps = []
    while iteration < max_iterations and consecutive_failures < max_consecutive_failures:
        iteration += 1
        print(f"\n{_get_guac_agent_emoji(consecutive_failures, max_consecutive_failures)} Agentic iteration {iteration} ")

        prompt = f"""
        USER REQUEST: {current_command} {next_step} 

        Here is the existing variable context:

        ```
        {existing_vars_context}
        ```
        PREVIOUS ATTEMPTS: ```{full_output[-1] if full_output else 'None'}```

        DO NOT SIMPLY COPY A PREVIOUS ATTEMPT.        
        
        Your goal is to generate Python code that BUILDS ON EXISTING VARIABLES to accomplish this task: {current_command}, with this next step planned: `{next_step} `

        
        You will notice in the local envs that there are functions for reading, editing, and loading files. 
        You should use these to your advantage as they will help you to clearly understand the user's system best.

        Here are all the previous steps: {steps}
        
        DO NOT redefine variables that already exist unless absolutely necessary.
        Use the existing variables and add/modify as needed.
        Be sure to generate logs and information  that once executed provide us with enough information to keep moving forward.
        log variables and behaviors so we can pinpoint fixes clearly rather than getting stuck in nonsensical problematic loops.
        
        Provide ONLY executable Python code without any explanations or markdown formatting.
        Focus on incremental changes rather than rewriting everything. Do not re-write any functions that are currently within the existing vars context or which appear to have no need to be changed.

        Do not include any leading ```python. Begin directly with the code.
        Do not write your code to include a __main__ check or portion unless the user asks.
        These should be functional components and building blocks that you and the user will take and build a great
        library of modules. Keep things straightforward and do not do unnecessary exception handling unless requested.
        Failing fast in research is important and so it is necessary to  
        No try except blocks unless requested.
        Determine and log information in a way that helps us move forward rather than by exception handling.
        Do not simply generate code that resembles the previous code. 
        While this code may one day be `production` code with such error handling,
        at the moment, we are simply in the business of experimentation.
        Do not use the python `input()` function. if you have a question, ask directly by typing <request_for_input> request </request_for_input>        
    
        users may ask you to edit code directly. do this by loading the code in and evaluating it. once it is evaluated, you may attempt to write changes to it.

        Always evaluate before attempting to fix. Read first. Gather information. Look at files. This will not be your final code, this is just part of 
        an ongoing workflow. 


        If a user is asking for help debugging, it's better to figure out what is wrong by attempting to run it yourself, and if they do not prefer that,
        then it's best to use static parsing methods and arguments based on deduction rather than attempting to just fix everything over and over. 

        Do not over- complicate the code.

        Do not include any '__name__'=='__main__' block.
        """
        
        npc_model = state.npc.model if state.npc and state.npc.model else state.chat_model
        npc_provider = state.npc.provider if state.npc and state.npc.provider else state.chat_provider
        
        llm_response = get_llm_response(prompt,
                                        npc=state.npc,
                                        stream=True,
                                        messages=state.messages)

        generated_code = print_and_process_stream(llm_response.get('response'),
                                                  npc_model,
                                                  npc_provider
                                                  )
        
        state.messages.append({'role':'user', 'content':current_command })
        state.messages.append({'role':'assistant', 'content': generated_code})        

        if '<request_for_input>' in generated_code:
            generated_code = generated_code.split('>')[1].split('<')[0]
            user_feedback = input("\n🤔 Agent requests feedback (press Enter to continue or type your input): ").strip()
            current_command = f"{current_command} - User feedback: {user_feedback}"
            max_iterations += int(max_iterations/2)
            continue

        if generated_code.startswith('```python'):
            generated_code = generated_code[len('```python'):].strip()
        if generated_code.endswith('```'):
            generated_code = generated_code[:-3].strip()
        
        try:
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                state, exec_output = execute_python_code(generated_code, 
                                                         state, 
                                                         locals_dict)

            captured_stdout = stdout_capture.getvalue()
            captured_stderr = stderr_capture.getvalue()
            if exec_output: print(exec_output)
            if captured_stdout: 
                print("\n📤 Captured stdout:\n", captured_stdout)
            if captured_stderr: 
                print("\n❌ Captured stderr:\n", captured_stderr)

            combined_output = f"{exec_output}\nstdout:\n{captured_stdout}\nstderr:\n{captured_stderr}"
            full_output.append(f"Iteration {iteration}:\nCode:\n{generated_code}\nOutput:\n{combined_output}")

            new_vars = []
            for var_name, var_value in locals_dict.items():
                if (not var_name.startswith('_') and 
                    var_name not in existing_vars_context and 
                    var_name not in ['In', 'Out', 'exit', 'quit', 'get_ipython']):
                    new_vars.append(var_name)
            if new_vars:
                existing_vars_context += f"\nNEW VARIABLES CREATED: {', '.join(new_vars)}\n"

            analysis_prompt = f"""
            CODE EXECUTION RESULTS: {combined_output}
            EXISTING VARIABLES: {existing_vars_context}
            EXECUTED_CODE: {generated_code}
            PREVIOUS_CODE: {previous_code}
            PREVIOUS ATTEMPTS: ```{full_output[-3:] if full_output else 'None'}```
            Here are the steps so far: {steps}
            ANALYSIS: 
            - Is there MEANINGFUL PROGRESS? Return 'progress' if making good progress.
            - Is there a PROBLEM? Return 'problem' if stuck or error occurred.
            - Is there an AMBIGUITY that should be resolved? Return 'question'.
            - Is the analysis COMPLETE enough to get feedback? If it's pretty much done, return 'complete'.
            - Return ONLY one of these words followed by a brief explanation for the next step.
            """
            analysis_response = get_llm_response(analysis_prompt, npc=state.npc, stream=False)
            analysis = analysis_response.get("response", "").strip().lower()
            next_step = analysis[8:]
            state.messages.append({'role':'assistant', 
                                   'content':f'Is there progress? is there a problem/ is there ambiguity? is it complete?\n {analysis}'})

            if analysis.startswith('complete'):
                print(f"✅ Task completed! {_get_guac_agent_emoji(0, max_consecutive_failures)}")
                break
            elif analysis.startswith('question'):
                print(f"🤔 Agent has a question: {next_step} {_get_guac_agent_emoji(consecutive_failures, max_consecutive_failures)}")
                break
            elif analysis.startswith('progress'):
                consecutive_failures = 0
                print(f"➡️ Making progress... {_get_guac_agent_emoji(consecutive_failures, max_consecutive_failures)}")
                current_command = next_step
            elif analysis.startswith('problem'):
                consecutive_failures += 1
                print(f"⚠️ Problem detected ({consecutive_failures}/{max_consecutive_failures}) {_get_guac_agent_emoji(consecutive_failures, max_consecutive_failures)}")
                current_command = f"{current_command} - PROBLEM: {analysis}"
            else:
                consecutive_failures += 1
                print(f"❓ Unexpected analysis, counting as failure ({consecutive_failures}/{max_consecutive_failures}) {_get_guac_agent_emoji(consecutive_failures, max_consecutive_failures)}")
        except KeyboardInterrupt:
            user_input = input('User input: ')
            current_command += user_input
        except Exception as e:
            consecutive_failures += 1
            error_msg = f"Error in iteration {iteration}: {str(e)} {_get_guac_agent_emoji(consecutive_failures, max_consecutive_failures)}"
            print(error_msg)
            full_output.append(error_msg)
            current_command = f"{current_command} - Error: {str(e)}"

    if consecutive_failures >= max_consecutive_failures:
        print(f"❌ Too many consecutive failures, stopping. {_get_guac_agent_emoji(consecutive_failures, max_consecutive_failures)}")

    return state, "# Agentic execution completed\n" + '\n'.join(full_output)

def print_guac_bowl():
    bowl_art = """
  🟢🟢🟢🟢🟢 
🟢          🟢
🟢  
🟢      
🟢      
🟢      🟢🟢🟢   🟢    🟢   🟢🟢🟢    🟢🟢🟢
🟢           🟢  🟢    🟢    ⚫⚫🟢  🟢
🟢           🟢  🟢    🟢  ⚫🥑🧅⚫  🟢
🟢           🟢  🟢    🟢  ⚫🥑🍅⚫  🟢
 🟢🟢🟢🟢🟢🟢    🟢🟢🟢🟢    ⚫⚫🟢   🟢🟢🟢 
"""
    print(bowl_art)

def get_guac_prompt_char(command_count: int, guac_refresh_period = 100) -> str:
    period = int(guac_refresh_period)
    period = max(1, period)
    stages = ["\U0001F951", "\U0001F951🔪", "\U0001F951🥣", "\U0001F951🥣🧂", "\U0001F958 REFRESH?"]
    divisor = max(1, period // (len(stages)-1) if len(stages) > 1 else period)
    stage_index = min(command_count // divisor, len(stages) - 1)
    return stages[stage_index]

def execute_guac_command(command: str, state: ShellState, locals_dict: Dict[str, Any], project_name: str, src_dir: Path, router) -> Tuple[ShellState, Any]:
    stripped_command = command.strip()
    output = None 
    cmd_parts = shlex.split(stripped_command)
    if cmd_parts and cmd_parts[0] in ["cd", "ls", "pwd"]:
        return execute_command(stripped_command, state, review=False, router=router)

    npc_team_dir = Path(state.team.team_path) if state.team and hasattr(state.team, 'team_path') else Path.cwd() / "npc_team"

    if not stripped_command:
        return state, None
    if stripped_command.lower() in ["exit", "quit", "exit()", "quit()"]:
        raise SystemExit("Exiting Guac Mode.")

    if stripped_command.startswith('run '):
        file_path = stripped_command[4:].strip()
        try:
            resolved_path = Path(file_path).resolve()
            if not resolved_path.exists():
                return state, f"Error: File '{file_path}' not found"
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            print(f"Running {resolved_path.name}...")
            state, exec_output = execute_python_code(file_content, state, locals_dict)
            return state, exec_output
            
        except Exception as e:
            return state, f"Error running file: {e}"


        
    # Check if this is a file drop (single file path)
    if _detect_file_drop(stripped_command):
        if stripped_command.startswith('run'):
            pass
        else:
            # Clean the path
            file_path = stripped_command.strip("'\"")
            expanded_path = Path(file_path).resolve()
            
            # Copy to workspace
            workspace_dirs = _get_workspace_dirs(npc_team_dir)
            _ensure_workspace_dirs(workspace_dirs)
            
            ext = expanded_path.suffix[1:].upper() if expanded_path.suffix else "OTHERS"
            category = EXTENSION_MAP.get(ext, "data_inputs")
            target_dir = workspace_dirs.get(category, workspace_dirs["data_inputs"])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_{expanded_path.name}"
            target_path = target_dir / new_filename
            
            try:
                shutil.copy2(expanded_path, target_path)
                print(f"📁 Copied {expanded_path.name} to workspace: {target_path}")
                
                # Generate and execute loading code
                loading_code = _generate_file_analysis_code(str(expanded_path), str(target_path))
                print(f"\n# Auto-generated file loading code:\n---\n{loading_code}\n---\n")
                
                state, exec_output = execute_python_code(loading_code, state, locals_dict)
                return state, exec_output
            except Exception as e:
                print(f"[ERROR] Failed to copy or load file: {e}")
                return state, f"Error loading file: {e}"

    # Handle file drops in text (multiple files or files with other text)
    processed_command, processed_files, file_paths = _handle_file_drop(stripped_command, npc_team_dir)
    if processed_files:
        print(f"📁 Processed {len(processed_files)} files")
        stripped_command = processed_command + 'Here are the files associated with the request'

    # Handle /refresh command
    if stripped_command == "/refresh":
        _handle_guac_refresh(state, project_name, src_dir)
        return state, "Refresh process initiated."

    # Handle mode switching commands
    if stripped_command in ["/agent", "/chat", "/cmd"]:
        state.current_mode = stripped_command[1:]
        return state, f"Switched to {state.current_mode.upper()} mode."



    # Check if it's a router command (starts with / and not a built-in command)
    if stripped_command.startswith('/') and stripped_command not in ["/refresh", "/agent", "/chat", "/cmd"]:
        return execute_command(stripped_command, state, review=True, router=router)
    if is_python_code(stripped_command):
        try:
            state, exec_output = execute_python_code(stripped_command, state, locals_dict)
            return state, exec_output
        except KeyboardInterrupt:
            print("\nExecution interrupted by user")
            return state, "Execution interrupted"
    if state.current_mode == "agent":
        return _run_agentic_mode(stripped_command, state, locals_dict, npc_team_dir) 
    if state.current_mode == "cmd":
       
        # If not Python, use LLM to generate Python code
        locals_context_string = "Current Python environment variables and functions:\n"
        if locals_dict:
            for k, v in locals_dict.items():
                if not k.startswith('__'):
                    try:
                        value_repr = repr(v)
                        if len(value_repr) > 200: 
                            value_repr = value_repr[:197] + "..."
                        locals_context_string += f"- {k} (type: {type(v).__name__}) = {value_repr}\n"
                    except Exception:
                        locals_context_string += f"- {k} (type: {type(v).__name__}) = <unrepresentable>\n"
            locals_context_string += "\n--- End of Environment Context ---\n"
        else:
            locals_context_string += "(Environment is empty)\n"

        # ADD CONTEXT ENHANCEMENT HERE:
        enhanced_prompt = stripped_command
        if any(word in stripped_command.lower() for word in ['plot', 'graph', 'chart', 'figure', 'visualiz']):
            plot_context = _get_plot_context(state.conversation_id, state.command_history.db_path)
            enhanced_prompt += f"\n\n{plot_context}"
        
        if any(word in stripped_command.lower() for word in ['file', 'data', 'load', 'variable', 'df']):
            file_context = _get_file_context(state.conversation_id, state.command_history.db_path)
            enhanced_prompt += f"\n\n{file_context}"

        prompt_cmd = f"""User input for Python CMD mode: '{enhanced_prompt}'.
            Generate ONLY executable Python code required to fulfill this. 
            Do not include any explanations, leading markdown like ```python, or any text other than the Python code itself.
            {locals_context_string}
            Begin directly with the code
            """
        # Ensure state.npc is not None before accessing .model or .provider
        npc_model = state.npc.model if state.npc and state.npc.model else state.chat_model
        npc_provider = state.npc.provider if state.npc and state.npc.provider else state.chat_provider

        llm_response = get_llm_response(prompt_cmd, 
                                        npc=state.npc, 
                                        stream=True, 
                                        messages=state.messages)
        response = print_and_process_stream(llm_response.get('response'),
                                            npc_model, 
                                            npc_provider )


        if response.startswith('```python'):
            generated_code = response.strip()[len('```python'):].strip()
            generated_code = generated_code.rsplit('```', 1)[0].strip()
        else:
            generated_code = response.strip()
        
        state.messages = llm_response.get("messages", state.messages) 
        
        if generated_code and not generated_code.startswith("# Error:"):
            print(f"\n# LLM Generated Code (Cmd Mode):\n---\n{generated_code}\n---\n")
            try:
                state, exec_output = execute_python_code(generated_code, state, locals_dict)
                output = f"# Code executed.\n# Output:\n{exec_output if exec_output else '(No direct output)'}"
            except KeyboardInterrupt:
                print("\nExecution interrupted by user")
                output = "Execution interrupted"
        else:
            output = generated_code if generated_code else "# Error: LLM did not generate Python code."
        
        if state.command_history:
            state.command_history.add_command(stripped_command, [str(output if output else "")], "", state.current_path)
            
        return state, output

    return execute_command(stripped_command, state, review=True, router=router)
def run_guac_repl(state: ShellState, project_name: str, package_root: Path, package_name: str):
    from npcsh.routes import router

    
    # Get workspace info 
    npc_team_dir = Path.cwd() / "npc_team"
    workspace_dirs = _get_workspace_dirs(npc_team_dir)
    _ensure_workspace_dirs(workspace_dirs)
    
    locals_dict = {}
    global _guac_monitor_thread, _guac_monitor_stop_event
    if _guac_monitor_thread is None or not (_guac_monitor_thread.is_alive()):
        _guac_monitor_stop_event = threading.Event()
        _guac_monitor_thread = threading.Thread(
            target=_file_drop_monitor,
            args=(workspace_dirs['workspace'].parent, state, locals_dict),
            kwargs={'poll_interval': 0.2},
            daemon=True
        )
        _guac_monitor_thread.start()

    try:
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        
        try:
            package_module = importlib.import_module(package_name)
            for name in dir(package_module):
                if not name.startswith('__'):
                    locals_dict[name] = getattr(package_module, name)
            print(f"Loaded package: {package_name}")
        except ImportError:
            print(f"Warning: Could not import package {package_name}")
            
    except Exception as e:
        print(f"Warning: Could not load package {package_name}: {e}", file=sys.stderr)
            
    from npcpy.data.load import load_file_contents

    def read_file(file_path, max_lines=10000, encoding='utf-8'):
        """
        Read and print file contents up to max_lines.
        Uses npcpy.data.load for specialized file types, falls back to text reading.
        Returns the content as a string for further processing.
        """
        path = Path(file_path).expanduser().resolve()
        
        if not path.exists():
            print(f"File not found: {path}")
            return None
        
        if not path.is_file():
            print(f"Not a file: {path}")
            return None
        
        try:
            # Try using npcpy's load_file_contents first for specialized formats
            file_ext = path.suffix.upper().lstrip('.')
            if file_ext in ['PDF', 'DOCX', 'PPTX', 'HTML', 'HTM', 'CSV', 'XLS', 'XLSX', 'JSON']:
                chunks = load_file_contents(str(path), chunk_size=10000)  # Large chunk to get full content
                if chunks and not chunks[0].startswith("Error") and not chunks[0].startswith("Unsupported"):
                    content = '\n'.join(chunks)
                    lines = content.split('\n')
                    
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                        print(f"File truncated at {max_lines} lines. Use windowed reading for larger files.")
                    
                    print(f"Reading {path.name} ({len(lines)} lines, {len(content)} chars)")
                    print("=" * 60)
                    
                    for i, line in enumerate(lines, 1):
                        print(f"{i:4d} | {line}")
                    
                    print("=" * 60)
                    print(f"End of {path.name}")
                    return content
            
            # Fall back to regular text reading
            with open(path, 'r', encoding=encoding) as f:
                lines = []
                for i, line in enumerate(f, 1):
                    if i > max_lines:
                        print(f"File truncated at {max_lines} lines. Use windowed reading for larger files.")
                        break
                    lines.append(line.rstrip('\n\r'))
            
            content = '\n'.join(lines)
            
            print(f"Reading {path.name} ({len(lines)} lines, {len(content)} chars)")
            print("=" * 60)
            
            for i, line in enumerate(lines, 1):
                print(f"{i:4d} | {line}")
            
            print("=" * 60)
            print(f"End of {path.name}")
            
            return content
            
        except UnicodeDecodeError:
            try:
                with open(path, 'rb') as f:
                    data = f.read(min(1024, max_lines * 80))
                print(f"Binary file {path.name} ({len(data)} bytes)")
                print("=" * 60)
                print(data.hex()[:1000] + ("..." if len(data) > 500 else ""))
                print("=" * 60)
                return data
            except Exception as e:
                print(f"Error reading file: {e}")
                return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def edit_file(file_path, content=None, line_number=None, new_line=None, insert_at=None, append=False, backup=True):
        """
        Edit file contents in various ways:
        - edit_file(path, content="new content") - replace entire file
        - edit_file(path, line_number=5, new_line="new text") - replace specific line
        - edit_file(path, insert_at=5, new_line="inserted text") - insert at line
        - edit_file(path, append=True, content="appended") - append to file
        """
        path = Path(file_path).expanduser().resolve()
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup original if it exists
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + '.backup')
            import shutil
            shutil.copy2(path, backup_path)
            print(f"Backup saved: {backup_path.name}")
        
        try:
            # Read existing content if file exists
            existing_lines = []
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    existing_lines = [line.rstrip('\n\r') for line in f]
            
            if content is not None:
                if append:
                    with open(path, 'a', encoding='utf-8') as f:
                        f.write('\n' + content if existing_lines else content)
                    print(f"Appended to {path.name}")
                else:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Wrote {path.name} ({len(content)} chars)")
            
            elif line_number is not None and new_line is not None:
                if line_number < 1:
                    print("Line numbers start at 1")
                    return False
                
                while len(existing_lines) < line_number:
                    existing_lines.append("")
                
                if line_number <= len(existing_lines):
                    old_line = existing_lines[line_number - 1] if line_number <= len(existing_lines) else ""
                    existing_lines[line_number - 1] = new_line
                    
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(existing_lines))
                    
                    print(f"Line {line_number} in {path.name}:")
                    print(f"  - OLD: {old_line}")
                    print(f"  + NEW: {new_line}")
                else:
                    print(f"File only has {len(existing_lines)} lines")
                    return False
            
            elif insert_at is not None and new_line is not None:
                if insert_at < 1:
                    insert_at = 1
                
                existing_lines.insert(insert_at - 1, new_line)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(existing_lines))
                
                print(f"Inserted at line {insert_at} in {path.name}: {new_line}")
            
            else:
                print("Must specify either 'content', or 'line_number + new_line', or 'insert_at + new_line'")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error editing file: {e}")
            return False

    def load_file(file_path):
        """
        Simple wrapper around npcpy's load_file_contents for direct data loading.
        Returns the loaded data in appropriate format.
        """
        path = Path(file_path).expanduser().resolve()
        
        if not path.exists():
            print(f"File not found: {path}")
            return None
        
        chunks = load_file_contents(str(path))
        if chunks and not chunks[0].startswith("Error") and not chunks[0].startswith("Unsupported"):
            content = '\n'.join(chunks)
            print(f"Loaded {path.name} using npcpy loader")
            return content
        else:
            print(f"Could not load {path.name}: {chunks[0] if chunks else 'Unknown error'}")
            return None
                
    core_imports = {
        'pd': pd, 'np': np, 'plt': plt, 'datetime': datetime, 
        'Path': Path, 'os': os, 'sys': sys, 'json': json, 
        'yaml': yaml, 're': re, 'traceback': traceback, 
        'edit_file': edit_file, 
        'read_file':read_file, 
        'load_file':load_file,
    }


    locals_dict.update(core_imports)
    locals_dict.update({f"guac_{k}": v for k, v in workspace_dirs.items()})
    
    print_guac_bowl()
    print(f"Welcome to Guac Mode! Current mode: {state.current_mode.upper()}. Type /agent, /chat, or /cmd to switch modes.")
    print(f"Workspace: {workspace_dirs['workspace']}")
    print("💡 You can drag and drop files into the terminal to automatically import them!")
    
    command_count = 0
    
    try:
        completer = make_completer(state, router)
        readline.set_completer(completer)
    except:
        pass
    
    while True:
        try:
            state.current_path = os.getcwd()
            
            display_model = state.chat_model
            # Ensure state.npc is not None before accessing .model or .provider
            if isinstance(state.npc, NPC) and state.npc.model:
                display_model = state.npc.model
            
            cwd_colored = colored(os.path.basename(state.current_path), "blue")
            npc_name = state.npc.name if state.npc and state.npc.name else "guac"
            prompt_char = get_guac_prompt_char(command_count)
            
            prompt_str = f"{cwd_colored}:{npc_name}:{display_model}{prompt_char}>  "
            prompt = readline_safe_prompt(prompt_str)
            
            user_input = get_multiline_input(prompt).strip()
            
            if not user_input:
                continue
            
            command_count += 1
            state, result = execute_guac_command(user_input, state, locals_dict, project_name, package_root, router)
            
            process_result(user_input, state, result, state.command_history)
            
        except EOFError:
            print("\nExiting Guac Mode...")
            try:
                readline.write_history_file(READLINE_HISTORY_FILE)
            except:
                pass
            if _guac_monitor_stop_event:
                _guac_monitor_stop_event.set()
            if _guac_monitor_thread:
                _guac_monitor_thread.join(timeout=1.0)
            break
        except SystemExit as e:
            try:
                readline.write_history_file(READLINE_HISTORY_FILE)
            except:
                pass
            print(f"\n{e}")

            if _guac_monitor_stop_event:
                _guac_monitor_stop_event.set()
            if _guac_monitor_thread:
                _guac_monitor_thread.join(timeout=1.0)
            break

        except Exception:
            print("An unexpected error occurred in the REPL:")
            traceback.print_exc()
            try:
                readline.write_history_file(READLINE_HISTORY_FILE)
            except:
                pass

            if _guac_monitor_stop_event:
                _guac_monitor_stop_event.set()
            if _guac_monitor_thread:
                _guac_monitor_thread.join(timeout=1.0)
            break




def enter_guac_mode(npc=None, 
                    team=None,
                    config_dir=None, 
                    plots_dir=None,
                    npc_team_dir=None,
                    refresh_period=None,
                    lang='python',
                    default_mode_choice=None):
    
    if refresh_period is not None:
        try:
            GUAC_REFRESH_PERIOD = int(refresh_period)
        except ValueError:
            pass
    
    setup_result = setup_guac_mode(
        config_dir=config_dir,
        plots_dir=plots_dir,
        npc_team_dir=npc_team_dir, 
        lang=lang,
        default_mode_choice=default_mode_choice
    )

    project_name = setup_result.get("package_name", "project")
    package_root = setup_result["package_root"]
    package_name = setup_result.get("package_name", "project")
    npc_team_dir = setup_result.get("npc_team_dir")

    # Always call setup_shell to build history, team, and default_npc
    command_history, default_team, default_npc = setup_shell()

    # 🔑 Ensure global guac gets loaded if npc is None
    if npc is None and default_npc is None:
        # Construct the path correctly based on where ensure_global_guac_team puts it
        guac_npc_path = Path(npc_team_dir) / "guac.npc" 
        if guac_npc_path.exists():
            npc = NPC(file=str(guac_npc_path), db_conn=command_history.engine)
            # Ensure the team is also correctly set to the global guac team
            team_ctx_path = Path(npc_team_dir) / "team.ctx"
            if team_ctx_path.exists():
                with open(team_ctx_path, "r") as f:
                    team_ctx = yaml.safe_load(f) or {}
                team = Team(team_path=str(npc_team_dir), forenpc=npc, jinxs={}) # Simplified team creation
                team.name = team_ctx.get("team_name", "guac_global_team")
        else:
            raise RuntimeError(f"No NPC loaded and {guac_npc_path} not found!")
    elif default_npc and npc is None:
        # If setup_shell provided a default_npc (e.g., sibiji), ensure it's used
        npc = default_npc


    state = ShellState(
        conversation_id=start_new_conversation(),
        stream_output=True,
        current_mode=setup_result.get("default_mode", "cmd"),
        chat_model=os.environ.get("NPCSH_CHAT_MODEL", "gemma3:4b"),
        chat_provider=os.environ.get("NPCSH_CHAT_PROVIDER", "ollama"),
        current_path=os.getcwd(),
        npc=npc, # This is now guaranteed to be an NPC object
        team=team or default_team # Use the correctly loaded team or default
    )
    
    state.command_history = command_history

    try:
        readline.read_history_file(READLINE_HISTORY_FILE)
        readline.set_history_length(1000)
        readline.parse_and_bind("set enable-bracketed-paste on")
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"Warning: Could not read readline history file {READLINE_HISTORY_FILE}: {e}")

    run_guac_repl(state, project_name, package_root, package_name)
        
def main():
    parser = argparse.ArgumentParser(description="Enter Guac Mode - Interactive Python with LLM assistance.")
    parser.add_argument("--config_dir", type=str, help="Guac configuration directory.")
    parser.add_argument("--plots_dir", type=str, help="Directory to save plots.")
    parser.add_argument("--npc_team_dir", type=str, default=None, 
                        help="NPC team directory for Guac. Defaults to ./npc_team")
    parser.add_argument("--refresh_period", type=int, help="Number of commands before suggesting /refresh.")
    parser.add_argument("--default_mode", type=str, choices=["agent", "chat", "cmd"], 
                        help="Default mode to start in.")
    
    args = parser.parse_args()

    enter_guac_mode(
        config_dir=args.config_dir,
        plots_dir=args.plots_dir,
        npc_team_dir=args.npc_team_dir,
        refresh_period=args.refresh_period,
        default_mode_choice=args.default_mode
    )

if __name__ == "__main__":
    main()
