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

plt.ioff()

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
from npcpy.npc_sysenv import render_markdown

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
    
    if state.command_history:
        state.command_history.add_command(code_str, [final_output_str if final_output_str else ""], "", state.current_path)
    return state, final_output_str

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
        response = get_llm_response(prompt, model=state.chat_model, provider=state.chat_provider, npc=state.npc, stream=False)
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
def setup_guac_mode(config_dir=None, plots_dir=None, npc_team_dir=None, lang='python', default_mode_choice=None):
    base_dir = Path.cwd()
    
    if config_dir is None:
        config_dir = base_dir / ".guac"
    else:
        config_dir = Path(config_dir)
        
    if plots_dir is None:
        plots_dir = base_dir / "plots"
    else:
        plots_dir = Path(plots_dir)
        
    if npc_team_dir is None:
        npc_team_dir = base_dir / "npc_team"
    else:
        npc_team_dir = Path(npc_team_dir)
    
    for p in [config_dir, plots_dir, npc_team_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Setup Guac workspace
    workspace_dirs = _get_workspace_dirs(npc_team_dir)
    _ensure_workspace_dirs(workspace_dirs)

    # Rest of existing setup_guac_mode code...
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
            response = input("Enter the path to your Python package root (press Enter for current directory): ").strip()
            package_root = response if response else str(base_dir)
            
            response = input("Enter your package name (press Enter to use 'project'): ").strip()
            package_name = response if response else "project"
        except EOFError:
            package_root = str(base_dir)
            package_name = "project"

    project_name = existing_ctx.get("GUAC_PROJECT_NAME")
    project_description = existing_ctx.get("GUAC_PROJECT_DESCRIPTION")
    
    if project_name is None:
        try:
            project_name = input("Enter the project name: ").strip() or "unknown_project"
        except EOFError:
            project_name = "unknown_project"
    if project_description is None:
        try:
            project_description = input("Enter a short description of the project: ").strip() or "No description provided."
        except EOFError:
            project_description = "No description provided."

    updated_ctx = {**existing_ctx}
    updated_ctx.update({
        "GUAC_TEAM_NAME": "guac_team",
        "GUAC_DESCRIPTION": f"A team of NPCs specialized in {lang} analysis for project {project_name}",
        "GUAC_FORENPC": "guac",
        "GUAC_PROJECT_NAME": project_name,
        "GUAC_PROJECT_DESCRIPTION": project_description,
        "GUAC_LANG": lang,
        "GUAC_PACKAGE_ROOT": package_root,
        "GUAC_PACKAGE_NAME": package_name,
        "GUAC_WORKSPACE_PATHS": {k: str(v) for k, v in workspace_dirs.items()},
    })

    with open(team_ctx_path, "w") as f:
        yaml.dump(updated_ctx, f, default_flow_style=False)
    print("Updated team.ctx with GUAC-specific information.")

    default_mode_val = default_mode_choice or "cmd"
    setup_npc_team(npc_team_dir, lang)
    
    print(f"\nGuac mode configured for package: {package_name} at {package_root}")
    print(f"Workspace created at: {workspace_dirs['workspace']}")

    return {
        "language": lang, 
        "package_root": Path(package_root), 
        "config_path": config_dir / "config.json",
        "plots_dir": plots_dir, 
        "npc_team_dir": npc_team_dir,
        "config_dir": config_dir, 
        "default_mode": default_mode_val,
        "project_name": project_name, 
        "project_description": project_description,
        "package_name": package_name
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
        "primary_directive": f"You are caug, a specialist in big data statistical methods in {lang}."
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
    print(f"[DEBUG] _detect_file_drop called with: '{input_text}'")
    
    stripped = input_text.strip()
    print(f"[DEBUG] Stripped input: '{stripped}'")
    
    # Remove quotes if present
    if stripped.startswith("'") and stripped.endswith("'"):
        stripped = stripped[1:-1]
        print(f"[DEBUG] Removed single quotes: '{stripped}'")
    elif stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1]
        print(f"[DEBUG] Removed double quotes: '{stripped}'")
    
    print(f"[DEBUG] Final stripped: '{stripped}'")
    print(f"[DEBUG] Number of words: {len(stripped.split())}")
    print(f"[DEBUG] Path exists: {Path(stripped).exists()}")
    print(f"[DEBUG] Is file: {Path(stripped).is_file()}")
    
    # Check if it's just a file path
    is_file_drop = (len(stripped.split()) == 1 and 
                   Path(stripped).exists() and 
                   Path(stripped).is_file())
    
    print(f"[DEBUG] _detect_file_drop returning: {is_file_drop}")
    return is_file_drop

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
    print(f"[DEBUG] _handle_file_drop called with input: '{input_text}'")
    
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
    
    print(f"[DEBUG] Found file paths: {file_paths}")
    
    if not file_paths:
        print(f"[DEBUG] No file paths found, returning original input")
        return input_text, processed_files
    
    modified_input = input_text
    for file_path in file_paths:
        expanded_path = Path(file_path.replace('~', str(Path.home()))).resolve()
        
        if expanded_path.exists() and expanded_path.is_file():
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
                processed_files.append(str(target_path))
                modified_input = modified_input.replace(file_path, str(target_path))
                print(f"📁 Copied {expanded_path.name} to workspace: {target_path}")
            except Exception as e:
                print(f"[ERROR] Failed to copy file: {e}")
    
    return modified_input, processed_files



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


def _run_agentic_mode(command: str, state: ShellState, locals_dict: Dict[str, Any], npc_team_dir: Path) -> Tuple[ShellState, Any]:
    """Run agentic mode with code execution, analysis, and iteration"""
    max_iterations = 3
    iteration = 0
    full_output = []
    current_command = command
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 Agentic iteration {iteration}/{max_iterations}")
        
        prompt = f"""
        User request: {current_command}
        Previous attempts: {full_output[-1] if full_output else 'None'}
        
        Generate Python code to accomplish this task. If you need to make assumptions, state them clearly in comments.
        Provide ONLY executable Python code without any explanations or markdown formatting.
        """
        
        llm_response = get_llm_response(prompt, model=state.chat_model, provider=state.chat_provider, npc=state.npc, stream=False)
        generated_code = llm_response.get("response", "").strip()
        
        if generated_code.startswith('```python'):
            generated_code = generated_code[len('```python'):].strip()
        if generated_code.endswith('```'):
            generated_code = generated_code[:-3].strip()
        
        print(f"\n# Generated Code (Iteration {iteration}):\n---\n{generated_code}\n---\n")
        
        try:
            state, exec_output = execute_python_code(generated_code, state, locals_dict)
            full_output.append(f"Iteration {iteration}:\nCode:\n{generated_code}\nOutput:\n{exec_output}")
            
            saved_figures = _save_matplotlib_figures(npc_team_dir)
            if saved_figures:
                full_output[-1] += f"\nSaved figures: {', '.join(saved_figures)}"
            
            analysis_prompt = f"""
            Code execution results: {exec_output}
            
            Was this successful? If not, what went wrong and how can we improve?
            If successful, should we continue with more iterations or is this sufficient?
            """
            
            analysis_response = get_llm_response(analysis_prompt, model=state.chat_model, provider=state.chat_provider, npc=state.npc, stream=False)
            analysis = analysis_response.get("response", "").strip()
            print(f"\n# Analysis:\n{analysis}")
            
            if "ask user" in analysis.lower() or "clarification" in analysis.lower() or iteration == max_iterations:
                user_feedback = input("\n🤔 Agent requests feedback (press Enter to continue or type your response): ").strip()
                if user_feedback:
                    current_command = f"{current_command} - User feedback: {user_feedback}"
                else:
                    break
            
            if "sufficient" in analysis.lower() or "complete" in analysis.lower():
                break
                
        except Exception as e:
            error_msg = f"Error in iteration {iteration}: {str(e)}"
            print(error_msg)
            full_output.append(error_msg)
            current_command = f"{current_command} - Error: {str(e)}"
    

    return state, "# Agentic execution completed"+'\n'.join(full_output)

    
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
    
    print(f"[DEBUG] execute_guac_command called with: '{stripped_command}'")
    
    if not stripped_command:
        return state, None
    if stripped_command.lower() in ["exit", "quit", "exit()", "quit()"]:
        raise SystemExit("Exiting Guac Mode.")

    # Get npc_team_dir from current working directory
    npc_team_dir = Path.cwd() / "npc_team"
    print(f"[DEBUG] npc_team_dir: {npc_team_dir}")

    # Check if this is a file drop (single file path)
    print(f"[DEBUG] Checking if file drop...")
    if _detect_file_drop(stripped_command):
        print(f"[DEBUG] File drop detected!")
        
        # Clean the path
        file_path = stripped_command.strip("'\"")
        expanded_path = Path(file_path).resolve()
        print(f"[DEBUG] Cleaned file path: {file_path}")
        print(f"[DEBUG] Expanded path: {expanded_path}")
        print(f"[DEBUG] Path exists: {expanded_path.exists()}")
        
        # Copy to workspace
        workspace_dirs = _get_workspace_dirs(npc_team_dir)
        _ensure_workspace_dirs(workspace_dirs)
        print(f"[DEBUG] Workspace dirs: {workspace_dirs}")
        
        ext = expanded_path.suffix[1:].upper() if expanded_path.suffix else "OTHERS"
        category = EXTENSION_MAP.get(ext, "data_inputs")
        target_dir = workspace_dirs.get(category, workspace_dirs["data_inputs"])
        print(f"[DEBUG] Extension: {ext}, Category: {category}, Target dir: {target_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{expanded_path.name}"
        target_path = target_dir / new_filename
        print(f"[DEBUG] Target path: {target_path}")
        
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
    else:
        print(f"[DEBUG] Not a file drop")

    # Handle file drops in text (multiple files or files with other text)
    processed_command, processed_files = _handle_file_drop(stripped_command, npc_team_dir)
    if processed_files:
        print(f"📁 Processed {len(processed_files)} files")
        stripped_command = processed_command

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

    # In agent mode, use the agentic workflow
    if state.current_mode == "agent":
        return _run_agentic_mode(stripped_command, state, locals_dict, npc_team_dir)

    # In cmd mode, prioritize Python execution
    if state.current_mode == "cmd":
        if is_python_code(stripped_command):
            try:
                state, exec_output = execute_python_code(stripped_command, state, locals_dict)
                saved_figures = _save_matplotlib_figures(npc_team_dir)
                if saved_figures:
                    exec_output += f"\n📊 Saved figures: {', '.join(saved_figures)}"
                return state, exec_output
            except KeyboardInterrupt:
                print("\nExecution interrupted by user")
                return state, "Execution interrupted"
        
        # If not Python, use LLM to generate Python code (existing logic)
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

        prompt_cmd = (
            f"User input for Python CMD mode: '{stripped_command}'.\n"
            f"Generate ONLY executable Python code required to fulfill this.\n"
            f"Do not include any explanations, leading markdown like ```python, or any text other than the Python code itself.\n"
            f"{locals_context_string}"
        )

        llm_response = get_llm_response(prompt_cmd, model=state.chat_model, provider=state.chat_provider, npc=state.npc, stream=False, messages=state.messages)
        
        if llm_response.get('response', '').startswith('```python'):
            generated_code = llm_response.get("response", "").strip()[len('```python'):].strip()
            generated_code = generated_code.rsplit('```', 1)[0].strip()
        else:
            generated_code = llm_response.get("response", "").strip()
        
        state.messages = llm_response.get("messages", state.messages) 
        
        if generated_code and not generated_code.startswith("# Error:"):
            print(f"\n# LLM Generated Code (Cmd Mode):\n---\n{generated_code}\n---\n")
            try:
                state, exec_output = execute_python_code(generated_code, state, locals_dict)
                saved_figures = _save_matplotlib_figures(npc_team_dir)
                if saved_figures:
                    exec_output += f"\n📊 Saved figures: {', '.join(saved_figures)}"
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
        
    core_imports = {
        'pd': pd, 'np': np, 'plt': plt, 'datetime': datetime, 
        'Path': Path, 'os': os, 'sys': sys, 'json': json, 
        'yaml': yaml, 're': re, 'traceback': traceback
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
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Guac Mode...")
            break
        except SystemExit as e:
            print(f"\n{e}")
            break
        except Exception:
            print("An unexpected error occurred in the REPL:")
            traceback.print_exc()





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

    project_name = setup_result.get("project_name", "project")
    package_root = setup_result["package_root"]
    package_name = setup_result.get("package_name", "project")

    command_history, default_team, default_npc = setup_shell()
    
    state = ShellState(
        conversation_id=start_new_conversation(),
        stream_output=True,
        current_mode=setup_result.get("default_mode", "cmd"),
        chat_model=os.environ.get("NPCSH_CHAT_MODEL", "gemma3:4b"),
        chat_provider=os.environ.get("NPCSH_CHAT_PROVIDER", "ollama"),
        current_path=os.getcwd(),
        npc=npc or default_npc,
        team=team or default_team
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