import re
import os
import sys
import code
import yaml
from pathlib import Path
import atexit
import traceback
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import io
import importlib.util 

from npcpy.memory.command_history import CommandHistory
from npcpy.npc_compiler import Team, NPC
from npcpy.llm_funcs import get_llm_response, check_llm_command, execute_llm_command
from npcsh._state import initial_state as npcsh_initial_state
from npcpy.npc_sysenv import render_markdown, print_and_process_stream_with_markdown

try:
    import readline
except ImportError:
    readline = None

GUAC_REFRESH_PERIOD = os.environ.get('GUAC_REFRESH_PERIOD', 100)
READLINE_HISTORY_FILE = os.path.expanduser("~/.guac_readline_history")
try:
    npcsh_initial_state.GUAC_REFRESH_PERIOD = int(GUAC_REFRESH_PERIOD)
except ValueError:
    npcsh_initial_state.GUAC_REFRESH_PERIOD = 100 

@dataclass
class GuacState:
    current_mode: str = "cmd" 
    current_path: str = field(default_factory=os.getcwd)
    npc: Optional[NPC] = None
    team: Optional[Team] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    locals: Dict[str, Any] = field(default_factory=dict)
    command_history: Optional[CommandHistory] = None
    chat_model: Optional[str] = npcsh_initial_state.chat_model
    chat_provider: Optional[str] = npcsh_initial_state.chat_provider
    stream_output: bool = True
    config_dir: Optional[Path] = None
    src_dir: Optional[Path] = None
    command_count: int = 0
    compile_buffer: List[str] = field(default_factory=list)

def get_multiline_input_guac(prompt_str: str, state: GuacState) -> str:
    lines = list(state.compile_buffer)
    current_prompt = prompt_str if not lines else "... "
    while True:
        try:
            line = input(current_prompt)
            lines.append(line)
            current_prompt = "... "
            if not line and len(lines) > 1 and not lines[-2].strip():
                lines.pop() 
                lines.pop() 
                break
            if not line and len(lines) == 1:
                lines.pop() 
                break
            if len(lines) == 1 and line.strip():
                temp_line = line.strip()
                is_block_starter = re.match(r"^\s*(def|class|for|while|if|try|with|@)", temp_line)
                ends_with_colon_for_block = temp_line.endswith(":") and is_block_starter
                if not is_block_starter and not ends_with_colon_for_block:
                    open_brackets = (temp_line.count('(') - temp_line.count(')') +
                                     temp_line.count('[') - temp_line.count(']') +
                                     temp_line.count('{') - temp_line.count('}'))
                    if open_brackets <= 0:
                        break
        except EOFError:
            print("\nGoodbye!")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
            state.compile_buffer.clear()
            return ""
    full_input = "\n".join(lines)
    state.compile_buffer.clear()
    return full_input

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
    except (OverflowError, ValueError): # Other potential compile errors
        return False


def setup_guac_readline(history_file: str):
    if not readline:
        return
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    except OSError:
        pass
    
    try:
        if sys.stdin.isatty():
            readline.set_history_length(1000)
            try:
                readline.parse_and_bind("set enable-bracketed-paste on")
            except Exception:
                pass 
    except Exception:
        pass

def save_guac_readline_history(history_file: str):
    if not readline:
        return
    try:
        readline.write_history_file(history_file)
    except OSError:
        pass
    except Exception:
        pass

def _load_guac_helpers_into_state(state: GuacState):
    if state.src_dir:
        main_module_path = state.src_dir / "main.py"
        if main_module_path.exists():
            try:
                p_path = str(state.src_dir.parent)
                s_path = str(state.src_dir)
                if p_path not in sys.path:
                    sys.path.insert(0, p_path)
                if s_path not in sys.path:
                    sys.path.insert(0, s_path)
                
                spec = importlib.util.spec_from_file_location("guac_main_helpers", main_module_path)
                if spec and spec.loader:
                    guac_main = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(guac_main)
                    for name in dir(guac_main):
                        if not name.startswith('__'):
                            state.locals[name] = getattr(guac_main, name)
                    
                    core_imports = {
                        'pd': pd, 'np': np, 'plt': plt, 'datetime': datetime, 
                        'Path': Path, 'os': os, 'sys': sys, 'json': json, 
                        'yaml': yaml, 're': re, 'traceback': traceback
                    }
                    state.locals.update(core_imports)
            except Exception as e:
                print(f"Warning: Could not load helpers from {main_module_path}: {e}", file=sys.stderr)

def setup_guac_mode(config_dir=None, 
                    plots_dir=None, 
                    npc_team_dir=None, 
                    lang='python',
                    ):
    home_dir = Path.home()
    config_dir = Path(config_dir) if config_dir else home_dir / ".npcsh" / "guac"
    plots_dir = Path(plots_dir) if plots_dir else config_dir / "plots"
    npc_team_dir = Path(npc_team_dir) if npc_team_dir else config_dir / "npc_team"
    src_dir = config_dir / "src"

    for p in [src_dir, plots_dir, npc_team_dir]:
        p.mkdir(parents=True, exist_ok=True)


    team_ctx = {
        "team_name": "guac_team",
        "description": f"A team of NPCs specialized in {lang} analysis",
        "forenpc": "guac",
        "model": os.environ.get("NPCSH_CHAT_MODEL", "llama3.2"),
        "provider": os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
    }

    with open(npc_team_dir / "team.ctx", "w") as f:
        yaml.dump(team_ctx, f, default_flow_style=False)


    if not (config_dir / "__init__.py").exists():
        (config_dir / "__init__.py").touch()

    config_file = config_dir / "config.json"
    default_mode_val = "cmd"
    current_config = {}

    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                current_config = json.load(f)
            default_mode_val = current_config.get("default_mode", "cmd")
        except json.JSONDecodeError:
            pass 

    if not current_config or \
       current_config.get("preferred_language") != lang or \
       current_config.get("default_mode") is None:
        current_config = {
            "preferred_language": lang,
            "plots_directory": str(plots_dir),
            "npc_team_directory": str(npc_team_dir),
            "default_mode": default_mode_val
        }
        with open(config_file, "w") as f:
            json.dump(current_config, f, indent=2)

    os.environ["NPCSH_GUAC_LANG"] = lang
    os.environ["NPCSH_GUAC_PLOTS"] = str(plots_dir)
    os.environ["NPCSH_GUAC_TEAM"] = str(npc_team_dir)
    npcsh_initial_state.GUAC_DEFAULT_MODE = default_mode_val

    if not (src_dir / "__init__.py").exists():
        with open(src_dir / "__init__.py", "w") as f:
            f.write("# Guac source directory\n")
    
    main_py_content = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from pathlib import Path

def save_plot(name=None, plots_dir=None):
    if plots_dir is None:
        plots_dir = os.environ.get("NPCSH_GUAC_PLOTS", Path.home() / ".npcsh" / "guac" / "plots")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{name}.png" if name else f"{timestamp}_plot.png"
    filepath = plots_dir / filename
    try:
        if plt.get_fignums():
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        else:
            print("No active matplotlib plot to save.")
            return None
    except Exception as e:
        print(f"Error saving plot: {e}")
        return None
    return filepath

def read_img(img_path):
    try:
        from PIL import Image
        img = Image.open(img_path)
        img.show()
    except ImportError:
        print("PIL (Pillow) not available. Please install it: pip install Pillow")
    except FileNotFoundError:
        print(f"Image file not found: {img_path}")
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
    return img_path
"""
    if not (src_dir / "main.py").exists():
        with open(src_dir / "main.py", "w") as f:
            f.write(main_py_content)
    
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))
    if str(config_dir.parent) not in sys.path:
        sys.path.insert(0, str(config_dir.parent))

    setup_npc_team(npc_team_dir, lang)
    return {
        "language": lang, "src_dir": src_dir, "config_path": config_file,
        "plots_dir": plots_dir, "npc_team_dir": npc_team_dir,
        "config_dir": config_dir, "default_mode": default_mode_val
    }

def setup_npc_team(npc_team_dir, lang):
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
        with open(npc_file, "w") as f:
            yaml.dump(npc_data, f, default_flow_style=False)


    team_ctx_model = os.environ.get("NPCSH_CHAT_MODEL", npcsh_initial_state.chat_model or "llama3.2")
    team_ctx_provider = os.environ.get("NPCSH_CHAT_PROVIDER", npcsh_initial_state.chat_provider or "ollama")
    team_ctx = {
        "team_name": "guac_team", "description": f"A team for {lang} analysis", "foreman": "guac",
        "model": team_ctx_model, "provider": team_ctx_provider
    }
    npcsh_initial_state.chat_model = team_ctx_model
    npcsh_initial_state.chat_provider = team_ctx_provider
    with open(npc_team_dir / "team.ctx", "w") as f:
        yaml.dump(team_ctx, f, default_flow_style=False)

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

def get_guac_prompt_char(command_count: int) -> str:
    period = int(npcsh_initial_state.GUAC_REFRESH_PERIOD)
    period = max(1, period)
    stages = ["\U0001F951", "\U0001F951🔪", "\U0001F951🥣", "\U0001F951🥣🧂", "\U0001F958 REFRESH?"]
    divisor = max(1, period // (len(stages)-1) if len(stages) > 1 else period)
    stage_index = min(command_count // divisor, len(stages) - 1)
    return stages[stage_index]

def _handle_guac_refresh(state: GuacState):
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
            main_py_path = state.src_dir / "main.py"
            with open(main_py_path, "a") as f:
                f.write("\n\n# --- Functions suggested by /refresh ---\n")
                f.write(suggested_functions_code)
                f.write("\n# --- End of suggested functions ---\n")
            print(f"Functions appended to {main_py_path}.")
            print("To use them in the current session: import importlib; importlib.reload(guac.src.main); from guac.src.main import *")
        else:
            print("Suggested functions not added.")
    except Exception as e:
        print(f"Error during /refresh: {e}")
        traceback.print_exc()

def execute_python_code(code_str: str, state: GuacState) -> Tuple[GuacState, Any]:
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
                exec_result = eval(compiled_expr, state.locals)
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
            exec(compiled_code, state.locals)

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

def execute_guac_command(command: str, state: GuacState) -> Tuple[GuacState, Any]:
    stripped_command = command.strip()
    output = None 
    
    if not stripped_command:
        return state, None
    if stripped_command.lower() in ["exit", "quit", "exit()", "quit()"]:
        raise SystemExit("Exiting Guac Mode.")

    # Check for shell-like commands first, before Python code detection
    parts = stripped_command.split(maxsplit=1)
    cmd_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    # Handle shell-like commands without / prefix
    if cmd_name == "ls":
        try:
            ls_path = args.strip() if args.strip() else state.current_path
            output = "\n".join(os.listdir(ls_path))
        except Exception as e:
            output = f"Error listing directory: {e}"
        if state.command_history:
            state.command_history.add_command(command, [str(output)], "", state.current_path)
        return state, output
    elif cmd_name == "pwd":
        output = state.current_path
        if state.command_history:
            state.command_history.add_command(command, [str(output)], "", state.current_path)
        return state, output
    elif cmd_name == "cd":
        target_dir = args.strip() if args.strip() else str(Path.home())
        try:
            os.chdir(target_dir)
            state.current_path = os.getcwd()
            output = f"Changed directory to {state.current_path}"
        except FileNotFoundError:
            output = f"Error: Directory not found: {target_dir}"
        except Exception as e:
            output = f"Error changing directory: {e}"
        if state.command_history:
            state.command_history.add_command(command, [str(output)], "", state.current_path)
        return state, output
    elif cmd_name == "run" and args.strip().endswith(".py"):
        script_path = Path(args.strip())
        if script_path.exists():
            try:
                with open(script_path, "r") as f:
                    script_code = f.read()
                _, script_exec_output = execute_python_code(script_code, state) 
                output = (f"Executed script '{script_path}'.\n"
                          f"Output from script:\n{script_exec_output if script_exec_output else '(No direct output)'}")
            except Exception as e:
                output = f"Error running script {script_path}: {e}"
        else:
            output = f"Error: Script not found: {script_path}"
        if state.command_history:
            state.command_history.add_command(command, [str(output)], "", state.current_path)
        return state, output

    # Now check if it's Python code
    if is_python_code(stripped_command):
        state, output = execute_python_code(stripped_command, state)
        return state, output

    # Handle / prefixed commands
    if stripped_command.startswith("/"):
        parts = stripped_command.split(maxsplit=1)
        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        is_core_cmd = True 
        
        if cmd_name == "/agent":
            state.current_mode = "agent"
            output = "Switched to AGENT mode."
        elif cmd_name == "/chat":
            state.current_mode = "chat"
            output = "Switched to CHAT mode."
        elif cmd_name == "/cmd":
            state.current_mode = "cmd"
            output = "Switched to CMD mode."
        elif cmd_name == "/ride":
            state.current_mode = "ride"
            output = "Switched to RIDE mode (placeholder)."
        elif cmd_name == "/refresh":
            _handle_guac_refresh(state)
            output = "Refresh process initiated." 
        elif cmd_name == "/mode":
            output = f"Current mode: {state.current_mode.upper()}"
        elif cmd_name == "/show_vars":
            temp_output_list = ["Current Python Environment Variables:"]
            if state.locals:
                for k, v_obj in state.locals.items():
                    if not k.startswith("__"):
                        try:
                            v_repr = repr(v_obj)
                            temp_output_list.append(f"  {k}: {v_repr[:100]}{'...' if len(v_repr) > 100 else ''}")
                        except Exception:
                             temp_output_list.append(f"  {k}: <Error representing value>")
            else:
                temp_output_list.append("  (empty)")
            output = "\n".join(temp_output_list)

        else:
            is_core_cmd = False 
        
        if is_core_cmd:
            if state.command_history:
                state.command_history.add_command(command, [str(output if output else "")], "", state.current_path)
            return state, output
    
    nl_input_for_llm = stripped_command 

    if state.current_mode == "agent":
        llm_result_dict = check_llm_command(
            command=nl_input_for_llm,
            model=state.chat_model,
            provider=state.chat_provider,
            npc=state.npc,
            team=state.team,
            messages=state.messages, # Pass current messages for context
            stream=state.stream_output,
            # tools and jinxs would be sourced from state.npc or state.team if check_llm_command uses them
        )
        output = llm_result_dict.get("output")
        state.messages = llm_result_dict.get("messages", state.messages) # Update messages from check_llm_command
        
        history_output = str(output) if not (state.stream_output and hasattr(output, '__iter__') and not isinstance(output, (str,bytes))) else "[Streamed Agent Response]"
        if state.command_history:
            state.command_history.add_command(nl_input_for_llm, [history_output], "", state.current_path)

    elif state.current_mode == "chat":
        llm_response_dict = get_llm_response(
            nl_input_for_llm, 
            model=state.chat_model,
            provider=state.chat_provider,
            npc=state.npc,
            messages=state.messages, # Pass current messages
            stream=state.stream_output 
        )
        output = llm_response_dict.get("response") 
        state.messages = llm_response_dict.get("messages", state.messages) # Update messages
        
        history_output = str(output) if not (state.stream_output and hasattr(output, '__iter__') and not isinstance(output, (str,bytes))) else "[Streamed Chat Response]"
        if state.command_history:
            state.command_history.add_command(nl_input_for_llm, [history_output], "", state.current_path)

    elif state.current_mode == "cmd":
        locals_context_string = "Current Python environment variables and functions:\n"
        if state.locals:
            for k, v in state.locals.items():
                if not k.startswith('__'): # Exclude Python built-ins and internal vars
                    try:
                        # Use repr() for a developer-friendly representation
                        value_repr = repr(v)
                        # Truncate long representations to prevent context window bloat
                        if len(value_repr) > 200: 
                            value_repr = value_repr[:197] + "..."
                        locals_context_string += f"- {k} (type: {type(v).__name__}) = {value_repr}\n"
                    except Exception:
                        locals_context_string += f"- {k} (type: {type(v).__name__}) = <unrepresentable>\n"
            # Add a clear separator for LLM to distinguish this context
            locals_context_string += "\n--- End of Environment Context ---\n"
        else:
            locals_context_string += "(Environment is empty)\n"

        prompt_cmd = (
            f"User input for Python CMD mode: '{nl_input_for_llm}'.\n"
            f"Generate ONLY executable Python code required to fulfill this.\n"
            f"Do not include any explanations, leading markdown like ```python, or any text other than the Python code itself.\n"
        )

        llm_response = get_llm_response(
            prompt_cmd,
            model=state.chat_model,
            provider=state.chat_provider,
            npc=state.npc,
            stream=False, 
            messages=state.messages # Pass messages for context if LLM uses them
        )
        if llm_response.get('response').startswith('```python'):
            generated_code = llm_response.get("response", "").strip()[len('```python'):].strip()
            generated_code = generated_code.rsplit('```', 1)[0].strip()
        else:
            generated_code = llm_response.get("response", "").strip()
        state.messages = llm_response.get("messages", state.messages) 
        
        if generated_code and not generated_code.startswith("# Error:"):
            print(f"\n# LLM Generated Code (Cmd Mode):\n---\n{generated_code}\n---\n")
            _, exec_output = execute_python_code(generated_code, state)
            output = f"# Code executed.\n# Output:\n{exec_output if exec_output else '(No direct output)'}"
        else:
            output = generated_code if generated_code else "# Error: LLM did not generate Python code."
        
        if state.command_history:
            state.command_history.add_command(nl_input_for_llm, [str(output if output else "")], "", state.current_path)

    elif state.current_mode == "ride":
        output = "RIDE mode is not yet implemented. Your input was: " + nl_input_for_llm
        if state.command_history:
            state.command_history.add_command(nl_input_for_llm, [str(output)], "", state.current_path)

    return state, output

def run_guac_repl(initial_guac_state: GuacState):
    state = initial_guac_state
    _load_guac_helpers_into_state(state) 
    print_guac_bowl()
    print(f"Welcome to Guac Mode! Current mode: {state.current_mode.upper()}. Type /agent, /chat, or /cmd to switch modes.")
    
    while True:
        try:
            state.current_path = os.getcwd()
            path_display = Path(state.current_path).name
            prompt_char = get_guac_prompt_char(state.command_count)
            mode_display = state.current_mode.upper()
            npc_display = f":{state.npc.name}" if state.npc and state.npc.name else ""
            prompt_str = f"[{path_display}|{mode_display}{npc_display}] {prompt_char} > "
            
            user_input = get_multiline_input_guac(prompt_str, state)
            if not user_input.strip() and not state.compile_buffer:
                if state.compile_buffer:
                    state.compile_buffer.clear()
                continue
            
            state.command_count +=1
            new_state, result = execute_guac_command(user_input, state)
            state = new_state 
            
            if result is not None:
                if state.stream_output and hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                    full_streamed_output_for_history = print_and_process_stream_with_markdown(result, state.chat_model, state.chat_provider)
                    if (state.current_mode == "chat" or state.current_mode == "agent") and \
                       state.messages and state.messages[-1].get("role") == "assistant": 
                         state.messages[-1]["content"] = full_streamed_output_for_history
                    
                    if state.command_history:
                        try:
                            last_entry_id = state.command_history.get_last_entry_id() 
                            if last_entry_id:
                                state.command_history.update_command_output(last_entry_id, [full_streamed_output_for_history])
                        except AttributeError: 
                            pass 
                elif isinstance(result, str): 
                    if result.strip():
                        render_markdown(result) 
                elif not (state.stream_output and hasattr(result, '__iter__')): 
                    if result:
                        print(str(result)) 
            print() 

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
                    lang=None, 
                    default_mode_choice=None): 
    
    if refresh_period is not None:
        try:
            npcsh_initial_state.GUAC_REFRESH_PERIOD = int(refresh_period)
        except ValueError:
            pass

    setup_result = setup_guac_mode(
        config_dir=config_dir,
        plots_dir=plots_dir,
        npc_team_dir=npc_team_dir
    )
    guac_config_dir = setup_result["config_dir"]
    guac_src_dir = setup_result["src_dir"]
    guac_npc_team_dir = setup_result["npc_team_dir"]
    guac_default_mode = default_mode_choice or setup_result.get("default_mode", "cmd")

    cmd_history = CommandHistory() 
    current_npc = npc
    current_team = team

    if current_npc is None and current_team is None: 
        try:
            current_team = Team(team_path=str(guac_npc_team_dir), db_conn=None)
            if current_team and current_team.npcs:
                 current_npc = current_team.get_npc("guac") 
                 if not current_npc:
                     current_npc = current_team.get_foreman() or next(iter(current_team.npcs.values()), None)
        except Exception as e:
            print(f"Warning: Could not load Guac NPC team from {guac_npc_team_dir}: {e}", file=sys.stderr)
    
    initial_guac_state = GuacState(
        current_mode=guac_default_mode,
        npc=current_npc,
        team=current_team,
        command_history=cmd_history,
        chat_model=npcsh_initial_state.chat_model,
        chat_provider=npcsh_initial_state.chat_provider,
        config_dir=guac_config_dir,
        src_dir=guac_src_dir,
        locals={}
    )

    try:
        setup_guac_readline(READLINE_HISTORY_FILE)
        atexit.register(save_guac_readline_history, READLINE_HISTORY_FILE)
    except Exception as e:
        print(f'Could not set up readline: {e}', file=sys.stderr)
    
    atexit.register(cmd_history.close) 
    run_guac_repl(initial_guac_state)

def main():
    parser = argparse.ArgumentParser(description="Enter Guac Mode - Interactive Python with LLM assistance.")
    parser.add_argument("--config_dir", type=str, help="Guac configuration directory.")
    parser.add_argument("--plots_dir", type=str, help="Directory to save plots.")
    parser.add_argument("--npc_team_dir", type=str, default=os.path.expanduser('~/.npcsh/guac/npc_team/'), 
                        help="NPC team directory for Guac.")
    parser.add_argument("--refresh_period", type=int, help="Number of commands before suggesting /refresh.")
    parser.add_argument("--default_mode", type=str, choices=["agent", "chat", "cmd", "ride"], 
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