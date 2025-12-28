import os
import sys
import argparse
import importlib.metadata
import warnings

# Suppress pydantic serialization warnings from litellm
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import platform
try:
    from termcolor import colored
except: 
    pass
from npcpy.npc_sysenv import (
    render_markdown,
)
from npcpy.memory.command_history import (
    CommandHistory,
    load_kg_from_db, 
    save_kg_to_db, 
)
from npcpy.npc_compiler import NPC
from npcpy.memory.knowledge_graph import (
    kg_evolve_incremental
)

try:
    import readline
except:
    print('no readline support, some features may not work as desired. ')

try:
    VERSION = importlib.metadata.version("npcsh")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"

from npcsh._state import (
    initial_state,
    orange,
    ShellState,
    execute_command,
    make_completer,
    process_result,
    readline_safe_prompt,
    setup_shell,
    get_multiline_input,
)


def display_usage(state: ShellState):
    """Display token usage and cost summary."""
    inp = state.session_input_tokens
    out = state.session_output_tokens
    cost = state.session_cost_usd
    turns = state.turn_count
    total = inp + out

    def fmt(n):
        return f"{n/1000:.1f}k" if n >= 1000 else str(n)

    def fmt_cost(c):
        if c == 0:
            return "free"
        elif c < 0.01:
            return f"${c:.4f}"
        else:
            return f"${c:.2f}"

    print(colored("\n─────────────────────────────", "cyan"))
    print(colored("📊 Session Usage", "cyan", attrs=["bold"]))
    print(f"   Tokens: {fmt(inp)} in / {fmt(out)} out ({fmt(total)} total)")
    print(f"   Cost:   {fmt_cost(cost)}")
    print(f"   Turns:  {turns}")
    print(colored("─────────────────────────────\n", "cyan"))


def print_welcome_art(npc=None):
    """Print welcome art - from NPC if available, otherwise default npcsh art."""
    BLUE = "\033[1;94m"
    RUST = "\033[1;38;5;202m"
    RESET = "\033[0m"

    # If NPC has ascii_art, display it with colors
    if npc and hasattr(npc, 'ascii_art') and npc.ascii_art:
        art = npc.ascii_art
        colors = getattr(npc, 'colors', {}) or {}

        if colors:
            top = colors.get("top", "255,255,255")
            bottom = colors.get("bottom", "255,255,255")
            lines = art.strip().split("\n")
            mid = len(lines) // 2

            try:
                tr, tg, tb = map(int, top.split(","))
                br, bg, bb = map(int, bottom.split(","))
            except:
                tr, tg, tb = 255, 255, 255
                br, bg, bb = 255, 255, 255

            for i, line in enumerate(lines):
                if i < mid:
                    print(f"\033[38;2;{tr};{tg};{tb}m{line}\033[0m")
                else:
                    print(f"\033[38;2;{br};{bg};{bb}m{line}\033[0m")
        else:
            print(art)
        print()
        return

    # Default npcsh art
    print(f"""
{BLUE}___________________________________________{RESET}

Welcome to {BLUE}npc{RESET}{RUST}sh{RESET}!
{BLUE}                    {RESET}{RUST}        _       \\\\{RESET}
{BLUE} _ __   _ __    ___ {RESET}{RUST}  ___  | |___    \\\\{RESET}
{BLUE}| '_ \\ | '_ \\  / __|{RESET}{RUST} / __/ | |_ _|    \\\\{RESET}
{BLUE}| | | || |_) |( |__ {RESET}{RUST} \\_  \\ | | | |    //{RESET}
{BLUE}|_| |_|| .__/  \\___/{RESET}{RUST} |___/ |_| |_|   //{RESET}
       {BLUE}|🤖|          {RESET}{RUST}               //{RESET}
       {BLUE}|🤖|{RESET}
       {BLUE}|🤖|{RESET}
{RUST}___________________________________________{RESET}

Begin by asking a question, issuing a bash command, or typing '/help' for more information.
""")


def run_repl(command_history: CommandHistory, initial_state: ShellState, router, launched_agent: str = None, launched_jinx: str = None):
    state = initial_state

    # Print welcome art - NPC art if launched with an agent, otherwise default
    if not launched_jinx:
        print_welcome_art(state.npc if launched_agent else None)

    # If launched with a jinx mode, auto-execute that jinx
    if launched_jinx:
        state, output = execute_command(f"/{launched_jinx}", state, router=router, command_history=command_history)
        process_result(f"/{launched_jinx}", state, output, command_history)
    else:
        render_markdown(f'- Using {state.current_mode} mode. Use /agent, /cmd, or /chat to switch to other modes')
    render_markdown(f'- To switch to a different NPC, type /npc <npc_name> or /n <npc_name> to switch to that NPC.')
    render_markdown('\n- Here are the current NPCs available in your team: ' + ', '.join([npc_name for npc_name in state.team.npcs.keys()]))
    # Show jinxs organized by folder using _source_path from jinx objects
    jinxs_by_folder = {}
    if hasattr(state.team, 'jinxs_dict'):
        for jinx_name, jinx_obj in state.team.jinxs_dict.items():
            folder = 'other'
            if hasattr(jinx_obj, '_source_path') and jinx_obj._source_path:
                parts = jinx_obj._source_path.split(os.sep)
                if 'jinxs' in parts:
                    idx = parts.index('jinxs')
                    if idx + 1 < len(parts) - 1:
                        folder = parts[idx + 1]
                    else:
                        folder = 'root'
            if folder not in jinxs_by_folder:
                jinxs_by_folder[folder] = []
            jinxs_by_folder[folder].append(jinx_name)

    if jinxs_by_folder:
        folder_order = ['bin', 'lib', 'npc_studio', 'root', 'other']
        sorted_folders = sorted(jinxs_by_folder.keys(), key=lambda x: (folder_order.index(x) if x in folder_order else 99, x))
        jinx_summary = []
        for folder in sorted_folders:
            count = len(jinxs_by_folder[folder])
            jinx_summary.append(f"{folder}/ ({count})")
        render_markdown('\n- Available Jinxs: ' + ', '.join(jinx_summary) + ' — use `/jinxs` for details')
    
    is_windows = platform.system().lower().startswith("win")
    try:
        completer = make_completer(state, router)
        readline.set_completer(completer)
    except:
        pass
    session_scopes = set()

    def exit_shell(current_state: ShellState):
        print("\nGoodbye!")
        print(colored("Processing and archiving all session knowledge...", "cyan"))
        
        engine = command_history.engine

        for team_name, npc_name, path in session_scopes:
            try:
                print(f"  -> Archiving knowledge for: T='{team_name}', N='{npc_name}', P='{path}'")
                
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

                current_kg = load_kg_from_db(engine, team_name, npc_name, path)
                
                evolved_kg, _ = kg_evolve_incremental(
                    existing_kg=current_kg,
                    new_content_text=full_text,
                    model=current_state.npc.model,
                    provider=current_state.npc.provider, 
                    npc= current_state.npc,
                    get_concepts=True,
                    link_concepts_facts = True, 
                    link_concepts_concepts = True, 
                    link_facts_facts = True, 
                )
                
                save_kg_to_db(engine,
                              evolved_kg,
                              team_name, 
                              npc_name, 
                              path)

            except Exception as e:
                import traceback
                print(colored(f"Failed to process KG for scope ({team_name}, {npc_name}, {path}): {e}", "red"))
                traceback.print_exc()

        sys.exit(0)

    while True:
        try:
            if state.messages is not None:
                if len(state.messages) > 20:
                    # Display usage before compacting
                    display_usage(state)

                    planning_state = {
                        "goal": "ongoing npcsh session",
                        "facts": [f"Working in {state.current_path}", f"Current mode: {state.current_mode}"],
                        "successes": [],
                        "mistakes": [],
                        "todos": [],
                        "constraints": ["Follow user requests", "Use appropriate mode for tasks"]
                    }
                    compressed_state = state.npc.compress_planning_state(planning_state)
                    state.messages = [{"role": "system", "content": f"Session context: {compressed_state}"}]

                try:
                    completer = make_completer(state, router)
                    readline.set_completer(completer)
                except:
                    pass

            display_model = state.chat_model
            if isinstance(state.npc, NPC) and state.npc.model:
                display_model = state.npc.model

            npc_name = state.npc.name if isinstance(state.npc, NPC) else "npcsh"
            team_name = state.team.name if state.team else ""

            # Check if model is local (ollama) or remote (has cost)
            provider = state.chat_provider
            if isinstance(state.npc, NPC) and state.npc.provider:
                provider = state.npc.provider
            is_local = provider and provider.lower() in ['ollama', 'transformers', 'local']

            # Build token/cost string for hint line
            if state.session_input_tokens > 0 or state.session_output_tokens > 0:
                usage_str = f"📊 {state.session_input_tokens:,} in / {state.session_output_tokens:,} out"
                if not is_local and state.session_cost_usd > 0:
                    usage_str += f" | ${state.session_cost_usd:.4f}"
                # Add elapsed time
                import time
                elapsed = time.time() - state.session_start_time
                if elapsed >= 3600:
                    hours = int(elapsed // 3600)
                    mins = int((elapsed % 3600) // 60)
                    usage_str += f" | {hours}h{mins}m"
                elif elapsed >= 60:
                    mins = int(elapsed // 60)
                    secs = int(elapsed % 60)
                    usage_str += f" | {mins}m{secs}s"
                else:
                    usage_str += f" | {int(elapsed)}s"
                token_hint = colored(usage_str, "white", attrs=["dark"])
            else:
                token_hint = ""

            if is_windows:
                print(f"cwd: {state.current_path}")
                status = f"{npc_name}"
                if team_name:
                    status += f" | {team_name}"
                status += f" | {display_model}"
                print(status)
                prompt = "> "
            else:
                # Line 1: cwd (full path)
                cwd_line = colored("📁 ", "blue") + colored(state.current_path, "blue")
                print(cwd_line)

                # Line 2: npc | team | model
                npc_colored = orange(npc_name) if isinstance(state.npc, NPC) else colored("npcsh", "cyan")
                parts = [colored("🤖 ", "yellow") + npc_colored]
                if team_name:
                    parts.append(colored("👥 ", "magenta") + colored(team_name, "magenta"))
                parts.append(colored(display_model, "white", attrs=["dark"]))
                print(" | ".join(parts))

                prompt = colored("> ", "green")

            user_input = get_multiline_input(prompt, state=state, router=router, token_hint=token_hint).strip()
          
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

            state, output = execute_command(user_input, 
                                            state, 
                                            review = False, 
                                            router=router, 
                                            command_history=command_history)

            process_result(user_input, 
                           state, 
                           output, 
                           command_history, 
                           )
        
        except KeyboardInterrupt:
            # Double Ctrl+C exits (handled in _input_with_hint_below)
            exit_shell(state)

        except EOFError:
            exit_shell(state)
        except Exception as e:            
            if is_windows and "EOF" in str(e).lower():
                print("\nHint: On Windows, use Ctrl+Z then Enter for EOF, or type 'exit'")
                continue
            raise
        

def main(npc_name: str = None) -> None:
    """
    Main entry point for npcsh.

    Args:
        npc_name: If provided, start with this NPC active. Used by agent-specific
                  entry points (guac, plonk, corca, etc.)
    """
    from npcsh.routes import router

    # If no npc_name provided, check how we were invoked
    if npc_name is None:
        invoked_as = os.path.basename(sys.argv[0])
        if invoked_as not in ('npcsh', 'npc'):
            npc_name = invoked_as

    parser = argparse.ArgumentParser(description="npcsh - An NPC-powered shell.")
    parser.add_argument(
        "-v", "--version", action="version", version=f"npcsh version {VERSION}"
    )
    parser.add_argument(
         "-c", "--command", type=str, help="Execute a single command and exit."
    )
    parser.add_argument(
         "-n", "--npc", type=str, help="Start with a specific NPC active."
    )
    args = parser.parse_args()

    command_history, team, default_npc = setup_shell()

    if team and hasattr(team, 'jinxs_dict'):
        for jinx_name, jinx_obj in team.jinxs_dict.items():
            router.register_jinx(jinx_obj)

    # Determine which NPC to start with
    # Special cases: these are jinxes/modes, not NPCs
    jinx_modes = {"yap", "spool", "wander"}
    target_npc_name = npc_name or args.npc

    if target_npc_name and target_npc_name.lower() in jinx_modes:
        # It's a jinx mode, use default NPC
        initial_state.npc = default_npc
    elif target_npc_name and team:
        target_npc = team.npcs.get(target_npc_name)
        if target_npc:
            initial_state.npc = target_npc
        else:
            print(f"Warning: NPC '{target_npc_name}' not found. Using default.")
            initial_state.npc = default_npc
    else:
        initial_state.npc = default_npc

    initial_state.team = team
    if args.command:
         state = initial_state
         state.current_path = os.getcwd()
         final_state, output = execute_command(args.command, state, router=router, command_history=command_history)
         if final_state.stream_output:
              for chunk in output:
                  print(str(chunk), end='')
              print()
         elif output is not None:
              print(output)
    else:
        # Determine if launching an NPC or a jinx mode
        if target_npc_name and target_npc_name.lower() in jinx_modes:
            run_repl(command_history, initial_state, router, launched_jinx=target_npc_name.lower())
        else:
            run_repl(command_history, initial_state, router, launched_agent=npc_name)
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()  # Clean exit on Ctrl+C without "KeyboardInterrupt" message
        sys.exit(0)