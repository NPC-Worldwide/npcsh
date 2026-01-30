import os
import sys
import shutil
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

Ask a question, run a command, or type '/help' to get started.
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

    # Print startup info panel
    BLUE = "\033[1;94m"
    RUST = "\033[1;38;5;202m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    term_width = min(shutil.get_terminal_size().columns, 80)

    def _wrap_names(names, indent=4, sep="  "):
        """Wrap a list of prefixed names to fit terminal width."""
        lines = []
        line = " " * indent
        for name in names:
            if len(line) + len(name) + len(sep) > term_width and line.strip():
                lines.append(line.rstrip())
                line = " " * indent
            line += name + sep
        if line.strip():
            lines.append(line.rstrip())
        return "\n".join(lines)

    if not launched_jinx:
        print(f"  {DIM}mode:{RESET} {BOLD}{state.current_mode}{RESET}  {DIM}│{RESET}  {DIM}switch:{RESET} /agent  /cmd  /chat")

    # NPCs with @ prefix
    npc_names = [f"@{n}" for n in state.team.npcs.keys()]
    print(f"  {DIM}npcs:{RESET} {BLUE}{('  ').join(npc_names)}{RESET}")

    # Jinxs - organized by group, hiding incognide, computer_use, and browser
    hidden_folders = {'incognide', 'computer_use', 'browser'}
    jinxs_by_group = {}
    if hasattr(state.team, 'jinxs_dict'):
        for jinx_name, jinx_obj in state.team.jinxs_dict.items():
            group = 'other'
            subgroup = None
            if hasattr(jinx_obj, '_source_path') and jinx_obj._source_path:
                parts = jinx_obj._source_path.split(os.sep)
                if 'jinxs' in parts:
                    idx = parts.index('jinxs')
                    remaining = parts[idx + 1:]  # e.g. ['lib', 'core', 'sh.jinx']
                    if any(seg in hidden_folders for seg in remaining):
                        continue
                    if len(remaining) > 1:
                        group = remaining[0]
                        # For lib, use the subfolder as subgroup
                        if group == 'lib' and len(remaining) > 2:
                            subgroup = remaining[1]
                    else:
                        group = 'root'
            key = (group, subgroup)
            if key not in jinxs_by_group:
                jinxs_by_group[key] = []
            jinxs_by_group[key].append(jinx_name)

    if jinxs_by_group:
        print()
        # Collect top-level groups and their subgroups
        group_order = ['bin', 'modes', 'lib', 'npc_studio', 'root', 'other']
        # Sort keys: by group order, then subgroup alphabetically
        sorted_keys = sorted(
            jinxs_by_group.keys(),
            key=lambda k: (group_order.index(k[0]) if k[0] in group_order else 99, k[1] or '')
        )
        last_group = None
        for group, subgroup in sorted_keys:
            names = sorted(jinxs_by_group[(group, subgroup)])
            prefixed = [f"/{n}" for n in names]
            if group != last_group:
                label = group
                if group == 'lib' and subgroup:
                    label = f"lib/{subgroup}"
                print(f"  {RUST}{label}/{RESET}")
            elif subgroup:
                print(f"  {RUST}lib/{subgroup}/{RESET}")
            else:
                print(f"  {RUST}{group}/{RESET}")
            print(_wrap_names(prefixed))
            last_group = group
        print(f"\n  {DIM}/jinxs for full list{RESET}")
    print()
    
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

        try:
            for team_name, npc_name, path in session_scopes:
                try:
                    print(f"  -> Archiving knowledge for: T='{team_name}', N='{npc_name}', P='{path}'")

                    # Only build KG from approved memories, not raw conversation
                    memory_examples = command_history.get_memory_examples_for_context(
                        npc=npc_name, team=team_name, directory_path=path
                    )
                    approved = memory_examples.get("approved", []) + memory_examples.get("edited", [])

                    if not approved:
                        print("     ...No approved memories for this scope, skipping.")
                        continue

                    memory_statements = []
                    for mem in approved:
                        statement = mem.get("final_memory") or mem.get("initial_memory")
                        if statement:
                            memory_statements.append(statement)

                    if not memory_statements:
                        continue

                    memory_text = "\n".join(f"- {s}" for s in memory_statements)
                    print(colored(f"     evolving KG from {len(memory_statements)} approved memories...", "cyan"))

                    current_kg = load_kg_from_db(engine, team_name, npc_name, path)

                    evolved_kg, _ = kg_evolve_incremental(
                        existing_kg=current_kg,
                        new_content_text=memory_text,
                        model=current_state.npc.model,
                        provider=current_state.npc.provider,
                        npc=current_state.npc,
                        get_concepts=True,
                        link_concepts_facts=True,
                        link_concepts_concepts=True,
                        link_facts_facts=True,
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
        except KeyboardInterrupt:
            print(colored("\nSkipping knowledge archival.", "yellow"))

        sys.exit(0)

    while True:
        try:
            if state.messages is not None:
                if len(state.messages) > 20:
                    # Display usage before compacting
                    display_usage(state)

                    # Pass actual messages so compress_planning_state summarizes real conversation
                    try:
                        compressed_state = state.npc.compress_planning_state(state.messages)
                    except Exception:
                        # Fallback: just keep recent messages if summarization fails
                        compressed_state = None

                    # Keep last 6 messages (3 exchanges) for immediate context
                    recent = state.messages[-6:]
                    if compressed_state:
                        state.messages = [{"role": "system", "content": f"Session context (earlier conversation summary): {compressed_state}"}] + recent
                    else:
                        state.messages = recent

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
    parser.add_argument(
         "--refresh", action="store_true", help="Force refresh of NPCs and jinxs from package."
    )
    args = parser.parse_args()

    # Handle refresh flag - reset initialization and re-copy files
    if args.refresh:
        from npcsh._state import initialize_base_npcs_if_needed
        import shutil

        npcshrc_path = os.path.expanduser("~/.npcshrc")
        if os.path.exists(npcshrc_path):
            with open(npcshrc_path, "r") as f:
                content = f.read()
            content = content.replace("export NPCSH_INITIALIZED=1", "export NPCSH_INITIALIZED=0")
            with open(npcshrc_path, "w") as f:
                f.write(content)

        os.environ["NPCSH_INITIALIZED"] = "0"

        # Remove existing jinxs and NPCs to force fresh copy
        user_npc_team = os.path.expanduser("~/.npcsh/npc_team")
        jinxs_dir = os.path.join(user_npc_team, "jinxs")
        if os.path.exists(jinxs_dir):
            shutil.rmtree(jinxs_dir)
            print("Cleared existing jinxs directory")

        for f in os.listdir(user_npc_team) if os.path.exists(user_npc_team) else []:
            if f.endswith(".npc"):
                os.remove(os.path.join(user_npc_team, f))
                print(f"Removed {f}")

        db_path = os.path.expanduser("~/npcsh_history.db")
        print("Reinitializing NPCs and jinxs...")
        initialize_base_npcs_if_needed(db_path)
        print("Refresh complete!")

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
         # Handle output - check if it's a dict (from jinx) or a stream
         if isinstance(output, dict):
              display_output = output.get('output') or output.get('response') or str(output)
              print(display_output)
         elif final_state.stream_output and output is not None:
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