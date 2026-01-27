import argparse
import sys
import os
import traceback
from typing import Optional

from npcsh._state import (
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_DB_PATH, 
    NPCSH_STREAM_OUTPUT,
    initial_state,
)
from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    render_markdown,
)
from npcpy.npc_compiler import NPC
from npcsh.routes import router
from sqlalchemy import create_engine

from npcsh._state import (
    setup_shell,
    execute_slash_command,
    execute_command,
)

def load_npc_by_name(npc_name: str = "sibiji", db_path: str = NPCSH_DB_PATH) -> Optional[NPC]:
    """Load NPC by name, with fallback logic matching npcsh"""
    if not npc_name:
        npc_name = "sibiji"

    project_npc_path = os.path.abspath(f"./npc_team/{npc_name}.npc")
    global_npc_path = os.path.expanduser(f"~/.npcsh/npc_team/{npc_name}.npc")

    chosen_path = None
    if os.path.exists(project_npc_path):
        chosen_path = project_npc_path
    elif os.path.exists(global_npc_path):
        chosen_path = global_npc_path
    elif os.path.exists(f"npcs/{npc_name}.npc"):
         chosen_path = f"npcs/{npc_name}.npc"

    if chosen_path:
        try:
            db_conn = create_engine(f'sqlite:///{NPCSH_DB_PATH}')
            npc = NPC(file=chosen_path, db_conn=db_conn)
            return npc
        except Exception as e:
            print(f"Warning: Failed to load NPC '{npc_name}' from {chosen_path}: {e}", file=sys.stderr)
            return None
    else:
        print(f"Warning: NPC file for '{npc_name}' not found in project or global paths.", file=sys.stderr)
        if npc_name != "sibiji":
            return load_npc_by_name("sibiji", db_path)
        return None
def main():
    from npcsh.routes import router
    
    parser = argparse.ArgumentParser(
        description=(
            "NPC Command Line Utilities. "
            "Call a command or provide a prompt for the default NPC."
        ),
        usage=(
            "npc <command> [command_args...] | "
            "<prompt> [--npc NAME] [--model MODEL] [--provider PROV]"
        )
    )
    parser.add_argument(
        "--model", 
        "-m", 
        help="LLM model to use (overrides NPC/defaults)", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--provider", 
        "-pr", 
        help="LLM provider to use (overrides NPC/defaults)", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "-n", 
        "--npc", 
        help="Name of the NPC to use (default: sibiji)", 
        type=str, 
        default="sibiji"
    )

    args, all_args = parser.parse_known_args()
    global_model = args.model
    global_provider = args.provider

    # Load team early so we can check for jinxs
    try:
        command_history, team, forenpc_obj = setup_shell()
        # Load jinxs into router so they're recognized as commands
        from npcsh._state import initialize_router_with_jinxs
        initialize_router_with_jinxs(team, router)
    except Exception as e:
        print(f"Warning: Could not set up full npcsh environment: {e}", file=sys.stderr)
        team = None
        forenpc_obj = None

    is_valid_command = False
    command_name = None

    if all_args:
        first_arg = all_args[0]
        if first_arg.startswith('/'):
            is_valid_command = True
            command_name = first_arg
            all_args = all_args[1:]
        elif first_arg in router.get_commands():
            is_valid_command = True
            command_name = '/' + first_arg
            all_args = all_args[1:]

    if is_valid_command:
        subparsers = parser.add_subparsers(
            dest="command", 
            title="Available Commands",
            help="Run 'npc <command> --help' for command-specific help"
        )

        for cmd_name, help_text in router.help_info.items():
            cmd_parser = subparsers.add_parser(
                cmd_name, 
                help=help_text, 
                add_help=False
            )
            cmd_parser.add_argument(
                'command_args', 
                nargs=argparse.REMAINDER,
                help='Arguments passed directly to the command handler'
            )

        args = parser.parse_args([command_name.lstrip('/')] + all_args)
        command_args = (
            args.command_args 
            if hasattr(args, 'command_args') 
            else []
        )
        unknown_args = []
    else:
        args.command = None
        command_args = []
        unknown_args = all_args

    if args.model is None:
        args.model = global_model
    if args.provider is None:
        args.provider = global_provider

    # Team already loaded above, just set up NPC
    if not forenpc_obj:
        forenpc_obj = load_npc_by_name(args.npc, NPCSH_DB_PATH)

    npc_instance = None
    if team and args.npc in team.npcs:
        npc_instance = team.npcs[args.npc]
    elif (
        team 
        and args.npc == team.forenpc.name 
        if team.forenpc 
        else False
    ):
        npc_instance = team.forenpc
    else:
        npc_instance = load_npc_by_name(args.npc, NPCSH_DB_PATH)

    if not npc_instance:
        print(f"Error: Could not load NPC '{args.npc}'", file=sys.stderr)
        sys.exit(1)

    shell_state = initial_state
    shell_state.npc = npc_instance
    shell_state.team = team
    shell_state.current_path = os.getcwd()
    shell_state.stream_output = NPCSH_STREAM_OUTPUT

    effective_model = (
        args.model 
        or (
            npc_instance.model 
            if npc_instance.model 
            else NPCSH_CHAT_MODEL
        )
    )
    effective_provider = (
        args.provider 
        or (
            npc_instance.provider 
            if npc_instance.provider 
            else NPCSH_CHAT_PROVIDER
        )
    )
    
    if args.model:
        npc_instance.model = effective_model
    if args.provider:
        npc_instance.provider = effective_provider
        
    try:
        if is_valid_command:
            full_command_str = command_name
            if command_args:
                full_command_str += " " + " ".join(command_args)
            
            print(f"Executing command: {full_command_str}")
            
            updated_state, result = execute_slash_command(
                full_command_str, 
                stdin_input=None, 
                state=shell_state, 
                stream=NPCSH_STREAM_OUTPUT, 
                router=router
            )

            if isinstance(result, dict):
                output = result.get("output") or result.get("response")
                model_for_stream = result.get('model', effective_model)
                provider_for_stream = result.get(
                    'provider', 
                    effective_provider
                )
                
                if (
                    NPCSH_STREAM_OUTPUT
                    and output is not None
                    and not isinstance(output, str)
                ):
                    print_and_process_stream_with_markdown(
                        output,
                        model_for_stream,
                        provider_for_stream
                    )
                elif output is not None:
                    render_markdown(str(output))
            elif result is not None:
                render_markdown(str(result))
            else:
                print(f"Command '{command_name}' executed.")

        else:
            prompt = " ".join(unknown_args)

            if not prompt:
                parser.print_help()
                sys.exit(1)

            print(
                f"Processing prompt: '{prompt}' with NPC: '{args.npc}'..."
            )

            # Use NPCSH_DEFAULT_MODE environment variable, default to 'agent' for tool execution
            shell_state.current_mode = os.environ.get('NPCSH_DEFAULT_MODE', 'agent')
            updated_state, result = execute_command(
                prompt, 
                shell_state, 
                router=router, 
                command_history=command_history
            )

            if isinstance(result, dict):
                output = result.get("output")
                model_for_stream = result.get('model', effective_model)
                provider_for_stream = result.get(
                    'provider', 
                    effective_provider
                )
                
                if (
                    hasattr(output, '__iter__')
                    and not isinstance(output, (str, bytes, dict, list))
                ):
                    print_and_process_stream_with_markdown(
                        output,
                        model_for_stream,
                        provider_for_stream,
                        show=True
                    )
                elif output is not None:
                    render_markdown(str(output))
            elif (
                hasattr(result, '__iter__') 
                and not isinstance(result, (str, bytes, dict, list))
            ):
                print_and_process_stream_with_markdown(
                    result,
                    effective_model,
                    effective_provider,
                    show=True
                )
            elif result is not None:
                render_markdown(str(result))


    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def jinx_main():
    """Entry point for bin jinxs called directly from CLI.

    Parses arguments as key=value pairs and executes the jinx.
    Example: nql show=1 model=daily_summary
    """
    import os

    # Get jinx name from command name
    jinx_name = os.path.basename(sys.argv[0])

    # Parse remaining args as key=value pairs
    args = sys.argv[1:]
    jinx_args = []

    for arg in args:
        if arg in ['-h', '--help']:
            print(f"Usage: {jinx_name} [key=value ...]")
            print(f"\nRun the '{jinx_name}' jinx with specified parameters.")
            print("\nExamples:")
            print(f"  {jinx_name} show=1")
            print(f"  {jinx_name} model=my_model db=~/mydb.db")
            print(f"\nOr use: npc {jinx_name} [key=value ...]")
            sys.exit(0)
        jinx_args.append(arg)

    # Build command string
    if jinx_args:
        command = f"/{jinx_name} " + " ".join(jinx_args)
    else:
        command = f"/{jinx_name}"

    try:
        _, team, forenpc_obj = setup_shell()

        from npcsh._state import initialize_router_with_jinxs
        initialize_router_with_jinxs(team, router)

        # Update the global initial_state with team/npc context
        initial_state.team = team
        initial_state.npc = forenpc_obj
        if forenpc_obj:
            initial_state.chat_model = forenpc_obj.model or NPCSH_CHAT_MODEL
            initial_state.chat_provider = forenpc_obj.provider or NPCSH_CHAT_PROVIDER

        _, result = execute_slash_command(
            command,
            stdin_input=None,
            state=initial_state,
            stream=NPCSH_STREAM_OUTPUT,
            router=router
        )

        if isinstance(result, dict):
            output = result.get("output") or result.get("response")
            if output is not None:
                render_markdown(str(output))
        elif result is not None:
            render_markdown(str(result))

    except Exception as e:
        print(f"Error executing jinx '{jinx_name}': {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()