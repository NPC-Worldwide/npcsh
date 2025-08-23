import argparse
import sys
import os
import traceback
from typing import Optional

from npcsh._state import (
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_API_URL, 
    NPCSH_DB_PATH, 
    NPCSH_STREAM_OUTPUT,
    initial_state,
)
from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    render_markdown,
)
from npcpy.npc_compiler import NPC, Team
from npcsh.routes import router
from npcpy.llm_funcs import check_llm_command
from sqlalchemy import create_engine

# Import the key functions from npcsh
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
        description="NPC Command Line Utilities. Call a command or provide a prompt for the default NPC.",
        usage="npc <command> [command_args...] | <prompt> [--npc NAME] [--model MODEL] [--provider PROV]"
    )
    parser.add_argument(
        "--model", "-m", help="LLM model to use (overrides NPC/defaults)", type=str, default=None
    )
    parser.add_argument(
        "--provider", "-pr", help="LLM provider to use (overrides NPC/defaults)", type=str, default=None
    )
    parser.add_argument(
        "-n", "--npc", help="Name of the NPC to use (default: sibiji)", type=str, default="sibiji"
    )

    # Parse arguments
    args, all_args = parser.parse_known_args()
    global_model = args.model
    global_provider = args.provider

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
        subparsers = parser.add_subparsers(dest="command", title="Available Commands",
                                         help="Run 'npc <command> --help' for command-specific help")

        for cmd_name, help_text in router.help_info.items():
            cmd_parser = subparsers.add_parser(cmd_name, help=help_text, add_help=False)
            cmd_parser.add_argument('command_args', nargs=argparse.REMAINDER,
                                    help='Arguments passed directly to the command handler')

        # Re-parse with command subparsers
        args = parser.parse_args([command_name.lstrip('/')] + all_args)
        command_args = args.command_args if hasattr(args, 'command_args') else []
        unknown_args = []
    else:
        # Treat all arguments as a prompt
        args.command = None
        command_args = []
        unknown_args = all_args

    if args.model is None:
        args.model = global_model
    if args.provider is None:
        args.provider = global_provider

    # Use npcsh's setup_shell to get proper team and NPC setup
    try:
        command_history, team, forenpc_obj = setup_shell()
    except Exception as e:
        print(f"Warning: Could not set up full npcsh environment: {e}", file=sys.stderr)
        print("Falling back to basic NPC loading...", file=sys.stderr)
        team = None
        forenpc_obj = load_npc_by_name(args.npc, NPCSH_DB_PATH)

    # Determine which NPC to use
    npc_instance = None
    if team and args.npc in team.npcs:
        npc_instance = team.npcs[args.npc]
    elif team and args.npc == team.forenpc.name if team.forenpc else False:
        npc_instance = team.forenpc
    else:
        npc_instance = load_npc_by_name(args.npc, NPCSH_DB_PATH)

    if not npc_instance:
        print(f"Error: Could not load NPC '{args.npc}'", file=sys.stderr)
        sys.exit(1)

    # Now check for jinxs if we haven't identified a command yet
    if not is_valid_command and all_args:
        first_arg = all_args[0]
        
        # Check if first argument is a jinx name
        jinx_found = False
        if team and first_arg in team.jinxs_dict:
            jinx_found = True
        elif isinstance(npc_instance, NPC) and hasattr(npc_instance, 'jinxs_dict') and first_arg in npc_instance.jinxs_dict:
            jinx_found = True
            
        if jinx_found:
            is_valid_command = True
            command_name = '/' + first_arg
            all_args = all_args[1:]

    # Create a shell state object similar to npcsh
    shell_state = initial_state
    shell_state.npc = npc_instance
    shell_state.team = team
    shell_state.current_path = os.getcwd()
    shell_state.stream_output = NPCSH_STREAM_OUTPUT

    # Override model/provider if specified
    effective_model = args.model or (npc_instance.model if npc_instance.model else NPCSH_CHAT_MODEL)
    effective_provider = args.provider or (npc_instance.provider if npc_instance.provider else NPCSH_CHAT_PROVIDER)
    
    # Update the NPC's model/provider for this session if overridden
    if args.model:
        npc_instance.model = effective_model
    if args.provider:
        npc_instance.provider = effective_provider
    try:
        if is_valid_command:
            # Handle slash command using npcsh's execute_slash_command
            full_command_str = command_name
            if command_args:
                full_command_str += " " + " ".join(command_args)
            
            print(f"Executing command: {full_command_str}")
            
            updated_state, result = execute_slash_command(
                full_command_str, 
                stdin_input=None, 
                state=shell_state, 
                stream=NPCSH_STREAM_OUTPUT, 
                router = router
            )

            # Process and display the result
            if isinstance(result, dict):
                output = result.get("output") or result.get("response")
                model_for_stream = result.get('model', effective_model)
                provider_for_stream = result.get('provider', effective_provider)
                
                if NPCSH_STREAM_OUTPUT and not isinstance(output, str):
                     print_and_process_stream_with_markdown(output, model_for_stream, provider_for_stream)
                elif output is not None:
                     render_markdown(str(output))
            elif result is not None:
                render_markdown(str(result))
            else:
                print(f"Command '{command_name}' executed.")

        else:
            # Process as a regular prompt using npcsh's execution logic
            prompt = " ".join(unknown_args)

            if not prompt:
                # If no prompt and no command, show help
                parser.print_help()
                sys.exit(1)

            print(f"Processing prompt: '{prompt}' with NPC: '{args.npc}'...")
            
            # Use npcsh's execute_command but force it to chat mode for simple prompts
            shell_state.current_mode = 'chat'
            updated_state, result = execute_command(prompt, shell_state)

            # Process and display the result
            if isinstance(result, dict):
                output = result.get("output")
                model_for_stream = result.get('model', effective_model)
                provider_for_stream = result.get('provider', effective_provider)
                
                if NPCSH_STREAM_OUTPUT and hasattr(output, '__iter__') and not isinstance(output, (str, bytes, dict, list)):
                    print_and_process_stream_with_markdown(output, model_for_stream, provider_for_stream)
                elif output is not None:
                    render_markdown(str(output))
            elif result is not None:
                 render_markdown(str(result))

    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()