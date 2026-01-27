"""
plonk - Vision-based GUI automation CLI entry point

This is a thin wrapper that executes the plonk.jinx through the jinx mechanism.
"""
import argparse
import sys

from npcsh._state import setup_shell


def main():
    parser = argparse.ArgumentParser(description="plonk - Vision-based GUI automation")
    parser.add_argument("task", nargs="*", help="Task description for GUI automation")
    parser.add_argument("--vmodel", type=str, help="Vision model to use (default: gpt-4o)")
    parser.add_argument("--vprovider", type=str, help="Vision provider (default: openai)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug output")
    args = parser.parse_args()

    if not args.task:
        parser.print_help()
        sys.exit(1)

    # Setup shell to get team and default NPC
    command_history, team, default_npc = setup_shell()

    if not team or "plonk" not in team.jinxs_dict:
        print("Error: plonk jinx not found. Ensure npc_team/jinxs/modes/plonk.jinx exists.")
        sys.exit(1)

    # Build context for jinx execution
    context = {
        "npc": default_npc,
        "team": team,
        "messages": [],
        "task": " ".join(args.task),
        "vmodel": args.vmodel,
        "vprovider": args.vprovider,
        "max_iterations": args.max_iterations,
        "debug": not args.no_debug,
    }

    # Execute the jinx
    plonk_jinx = team.jinxs_dict["plonk"]
    result = plonk_jinx.execute(context=context, npc=default_npc)

    if isinstance(result, dict) and result.get("output"):
        print(result["output"])


if __name__ == "__main__":
    main()
