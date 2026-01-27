"""
guac - Python data analysis mode CLI entry point

This is a thin wrapper that executes the guac.jinx through the jinx mechanism.
"""
import argparse
import sys

from npcsh._state import setup_shell


def main():
    parser = argparse.ArgumentParser(description="guac - Python data analysis mode")
    parser.add_argument("--model", "-m", type=str, help="LLM model to use")
    parser.add_argument("--provider", "-p", type=str, help="LLM provider to use")
    parser.add_argument("--plots-dir", type=str, help="Directory to save plots")
    args = parser.parse_args()

    # Setup shell to get team and default NPC
    command_history, team, default_npc = setup_shell()

    if not team or "guac" not in team.jinxs_dict:
        print("Error: guac jinx not found. Ensure npc_team/jinxs/modes/guac.jinx exists.")
        sys.exit(1)

    # Build context for jinx execution
    context = {
        "npc": default_npc,
        "team": team,
        "messages": [],
        "model": args.model,
        "provider": args.provider,
        "plots_dir": args.plots_dir,
    }

    # Execute the jinx
    guac_jinx = team.jinxs_dict["guac"]
    result = guac_jinx.execute(context=context, npc=default_npc)

    if isinstance(result, dict) and result.get("output"):
        print(result["output"])


if __name__ == "__main__":
    main()
