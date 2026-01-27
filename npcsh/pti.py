"""
pti - Pardon-The-Interruption mode CLI entry point

This is a thin wrapper that executes the pti.jinx through the jinx mechanism.
"""
import argparse
import sys

from npcsh._state import setup_shell


def main():
    parser = argparse.ArgumentParser(description="pti - Human-in-the-loop reasoning mode")
    parser.add_argument("prompt", nargs="*", help="Initial prompt to start with")
    parser.add_argument("--model", "-m", type=str, help="LLM model to use")
    parser.add_argument("--provider", "-p", type=str, help="LLM provider to use")
    parser.add_argument("--files", "-f", nargs="*", help="Files to load into context")
    parser.add_argument("--reasoning-model", type=str, help="Model for reasoning (may differ from chat)")
    args = parser.parse_args()

    # Setup shell to get team and default NPC
    command_history, team, default_npc = setup_shell()

    if not team or "pti" not in team.jinxs_dict:
        print("Error: pti jinx not found. Ensure npc_team/jinxs/modes/pti.jinx exists.")
        sys.exit(1)

    # Build context for jinx execution
    context = {
        "npc": default_npc,
        "team": team,
        "messages": [],
        "model": args.model,
        "provider": args.provider,
        "files": ",".join(args.files) if args.files else None,
        "reasoning_model": args.reasoning_model,
    }

    # If initial prompt provided, add it
    if args.prompt:
        initial = " ".join(args.prompt)
        context["messages"] = [{"role": "user", "content": initial}]

    # Execute the jinx
    pti_jinx = team.jinxs_dict["pti"]
    result = pti_jinx.execute(context=context, npc=default_npc)

    if isinstance(result, dict) and result.get("output"):
        print(result["output"])


if __name__ == "__main__":
    main()
