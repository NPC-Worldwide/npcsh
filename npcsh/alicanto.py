"""
alicanto - Deep research mode CLI entry point

This is a thin wrapper that executes the alicanto.jinx through the jinx mechanism.
"""
import argparse

import sys

from npcsh._state import setup_shell


def main():
    parser = argparse.ArgumentParser(description="alicanto - Deep research with multiple perspectives")
    parser.add_argument("query", nargs="*", help="Research query")
    parser.add_argument("--model", "-m", type=str, help="LLM model to use")
    parser.add_argument("--provider", "-p", type=str, help="LLM provider to use")
    parser.add_argument("--num-npcs", type=int, default=5, help="Number of research perspectives")
    parser.add_argument("--depth", type=int, default=3, help="Research depth")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum research steps")
    parser.add_argument("--exploration", type=float, default=0.3, help="Exploration factor (0-1)")
    parser.add_argument("--creativity", type=float, default=0.5, help="Creativity factor (0-1)")
    parser.add_argument("--format", type=str, default="report", choices=["report", "summary", "full"],
                        help="Output format")
    parser.add_argument("--with-research", action="store_true", help="Include web research")
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        sys.exit(1)

    # Setup shell to get team and default NPC
    _, team, default_npc = setup_shell()

    if not team or "alicanto" not in team.jinxs_dict:
        print("Error: alicanto jinx not found. Ensure npc_team/jinxs/modes/alicanto.jinx exists.")
        sys.exit(1)

    # Build context for jinx execution
    context = {
        "npc": default_npc,
        "team": team,
        "messages": [],
        "query": " ".join(args.query),
        "model": args.model,
        "provider": args.provider,
        "num_npcs": args.num_npcs,
        "depth": args.depth,
        "max_steps": args.max_steps,
        "exploration": args.exploration,
        "creativity": args.creativity,
        "format": args.format,
        "skip_research": not args.with_research,
    }

    # Execute the jinx
    alicanto_jinx = team.jinxs_dict["alicanto"]
    result = alicanto_jinx.execute(context=context, npc=default_npc)

    if isinstance(result, dict) and result.get("output"):
        print(result["output"])


if __name__ == "__main__":
    main()
