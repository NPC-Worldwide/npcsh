"""
wander - Experimental wandering mode CLI entry point

This is a thin wrapper that executes the wander.jinx through the jinx mechanism.
"""
import argparse
import sys

from npcsh._state import setup_shell


def main():
    parser = argparse.ArgumentParser(description="wander - Creative exploration with varied temperatures")
    parser.add_argument("problem", nargs="*", help="Problem to explore through wandering")
    parser.add_argument("--model", "-m", type=str, help="LLM model to use")
    parser.add_argument("--provider", "-p", type=str, help="LLM provider to use")
    parser.add_argument("--environment", type=str, help="Metaphorical environment for wandering")
    parser.add_argument("--low-temp", type=float, default=0.5, help="Low temperature setting")
    parser.add_argument("--high-temp", type=float, default=1.9, help="High temperature setting")
    parser.add_argument("--n-streams", type=int, default=5, help="Number of exploration streams")
    parser.add_argument("--include-events", action="store_true", help="Include random events")
    parser.add_argument("--num-events", type=int, default=3, help="Number of events per stream")
    args = parser.parse_args()

    if not args.problem:
        parser.print_help()
        sys.exit(1)

    # Setup shell to get team and default NPC
    command_history, team, default_npc = setup_shell()

    if not team or "wander" not in team.jinxs_dict:
        print("Error: wander jinx not found. Ensure npc_team/jinxs/modes/wander.jinx exists.")
        sys.exit(1)

    # Build context for jinx execution
    context = {
        "npc": default_npc,
        "team": team,
        "messages": [],
        "problem": " ".join(args.problem),
        "model": args.model,
        "provider": args.provider,
        "environment": args.environment,
        "low_temp": args.low_temp,
        "high_temp": args.high_temp,
        "n_streams": args.n_streams,
        "include_events": args.include_events,
        "num_events": args.num_events,
    }

    # Execute the jinx
    wander_jinx = team.jinxs_dict["wander"]
    result = wander_jinx.execute(context=context, npc=default_npc)

    if isinstance(result, dict) and result.get("output"):
        print(result["output"])


if __name__ == "__main__":
    main()
