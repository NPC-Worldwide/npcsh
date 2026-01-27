"""
corca - MCP-powered agentic shell CLI entry point

This is a thin wrapper that executes the corca.jinx through the jinx mechanism.
"""
import argparse
import sys

from npcsh._state import setup_shell


def main():
    parser = argparse.ArgumentParser(description="corca - MCP-powered agentic shell")
    parser.add_argument("command", nargs="*", help="Optional one-shot command to execute")
    parser.add_argument("--model", "-m", type=str, help="LLM model to use")
    parser.add_argument("--provider", "-p", type=str, help="LLM provider to use")
    parser.add_argument("--mcp-server", type=str, help="Path to MCP server script")
    args = parser.parse_args()

    # Setup shell to get team and default NPC
    command_history, team, default_npc = setup_shell()

    if not team or "corca" not in team.jinxs_dict:
        print("Error: corca jinx not found. Ensure npc_team/jinxs/modes/corca.jinx exists.")
        sys.exit(1)

    # Build context for jinx execution
    initial_command = " ".join(args.command) if args.command else None

    context = {
        "npc": default_npc,
        "team": team,
        "messages": [],
        "model": args.model,
        "provider": args.provider,
        "mcp_server_path": args.mcp_server,
        "initial_command": initial_command,
    }

    # Execute the jinx
    corca_jinx = team.jinxs_dict["corca"]
    result = corca_jinx.execute(context=context, npc=default_npc)

    if isinstance(result, dict) and result.get("output"):
        print(result["output"])


if __name__ == "__main__":
    main()
