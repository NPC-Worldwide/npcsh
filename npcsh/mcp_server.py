"""
Thin wrapper — the MCP server lives in npcpy now.
Kept for backward compat with deployed scripts.
"""
from npcpy.mcp_server import (  # noqa: F401
    NPCServerState,
    build_server,
    discover_team_path,
    main,
)

if __name__ == "__main__":
    main()
