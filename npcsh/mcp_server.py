"""
Thin wrapper — the MCP server lives in npcpy now.
Kept for backward compat with deployed scripts.
"""
from npcpy.mcp_server import (
    main,
)

if __name__ == "__main__":
    main()
