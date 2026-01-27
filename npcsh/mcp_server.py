
"""
Enhanced MCP server that incorporates functionality from npcpy.routes, 
npcpy.llm_funcs, and npcpy.npc_compiler as tools.
"""

import os
import subprocess
import json

from typing import List, Callable

from mcp.server.fastmcp import FastMCP
import importlib

from sqlalchemy import text


import os
import subprocess
import json
try:
    import inspect
except: 
    pass
from typing import List, Callable

from functools import wraps

from npcpy.llm_funcs import (gen_image)
from npcpy.data.load import load_file_contents
from npcpy.memory.command_history import CommandHistory
from npcpy.data.image import capture_screenshot
from npcpy.data.web import search_web

from npcsh._state import NPCSH_DB_PATH

command_history = CommandHistory(db=NPCSH_DB_PATH)

mcp = FastMCP("npcsh_mcp")


DEFAULT_WORKSPACE = os.path.join(os.getcwd(), "workspace")
os.makedirs(DEFAULT_WORKSPACE, exist_ok=True)

@mcp.tool()
async def add_memory(
    npc_name: str,
    team_name: str,
    content: str,
    memory_type: str = "observation",
    directory_path: str = None
) -> str:
    """
    Add a memory entry to the database.
    
    Args:
        npc_name: Name of the NPC this memory belongs to
        team_name: Name of the team the NPC belongs to
        content: The memory content to store
        memory_type: Type of memory (observation, preference, achievement, etc.)
        directory_path: Directory path context (defaults to current working directory)
        
    Returns:
        Success message with memory ID or error message
    """
    if directory_path is None:
        directory_path = os.getcwd()
    
    try:
        from npcpy.memory.command_history import generate_message_id
        message_id = generate_message_id()
        
        memory_id = command_history.add_memory_to_database(
            message_id=message_id,
            conversation_id='mcp_direct',
            npc=npc_name,
            team=team_name,
            directory_path=directory_path,
            initial_memory=content,
            status='active',
            model=None,
            provider=None
        )
        return f"Memory created successfully with ID: {memory_id}"
    except Exception as e:
        return f"Error creating memory: {str(e)}"

@mcp.tool()
async def search_memory(
    query: str,
    npc_name: str = None,
    team_name: str = None,
    directory_path: str = None,
    status_filter: str = None,
    limit: int = 10
) -> str:
    """
    Search memories in the database.
    
    Args:
        query: Search query text
        npc_name: Filter by specific NPC (optional)
        team_name: Filter by specific team (optional)
        directory_path: Filter by directory path (optional)
        status_filter: Filter by memory status (active, archived, etc.)
        limit: Maximum number of results to return
        
    Returns:
        JSON string of matching memories or error message
    """
    if directory_path is None:
        directory_path = os.getcwd()
    
    try:
        results = command_history.search_memory(
            query=query,
            npc=npc_name,
            team=team_name,
            directory_path=directory_path,
            status_filter=status_filter,
            limit=limit
        )
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching memories: {str(e)}"

@mcp.tool()
async def query_npcsh_database(sql_query: str) -> str:
    """
    Execute a SQL query against the npcsh_history.db database.
    
    Args:
        sql_query: SQL query to execute (SELECT statements only for safety)
        
    Returns:
        JSON string of query results or error message
    """
    # Safety check - only allow SELECT queries
    if not sql_query.strip().upper().startswith('SELECT'):
        return "Error: Only SELECT queries are allowed for safety"
    
    try:
        with command_history.engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            
            if not rows:
                return "Query executed successfully but returned no results"
            
            # Convert to list of dictionaries
            columns = result.keys()
            results = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                results.append(row_dict)
            
            return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return f"Database query error: {str(e)}"
@mcp.tool()
async def run_server_command(command: str, wd: str) -> str:
    """
    Run a terminal command in the workspace.
    
    Args:
        command: The shell command to run
        wd: The working directory to run the command in
        
    Returns:
        The command output or an error message.
    """
    try:
        result = subprocess.run(
            command, 
            cwd=wd,
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        return result.stdout or result.stderr or "Command completed with no output"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return str(e)



def make_async_wrapper(func: Callable) -> Callable:
    """Create an async wrapper for sync functions."""
    
    @wraps(func)
    async def async_wrapper(**kwargs):
        func_name = func.__name__
        print(f"MCP SERVER DEBUG: {func_name} called with kwargs={kwargs}", flush=True)
        
        try:
            result = func(**kwargs)
            print(f"MCP SERVER DEBUG: {func_name} returned type={type(result)}, result={result[:500] if isinstance(result, str) else result}", flush=True)
            return result
                
        except Exception as e:
            print(f"MCP SERVER DEBUG: {func_name} exception: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return f"Error in {func_name}: {e}"
    
    async_wrapper.__name__ = func.__name__
    async_wrapper.__doc__ = func.__doc__
    async_wrapper.__annotations__ = func.__annotations__
    
    return async_wrapper



def register_module_tools(module_name: str) -> None:
    """
    Register all suitable functions from a module as MCP tools with improved argument handling.
    """
    functions = load_module_functions(module_name)
    for func in functions:
      
        if not func.__doc__:
            print(f"Skipping function without docstring: {func.__name__}")
            continue
            
      
        async_func = make_async_wrapper(func)
        
      
        try:
            mcp.tool()(async_func)
            print(f"Registered tool: {func.__name__}")
        except Exception as e:
            print(f"Failed to register {func.__name__}: {e}")
def load_module_functions(module_name: str) -> List[Callable]:
    """
    Dynamically load functions from a module.
    """
    try:
        module = importlib.import_module(module_name)
      
        functions = []
        for name, func in inspect.getmembers(module, callable):
            if not name.startswith('_'):
              
                if inspect.isfunction(func) or inspect.ismethod(func):
                    functions.append(func)
        return functions
    except ImportError as e:
        print(f"Warning: Could not import module {module_name}: {e}")
        return []

print("Loading tools from npcpy modules...")





def register_selected_npcpy_tools():
    tools = [              
             gen_image, 
             load_file_contents, 
             capture_screenshot, 
             search_web ]

    for func in tools:
      
        if not (getattr(func, "__doc__", None) and func.__doc__.strip()):
            fallback_doc = f"Tool wrapper for {func.__name__}."
            try:
                func.__doc__ = fallback_doc
            except Exception:
                pass

        try:
            async_func = make_async_wrapper(func)
            mcp.tool()(async_func)
            print(f"Registered npcpy tool: {func.__name__}")
        except Exception as e:
            print(f"Failed to register npcpy tool {func.__name__}: {e}")
register_selected_npcpy_tools()






if __name__ == "__main__":
    print("Starting enhanced NPCPY MCP server...")
    print(f"Workspace: {DEFAULT_WORKSPACE}")
    
  
    mcp.run(transport="stdio")