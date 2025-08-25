#!/usr/bin/env python
"""
Enhanced MCP server that incorporates functionality from npcpy.routes, 
npcpy.llm_funcs, and npcpy.npc_compiler as tools.
"""

import os
import subprocess
import json
import asyncio

from typing import Optional, Dict, Any, List, Union, Callable
# MCP imports
from mcp.server.fastmcp import FastMCP
import importlib
# npcpy imports


import os
import subprocess
import json
import asyncio
try:
    import inspect
except: 
    pass
from typing import Optional, Dict, Any, List, Union, Callable, get_type_hints
# Add these imports to the top of your file
from functools import wraps
import sys 

from npcpy.llm_funcs import  generate_group_candidates, abstract, extract_facts, zoom_in,   execute_llm_command, gen_image
from npcpy.memory.search import search_similar_texts, execute_search_command, execute_rag_command,  answer_with_rag, execute_brainblast_command
from npcpy.data.load import load_file_contents
from npcpy.memory.command_history import CommandHistory
from npcpy.data.image import capture_screenshot
from npcpy.data.web import search_web

from npcsh._state import NPCSH_DB_PATH

command_history = CommandHistory(db=NPCSH_DB_PATH)

# Initialize the MCP server
mcp = FastMCP("npcsh_mcp")

# Define the default workspace
DEFAULT_WORKSPACE = os.path.join(os.getcwd(), "workspace")
os.makedirs(DEFAULT_WORKSPACE, exist_ok=True)

# ==================== SYSTEM TOOLS ====================
@mcp.tool()
async def run_server_command(command: str) -> str:
    """
    Run a terminal command in the workspace.
    
    Args:
        command: The shell command to run
        
    Returns:
        The command output or an error message.
    """
    try:
        result = subprocess.run(
            command, 
            cwd=DEFAULT_WORKSPACE, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30  # Add timeout to prevent hanging
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
        # Skip functions that don't have docstrings
        if not func.__doc__:
            print(f"Skipping function without docstring: {func.__name__}")
            continue
            
        # Create async wrapper with improved argument handling
        async_func = make_async_wrapper(func)
        
        # Register as MCP tool
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
        # Get all callables from the module that don't start with underscore
        functions = []
        for name, func in inspect.getmembers(module, callable):
            if not name.startswith('_'):
                # Check if it's a function, not a class
                if inspect.isfunction(func) or inspect.ismethod(func):
                    functions.append(func)
        return functions
    except ImportError as e:
        print(f"Warning: Could not import module {module_name}: {e}")
        return []

print("Loading tools from npcpy modules...")





def register_selected_npcpy_tools():
    tools = [generate_group_candidates, 
             abstract, 
             extract_facts, 
             zoom_in, 
             execute_llm_command, 
             gen_image, 
             load_file_contents, 
             capture_screenshot, 
             search_web, ]

    for func in tools:
        # Ensure a docstring exists for schema generation
        if not (getattr(func, "__doc__", None) and func.__doc__.strip()):
            fallback_doc = f"Tool wrapper for {func.__name__}."
            try:
                func.__doc__ = fallback_doc
            except Exception:
                pass  # Some builtins may not allow setting __doc__

        try:
            async_func = make_async_wrapper(func)
            mcp.tool()(async_func)
            print(f"Registered npcpy tool: {func.__name__}")
        except Exception as e:
            print(f"Failed to register npcpy tool {func.__name__}: {e}")
register_selected_npcpy_tools()




# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    print(f"Starting enhanced NPCPY MCP server...")
    print(f"Workspace: {DEFAULT_WORKSPACE}")
    
    # Run the server
    mcp.run(transport="stdio")