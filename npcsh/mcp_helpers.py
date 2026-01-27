#!/usr/bin/env python
"""
Raw MCP client with no exception handling and full visibility.
"""

import asyncio
import sys
import json
try:
    import inspect
except: 
    pass
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


from npcpy.gen.response import get_litellm_response
from npcsh._state import (
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_API_URL,
)

class MCPClient:
    """
    Raw MCP Client with no exception handling.
    """
    
    def __init__(
        self,
        model: str = NPCSH_CHAT_MODEL,
        provider: str = NPCSH_CHAT_PROVIDER,
        api_url: str = NPCSH_API_URL,
        api_key: Optional[str] = None,
        debug: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.debug = debug
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        self.available_tools = []
        
    def _log(self, message: str) -> None:
        """Log debug messages."""
        if self.debug:
            print(f"[MCP Client] {message}")

    async def connect_to_server(self, server_script_path: str) -> None:
        """
        Connect to an MCP server.
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        self._log(f"Connecting to server: {server_script_path}")
        
      
        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
      
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        
      
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        
      
        await self.session.initialize()
        
      
        response = await self.session.list_tools()
        self.tools = response.tools
        
      
        for tool in self.tools:
            print(f"\nJinx: {tool.name}")
            print(f"Description: {tool.description}")
            
          
            for attribute_name in dir(tool):
                if not attribute_name.startswith('_'):
                    attribute = getattr(tool, attribute_name)
                    if not callable(attribute):
                        print(f"  {attribute_name}: {attribute}")
            
          
            if hasattr(tool, 'source'):
                print(f"Source: {tool.source}")
                
          
            try:
                tool_module = inspect.getmodule(tool)
                if tool_module:
                    print(f"Module: {tool_module.__name__}")
                    if hasattr(tool_module, tool.name):
                        tool_func = getattr(tool_module, tool.name)
                        if callable(tool_func):
                            print(f"Function signature: {inspect.signature(tool_func)}")
            except:
                pass
        
      
        self.available_tools = []
        for tool in self.tools:
          
            schema = getattr(tool, "inputSchema", {})
            
          
            tool_info = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema
                }
            }
            self.available_tools.append(tool_info)
            
          
            print(f"\nJinx schema for {tool.name}:")
            print(json.dumps(schema, indent=2))
        
        tool_names = [tool.name for tool in self.tools]
        self._log(f"Available tools: {', '.join(tool_names)}")

    async def process_query(
        self,
        query: str,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query using the LLM and available tools.
        
        Args:
            query: User query
            messages: Optional conversation history
            stream: Whether to stream the response
            
        Returns:
            Dict with response text and updated messages
        """
        self._log(f"Processing query: {query}")
        
      
        if messages is None:
            messages = []
            
        current_messages = messages.copy()
        if not current_messages or current_messages[-1]["role"] != "user":
            current_messages.append({"role": "user", "content": query})
        elif current_messages[-1]["role"] == "user":
            current_messages[-1]["content"] = query
            
      
        self._log("Making initial LLM call with tools")
        response = get_litellm_response(
            model=self.model,
            provider=self.provider,
            api_url=self.api_url,
            api_key=self.api_key,
            messages=current_messages,
            tools=self.available_tools,
            stream=False
        )
        
      
        print("\nLLM Response:")
        print(json.dumps(response, indent=2, default=str))
        
      
        response_content = response.get("response", "")
        tool_calls = response.get("tool_calls", [])
        
      
        print("\nJinx Calls:")
        print(json.dumps(tool_calls, indent=2, default=str))
        
      
        final_text = []
        
      
        if response_content and not tool_calls:
            final_text.append(response_content)
            
          
            current_messages.append({
                "role": "assistant",
                "content": response_content
            })
        
      
        if tool_calls:
            self._log(f"Processing {len(tool_calls)} tool calls")
            
          
            assistant_message = {
                "role": "assistant",
                "content": response_content if response_content else None,
                "tool_calls": []
            }
            
          
            for tool_call in tool_calls:
              
                if isinstance(tool_call, dict):
                    tool_id = tool_call.get("id", "")
                    tool_name = tool_call.get("function", {}).get("name", "")
                    tool_args = tool_call.get("function", {}).get("arguments", {})
                else:
                  
                    tool_id = getattr(tool_call, "id", "")
                    tool_name = getattr(tool_call.function, "name", "")
                    tool_args = getattr(tool_call.function, "arguments", {})
                
              
                if isinstance(tool_args, str):
                    print(f"\nJinx args is string: {tool_args}")
                    tool_args = json.loads(tool_args)
                    print(f"Parsed to: {tool_args}")
                
              
                assistant_message["tool_calls"].append({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
                    }
                })
                
              
                self._log(f"Executing tool: {tool_name} with args: {tool_args}")
                print("\nExecuting tool call:")
                print(f"  Jinx name: {tool_name}")
                print(f"  Jinx args: {tool_args}")
                print(f"  Jinx args type: {type(tool_args)}")
                
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
              
                result = await self.session.call_tool(tool_name, tool_args)
                
              
                print("\nJinx Result:")
                print(f"  Result: {result}")
                print(f"  Content: {result.content}")
                print(f"  Content type: {type(result.content)}")
                
                tool_result = result.content
                
              
                if hasattr(tool_result, 'text'):
                    print(f"  TextContent detected, text: {tool_result.text}")
                    tool_result = tool_result.text
                elif isinstance(tool_result, list) and all(hasattr(item, 'text') for item in tool_result):
                    print("  List of TextContent detected")
                    tool_result = [item.text for item in tool_result]
                
              
                current_messages.append(assistant_message)
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(tool_result) if not isinstance(tool_result, str) else str(tool_result)
                })
            
          
            print("\nUpdated Messages:")
            print(json.dumps(current_messages, indent=2, default=str))
            
          
            self._log("Getting final response after tool calls")
            final_response = get_litellm_response(
                model=self.model,
                provider=self.provider,
                api_url=self.api_url,
                api_key=self.api_key,
                messages=current_messages,
                stream=stream
            )
            
            final_text.append(final_response.get("response", ""))
            
          
            current_messages.append({
                "role": "assistant",
                "content": final_response.get("response", "")
            })
        
        return {
            "response": "\n".join(final_text),
            "messages": current_messages
        }

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        messages = []
        
        while True:
            query = input("\nQuery: ").strip()
            
            if query.lower() == 'quit':
                break
            
          
            result = await self.process_query(query, messages)
            messages = result.get("messages", [])
            
          
            print("\nResponse:")
            print(result.get("response", ""))

    async def cleanup(self):
        """Clean up resources"""
        self._log("Cleaning up resources")
        await self.exit_stack.aclose()

async def main():
    """Entry point for the MCP client."""
    if len(sys.argv) < 2:
        print("Usage: python raw_mcp_client.py <path_to_server_script>")
        sys.exit(1)
        
    server_script = sys.argv[1]
    
  
    client = MCPClient()
    
  
    await client.connect_to_server(server_script)
    
  
    await client.chat_loop()
    
  
    await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())