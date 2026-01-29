from typing import Callable, Dict, Any, List, Optional
import functools
import os
import traceback
import inspect

from npcpy.npc_compiler import Jinx, extract_jinx_inputs


class CommandRouter:
    def __init__(self):
        self.routes = {}
        self.help_info = {}
        self.jinx_routes = {}
        self.jinx_objects = {}  # command_name -> Jinx object

    def route(self, command: str, help_text: str = "") -> Callable:
        def wrapper(func):
            self.routes[command] = func
            self.help_info[command] = help_text

            @functools.wraps(func)
            def wrapped_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_func
        return wrapper

    def load_jinx_routes(self, jinxs_dir: str):
        if not os.path.exists(jinxs_dir):
            print(f"Jinxs directory not found: {jinxs_dir}")
            return
        
        for root, dirs, files in os.walk(jinxs_dir):
            for filename in files:
                if filename.endswith('.jinx'):
                    jinx_path = os.path.join(root, filename)
                    try:
                        jinx = Jinx(jinx_path=jinx_path)
                        self.register_jinx(jinx)
                    except Exception as e:
                        print(f"Error loading jinx {filename}: {e}")
    
    def register_jinx(self, jinx: Jinx):
        command_name = jinx.jinx_name

        def jinx_handler(command: str, **kwargs):
            return self._execute_jinx(jinx, command, **kwargs)

        self.jinx_routes[command_name] = jinx_handler
        self.jinx_objects[command_name] = jinx
        self.help_info[command_name] = jinx.description or "Jinx command"

    def is_interactive(self, command_name: str) -> bool:
        jinx = self.jinx_objects.get(command_name)
        return bool(jinx and getattr(jinx, 'interactive', False))
    
    def _execute_jinx(self, jinx: Jinx, command: str, **kwargs):
        messages = kwargs.get("messages", [])
        npc = kwargs.get('npc')
        
        try:
            import shlex
            
            try:
                parts = shlex.split(command)
            except ValueError:
                parts = command.split()
            args = parts[1:] if len(parts) > 1 else []
            
            # Use extract_jinx_inputs
            input_values = extract_jinx_inputs(args, jinx)
            
            # Build extra_globals for jinx execution
            from npcpy.memory.command_history import CommandHistory, load_kg_from_db
            from npcpy.memory.search import execute_rag_command, execute_brainblast_command
            from npcpy.data.load import load_file_contents
            from npcpy.data.web import search_web
            
            application_globals_for_jinx = {
                "CommandHistory": CommandHistory,
                "load_kg_from_db": load_kg_from_db,
                "execute_rag_command": execute_rag_command,
                "execute_brainblast_command": execute_brainblast_command,
                "load_file_contents": load_file_contents,
                "search_web": search_web,
                'state': kwargs.get('state')
            }
            
            # Add functions from _state module if available
            try:
                from npcsh import _state
                for name, func in inspect.getmembers(_state, inspect.isfunction):
                    application_globals_for_jinx[name] = func
            except:
                pass
            
            jinx_output = jinx.execute(
                input_values=input_values,
                npc=npc,
                messages=messages,
                extra_globals=application_globals_for_jinx
            )
            
            if isinstance(jinx_output, dict):
                return {
                    "output": jinx_output.get('output', str(jinx_output)),
                    "messages": jinx_output.get('messages', messages)
                }
            else:
                return {"output": str(jinx_output), "messages": messages}
                
        except Exception as e:
            traceback.print_exc()
            return {
                "output": f"Error executing jinx '{jinx.jinx_name}': {e}",
                "messages": messages
            }

    def get_route(self, command: str) -> Optional[Callable]:
        if command in self.routes:
            return self.routes[command]
        elif command in self.jinx_routes:
            return self.jinx_routes[command]
        return None

    def execute(self, command_str: str, **kwargs) -> Any:
        command_name = command_str.split()[0].lstrip('/')
        route_func = self.get_route(command_name)
        if route_func:
            return route_func(command=command_str, **kwargs)
        return None

    def get_commands(self) -> List[str]:
        all_commands = list(self.routes.keys()) + list(self.jinx_routes.keys())
        return sorted(set(all_commands))

    def get_help(self, command: str = None) -> Dict[str, str]:
        if command:
            if command in self.help_info:
                return {command: self.help_info[command]}
            return {}
        return self.help_info


router = CommandRouter()