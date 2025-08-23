import os
import sys
import shlex
import argparse
from typing import Dict, List, Any, Optional

from termcolor import colored

from npcpy.memory.command_history import CommandHistory, save_conversation_message
from npcpy.npc_sysenv import (
    render_markdown
)
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_compiler import NPC
from npcpy.data.load import load_file_contents

from npcsh._state import (
    ShellState,
    setup_shell,
    get_multiline_input,
    readline_safe_prompt, 
    get_npc_path
)

ice = "🧊"
bear = "🐻‍❄️"
def print_pti_welcome_message():
    
    print(f"""
Welcome to PTI Mode!

{ice}{ice}{ice}   {ice}{ice}{ice}   {bear}
{ice}  {ice}     {ice}     {bear}
{ice}{ice}{ice}     {ice}     {bear}
{ice}         {ice}     {bear}
{ice}         {ice}     {bear}

Pardon-The-Interruption for human-in-the-loop reasoning.
Type 'exit' or 'quit' to return to the main shell.
        """)

def enter_pti_mode(command: str, **kwargs):
    state: ShellState = kwargs.get('shell_state')
    command_history: CommandHistory = kwargs.get('command_history')

    if not state or not command_history:
        return {"output": "Error: PTI mode requires shell state and history.", "messages": kwargs.get('messages', [])}

    all_command_parts = shlex.split(command)
    parsed_args_list = all_command_parts[1:]
    
    parser = argparse.ArgumentParser(prog="/pti", description="Enter PTI mode for human-in-the-loop reasoning.")
    parser.add_argument('initial_prompt', nargs='*', help="Initial prompt to start the session.")
    parser.add_argument("-f", "--files", nargs="*", default=[], help="Files to load into context.")
    
    try:
        args = parser.parse_args(parsed_args_list)
    except SystemExit:
         return {"output": "Invalid arguments for /pti. Usage: /pti [initial prompt] [-f file1 file2 ...]", "messages": state.messages}

    print_pti_welcome_message()

    frederic_path = get_npc_path("frederic", command_history.db_path)
    state.npc = NPC(file=frederic_path)
    print(colored("Defaulting to NPC: frederic", "cyan"))
    state.npc = NPC(name="frederic")

    pti_messages = list(state.messages)
    loaded_content = {}
    
    if args.files:
        for file_path in args.files:
            try:
                content_chunks = load_file_contents(file_path)
                loaded_content[file_path] = "\n".join(content_chunks)
                print(colored(f"Successfully loaded content from: {file_path}", "green"))
            except Exception as e:
                print(colored(f"Error loading {file_path}: {e}", "red"))
    
    user_input = " ".join(args.initial_prompt)

    while True:
        try:
            if not user_input:
                npc_name = state.npc.name if state.npc and isinstance(state.npc, NPC) else "frederic"
                model_name = state.reasoning_model
                
                prompt_str = f"{colored(os.path.basename(state.current_path), 'blue')}:{npc_name}:{model_name}{bear}> "
                prompt = readline_safe_prompt(prompt_str)
                user_input = get_multiline_input(prompt).strip()

            if user_input.lower() in ["exit", "quit", "done"]:
                break
            
            if not user_input:
                continue
            
            prompt_for_llm = user_input
            if loaded_content:
                context_str = "\n".join([f"--- Content from {fname} ---\n{content}" for fname, content in loaded_content.items()])
                prompt_for_llm += f"\n\nUse the following context to inform your answer:\n{context_str}"
            
            prompt_for_llm += "\n\nThink step-by-step using <think> tags. When you need more information from me, enclose your question in <request_for_input> tags."

            save_conversation_message(
                command_history,
                state.conversation_id,
                "user",
                user_input,
                wd=state.current_path,
                model=state.reasoning_model,
                provider=state.reasoning_provider,
                npc=state.npc.name if isinstance(state.npc, NPC) else None,
            )
            pti_messages.append({"role": "user", "content": user_input})

            try:
                response_dict = get_llm_response(
                    prompt=prompt_for_llm,
                    model=state.reasoning_model,
                    provider=state.reasoning_provider,
                    messages=pti_messages,
                    stream=True,
                    npc=state.npc
                )
                stream = response_dict.get('response')
                
                response_chunks = []
                request_found = False
                
                for chunk in stream:
                    chunk_content = ""
                    if state.reasoning_provider == "ollama":
                        chunk_content = chunk.get("message", {}).get("content", "")
                    else:
                        chunk_content = "".join(
                            choice.delta.content
                            for choice in chunk.choices
                            if choice.delta.content is not None
                        )

                    print(chunk_content, end='')
                    sys.stdout.flush()
                    response_chunks.append(chunk_content)
                    
                    combined_text = "".join(response_chunks)
                    if "</request_for_input>" in combined_text:
                        request_found = True
                        break

                full_response_text = "".join(response_chunks)

                save_conversation_message(
                    command_history,
                    state.conversation_id,
                    "assistant",
                    full_response_text,
                    wd=state.current_path,
                    model=state.reasoning_model,
                    provider=state.reasoning_provider,
                    npc=state.npc.name if isinstance(state.npc, NPC) else None,
                )
                pti_messages.append({"role": "assistant", "content": full_response_text})
                
                print() 
                user_input = None 
                continue

            except KeyboardInterrupt:
                print(colored("\n\n--- Stream Interrupted ---", "yellow"))
                interrupt_text = input('🐻‍❄️> ').strip()
                if interrupt_text:
                    user_input = interrupt_text
                else:
                    user_input = None
                continue

        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print("\nExiting PTI Mode.")
            break

    render_markdown("\n# Exiting PTI Mode")
    return {"output": "", "messages": pti_messages}

def main():
    parser = argparse.ArgumentParser(description="PTI - Pardon-The-Interruption human-in-the-loop shell.")
    parser.add_argument('initial_prompt', nargs='*', help="Initial prompt to start the session.")
    parser.add_argument("-f", "--files", nargs="*", default=[], help="Files to load into context.")
    args = parser.parse_args()

    command_history, team, default_npc = setup_shell()

    from npcsh._state import initial_state
    initial_shell_state = initial_state
    initial_shell_state.team = team
    initial_shell_state.npc = default_npc
    
    fake_command_str = "/pti " + " ".join(args.initial_prompt)
    if args.files:
        fake_command_str += " --files " + " ".join(args.files)
        
    kwargs = {
        'command': fake_command_str,
        'shell_state': initial_shell_state,
        'command_history': command_history
    }
    
    enter_pti_mode(**kwargs)

if __name__ == "__main__":
    main()