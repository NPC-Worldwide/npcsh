from npcpy.memory.command_history import CommandHistory, start_new_conversation, save_conversation_message
from npcpy.data.load import load_file_contents
from npcpy.data.image import capture_screenshot
from npcpy.data.text import rag_search

import os
import sys
from npcpy.npc_sysenv import (    
    print_and_process_stream_with_markdown,
    get_system_message, 
    render_markdown,
)
from npcsh._state import (
    orange, 
    ShellState,
    execute_command,
    get_multiline_input,
    readline_safe_prompt,
    setup_shell,
    get_npc_path,
    process_result,
    initial_state,
)
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_compiler import NPC
from typing import Any, List, Dict, Union
from npcsh.yap import enter_yap_mode
from termcolor import colored
def print_spool_ascii():
    spool_art = """
 ██████╗██████╗  ████████╗   ████████╗  ██╗     
██╔════╝██╔══██╗██╔🧵🧵🧵██ ██╔🧵🧵🧵██ ██║     
╚█████╗ ██████╔╝██║🧵🔴🧵██ ██║🧵🔴🧵██ ██║     
 ╚═══██╗██╔═══╝ ██║🧵🧵🧵██ ██║🧵🧵🧵██ ██║     
██████╔╝██║     ██╚══════██ ██ ══════██ ██║
╚═════╝ ╚═╝     ╚═████████   ███████═╝ █████████╗
"""
    print(spool_art)
def enter_spool_mode(
    npc: NPC = None,    
    team = None,
    model: str = None, 
    provider: str = None,
    vmodel: str = None,
    vprovider: str = None,
    attachments: List[str] = None,
    rag_similarity_threshold: float = 0.3,
    messages: List[Dict] = None,
    conversation_id: str = None,
    stream: bool = None,
    **kwargs,
) -> Dict:
    print_spool_ascii()
    # Initialize state using existing infrastructure
    command_history, state_team, default_npc = setup_shell()
    
    # Create spool state, inheriting from initial_state
    spool_state = ShellState(
        npc=npc or default_npc,
        team=team or state_team,
        messages=messages.copy() if messages else [],
        conversation_id=conversation_id or start_new_conversation(),
        current_path=os.getcwd(),
        stream_output=stream if stream is not None else initial_state.stream_output,
        attachments=None,
    )
    
    # Override models/providers if specified
    if model:
        spool_state.chat_model = model
    if provider:
        spool_state.chat_provider = provider
    if vmodel:
        spool_state.vision_model = vmodel
    if vprovider:
        spool_state.vision_provider = vprovider

    npc_info = f" (NPC: {spool_state.npc.name})" if spool_state.npc else ""
    print(f"🧵 Entering spool mode{npc_info}. Type '/sq' to exit spool mode.")
    print("💡 Tip: Press Ctrl+C during streaming to interrupt and continue with a new message.")

    # Handle file loading
    loaded_chunks = {}
    if attachments:
        if isinstance(attachments, str):
            attachments = [f.strip() for f in attachments.split(',')]
        
        for file_path in attachments:
            file_path = os.path.expanduser(file_path)
            if not os.path.exists(file_path):
                print(colored(f"Error: File not found at {file_path}", "red"))
                continue
            try:
                chunks = load_file_contents(file_path)
                loaded_chunks[file_path] = chunks
                print(colored(f"Loaded {len(chunks)} chunks from: {file_path}", "green"))
            except Exception as e:
                print(colored(f"Error loading {file_path}: {str(e)}", "red"))

    # Initialize context with system message if needed
    if not spool_state.messages or spool_state.messages[0].get("role") != "system":
        system_message = get_system_message(spool_state.npc) if spool_state.npc else "You are a helpful assistant."
        spool_state.messages.insert(0, {"role": "system", "content": system_message})

    while True:
        try:
            # Use consistent prompt styling with npcsh
            npc_name = spool_state.npc.name if spool_state.npc else "chat"
            display_model = spool_state.npc.model if spool_state.npc and spool_state.npc.model else spool_state.chat_model
            
            prompt_str = f"{orange(npc_name)}:{display_model}🧵> "
            prompt = readline_safe_prompt(prompt_str)
            user_input = get_multiline_input(prompt).strip()

            if not user_input:
                continue
                
            if user_input.lower() == "/sq":
                print("Exiting spool mode.")
                break
                
            if user_input.lower() == "/yap":
                spool_state.messages = enter_yap_mode(spool_state.messages, spool_state.npc)
                continue

            # Handle vision commands
            if user_input.startswith("/ots"):
                command_parts = user_input.split()
                image_paths = []
                
                if len(command_parts) > 1:
                    for img_path in command_parts[1:]:
                        full_path = os.path.expanduser(img_path)
                        if os.path.exists(full_path): 
                            image_paths.append(full_path)
                        else: 
                            print(colored(f"Error: Image file not found at {full_path}", "red"))
                else:
                    screenshot = capture_screenshot()
                    if screenshot and "file_path" in screenshot:
                        image_paths.append(screenshot["file_path"])
                        print(colored(f"Screenshot captured: {screenshot['filename']}", "green"))
                
                if not image_paths: 
                    continue
                
                vision_prompt = input("Prompt for image(s) (or press Enter): ").strip() or "Describe these images."
                
                # Use vision models for image processing
                response = get_llm_response(
                    vision_prompt,
                    model=spool_state.vision_model,
                    provider=spool_state.vision_provider,
                    messages=spool_state.messages,
                    images=image_paths,
                    stream=spool_state.stream_output,
                    npc=spool_state.npc, 
                    **kwargs
                    
                )
                
                spool_state.messages = response.get('messages', spool_state.messages)
                output = response.get('response')
                
                # Process and display the result
                process_result(vision_prompt, spool_state, {'output': output}, command_history)
                continue
            
            # Handle RAG context if files are loaded
            current_prompt = user_input
            if loaded_chunks:
                context_content = ""
                for filename, chunks in loaded_chunks.items():
                    full_content_str = "\n".join(chunks)
                    retrieved_docs = rag_search(
                        user_input,
                        full_content_str,
                        similarity_threshold=rag_similarity_threshold,
                    )
                    if retrieved_docs:
                        context_content += f"\n\nContext from: {filename}\n{retrieved_docs}\n"
                
                if context_content:
                    current_prompt += f"\n\n--- Relevant context from loaded files ---\n{context_content}"
            
            # Use standard LLM processing
            response = get_llm_response(
                current_prompt,
                model=spool_state.npc.model if spool_state.npc and spool_state.npc.model else spool_state.chat_model,
                provider=spool_state.npc.provider if spool_state.npc and spool_state.npc.provider else spool_state.chat_provider,
                messages=spool_state.messages,
                stream=spool_state.stream_output,
                npc=spool_state.npc, 
                **kwargs
            )
            
            spool_state.messages = response.get('messages', spool_state.messages)
            output = response.get('response')
            
            # Use existing result processing
            process_result(current_prompt, spool_state, {'output': output}, command_history)

        except (EOFError,):
            print("\nExiting spool mode.")
            break
        except KeyboardInterrupt:
            print("\n🔄 Use '/sq' to exit or continue with a new message.")
            continue

    return {"messages": spool_state.messages, "output": "Exited spool mode."}


def main():
    import argparse    
    parser = argparse.ArgumentParser(description="Enter spool mode for chatting with an LLM")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--provider", help="Provider to use")
    parser.add_argument("--attachments", nargs="*", help="Files to load into context")
    parser.add_argument("--stream", default="true", help="Use streaming mode")
    parser.add_argument("--npc", type=str, help="NPC name or path to NPC file", default='sibiji',)
    
    args = parser.parse_args()
    
    # Use existing infrastructure to get NPC
    command_history, team, default_npc = setup_shell()
    
    npc = None
    if args.npc:
        if os.path.exists(os.path.expanduser(args.npc)):
            npc = NPC(file=args.npc)
        elif team and args.npc in team.npcs:
            npc = team.npcs[args.npc]
        else:
            try:
                npc_path = get_npc_path(args.npc, command_history.db_path)
                npc = NPC(file=npc_path)
            except ValueError:
                print(colored(f"NPC '{args.npc}' not found. Using default.", "yellow"))
                npc = default_npc
    else:
        npc = default_npc

    enter_spool_mode(
        npc=npc,
        team=team,
        model=args.model,
        provider=args.provider,
        attachments=args.attachments,
        stream=args.stream.lower() == "true",
    )

if __name__ == "__main__":
    main()