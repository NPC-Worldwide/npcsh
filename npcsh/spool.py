from npcpy.memory.command_history import CommandHistory, start_new_conversation, save_conversation_message
from npcpy.data.load import load_file_contents
from npcpy.data.image import capture_screenshot
from npcpy.data.text import rag_search

import os
from npcpy.npc_sysenv import (    
    print_and_process_stream_with_markdown,
)
from npcpy.npc_sysenv import (
        get_system_message, 
        render_markdown,

)
from npcsh._state import    (
    orange, 
    NPCSH_VISION_MODEL, 
    NPCSH_VISION_PROVIDER, 
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_STREAM_OUTPUT
)
from npcpy.llm_funcs import (get_llm_response,)

from npcpy.npc_compiler import NPC
from typing import Any, List, Dict, Union
from npcsh.yap import enter_yap_mode


def enter_spool_mode(
    npc: NPC = None,    
    team = None,
    model: str = None, 
    provider: str = None,
    vision_model:str = None,
    vision_provider:str = None,
    attachments: List[str] = None,
    rag_similarity_threshold: float = 0.3,
    messages: List[Dict] = None,
    conversation_id: str = None,
    stream: bool = NPCSH_STREAM_OUTPUT,
    **kwargs,
) -> Dict:
    
    session_model = model or (npc.model if npc else NPCSH_CHAT_MODEL)
    session_provider = provider or (npc.provider if npc else NPCSH_CHAT_PROVIDER)
    session_vision_model = vision_model or NPCSH_VISION_MODEL
    session_vision_provider = vision_provider or NPCSH_VISION_PROVIDER

    npc_info = f" (NPC: {npc.name})" if npc else ""
    print(f"Entering spool mode{npc_info}. Type '/sq' to exit spool mode.")
    print("💡 Tip: Press Ctrl+C during streaming to interrupt and continue with a new message.")

    spool_context = messages.copy() if messages else []
    loaded_chunks = {}

    if not conversation_id:
        conversation_id = start_new_conversation()

    command_history = CommandHistory()
    
    files_to_load = attachments
    if files_to_load:
        if isinstance(files_to_load, str):
            files_to_load = [f.strip() for f in files_to_load.split(',')]
        
        for file_path in files_to_load:
            file_path = os.path.expanduser(file_path)
            if not os.path.exists(file_path):
                print(f"Error: File not found at {file_path}")
                continue
            try:
                chunks = load_file_contents(file_path)
                loaded_chunks[file_path] = chunks
                print(f"Loaded {len(chunks)} chunks from: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if not spool_context or spool_context[0].get("role") != "system":
        spool_context.insert(0, {"role": "system", "content": system_message})

    if loaded_chunks:
        initial_file_context = "\n\n--- The user has loaded the following files for this session ---\n"
        for filename, chunks in loaded_chunks.items():
            initial_file_context += f"\n\n--- Start of content from {filename} ---\n"
            initial_file_context += "\n".join(chunks)
            initial_file_context += f"\n--- End of content from {filename} ---\n"

    def _handle_llm_interaction(
        prompt, 
        current_context, 
        model_to_use, 
        provider_to_use, 
        images_to_use=None
    ):
        
        current_context.append({"role": "user", "content": prompt})

        save_conversation_message(
            command_history, 
            conversation_id, 
            "user", 
            prompt,
            wd=os.getcwd(), 
            model=model_to_use, 
            provider=provider_to_use,
            npc=npc.name if npc else None, 
            team=team.name if team else None,
        )
        
        assistant_reply = ""
        
        try:
            response = get_llm_response(
                prompt,
                model=model_to_use, 
                provider=provider_to_use,
                messages=current_context, 
                images=images_to_use, 
                stream=stream, 
                npc=npc
            )
            assistant_reply = response.get('response')

            if stream:
                print(orange(f'{npc.name if npc else "🧵"}....> '), end='', flush=True)
                
                # The streaming function now handles KeyboardInterrupt internally
                assistant_reply = print_and_process_stream_with_markdown(
                    assistant_reply, 
                    model=model_to_use, 
                    provider=provider_to_use
                )
            else:
                render_markdown(assistant_reply)
        
        except Exception as e:
            assistant_reply = f"[Error during response generation: {str(e)}]"
            print(f"\n❌ Error: {str(e)}")
        
        current_context.append({"role": "assistant", "content": assistant_reply})
        
        if assistant_reply and assistant_reply.count("```") % 2 != 0:
            assistant_reply += "```"

        save_conversation_message(
            command_history, 
            conversation_id, 
            "assistant", 
            assistant_reply,
            wd=os.getcwd(), 
            model=model_to_use, 
            provider=provider_to_use,
            npc=npc.name if npc else None, 
            team=team.name if team else None,
        )
        
        return current_context

    while True:
        try:
            prompt_text = orange(f"🧵:{npc.name if npc else 'chat'}:{session_model}> ")
            user_input = input(prompt_text).strip()

            if not user_input:
                continue
            if user_input.lower() == "/sq":
                print("Exiting spool mode.")
                break
            if user_input.lower() == "/yap":
                spool_context = enter_yap_mode(spool_context, npc)
                continue

            if user_input.startswith("/ots"):
                command_parts = user_input.split()
                image_paths = []
                
                if len(command_parts) > 1:
                    for img_path in command_parts[1:]:
                        full_path = os.path.expanduser(img_path)
                        if os.path.exists(full_path): image_paths.append(full_path)
                        else: print(f"Error: Image file not found at {full_path}")
                else:
                    screenshot = capture_screenshot()
                    if screenshot and "file_path" in screenshot:
                        image_paths.append(screenshot["file_path"])
                        print(f"Screenshot captured: {screenshot['filename']}")
                
                if not image_paths: continue
                
                vision_prompt = input("Prompt for image(s) (or press Enter): ").strip() or "Describe these images."
                spool_context = _handle_llm_interaction(
                    vision_prompt, 
                    spool_context, 
                    session_vision_model, 
                    session_vision_provider, 
                    images_to_use=image_paths
                )
                continue
            
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
                print(f'prepped  context_content : {context_content}')
            
            spool_context = _handle_llm_interaction(
                current_prompt, 
                spool_context, 
                session_model, 
                session_provider
            )

        except (EOFError,):
            print("\nExiting spool mode.")
            break
        except KeyboardInterrupt:
            # This handles Ctrl+C at the input prompt (not during streaming)
            print("\n🔄 Use '/sq' to exit or continue with a new message.")
            continue

    return {"messages": spool_context, "output": "Exited spool mode."}


def main():
    import argparse    
    parser = argparse.ArgumentParser(description="Enter spool mode for chatting with an LLM")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--provider", help="Provider to use")
    parser.add_argument("--attachments", nargs="*", help="Files to load into context")
    parser.add_argument("--stream", default="true", help="Use streaming mode")
    parser.add_argument("--npc", type=str, default=os.path.expanduser('~/.npcsh/npc_team/sibiji.npc'), help="Path to NPC file")
    
    args = parser.parse_args()
    
    npc = NPC(file=args.npc) if os.path.exists(os.path.expanduser(args.npc)) else None

    enter_spool_mode(
        npc=npc,
        model=args.model,
        provider=args.provider,
        attachments=args.attachments,
        stream=args.stream.lower() == "true",
    )

if __name__ == "__main__":
    main()