from npcpy.memory.command_history import CommandHistory, start_new_conversation, save_conversation_message
from npcpy.data.load import load_pdf, load_csv, load_json, load_excel, load_txt
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
    npc = None,    
    team = None,
    model: str = NPCSH_CHAT_MODEL, 
    provider: str =  NPCSH_CHAT_PROVIDER,
    vision_model:str = NPCSH_VISION_MODEL,
    vision_provider:str = NPCSH_VISION_PROVIDER,
    files: List[str] = None,
    rag_similarity_threshold: float = 0.3,
    messages: List[Dict] = None,
    conversation_id: str = None,
    stream: bool = NPCSH_STREAM_OUTPUT,
) -> Dict:
    """
    Function Description:
        This function is used to enter the spool mode where files can be loaded into memory.
    Args:

        npc : Any : The NPC object.
        files : List[str] : List of file paths to load into the context.
    Returns:
        Dict : The messages and output.
    """

    npc_info = f" (NPC: {npc.name})" if npc else ""
    print(f"Entering spool mode{npc_info}. Type '/sq' to exit spool mode.")

    spool_context = (
        messages.copy() if messages else []
    )  # Initialize context with messages

    loaded_content = {}  # New dictionary to hold loaded content

    # Create conversation ID if not provided
    if not conversation_id:
        conversation_id = start_new_conversation()

    command_history = CommandHistory()
    # Load specified files if any
    if files:
        for file in files:
            extension = os.path.splitext(file)[1].lower()
            try:
                if extension == ".pdf":
                    content = load_pdf(file)["texts"].iloc[0]
                elif extension == ".csv":
                    content = load_csv(file)
                else:
                    print(f"Unsupported file type: {file}")
                    continue
                loaded_content[file] = content
                print(f"Loaded content from: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    # Add system message to context
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if len(spool_context) > 0:
        if spool_context[0]["role"] != "system":
            spool_context.insert(0, {"role": "system", "content": system_message})
    else:
        spool_context.append({"role": "system", "content": system_message})
    # Inherit last n messages if specified
    if npc is not None:
        if model is None:
            model = npc.model
        if provider is None:
            provider = npc.provider

    while True:
        kwargs_to_pass = {}
        if npc:
            kwargs_to_pass["npc"] = npc

        try:
            
            user_input = input("spool:in> ").strip()
            if len(user_input) == 0:
                continue
            if user_input.lower() == "/sq":
                print("Exiting spool mode.")
                break

            if user_input.lower() == "/whisper":  # Check for whisper command
                messages = enter_yap_mode(spool_context, npc)
                continue

            if user_input.startswith("/ots"):
                command_parts = user_input.split()
                image_paths = []
                print('using vision model: ', vision_model)
                
                # Handle image loading/capturing
                if len(command_parts) > 1:
                    # User provided image path(s)
                    for img_path in command_parts[1:]:
                        full_path = os.path.join(os.getcwd(), img_path)
                        if os.path.exists(full_path):
                            image_paths.append(full_path)
                        else:
                            print(f"Error: Image file not found at {full_path}")
                else:
                    # Capture screenshot
                    output = capture_screenshot(npc=npc)
                    if output and "file_path" in output:
                        image_paths.append(output["file_path"])
                        print(f"Screenshot captured: {output['filename']}")
                
                if not image_paths:
                    print("No valid images provided.")
                    continue
                
                # Get user prompt about the image(s)
                user_prompt = input(
                    "Enter a prompt for the LLM about these images (or press Enter to skip): "
                )
                if not user_prompt:
                    user_prompt = "Please analyze these images."
                
                model= vision_model
                provider= vision_provider
                # Save the user message
                message_id = save_conversation_message(
                    command_history,
                    conversation_id,
                    "user",
                    user_prompt,
                    wd=os.getcwd(),
                    model=vision_model,
                    provider=vision_provider,
                    npc=npc.name if npc else None,
                    team=team.name if team else None,
                    
                )
                
                # Process the request with our unified approach
                response = get_llm_response(
                    user_prompt, 
                    model=vision_model,
                    provider=provider,
                    messages=spool_context,
                    images=image_paths,
                    stream=stream, 
                    **kwargs_to_pass
                )
                
                # Extract the assistant's response
                assistant_reply = response['response']
                
                spool_context = response['messages']
                
                if stream:
                    print(orange(f'spool:{npc.name}:{vision_model}>'), end='', flush=True)
                    
                    assistant_reply = print_and_process_stream_with_markdown(assistant_reply, model=model, provider=provider)
                
                    spool_context.append({"role": "assistant", "content": assistant_reply})
                if assistant_reply.count("```") % 2 != 0:
                    assistant_reply = assistant_reply + "```"
                # Save the assistant's response
                save_conversation_message(
                    command_history,
                    conversation_id,
                    "assistant",
                    assistant_reply,
                    wd=os.getcwd(),
                    model=vision_model,
                    provider=vision_provider,
                    npc=npc.name if npc else None,
                    team=team.name if team else None,
                    

                )
                                

                
                # Display the response
                if not stream:
                    render_markdown(assistant_reply)
                
                continue


            
            # Handle RAG context
            if loaded_content:
                context_content = ""
                for filename, content in loaded_content.items():
                    retrieved_docs = rag_search(
                        user_input,
                        content,
                        similarity_threshold=rag_similarity_threshold,
                    )
                    if retrieved_docs:
                        context_content += (
                            f"\n\nLoaded content from: {filename}\n{content}\n\n"
                        )
                if len(context_content) > 0:
                    user_input += f"""
                    Here is the loaded content that may be relevant to your query:
                        {context_content}
                    Please reference it explicitly in your response and use it for answering.
                    """

            # Save user message
            message_id = save_conversation_message(
                command_history,
                conversation_id,
                "user",
                user_input,
                wd=os.getcwd(),
                model=model,
                provider=provider,
                npc=npc.name if npc else None,
                team=team.name if team else None,
                
            )
            
            response = get_llm_response(
                user_input, 
                model=model, 
                provider=provider,
                messages=spool_context, 
                stream=stream,
                **kwargs_to_pass
            )

            assistant_reply, spool_context = response['response'], response['messages']
            if stream:
                print(orange(f'{npc.name if npc else "spool"}:{npc.model if npc else model}>'), end='', flush=True)
                assistant_reply = print_and_process_stream_with_markdown(assistant_reply, model=model, provider=provider)
            # Save assistant message
            save_conversation_message(
                command_history,
                conversation_id,
                "assistant",
                assistant_reply,
                wd=os.getcwd(),
                model=model,
                provider=provider,
                npc=npc.name if npc else None,
                team=team.name if team else None,
                
            )

            # Fix unfinished markdown notation
            if assistant_reply.count("```") % 2 != 0:
                assistant_reply = assistant_reply + "```"

            if not stream:
                render_markdown(assistant_reply)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting spool mode.")
            break

    return {
        "messages": spool_context,
        "output": "\n".join(
            [msg["content"] for msg in spool_context if msg["role"] == "assistant"]
        ),
    }
def main():
    # Example usage
    import argparse    
    parser = argparse.ArgumentParser(description="Enter spool mode for chatting with an LLM")
    parser.add_argument("--model", default=NPCSH_CHAT_MODEL, help="Model to use")
    parser.add_argument("--provider", default=NPCSH_CHAT_PROVIDER, help="Provider to use")
    parser.add_argument("--files", nargs="*", help="Files to load into context")
    parser.add_argument("--stream", default="true", help="Use streaming mode")
    parser.add_argument("--npc", type=str, default=os.path.expanduser('~/.npcsh/npc_team/sibiji.npc'), help="Path to NPC file")
    
    
    args = parser.parse_args()
    
    npc = NPC(file=args.npc)
    print('npc: ', args.npc)
    print(args.stream)
    # Enter spool mode
    enter_spool_mode(
        npc=npc,
        model=args.model,
        provider=args.provider,
        files=args.files,
        stream= args.stream.lower() == "true",
    )

if __name__ == "__main__":
    main()