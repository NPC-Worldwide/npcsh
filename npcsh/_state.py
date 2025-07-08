from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    NPCSH_STREAM_OUTPUT,
    NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER,
    NPCSH_VISION_MODEL, NPCSH_VISION_PROVIDER,
    NPCSH_EMBEDDING_MODEL, NPCSH_EMBEDDING_PROVIDER,
    NPCSH_REASONING_MODEL, NPCSH_REASONING_PROVIDER,
    NPCSH_IMAGE_GEN_MODEL, NPCSH_IMAGE_GEN_PROVIDER,
    NPCSH_VIDEO_GEN_MODEL, NPCSH_VIDEO_GEN_PROVIDER,
    NPCSH_API_URL,
    NPCSH_DEFAULT_MODE, 

) 
from npcpy.memory.command_history import (
    start_new_conversation,
)
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from npcpy.npc_compiler import NPC, Team
import os
@dataclass
class ShellState:
    npc: Optional[Union[NPC, str]] = None
    team: Optional[Team] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    mcp_client: Optional[Any] = None
    conversation_id: Optional[int] = None
    chat_model: str = NPCSH_CHAT_MODEL
    chat_provider: str = NPCSH_CHAT_PROVIDER
    vision_model: str = NPCSH_VISION_MODEL
    vision_provider: str = NPCSH_VISION_PROVIDER
    embedding_model: str = NPCSH_EMBEDDING_MODEL
    embedding_provider: str = NPCSH_EMBEDDING_PROVIDER
    reasoning_model: str = NPCSH_REASONING_MODEL
    reasoning_provider: str = NPCSH_REASONING_PROVIDER
    image_gen_model: str = NPCSH_IMAGE_GEN_MODEL
    image_gen_provider: str = NPCSH_IMAGE_GEN_PROVIDER
    video_gen_model: str = NPCSH_VIDEO_GEN_MODEL
    video_gen_provider: str = NPCSH_VIDEO_GEN_PROVIDER
    current_mode: str = NPCSH_DEFAULT_MODE
    api_key: Optional[str] = None
    api_url: Optional[str] = NPCSH_API_URL
    current_path: str = field(default_factory=os.getcwd)
    stream_output: bool = NPCSH_STREAM_OUTPUT
    attachments: Optional[List[Any]] = None
    def get_model_for_command(self, model_type: str = "chat"):
        if model_type == "chat":
            return self.chat_model, self.chat_provider
        elif model_type == "vision":
            return self.vision_model, self.vision_provider
        elif model_type == "embedding":
            return self.embedding_model, self.embedding_provider
        elif model_type == "reasoning":
            return self.reasoning_model, self.reasoning_provider
        elif model_type == "image_gen":
            return self.image_gen_model, self.image_gen_provider
        elif model_type == "video_gen":
            return self.video_gen_model, self.video_gen_provider
        else:
            return self.chat_model, self.chat_provider # Default fallback
initial_state = ShellState(
    conversation_id=start_new_conversation(),
    stream_output=NPCSH_STREAM_OUTPUT,
    current_mode=NPCSH_DEFAULT_MODE,
    chat_model=NPCSH_CHAT_MODEL,
    chat_provider=NPCSH_CHAT_PROVIDER,
    vision_model=NPCSH_VISION_MODEL, 
    vision_provider=NPCSH_VISION_PROVIDER,
    embedding_model=NPCSH_EMBEDDING_MODEL, 
    embedding_provider=NPCSH_EMBEDDING_PROVIDER,
    reasoning_model=NPCSH_REASONING_MODEL, 
    reasoning_provider=NPCSH_REASONING_PROVIDER,
    image_gen_model=NPCSH_IMAGE_GEN_MODEL, 
    image_gen_provider=NPCSH_IMAGE_GEN_PROVIDER,
    video_gen_model=NPCSH_VIDEO_GEN_MODEL,
    video_gen_provider=NPCSH_VIDEO_GEN_PROVIDER,
    api_url=NPCSH_API_URL,
)
