# Standard library imports
import atexit
import base64
import os
from dataclasses import dataclass, field
from datetime import datetime
import filecmp
import inspect

import logging

from pathlib import Path
import platform
import re
import select
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import textwrap
import readline
import json
from typing import Dict, List, Any, Tuple, Union, Optional, Callable
import yaml

# Setup logging - INFO by default, DEBUG if NPCSH_DEBUG=1
def _setup_logging():
    level = logging.DEBUG if os.environ.get("NPCSH_DEBUG", "0") == "1" else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(message)s',  # Simple format - just the message
        datefmt='%H:%M:%S'
    )
    # Always show tool calls from llm_funcs at INFO level
    logging.getLogger("npcpy.llm_funcs").setLevel(level)
    logging.getLogger("npcsh.state").setLevel(level)

_setup_logging()

# Platform-specific imports
try:
    import pty
    import tty
    import termios

except ImportError:
    
    pty = None
    tty = None
    termios = None

# Optional dependencies
try:
    import chromadb
except ImportError:
    chromadb = None

try:
    import ollama
except ImportError:
    ollama = None

# Third-party imports
from colorama import Style
from litellm import RateLimitError
import numpy as np
from termcolor import colored

# npcpy imports
from npcpy.data.load import load_file_contents
from npcpy.data.web import search_web

from npcpy.llm_funcs import (
    check_llm_command,
    get_llm_response,
    execute_llm_command,
    breathe,
)
from npcpy.memory.command_history import (
    CommandHistory,
    start_new_conversation,
    save_conversation_message,
    load_kg_from_db,
    save_kg_to_db,
    format_memory_context,
)
from npcpy.memory.knowledge_graph import kg_evolve_incremental
from npcpy.memory.search import execute_rag_command, execute_brainblast_command
from npcpy.npc_compiler import NPC, Team, build_jinx_tool_catalog
from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    render_markdown,
    get_model_and_provider,
    get_locally_available_models,

)
from npcpy.tools import auto_tools
from npcpy.gen.embeddings import get_embeddings

# Local module imports
from .config import (
    DEFAULT_NPC_TEAM_PATH,
    PROJECT_NPC_TEAM_PATH,
    READLINE_HISTORY_FILE,
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_DB_PATH,
    NPCSH_VECTOR_DB_PATH,
    NPCSH_DEFAULT_MODE,
    NPCSH_VISION_MODEL,
    NPCSH_VISION_PROVIDER,
    NPCSH_IMAGE_GEN_MODEL,
    NPCSH_IMAGE_GEN_PROVIDER,
    NPCSH_VIDEO_GEN_MODEL,
    NPCSH_VIDEO_GEN_PROVIDER,
    NPCSH_EMBEDDING_MODEL,
    NPCSH_EMBEDDING_PROVIDER,
    NPCSH_REASONING_MODEL,
    NPCSH_REASONING_PROVIDER,
    NPCSH_STREAM_OUTPUT,
    NPCSH_API_URL,
    NPCSH_SEARCH_PROVIDER,
    NPCSH_BUILD_KG,
    NPCSH_EDIT_APPROVAL,
    setup_npcsh_config,
    is_npcsh_initialized,
    set_npcsh_initialized,
    set_npcsh_config_value,
)
from .ui import SpinnerContext, orange, get_file_color, format_file_listing, wrap_text
from .parsing import split_by_pipes, parse_command_safely, parse_generic_command_flags
from .execution import (
    TERMINAL_EDITORS,
    INTERACTIVE_COMMANDS as interactive_commands,
    validate_bash_command,
    handle_bash_command,
    handle_cd_command,
    handle_interactive_command,
    open_terminal_editor,
    list_directory,
)
from .completion import setup_readline, save_readline_history, make_completer, get_slash_commands



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
    search_provider: str = NPCSH_SEARCH_PROVIDER
    image_gen_model: str = NPCSH_IMAGE_GEN_MODEL
    image_gen_provider: str = NPCSH_IMAGE_GEN_PROVIDER
    video_gen_model: str = NPCSH_VIDEO_GEN_MODEL
    video_gen_provider: str = NPCSH_VIDEO_GEN_PROVIDER
    current_mode: str = NPCSH_DEFAULT_MODE
    build_kg: bool = NPCSH_BUILD_KG
    kg_link_facts: bool = False      # Link facts to concepts (requires LLM calls)
    kg_link_concepts: bool = False   # Link concepts to concepts (requires LLM calls)
    kg_link_facts_facts: bool = False  # Link facts to facts (requires LLM calls)
    api_key: Optional[str] = None
    api_url: Optional[str] = NPCSH_API_URL
    current_path: str = field(default_factory=os.getcwd)
    stream_output: bool = NPCSH_STREAM_OUTPUT
    attachments: Optional[List[Any]] = None
    turn_count: int = 0
    # Token usage tracking
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    session_cost_usd: float = 0.0
    # Session timing
    session_start_time: float = field(default_factory=lambda: __import__('time').time())
    # Logging level: "silent", "normal", "verbose"
    log_level: str = "normal"
    # Edit approval mode: "off", "interactive", "auto"
    edit_approval: str = NPCSH_EDIT_APPROVAL
    # Pending file edits for approval
    pending_edits: Dict[str, Dict[str, str]] = field(default_factory=dict)

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
            return self.chat_model, self.chat_provider

    def set_log_level(self, level: str) -> str:
        """Set the logging level and configure npcpy loggers accordingly."""
        level = level.lower()
        if level not in ("silent", "normal", "verbose"):
            return f"Invalid log level: {level}. Use 'silent', 'normal', or 'verbose'."

        self.log_level = level

        # Map to Python logging levels
        level_map = {
            "silent": logging.WARNING,
            "normal": logging.INFO,
            "verbose": logging.DEBUG,
        }
        log_level = level_map[level]

        # Configure npcpy loggers
        for logger_name in ["npcpy", "npcpy.gen", "npcpy.gen.response", "npcsh"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)

        # Also set root logger for npcpy
        logging.getLogger("npcpy").setLevel(log_level)

        return f"Log level set to: {level}"

CONFIG_KEY_MAP = {
  
    "model": "NPCSH_CHAT_MODEL",
    "chatmodel": "NPCSH_CHAT_MODEL",
    "provider": "NPCSH_CHAT_PROVIDER",
    "chatprovider": "NPCSH_CHAT_PROVIDER",

  
    "vmodel": "NPCSH_VISION_MODEL",
    "visionmodel": "NPCSH_VISION_MODEL",
    "vprovider": "NPCSH_VISION_PROVIDER",
    "visionprovider": "NPCSH_VISION_PROVIDER",

  
    "emodel": "NPCSH_EMBEDDING_MODEL",
    "embeddingmodel": "NPCSH_EMBEDDING_MODEL",
    "eprovider": "NPCSH_EMBEDDING_PROVIDER",
    "embeddingprovider": "NPCSH_EMBEDDING_PROVIDER",

  
    "rmodel": "NPCSH_REASONING_MODEL",
    "reasoningmodel": "NPCSH_REASONING_MODEL",
    "rprovider": "NPCSH_REASONING_PROVIDER",
    "reasoningprovider": "NPCSH_REASONING_PROVIDER",

  
    "igmodel": "NPCSH_IMAGE_GEN_MODEL",
    "imagegenmodel": "NPCSH_IMAGE_GEN_MODEL",
    "igprovider": "NPCSH_IMAGE_GEN_PROVIDER",
    "imagegenprovider": "NPCSH_IMAGE_GEN_PROVIDER",

  
    "vgmodel": "NPCSH_VIDEO_GEN_MODEL",
    "videogenmodel": "NPCSH_VIDEO_GEN_MODEL",
    "vgprovider": "NPCSH_VIDEO_GEN_PROVIDER",
    "videogenprovider": "NPCSH_VIDEO_GEN_PROVIDER",

  
    "sprovider": "NPCSH_SEARCH_PROVIDER",
    "mode": "NPCSH_DEFAULT_MODE",
    "stream": "NPCSH_STREAM_OUTPUT",
    "apiurl": "NPCSH_API_URL",
    "buildkg": "NPCSH_BUILD_KG",
    "editapproval": "NPCSH_EDIT_APPROVAL",
    "approval": "NPCSH_EDIT_APPROVAL",
}


def set_npcsh_config_value(key: str, value: str):
    """
    Set NPCSH config values at runtime using shorthand (case-insensitive) or full keys.
    Updates os.environ, globals, and ShellState defaults.
    """
  
    env_key = CONFIG_KEY_MAP.get(key.lower(), key)

  
    os.environ[env_key] = value

  
    if env_key in ["NPCSH_STREAM_OUTPUT", "NPCSH_BUILD_KG"]:
        parsed_val = value.strip().lower() in ["1", "true", "yes"]
    elif env_key.endswith("_PATH"):
        parsed_val = os.path.expanduser(value)
    else:
        parsed_val = value

  
    globals()[env_key] = parsed_val

  
    field_map = {
        "NPCSH_CHAT_MODEL": "chat_model",
        "NPCSH_CHAT_PROVIDER": "chat_provider",
        "NPCSH_VISION_MODEL": "vision_model",
        "NPCSH_VISION_PROVIDER": "vision_provider",
        "NPCSH_EMBEDDING_MODEL": "embedding_model",
        "NPCSH_EMBEDDING_PROVIDER": "embedding_provider",
        "NPCSH_REASONING_MODEL": "reasoning_model",
        "NPCSH_REASONING_PROVIDER": "reasoning_provider",
        "NPCSH_SEARCH_PROVIDER": "search_provider",
        "NPCSH_IMAGE_GEN_MODEL": "image_gen_model",
        "NPCSH_IMAGE_GEN_PROVIDER": "image_gen_provider",
        "NPCSH_VIDEO_GEN_MODEL": "video_gen_model",
        "NPCSH_VIDEO_GEN_PROVIDER": "video_gen_provider",
        "NPCSH_DEFAULT_MODE": "current_mode",
        "NPCSH_BUILD_KG": "build_kg",
        "NPCSH_API_URL": "api_url",
        "NPCSH_STREAM_OUTPUT": "stream_output",
        "NPCSH_EDIT_APPROVAL": "edit_approval",
    }
    if env_key in field_map:
        setattr(ShellState, field_map[env_key], parsed_val)

    # Persist to ~/.npcshrc
    npcshrc_path = os.path.expanduser("~/.npcshrc")
    try:
        existing_lines = []
        if os.path.exists(npcshrc_path):
            with open(npcshrc_path, 'r') as f:
                existing_lines = f.readlines()

        # Update or add the export line
        export_line = f"export {env_key}=\"{value}\"\n"
        found = False
        for i, line in enumerate(existing_lines):
            if line.strip().startswith(f"export {env_key}="):
                existing_lines[i] = export_line
                found = True
                break

        if not found:
            existing_lines.append(export_line)

        with open(npcshrc_path, 'w') as f:
            f.writelines(existing_lines)
    except Exception as e:
        print(f"Warning: Could not persist config to {npcshrc_path}: {e}")


def get_npc_path(npc_name: str, db_path: str) -> str:
    project_npc_team_dir = os.path.abspath("./npc_team")
    project_npc_path = os.path.join(project_npc_team_dir, f"{npc_name}.npc")
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")
    global_npc_path = os.path.join(user_npc_team_dir, f"{npc_name}.npc")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            query = f"SELECT source_path FROM compiled_npcs WHERE name = '{npc_name}'"
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                return result[0]

    except Exception:
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                query = f"SELECT source_path FROM compiled_npcs WHERE name = {npc_name}"
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    return result[0]
        except Exception as e:
            print(f"Database query error: {e}")

  
    if os.path.exists(project_npc_path):
        return project_npc_path

    if os.path.exists(global_npc_path):
        return global_npc_path

    raise ValueError(f"NPC file not found: {npc_name}")

def initialize_base_npcs_if_needed(db_path: str) -> None:
    """
    Function Description:
        This function initializes the base NPCs if they are not already in the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:

        None
    Returns:
        None
    """

    already_initialized = is_npcsh_initialized()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_npcs (
            name TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            compiled_content TEXT
        )
        """
    )

    # Package directories - use helper that handles PyInstaller bundles
    package_dir = get_package_dir()
    package_npc_team_dir = os.path.join(package_dir, "npc_team")

    # Debug logging for package path resolution
    if os.environ.get("NPCSH_DEBUG", "0") == "1":
        print(f"[DEBUG] Package dir: {package_dir}")
        print(f"[DEBUG] Package npc_team dir: {package_npc_team_dir}")
        print(f"[DEBUG] npc_team exists: {os.path.exists(package_npc_team_dir)}")
        if os.path.exists(package_npc_team_dir):
            print(f"[DEBUG] npc_team contents: {os.listdir(package_npc_team_dir)}")

    if not os.path.exists(package_npc_team_dir):
        print(f"Warning: Package npc_team directory not found at {package_npc_team_dir}")
        # For bundled executables, try to find it
        if getattr(sys, 'frozen', False):
            print(f"Running as frozen executable, _MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}")
            if hasattr(sys, '_MEIPASS'):
                print(f"Contents of _MEIPASS: {os.listdir(sys._MEIPASS)}")
        return

    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")

    user_jinxs_dir = os.path.join(user_npc_team_dir, "jinxs")
    user_templates_dir = os.path.join(user_npc_team_dir, "templates")
    os.makedirs(user_npc_team_dir, exist_ok=True)
    os.makedirs(user_jinxs_dir, exist_ok=True)
    os.makedirs(user_templates_dir, exist_ok=True)

    # Copy .npc and .ctx files
    for filename in os.listdir(package_npc_team_dir):
        if filename.endswith(".npc"):
            source_path = os.path.join(package_npc_team_dir, filename)
            destination_path = os.path.join(user_npc_team_dir, filename)
            if not os.path.exists(destination_path) or file_has_changed(
                source_path, destination_path
            ):
                shutil.copy2(source_path, destination_path)
                print(f"Copied NPC {filename} to {destination_path}")
        if filename.endswith(".ctx"):
            source_path = os.path.join(package_npc_team_dir, filename)
            destination_path = os.path.join(user_npc_team_dir, filename)
            if not os.path.exists(destination_path) or file_has_changed(
                source_path, destination_path
            ):
                shutil.copy2(source_path, destination_path)
                print(f"Copied ctx {filename} to {destination_path}")

    # Copy jinxs directory RECURSIVELY with manifest tracking
    # This ensures we only sync package jinxs and can clean up old ones
    package_jinxs_dir = os.path.join(package_npc_team_dir, "jinxs")
    manifest_path = os.path.join(user_jinxs_dir, ".package_manifest.json")

    # Load existing manifest of package-synced jinxs
    old_package_jinxs = set()
    if os.path.exists(manifest_path):
        try:

            with open(manifest_path, 'r') as f:
                old_package_jinxs = set(json.load(f).get('jinxs', []))
        except Exception:
            pass

    # Track current package jinxs
    current_package_jinxs = set()

    if os.path.exists(package_jinxs_dir):
        for root, dirs, files in os.walk(package_jinxs_dir):
            # Calculate relative path from package_jinxs_dir
            rel_path = os.path.relpath(root, package_jinxs_dir)

            # Create corresponding directory in user jinxs
            if rel_path == '.':
                dest_dir = user_jinxs_dir
            else:
                dest_dir = os.path.join(user_jinxs_dir, rel_path)
            os.makedirs(dest_dir, exist_ok=True)

            # Copy all .jinx files in this directory
            for filename in files:
                if filename.endswith(".jinx"):
                    source_jinx_path = os.path.join(root, filename)
                    destination_jinx_path = os.path.join(dest_dir, filename)
                    jinx_rel_path = os.path.join(rel_path, filename) if rel_path != '.' else filename
                    current_package_jinxs.add(jinx_rel_path)

                    if not os.path.exists(destination_jinx_path) or file_has_changed(
                        source_jinx_path, destination_jinx_path
                    ):
                        shutil.copy2(source_jinx_path, destination_jinx_path)
                        print(f"Copied jinx {jinx_rel_path} to {destination_jinx_path}")

    # Clean up old package jinxs that are no longer in the package
    # (but preserve user-created jinxs that were never in the manifest)
    stale_jinxs = old_package_jinxs - current_package_jinxs
    for stale_jinx in stale_jinxs:
        stale_path = os.path.join(user_jinxs_dir, stale_jinx)
        if os.path.exists(stale_path):
            try:
                os.remove(stale_path)
                print(f"Removed stale package jinx: {stale_jinx}")
                # Remove empty parent directories
                parent_dir = os.path.dirname(stale_path)
                while parent_dir != user_jinxs_dir:
                    if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                        print(f"Removed empty directory: {parent_dir}")
                    parent_dir = os.path.dirname(parent_dir)
            except Exception as e:
                print(f"Could not remove stale jinx {stale_jinx}: {e}")

    # Save updated manifest
    try:
        
        with open(manifest_path, 'w') as f:
            json.dump({'jinxs': list(current_package_jinxs), 'updated': str(__import__('datetime').datetime.now())}, f, indent=2)
    except Exception as e:
        print(f"Could not save jinx manifest: {e}")

    # Copy templates directory
    templates = os.path.join(package_npc_team_dir, "templates")
    if os.path.exists(templates):
        for folder in os.listdir(templates):
            os.makedirs(os.path.join(user_templates_dir, folder), exist_ok=True)
            for file in os.listdir(os.path.join(templates, folder)):
                if file.endswith(".npc"):
                    source_template_path = os.path.join(templates, folder, file)

                    destination_template_path = os.path.join(
                        user_templates_dir, folder, file
                    )
                    if not os.path.exists(
                        destination_template_path
                    ) or file_has_changed(
                        source_template_path, destination_template_path
                    ):
                        shutil.copy2(source_template_path, destination_template_path)
                        print(f"Copied template {file} to {destination_template_path}")
    conn.commit()
    conn.close()

    if not already_initialized:
        set_npcsh_initialized()
        add_npcshrc_to_shell_config()


def get_shell_config_file() -> str:
    """

    Function Description:
        This function returns the path to the shell configuration file.
    Args:
        None
    Keyword Args:
        None
    Returns:
        The path to the shell configuration file.
    """
  
    shell = os.environ.get("SHELL", "")

    if "zsh" in shell:
        return os.path.expanduser("~/.zshrc")
    elif "bash" in shell:
      
        if platform.system() == "Darwin":
            return os.path.expanduser("~/.bash_profile")
        else:
            return os.path.expanduser("~/.bashrc")
    else:
      
        return os.path.expanduser("~/.bashrc")


def get_team_ctx_path(team_path: str) -> Optional[str]:
    """Find the first .ctx file in the team directory"""
    team_dir = Path(team_path)
    ctx_files = list(team_dir.glob("*.ctx"))
    return str(ctx_files[0]) if ctx_files else None


from npcpy.memory.memory_processor import  memory_approval_ui
from npcpy.ft.memory_trainer import MemoryTrainer
from npcpy.llm_funcs import get_facts

def get_relevant_memories(
    command_history: CommandHistory,
    npc_name: str,
    team_name: str,
    path: str,
    query: Optional[str] = None,
    max_memories: int = 10,
    state: Optional[ShellState] = None
) -> List[Dict]:
    all_memories = command_history.get_memories_for_scope(
        npc=npc_name,
        team=team_name,
        directory_path=path,
    )
    
    if not all_memories:
        return []
    
    if len(all_memories) <= max_memories and not query:
        return all_memories
    
    if query:
        query_lower = query.lower()
        keyword_matches = [
            m for m in all_memories 
            if query_lower in (m.get('final_memory') or m.get('initial_memory') or '').lower()
        ]
        
        if keyword_matches:
            return keyword_matches[:max_memories]

    if state and state.embedding_model and state.embedding_provider:
        try:
            
            
            search_text = query if query else "recent context"
            query_embedding = get_embeddings(
                [search_text],
                state.embedding_model,
                state.embedding_provider
            )[0]
            
            memory_texts = [
                m.get('final_memory', '') for m in all_memories
            ]
            memory_embeddings = get_embeddings(
                memory_texts,
                state.embedding_model,
                state.embedding_provider
            )
            

            similarities = []
            for mem_emb in memory_embeddings:
                similarity = np.dot(query_embedding, mem_emb) / (
                    np.linalg.norm(query_embedding) * 
                    np.linalg.norm(mem_emb)
                )
                similarities.append(similarity)
            
            sorted_indices = np.argsort(similarities)[::-1]
            return [all_memories[i] for i in sorted_indices[:max_memories]]
            
        except Exception as e:
            print(colored(
                f"RAG search failed, using recent: {e}", 
                "yellow"
            ))
    
    return all_memories[-max_memories:]



def add_npcshrc_to_shell_config() -> None:
    """
    Function Description:
        This function adds the sourcing of the .npcshrc file to the user's shell configuration file.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    if os.getenv("NPCSH_INITIALIZED") is not None:
        return
    config_file = get_shell_config_file()
    npcshrc_line = "\n# Source NPCSH configuration\nif [ -f ~/.npcshrc ]; then\n    . ~/.npcshrc\nfi\n"

    with open(config_file, "a+") as shell_config:
        shell_config.seek(0)
        content = shell_config.read()
        if "source ~/.npcshrc" not in content and ". ~/.npcshrc" not in content:
            shell_config.write(npcshrc_line)
            print(f"Added .npcshrc sourcing to {config_file}")
        else:
            print(f".npcshrc already sourced in {config_file}")

def ensure_npcshrc_exists() -> str:
    """
    Function Description:
        This function ensures that the .npcshrc file exists in the user's home directory.
    Args:
        None
    Keyword Args:
        None
    Returns:
        The path to the .npcshrc file.
    """

    npcshrc_path = os.path.expanduser("~/.npcshrc")
    if not os.path.exists(npcshrc_path):
        with open(npcshrc_path, "w") as npcshrc:
            npcshrc.write("# NPCSH Configuration File\n")
            npcshrc.write("export NPCSH_INITIALIZED=0\n")
            npcshrc.write("export NPCSH_DEFAULT_MODE='agent'\n")
            npcshrc.write("export NPCSH_BUILD_KG=1")
            npcshrc.write("export NPCSH_CHAT_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_CHAT_MODEL='gemma3:4b'\n")
            npcshrc.write("export NPCSH_REASONING_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_REASONING_MODEL='deepseek-r1'\n")
            npcshrc.write("export NPCSH_EMBEDDING_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_EMBEDDING_MODEL='nomic-embed-text'\n")
            npcshrc.write("export NPCSH_VISION_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_VISION_MODEL='llava7b'\n")
            npcshrc.write(
                "export NPCSH_IMAGE_GEN_MODEL='runwayml/stable-diffusion-v1-5'\n"
            )

            npcshrc.write("export NPCSH_IMAGE_GEN_PROVIDER='diffusers'\n")
            npcshrc.write(
                "export NPCSH_VIDEO_GEN_MODEL='runwayml/stable-diffusion-v1-5'\n"
            )

            npcshrc.write("export NPCSH_VIDEO_GEN_PROVIDER='diffusers'\n")

            npcshrc.write("export NPCSH_API_URL=''\n")
            npcshrc.write("export NPCSH_DB_PATH='~/npcsh_history.db'\n")
            npcshrc.write("export NPCSH_VECTOR_DB_PATH='~/npcsh_chroma.db'\n")
            npcshrc.write("export NPCSH_STREAM_OUTPUT=0")
    return npcshrc_path



def setup_npcsh_config() -> None:
    """
    Function Description:
        This function initializes the NPCSH configuration.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    ensure_npcshrc_exists()
    add_npcshrc_to_shell_config()



CANONICAL_ARGS = [
    'model',            
    'provider',         
    'output_file',           
    'attachments',     
    'format',    
    'temperature',
    'top_k',
    'top_p',
    'max_tokens',
    'messages',    
    'npc',
    'team',
    'height',
    'width',
    'num_frames',
    'sprovider',
    'emodel',
    'eprovider',
    'igmodel',
    'igprovider',
    'vmodel',
    'vprovider',
    'rmodel',
    'rprovider',
    'num_npcs',
    'depth',
    'exploration',
    'creativity',
    'port',
    'cors',
    'config_dir',
    'plots_dir',
    'refresh_period',
    'lang',
]

def get_argument_help() -> Dict[str, List[str]]:
    """
    Analyzes CANONICAL_ARGS to generate a map of canonical arguments
    to all their possible shorthands.
    
    Returns -> {'model': ['m', 'mo', 'mod', 'mode'], 'provider': ['p', 'pr', ...]}
    """
    arg_map = {arg: [] for arg in CANONICAL_ARGS}
    
    for arg in CANONICAL_ARGS:
      
        for i in range(1, len(arg)):
            prefix = arg[:i]
            
          
            matches = [canonical for canonical in CANONICAL_ARGS if canonical.startswith(prefix)]
            
          
            if len(matches) == 1 and matches[0] == arg:
                arg_map[arg].append(prefix)

    return arg_map




def normalize_and_expand_flags(parsed_flags: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expands argument aliases based on the priority order of CANONICAL_ARGS.
    The first matching prefix in the list wins.
    """
    normalized = {}
    for key, value in parsed_flags.items():
        if key in CANONICAL_ARGS:
            if key in normalized:
                print(colored(f"Warning: Argument '{key}' specified multiple times. Using last value.", "yellow"))
            normalized[key] = value
            continue
        first_match = next((arg for arg in CANONICAL_ARGS if arg.startswith(key)), None)
        if first_match:
            if first_match in normalized:
                print(colored(f"Warning: Argument '{first_match}' specified multiple times (via alias '{key}'). Using last value.", "yellow"))
            normalized[first_match] = value
        else:
            normalized[key] = value
    return normalized


BASH_COMMANDS = [
    "npc",
    "npm",
    "npx",
    "open",
    "alias",
    "bg",
    "bind",
    "break",
    "builtin",
    "case",
    "command",
    "compgen",
    "complete",
    "declare",
    "dirs",
    "disown",
    "echo",
    "enable",
    "eval",
    "exec",
    "exit",
    "export",
    "fc",
    "fg",
    "getopts",
    "hash",
    "history",
    "if",
    "jobs",
    "kill",
    "let",
    "local",
    "logout",
    "ollama",
    "popd",
    "printf",
    "pushd",
    "pwd",
    "read",
    "readonly",
    "return",
    "set",
    "shift",
    "shopt",
    "source",
    "suspend",
    "test",
    "times",
    "trap",
    "type",
    "typeset",
    "ulimit",
    "umask",
    "unalias",
    "unset",
    "until",
    "wait",
    "while",
  
    "ls",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "rmdir",
    "touch",
    "cat",
    "less",
    "more",
    "head",
    "tail",
    "grep",
    "find",
    "sed",
    "awk",
    "sort",
    "uniq",
    "wc",
    "diff",
    "chmod",
    "chown",
    "chgrp",
    "ln",
    "tar",
    "gzip",
    "gunzip",
    "zip",
    "unzip",
    "ssh",
    "scp",
    "rsync",
    "wget",
    "curl",
    "ping",
    "netstat",
    "ifconfig",
    "route",
    "traceroute",
    "ps",
    "top",
    "htop",
    "kill",
    "killall",
    "su",
    "sudo",
    "whoami",
    "who",
    "last",
    "finger",
    "uptime",
    "free",
    "df",
    "du",
    "mount",
    "umount",
    "fdisk",
    "mkfs",
    "fsck",
    "dd",
    "cron",
    "at",
    "systemctl",
    "service",
    "journalctl",
    "man",
    "info",
    "whatis",
    "whereis",
    "date",
    "cal",
    "bc",
    "expr",
    "screen",
    "tmux",
    "git",
    "vim",
    "emacs",
    "nano",
    "pip",
]


# interactive_commands imported from .execution


def start_interactive_session(command: str) -> int:
    """
    Starts an interactive session. Only works on Unix. On Windows, print a message and return 1.
    """
    ON_WINDOWS = platform.system().lower().startswith("win")
    if ON_WINDOWS or termios is None or tty is None or pty is None or select is None or signal is None or tty is None:
        print("Interactive terminal sessions are not supported on Windows.")
        return 1
  
    old_tty = termios.tcgetattr(sys.stdin)
    try:
      
        master_fd, slave_fd = pty.openpty()

      
        p = subprocess.Popen(
            command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            shell=True,
            preexec_fn=os.setsid,
        )

      
        tty.setraw(sys.stdin.fileno())

        def handle_timeout(signum, frame):
            raise TimeoutError("Process did not terminate in time")

        while p.poll() is None:
            r, w, e = select.select([sys.stdin, master_fd], [], [], 0.1)
            if sys.stdin in r:
                d = os.read(sys.stdin.fileno(), 10240)
                os.write(master_fd, d)
            elif master_fd in r:
                o = os.read(master_fd, 10240)
                if o:
                    os.write(sys.stdout.fileno(), o)
                else:
                    break

      
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(5)
        try:
            p.wait()
        except TimeoutError:
            print("\nProcess did not terminate. Force killing...")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            time.sleep(1)
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        finally:
            signal.alarm(0)

    finally:
      
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_tty)

    return p.returncode

def validate_bash_command(command_parts: list) -> bool:
    """
    Function Description:
        Validate if the command sequence is a valid bash command with proper arguments/flags.
        Simplified to be less strict and allow bash to handle argument specifics for common commands.
    Args:
        command_parts : list : Command parts
    Keyword Args:
        None
    Returns:
        bool : bool : Boolean
    """
    if not command_parts:
        return False

    base_command = command_parts[0]

    # Commands that are always considered valid for direct execution
    ALWAYS_VALID_COMMANDS = BASH_COMMANDS + list(interactive_commands.keys()) + TERMINAL_EDITORS

    if base_command in ALWAYS_VALID_COMMANDS:
        return True
    
    # Specific checks for commands that might be misinterpreted or need special handling
    if base_command == 'which':
        return True # 'which' is a valid bash command

    # If it's not in our explicit list, it's not a bash command we want to validate strictly
    return False # If it reaches here, it's not a recognized bash command for strict validation.

def is_npcsh_initialized() -> bool:
    """
    Function Description:
        This function checks if the NPCSH initialization flag is set.
    Args:
        None
    Keyword Args:
        None
    Returns:
        A boolean indicating whether NPCSH is initialized.
    """

    return os.environ.get("NPCSH_INITIALIZED", None) == "1"


def execute_set_command(command: str, value: str) -> str:
    """
    Function Description:
        This function sets a configuration value in the .npcshrc file.
    Args:
        command: The command to execute.
        value: The value to set.
    Keyword Args:
        None
    Returns:
        A message indicating the success or failure of the operation.
    """

    config_path = os.path.expanduser("~/.npcshrc")

  
    var_map = {
        "model": "NPCSH_CHAT_MODEL",
        "provider": "NPCSH_CHAT_PROVIDER",
        "db_path": "NPCSH_DB_PATH",
    }

    if command not in var_map:
        return f"Unknown setting: {command}"

    env_var = var_map[command]

  
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

  
    property_exists = False
    for i, line in enumerate(lines):
        if line.startswith(f"export {env_var}="):
            lines[i] = f"export {env_var}='{value}'\n"
            property_exists = True
            break

    if not property_exists:
        lines.append(f"export {env_var}='{value}'\n")

  
    with open(config_path, "w") as f:
        f.writelines(lines)

    return f"{command.capitalize()} has been set to: {value}"


def set_npcsh_initialized() -> None:
    """
    Function Description:
        This function sets the NPCSH initialization flag in the .npcshrc file.
    Args:
        None
    Keyword Args:
        None
    Returns:

        None
    """

    npcshrc_path = ensure_npcshrc_exists()

    with open(npcshrc_path, "r+") as npcshrc:
        content = npcshrc.read()
        if "export NPCSH_INITIALIZED=0" in content:
            content = content.replace(
                "export NPCSH_INITIALIZED=0", "export NPCSH_INITIALIZED=1"
            )
            npcshrc.seek(0)
            npcshrc.write(content)
            npcshrc.truncate()

  
    os.environ["NPCSH_INITIALIZED"] = "1"
    print("NPCSH initialization flag set in .npcshrc")



def get_package_dir() -> str:
    """
    Get the package directory, handling both normal Python and PyInstaller executables.

    For normal Python: returns os.path.dirname(__file__)
    For PyInstaller: returns the bundled data directory (sys._MEIPASS/npcsh)
    """
    # Check if running as a PyInstaller bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle - look for npcsh folder in _MEIPASS
        meipass = sys._MEIPASS
        # The package data should be at _MEIPASS/npcsh (based on PyInstaller config)
        bundled_path = os.path.join(meipass, 'npcsh')
        if os.path.exists(bundled_path):
            return bundled_path
        # Fallback: check if npc_team is directly in _MEIPASS
        if os.path.exists(os.path.join(meipass, 'npc_team')):
            return meipass
        # Last resort: return meipass and let caller handle
        return meipass
    else:
        # Normal Python execution
        return os.path.dirname(__file__)


def file_has_changed(source_path: str, destination_path: str) -> bool:
    """
    Function Description:
        This function compares two files to determine if they are different.
    Args:
        source_path: The path to the source file.
        destination_path: The path to the destination file.
    Keyword Args:
        None
    Returns:
        A boolean indicating whether the files are different
    """


    return not filecmp.cmp(source_path, destination_path, shallow=False)


def list_directory(args: List[str]) -> None:
    """
    Function Description:
        This function lists the contents of a directory.
    Args:
        args: The command arguments.
    Keyword Args:
        None
    Returns:
        None
    """
    directory = args[0] if args else "."
    try:
        files = os.listdir(directory)
        for f in files:
            print(f)
    except Exception as e:
        print(f"Error listing directory: {e}")



def change_directory(command_parts: list, messages: list) -> dict:
    """
    Function Description:
        Changes the current directory.
    Args:
        command_parts : list : Command parts
        messages : list : Messages
    Keyword Args:
        None
    Returns:
        dict : dict : Dictionary

    """

    try:
        if len(command_parts) > 1:
            new_dir = os.path.expanduser(command_parts[1])
        else:
            new_dir = os.path.expanduser("~")
        os.chdir(new_dir)
        return {
            "messages": messages,
            "output": f"Changed directory to {os.getcwd()}",
        }
    except FileNotFoundError:
        return {
            "messages": messages,
            "output": f"Directory not found: {new_dir}",
        }
    except PermissionError:
        return {"messages": messages, "output": f"Permission denied: {new_dir}"}


def orange(text: str) -> str:
    """
    Function Description:
        Returns orange text.
    Args:
        text : str : Text
    Keyword Args:
        None
    Returns:
        text : str : Text

    """
    return f"\033[38;2;255;165;0m{text}{Style.RESET_ALL}"


def get_npcshrc_path_windows():
    return Path.home() / ".npcshrc"


def read_rc_file_windows(path):
    """Read shell-style rc file"""
    config = {}
    if not path.exists():
        return config

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
              
                match = re.match(r'^([A-Z_]+)\s*=\s*[\'"](.*?)[\'"]$', line)
                if match:
                    key, value = match.groups()
                    config[key] = value
    return config


def get_setting_windows(key, default=None):
  
    if env_value := os.getenv(key):
        return env_value

  
    config = read_rc_file_windows(get_npcshrc_path_windows())
    return config.get(key, default)


def setup_readline() -> str:

    if readline is None:
        return None
    try:
        readline.read_history_file(READLINE_HISTORY_FILE)
        readline.set_history_length(1000)
        readline.parse_and_bind("set enable-bracketed-paste on")
        readline.parse_and_bind(r'"\e[A": history-search-backward')
        readline.parse_and_bind(r'"\e[B": history-search-forward')
        readline.parse_and_bind(r'"\C-r": reverse-search-history')
        readline.parse_and_bind(r'\C-e: end-of-line')
        readline.parse_and_bind(r'\C-a: beginning-of-line')
        if sys.platform == "darwin":
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        return READLINE_HISTORY_FILE
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"Warning: Could not read readline history file {READLINE_HISTORY_FILE}: {e}")

def save_readline_history():
    if readline is None:
        return
    try:
        readline.write_history_file(READLINE_HISTORY_FILE)
    except OSError as e:
        print(f"Warning: Could not write readline history file {READLINE_HISTORY_FILE}: {e}")



# ChromaDB client (lazy init)
EMBEDDINGS_DB_PATH = NPCSH_VECTOR_DB_PATH

try:
    chroma_client = chromadb.PersistentClient(path=EMBEDDINGS_DB_PATH) if chromadb else None
except Exception as e:
    print(f"Warning: Failed to initialize ChromaDB client at {EMBEDDINGS_DB_PATH}: {e}")
    chroma_client = None




def get_path_executables() -> List[str]:
    """Get executables from PATH (cached for performance)"""
    if not hasattr(get_path_executables, '_cache'):
        executables = set()
        path_dirs = os.environ.get('PATH', '').split(os.pathsep)
        for path_dir in path_dirs:
            if os.path.isdir(path_dir):
                try:
                    for item in os.listdir(path_dir):
                        item_path = os.path.join(path_dir, item)
                        if os.path.isfile(item_path) and os.access(item_path, os.X_OK):
                            executables.add(item)
                except (PermissionError, OSError):
                    continue
        get_path_executables._cache = sorted(list(executables))
    return get_path_executables._cache


import logging


completion_logger = logging.getLogger('npcsh.completion')
completion_logger.setLevel(logging.WARNING)


if not completion_logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('[%(name)s] %(message)s')
    handler.setFormatter(formatter)
    completion_logger.addHandler(handler)
def make_completer(shell_state: ShellState, router: Any):
    slash_hint_cache = {"last_key": None}

    def complete(text: str, state_index: int) -> Optional[str]:
        """Main completion function"""
        try:
            buffer = readline.get_line_buffer()
            begidx = readline.get_begidx()
            endidx = readline.get_endidx()
            
            # The word currently being completed (e.g., "lor" in "ls lor")
            word_under_cursor = buffer[begidx:endidx] 

            # The very first word/token in the entire buffer (e.g., "ls" in "ls lor")
            first_token_of_buffer = ""
            if buffer.strip():
                match = re.match(r'^(\S+)', buffer.strip())
                if match:
                    first_token_of_buffer = match.group(1)

            matches = []

            # Determine if we are in a "slash command context"
            # This is true if the *entire buffer starts with a slash* AND
            # the current completion is for that initial slash command (begidx == 0).

            is_slash_command_context = (begidx <=1 and first_token_of_buffer.startswith('/'))

            if is_slash_command_context:
                slash_commands = get_slash_commands(shell_state, router)
                
                if first_token_of_buffer == '/': # If just '/' is typed
                    matches = [cmd[1:] for cmd in slash_commands]
                else: # If '/ag' is typed
                    matching_commands = [cmd for cmd in slash_commands if cmd.startswith(first_token_of_buffer)]
                    matches = [cmd[1:] for cmd in matching_commands]
                
                # Only print hints if this is the first completion attempt (state_index == 0)
                # and the hints haven't been printed for this specific input yet.
                if matches and state_index == 0:
                    key = (buffer, first_token_of_buffer) # Use full buffer for cache key
                    if slash_hint_cache["last_key"] != key:
                        print("\nAvailable slash commands: " + ", ".join(slash_commands))
                        try:
                            readline.redisplay()
                        except Exception:
                            pass
                        slash_hint_cache["last_key"] = key
            
            # If not a slash command context, then it's either a regular command or an argument.
            elif begidx == 0: # Completing a regular command (e.g., "ls", "pyt")
                bash_matches = [cmd for cmd in BASH_COMMANDS if cmd.startswith(word_under_cursor)]
                matches.extend(bash_matches)
                
                interactive_matches = [cmd for cmd in interactive_commands.keys() if cmd.startswith(word_under_cursor)]
                matches.extend(interactive_matches)
                
                if len(word_under_cursor) >= 1:
                    path_executables = get_path_executables()
                    exec_matches = [cmd for cmd in path_executables if cmd.startswith(word_under_cursor)]
                    matches.extend(exec_matches[:20])
            
            else: # Completing a file or directory path (e.g., "ls doc/my_f")
                matches = get_file_completions(word_under_cursor)
            
            matches = sorted(list(set(matches)))
            
            if state_index < len(matches):
                return matches[state_index]
            else:
                return None # readline expects None when no more completions
            
        except Exception:
            # Using completion_logger for internal debugging, not printing to stdout for user.
            # completion_logger.error(f"Exception in completion: {e}", exc_info=True) 
            return None
    
    return complete

def get_slash_commands(state: ShellState, router: Any) -> List[str]:
    """Get available slash commands from the provided router and team"""
    commands = []
    
    if router and hasattr(router, 'routes'):
        router_cmds = [f"/{cmd}" for cmd in router.routes.keys()]
        commands.extend(router_cmds)
        completion_logger.debug(f"Router commands: {router_cmds}")
    
  
    if state.team and hasattr(state.team, 'jinxs_dict'):
        jinx_cmds = [f"/{jinx}" for jinx in state.team.jinxs_dict.keys()]
        commands.extend(jinx_cmds)
        completion_logger.debug(f"Jinx commands: {jinx_cmds}")
    
  
    if state.team and hasattr(state.team, 'npcs'):
        npc_cmds = [f"/{npc}" for npc in state.team.npcs.keys()]
        commands.extend(npc_cmds)
        completion_logger.debug(f"NPC commands: {npc_cmds}")
    
  
    mode_cmds = ['/cmd', '/agent', '/chat']
    commands.extend(mode_cmds)
    completion_logger.debug(f"Mode commands: {mode_cmds}")
    
    result = sorted(commands)
    completion_logger.debug(f"Final slash commands: {result}")
    return result
def get_file_completions(text: str) -> List[str]:
    """Get file/directory completions, including for subfolders."""
    try:
        # Determine the base directory and the prefix to match
        if '/' in text:
            basedir = os.path.dirname(text)
            prefix = os.path.basename(text)
        else:
            basedir = '.'
            prefix = text
        
        # If basedir is empty (e.g., text is "folder/"), it should be current dir
        if not basedir:
            basedir = '.'

        # Handle absolute paths
        if text.startswith('/'):
            # Ensure absolute path starts with / and handle cases like "/something"
            if basedir.startswith('/'):
                pass # already absolute
            else:
                basedir = '/' + basedir.lstrip('/') 
            if basedir == '/': # If text was just "/something", basedir is "/"
                prefix = os.path.basename(text)

        # Resolve the actual path to list
        if basedir == '.':
            current_path_to_list = os.getcwd()
        else:
            # If basedir is relative, join it with current working directory
            if not os.path.isabs(basedir):
                current_path_to_list = os.path.join(os.getcwd(), basedir)
            else:
                current_path_to_list = basedir

            if not os.path.isdir(current_path_to_list): # If the base path doesn't exist yet, no completions
                return []

        matches = []
        try:
            for item in os.listdir(current_path_to_list):
                if item.startswith(prefix):
                    full_item_path = os.path.join(current_path_to_list, item)
                    
                    # Construct the completion string relative to the input 'text'
                    # This ensures that if the input was 'folder/s', the completion is 'folder/subfolder/'
                    if basedir == '.':
                        completion = item
                    else:
                        # Reconstruct the path fragment before the prefix
                        path_fragment_before_prefix = text[:len(text) - len(prefix)]
                        completion = os.path.join(path_fragment_before_prefix, item)

                    if os.path.isdir(full_item_path):
                        matches.append(completion + '/')
                    else:
                        matches.append(completion)
        except (PermissionError, OSError):
            pass
        
        return sorted(matches)
    except Exception as e:
        completion_logger.error(f"Error in get_file_completions for text '{text}': {e}", exc_info=True)
        return []


def is_command_position(buffer: str, begidx: int) -> bool:
    """Determine if cursor is at a command position"""
  
    before_word = buffer[:begidx]
    
  
    parts = re.split(r'[|;&]', before_word)
    current_command_part = parts[-1].strip()
    
  
  
    return len(current_command_part) == 0


def readline_safe_prompt(prompt: str) -> str:
    ansi_escape = re.compile(r"(\033\[[0-9;]*[a-zA-Z])")
    return ansi_escape.sub(r"\001\1\002", prompt)

def print_jinxs(jinxs):
    output = "Available jinxs:\n"
    for jinx in jinxs:
        output += f"  {jinx.jinx_name}\n"
        output += f"   Description: {jinx.description}\n"
        output += f"   Inputs: {jinx.inputs}\n"
    return output

def open_terminal_editor(command: str) -> str:
    try:
        os.system(command)
        return 'Terminal editor closed.'
    except Exception as e:
        return f"Error opening terminal editor: {e}"

def get_multiline_input(prompt: str, state=None, router=None, token_hint: str = "") -> str:
    """Get input with hint line below prompt."""
    lines = []
    current_prompt = prompt
    while True:
        try:
            line = _input_with_hint_below(current_prompt, state, router, token_hint)
            if line.endswith("\\"):
                lines.append(line[:-1])
                current_prompt = "> "
                token_hint = ""
            else:
                lines.append(line)
                break
        except EOFError:
            print("Goodbye!")
            sys.exit(0)
    return "\n".join(lines)


def _input_with_hint_below(prompt: str, state=None, router=None, token_hint: str = "") -> str:
    """Custom input with hint displayed below. Arrow keys work for history."""
    try:
        import termios
        import tty
    
    except ImportError:
        return input(prompt)

    if not sys.stdin.isatty():
        return input(prompt)

    # Get history from readline
    hist_len = readline.get_current_history_length()
    history = [readline.get_history_item(i) for i in range(1, hist_len + 1)]
    history_idx = len(history)
    saved_line = ""

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    buf = ""
    pos = 0  # cursor position in buf

    # Calculate visible prompt length (strip ANSI codes)
    import re
    prompt_visible_len = len(re.sub(r'\x1b\[[0-9;]*m|\x01|\x02', '', prompt))

    def current_hint():
        if buf.startswith('/') and len(buf) >= 1:
            h = _get_slash_hints(state, router, buf)
            return h if h else token_hint
        elif buf.startswith('@') and len(buf) >= 1:
            h = _get_npc_hints(state, buf)
            return h if h else token_hint
        return token_hint

    # Get terminal width
    try:
        import shutil
        term_width = shutil.get_terminal_size().columns
    except json.JSONDecodeError:
        term_width = 80

    def draw():
        # Calculate how many lines the input takes
        total_len = prompt_visible_len + len(buf)
        num_lines = (total_len // term_width) + 1

        # Move to start of input (may need to go up multiple lines)
        # First go to column 0
        sys.stdout.write('\r')
        # Move up for each wrapped line we're on
        cursor_total = prompt_visible_len + pos
        # Go up to the first line of input
        for _ in range(num_lines - 1):
            sys.stdout.write('\033[A')

        # Clear from cursor to end of screen (clears all wrapped lines + hint)
        sys.stdout.write('\033[J')

        # Print prompt and buffer
        sys.stdout.write(prompt + buf)

        # Print hint on next line
        sys.stdout.write('\n\033[K' + current_hint())

        # Now position cursor back to correct spot
        # Go back up to the line where cursor should be
        lines_after_cursor = (total_len // term_width) - (cursor_total // term_width) + 1  # +1 for hint line
        for _ in range(lines_after_cursor):
            sys.stdout.write('\033[A')

        # Position cursor in correct column
        cursor_col = cursor_total % term_width
        sys.stdout.write('\r')
        if cursor_col > 0:
            sys.stdout.write('\033[' + str(cursor_col) + 'C')

        sys.stdout.flush()

    # Enable bracketed paste mode - terminal will wrap pastes with escape sequences
    sys.stdout.write('\033[?2004h')

    # Print prompt and reserve hint line
    sys.stdout.write(prompt + '\n' + (token_hint or '') + '\033[A\r')
    if prompt_visible_len > 0:
        sys.stdout.write('\033[' + str(prompt_visible_len) + 'C')
    sys.stdout.flush()

    # Store pasted content separately
    pasted_content = None
    in_paste = False
    paste_buffer = ""

    # Track Ctrl+C for double-press exit
    import time
    last_ctrl_c_time = 0

    try:
        tty.setcbreak(fd)

        while True:
            c = sys.stdin.read(1)

            if not c:  # EOF/stdin closed
                sys.stdout.write('\033[?2004l')  # Disable bracketed paste
                sys.stdout.write('\n\033[K')
                sys.stdout.flush()
                raise EOFError

            # Check for bracketed paste start: ESC [ 2 0 0 ~
            if c == '\x1b':
                c2 = sys.stdin.read(1)
                if c2 == '[':
                    c3 = sys.stdin.read(1)
                    if c3 == '2':
                        c4 = sys.stdin.read(1)
                        if c4 == '0':
                            c5 = sys.stdin.read(1)
                            if c5 == '0':
                                c6 = sys.stdin.read(1)
                                if c6 == '~':
                                    # Start of bracketed paste
                                    in_paste = True
                                    paste_buffer = ""
                                    continue
                            elif c5 == '1':
                                c6 = sys.stdin.read(1)
                                if c6 == '~':
                                    # End of bracketed paste ESC [ 2 0 1 ~
                                    in_paste = False
                                    if paste_buffer:
                                        # Check if this looks like binary/image data
                                        # Image signatures: PNG (\x89PNG), JPEG (\xff\xd8\xff), GIF (GIF8), BMP (BM)
                                        # Also check for high ratio of non-printable chars

                                        if len(paste_buffer) > 4:
                                            # Check for common image magic bytes
                                            if paste_buffer[:4] == '\x89PNG' or paste_buffer[:8] == '\x89PNG\r\n\x1a\n':
                                                is_binary = True
                                            elif paste_buffer[:2] == '\xff\xd8':  # JPEG
                                                is_binary = True
                                            elif paste_buffer[:4] == 'GIF8':  # GIF
                                                is_binary = True
                                            elif paste_buffer[:2] == 'BM':  # BMP
                                                is_binary = True
                                            elif paste_buffer.startswith('data:image/'):  # Base64 data URL
                                                is_binary = True
                                            else:
                                                # Check for high ratio of non-printable characters
                                                non_printable = sum(1 for c in paste_buffer[:100] if ord(c) < 32 and c not in '\n\r\t')
                                                if non_printable > 10:  # More than 10% non-printable in first 100 chars
                                                    is_binary = True

                                        if is_binary:
                                            # Save image data to temp file

                                            
                                            try:
                                                # Determine extension from magic bytes
                                                ext = '.bin'
                                                if '\x89PNG' in paste_buffer[:8]:
                                                    ext = '.png'
                                                elif paste_buffer[:2] == '\xff\xd8':
                                                    ext = '.jpg'
                                                elif paste_buffer[:4] == 'GIF8':
                                                    ext = '.gif'
                                                elif paste_buffer.startswith('data:image/'):
                                                    # Extract from data URL
                                                    if 'png' in paste_buffer[:30]:
                                                        ext = '.png'
                                                    elif 'jpeg' in paste_buffer[:30] or 'jpg' in paste_buffer[:30]:
                                                        ext = '.jpg'
                                                    elif 'gif' in paste_buffer[:30]:
                                                        ext = '.gif'

                                                fd, temp_path = tempfile.mkstemp(suffix=ext, prefix='npcsh_paste_')
                                                with os.fdopen(fd, 'wb') as f:
                                                    if paste_buffer.startswith('data:image/'):
                                                        # Decode base64 data URL

                                                        _, data = paste_buffer.split(',', 1)
                                                        f.write(base64.b64decode(data))
                                                    else:
                                                        f.write(paste_buffer.encode('latin-1'))
                                                pasted_content = temp_path  # Store path to image
                                                placeholder = f"[pasted image: {temp_path}]"
                                            except Exception:
                                                pasted_content = None
                                                placeholder = "[pasted image: failed to save]"
                                        else:
                                            pasted_content = paste_buffer.rstrip('\r\n')
                                            line_count = pasted_content.count('\n') + 1
                                            char_count = len(pasted_content)
                                            if line_count > 1:
                                                placeholder = f"[pasted: {line_count} lines, {char_count} chars]"
                                            else:
                                                # Single line paste - just insert it directly
                                                buf = buf[:pos] + pasted_content + buf[pos:]
                                                pos += len(pasted_content)
                                                pasted_content = None  # Clear so we don't replace on submit
                                                draw()
                                                continue
                                        buf = buf[:pos] + placeholder + buf[pos:]
                                        pos += len(placeholder)
                                        draw()
                                    continue
                    # Handle arrow keys and other escape sequences
                    if c3 == 'A':  # Up
                        if history_idx > 0:
                            if history_idx == len(history):
                                saved_line = buf
                            history_idx -= 1
                            buf = history[history_idx] or ''
                            pos = len(buf)
                            draw()
                        continue
                    elif c3 == 'B':  # Down
                        if history_idx < len(history):
                            history_idx += 1
                            buf = saved_line if history_idx == len(history) else (history[history_idx] or '')
                            pos = len(buf)
                            draw()
                        continue
                    elif c3 == 'C':  # Right
                        if pos < len(buf):
                            pos += 1
                            sys.stdout.write('\033[C')
                            sys.stdout.flush()
                        continue
                    elif c3 == 'D':  # Left
                        if pos > 0:
                            pos -= 1
                            sys.stdout.write('\033[D')
                            sys.stdout.flush()
                        continue
                    elif c3 == '3':  # Del
                        sys.stdin.read(1)  # ~
                        if pos < len(buf):
                            buf = buf[:pos] + buf[pos+1:]
                            draw()
                        continue
                    elif c3 == 'H':  # Home
                        pos = 0
                        draw()
                        continue
                    elif c3 == 'F':  # End
                        pos = len(buf)
                        draw()
                        continue
                elif c2 == '\x1b':  # Double ESC
                    sys.stdout.write('\033[?2004l')  # Disable bracketed paste
                    sys.stdout.write('\n\033[K')
                    sys.stdout.flush()
                    return '\x1b'
                continue

            # If we're in a paste, accumulate to paste buffer
            if in_paste:
                paste_buffer += c
                continue

            if c in ('\n', '\r'):
                # Clear hint and newline
                sys.stdout.write('\033[?2004l')  # Disable bracketed paste
                sys.stdout.write('\n\033[K')
                sys.stdout.flush()
                # If we have pasted content, replace placeholder with actual content
                if pasted_content is not None:
                    import re
                    # Escape pipe characters in pasted content so they aren't parsed as pipeline operators
                    escaped_content = pasted_content.replace('|', '\\|')
                    # If pasted content is at the start of command, escape @ and / to prevent
                    # them being interpreted as delegation or slash commands
                    placeholder_pattern = r'\[pasted: \d+ lines?, \d+ chars?\]'
                    if re.match(placeholder_pattern, buf.lstrip()):
                        if escaped_content.startswith('@'):
                            escaped_content = '\\@' + escaped_content[1:]
                        elif escaped_content.startswith('/'):
                            escaped_content = '\\/' + escaped_content[1:]
                    # Use lambda to avoid backreference issues in replacement string
                    result = re.sub(placeholder_pattern, lambda m: escaped_content, buf)
                    if result.strip():
                        readline.add_history(result)
                    return result
                if buf.strip():
                    readline.add_history(buf)
                return buf

            elif c == '\x7f' or c == '\x08':  # Backspace
                if pos > 0:
                    buf = buf[:pos-1] + buf[pos:]
                    pos -= 1
                    draw()

            elif c == '\x03':  # Ctrl-C
                current_time = time.time()
                if current_time - last_ctrl_c_time < 1.0:  # Double Ctrl+C within 1 second
                    # Exit
                    sys.stdout.write('\033[?2004l')  # Disable bracketed paste
                    sys.stdout.write('\n\033[K')
                    sys.stdout.flush()
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    raise KeyboardInterrupt
                else:
                    # First Ctrl+C - clear the line
                    last_ctrl_c_time = current_time
                    buf = ""
                    pos = 0
                    pasted_content = None
                    sys.stdout.write('\r\033[K')  # Clear current line
                    sys.stdout.write('^C\n')
                    # Redraw prompt
                    sys.stdout.write(prompt + '\n' + current_hint() + '\033[A\r')
                    if prompt_visible_len > 0:
                        sys.stdout.write('\033[' + str(prompt_visible_len) + 'C')
                    sys.stdout.flush()

            elif c == '\x04':  # Ctrl-D
                if not buf:
                    sys.stdout.write('\n\033[K')
                    sys.stdout.flush()
                    raise EOFError

            elif c == '\x01':  # Ctrl-A
                pos = 0
                draw()

            elif c == '\x05':  # Ctrl-E
                pos = len(buf)
                draw()

            elif c == '\x15':  # Ctrl-U
                buf = buf[pos:]
                pos = 0
                draw()

            elif c == '\x0b':  # Ctrl-K
                buf = buf[:pos]
                draw()

            elif c == '\x17':  # Ctrl-W - delete word back
                while pos > 0 and buf[pos-1] == ' ':
                    buf = buf[:pos-1] + buf[pos:]
                    pos -= 1
                while pos > 0 and buf[pos-1] != ' ':
                    buf = buf[:pos-1] + buf[pos:]
                    pos -= 1
                draw()

            elif c == '\t':  # Tab - do nothing for now
                pass

            elif c == '\x0f':  # Ctrl-O - show last tool call args
                try:
                    import builtins
                    last_call = getattr(builtins, '_npcsh_last_tool_call', None)
                    if last_call:
                        from termcolor import colored
                        # Save cursor, move down past hint, show args, restore
                        sys.stdout.write('\n\033[K')  # New line, clear
                        sys.stdout.write(colored(f"─── {last_call['name']} ───\n", "cyan"))
                        args = last_call.get('arguments', {})
                        for k, v in args.items():
                            v_str = str(v)
                            # Show with syntax highlighting for code
                            if '\n' in v_str:
                                sys.stdout.write(colored(f"{k}:\n", "yellow"))
                                for line in v_str.split('\n')[:30]:  # Limit lines
                                    sys.stdout.write(f"  {line}\n")
                                if v_str.count('\n') > 30:
                                    sys.stdout.write(colored(f"  ... ({v_str.count(chr(10)) - 30} more lines)\n", "white", attrs=["dark"]))
                            else:
                                sys.stdout.write(colored(f"{k}: ", "yellow") + f"{v_str}\n")
                        sys.stdout.write(colored("─" * 40 + "\n", "cyan"))
                        # Redraw prompt
                        sys.stdout.write(prompt)
                        sys.stdout.write(buf)
                        sys.stdout.write('\n' + (token_hint or ''))
                        sys.stdout.write('\033[A\r')
                        if prompt_visible_len > 0:
                            sys.stdout.write('\033[' + str(prompt_visible_len + pos) + 'C')
                        sys.stdout.flush()
                    else:
                        pass  # No tool call to show
                except Exception:
                    pass

            elif c and ord(c) >= 32:  # Printable
                buf = buf[:pos] + c + buf[pos:]
                pos += 1
                draw()

    finally:
        sys.stdout.write('\033[?2004l')  # Disable bracketed paste mode
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _get_slash_hints(state, router, prefix='/') -> str:
    """Slash command hints - fits terminal width."""
    cmds = {'help', 'set', 'agent', 'chat', 'cmd', 'sq', 'quit', 'exit', 'clear', 'npc'}
    if state and state.team and hasattr(state.team, 'jinxs_dict'):
        cmds.update(state.team.jinxs_dict.keys())
    if router and hasattr(router, 'jinx_routes'):
        cmds.update(router.jinx_routes.keys())
    if len(prefix) > 1:
        f = prefix[1:].lower()
        cmds = {c for c in cmds if c.lower().startswith(f)}
    if cmds:
        # Get terminal width, default 80
        try:
            import shutil
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 80

        # Build hint string that fits in terminal
        sorted_cmds = sorted(cmds)
        hint_parts = []
        current_len = 2  # leading spaces
        for c in sorted_cmds:
            item = '/' + c
            if current_len + len(item) + 2 > term_width - 5:  # leave margin
                break
            hint_parts.append(item)
            current_len += len(item) + 2

        if hint_parts:
            return colored('  ' + '  '.join(hint_parts), 'white', attrs=['dark'])
    return ""


def _get_npc_hints(state, prefix='@') -> str:
    """NPC hints."""
    npcs = set()
    if state and state.team:
        if hasattr(state.team, 'npcs') and state.team.npcs:
            npcs.update(state.team.npcs.keys())
        if hasattr(state.team, 'forenpc') and state.team.forenpc:
            npcs.add(state.team.forenpc.name)
    if not npcs:
        npcs = {'sibiji', 'guac', 'corca', 'kadiefa', 'plonk'}
    if len(prefix) > 1:
        f = prefix[1:].lower()
        npcs = {n for n in npcs if n.lower().startswith(f)}
    if npcs:
        return colored('  ' + '  '.join('@' + n for n in sorted(npcs)), 'cyan')
    return ""



def split_by_pipes(command: str) -> List[str]:
    parts = []
    current = ""
    in_single_quote = False
    in_double_quote = False
    escape = False

    for char in command:
        if escape:
            current += char
            escape = False
        elif char == '\\':
            escape = True
            current += char
        elif char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current += char
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_single_quote
            current += char
        elif char == '|' and not in_single_quote and not in_double_quote:
            parts.append(current.strip())
            current = ""
        else:
            current += char

    if current:
        parts.append(current.strip())
    return parts

def parse_command_safely(cmd: str) -> List[str]:
    try:
        return shlex.split(cmd)
    except ValueError as e:
        if "No closing quotation" in str(e):
            if cmd.count('"') % 2 == 1:
                cmd += '"'
            elif cmd.count("'") % 2 == 1:
                cmd += "'"
            try:
                return shlex.split(cmd)
            except ValueError:
                return cmd.split()
        else:
            return cmd.split()

def get_file_color(filepath: str) -> tuple:
    if not os.path.exists(filepath):
         return "grey", []
    if os.path.isdir(filepath):
        return "blue", ["bold"]
    elif os.access(filepath, os.X_OK) and not os.path.isdir(filepath):
        return "green", ["bold"]
    elif filepath.endswith((".zip", ".tar", ".gz", ".bz2", ".xz", ".7z")):
        return "red", []
    elif filepath.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")):
        return "magenta", []
    elif filepath.endswith((".py", ".pyw")):
        return "yellow", []
    elif filepath.endswith((".sh", ".bash", ".zsh")):
        return "green", []
    elif filepath.endswith((".c", ".cpp", ".h", ".hpp")):
        return "cyan", []
    elif filepath.endswith((".js", ".ts", ".jsx", ".tsx")):
        return "yellow", []
    elif filepath.endswith((".html", ".css", ".scss", ".sass")):
        return "magenta", []
    elif filepath.endswith((".md", ".txt", ".log")):
        return "white", []
    elif os.path.basename(filepath).startswith("."):
        return "cyan", []
    else:
        return "white", []

def format_file_listing(output: str) -> str:
    colored_lines = []
    current_dir = os.getcwd()
    for line in output.strip().split("\n"):
        parts = line.split()
        if not parts:
            colored_lines.append(line)
            continue

        filepath_guess = parts[-1]
        potential_path = os.path.join(current_dir, filepath_guess)

        color, attrs = get_file_color(potential_path)
        colored_filepath = colored(filepath_guess, color, attrs=attrs)

        if len(parts) > 1 :
           
             colored_line = " ".join(parts[:-1] + [colored_filepath])
        else:
           
             colored_line = colored_filepath

        colored_lines.append(colored_line)

    return "\n".join(colored_lines)

def wrap_text(text: str, width: int = 80) -> str:
    lines = []
    for paragraph in text.split("\n"):
        if len(paragraph) > width:
             lines.extend(textwrap.wrap(paragraph, width=width, replace_whitespace=False, drop_whitespace=False))
        else:
             lines.append(paragraph)
    return "\n".join(lines)




        
def store_command_embeddings(command: str, output: Any, state: ShellState):
    if not chroma_client or not state.embedding_model or not state.embedding_provider:
        if not chroma_client:
            print("Warning: ChromaDB client not available for embeddings.", file=sys.stderr)
        return
    if not command and not output:
        return

    try:
        output_str = str(output) if output else ""
        if not command and not output_str:
            return

        texts_to_embed = [command, output_str]

        embeddings = get_embeddings(
            texts_to_embed,
            state.embedding_model,
            state.embedding_provider,
        )

        if not embeddings or len(embeddings) != 2:
             print(f"Warning: Failed to generate embeddings for command: {command[:50]}...", file=sys.stderr)
             return

        timestamp = datetime.now().isoformat()
        npc_name = state.npc.name if isinstance(state.npc, NPC) else state.npc

        metadata = [
            {
                "type": "command", "timestamp": timestamp, "path": state.current_path,
                "npc": npc_name, "conversation_id": state.conversation_id,
            },
            {
                "type": "response", "timestamp": timestamp, "path": state.current_path,
                "npc": npc_name, "conversation_id": state.conversation_id,
            },
        ]

        collection_name = f"{state.embedding_provider}_{state.embedding_model}_embeddings"
        try:
            collection = chroma_client.get_or_create_collection(collection_name)
            ids = [f"cmd_{timestamp}_{hash(command)}", f"resp_{timestamp}_{hash(output_str)}"]

            collection.add(
                embeddings=embeddings,
                documents=texts_to_embed,
                metadatas=metadata,
                ids=ids,
            )
        except Exception as e:
            print(f"Warning: Failed to add embeddings to collection '{collection_name}': {e}", file=sys.stderr)

    except Exception as e:
        print(f"Warning: Failed to store embeddings: {e}", file=sys.stderr)


def handle_interactive_command(cmd_parts: List[str], state: ShellState) -> Tuple[ShellState, str]:
    command_name = cmd_parts[0]
    print(f"Starting interactive {command_name} session...")
    try:
      
        full_command_str = " ".join(cmd_parts)
        return_code = start_interactive_session(full_command_str)
        output = f"Interactive {command_name} session ended with return code {return_code}"
    except Exception as e:
        output = f"Error starting interactive session {command_name}: {e}"
    return state, output

def handle_cd_command(cmd_parts: List[str], state: ShellState) -> Tuple[ShellState, str]:
    original_path = os.getcwd()
    target_path = cmd_parts[1] if len(cmd_parts) > 1 else os.path.expanduser("~")
    try:
        os.chdir(target_path)
        state.current_path = os.getcwd()
        output = f"Changed directory to {state.current_path}"
    except FileNotFoundError:
        output = colored(f"cd: no such file or directory: {target_path}", "red")
    except Exception as e:
        output = colored(f"cd: error changing directory: {e}", "red")
        os.chdir(original_path) 

    return state, output


def handle_bash_command(
    cmd_parts: List[str],
    cmd_str: str,
    stdin_input: Optional[str],
    state: ShellState,
) -> Tuple[bool, str]:
    try:
        process = subprocess.Popen(
            cmd_parts,
            stdin=subprocess.PIPE if stdin_input is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=state.current_path
        )
        stdout, stderr = process.communicate(input=stdin_input)

        if process.returncode != 0:
            return False, stderr.strip() if stderr else f"Command '{cmd_str}' failed with return code {process.returncode}."

        if stderr.strip():
            print(colored(f"stderr: {stderr.strip()}", "yellow"), file=sys.stderr)
        
        if cmd_parts[0] in ["ls", "find", "dir"]:
            return True, format_file_listing(stdout.strip())

        return True, stdout.strip()

    except FileNotFoundError:
        return False, f"Command not found: {cmd_parts[0]}"
    except PermissionError:
        return False, f"Permission denied: {cmd_str}"

def _try_convert_type(value: str) -> Union[str, int, float, bool]:
    """Helper to convert string values to appropriate types."""
    if value.lower() in ['true', 'yes']:
        return True
    if value.lower() in ['false', 'no']:
        return False
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value

def parse_generic_command_flags(parts: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Parses a list of command parts into a dictionary of keyword arguments and a list of positional arguments.
    Handles: -f val, --flag val, --flag=val, flag=val, --boolean-flag
    """
    parsed_kwargs = {}
    positional_args = []
    i = 0
    while i < len(parts):
        part = parts[i]
        
        if part.startswith('--'):
            key_part = part[2:]
            if '=' in key_part:
                key, value = key_part.split('=', 1)
                parsed_kwargs[key] = _try_convert_type(value)
            else:
              
                if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                    parsed_kwargs[key_part] = _try_convert_type(parts[i + 1])
                    i += 1 
                else:
                    parsed_kwargs[key_part] = True 
        
        elif part.startswith('-'):
            key = part[1:]
          
            if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                parsed_kwargs[key] = _try_convert_type(parts[i + 1])
                i += 1 
            else:
                parsed_kwargs[key] = True 
        
        elif '=' in part and not part.startswith('-'):
             key, value = part.split('=', 1)
             parsed_kwargs[key] = _try_convert_type(value)
        
        else:
            positional_args.append(part)
        
        i += 1
        
    return parsed_kwargs, positional_args

def _ollama_supports_tools(model: str) -> Optional[bool]:
    """
    Best-effort check for tool-call support on an Ollama model by inspecting its template/metadata.
    Mirrors the lightweight check used in the Flask serve path.
    """


    try:
        details = ollama.show(model)
        template = details.get("template") or ""
        metadata = details.get("metadata") or {}
        if any(token in template for token in ["{{- if .Tools", "{{- range .Tools", "{{- if .ToolCalls"]):
            return True
        if metadata.get("tools") or metadata.get("tool_calls"):
            return True
        return False
    except Exception:
        return None


def model_supports_tool_calls(model: Optional[str], provider: Optional[str]) -> bool:
    """
    Decide whether to attempt tool-calling for the given model/provider.
    Uses Ollama template inspection when possible and falls back to name heuristics.
    """
    if not model:
        return False

    provider = (provider or "").lower()
    model_lower = model.lower()

    if provider == "ollama":
        ollama_support = _ollama_supports_tools(model)
        if ollama_support is not None:
            return ollama_support

    toolish_markers = [
        "gpt",
        "claude",
        "qwen",
        "mistral",
        "llama-3.1",
        "llama3.1",
        "llama-3.2",
        "llama3.2",
        "gemini",
        "tool",
    ]
    return any(marker in model_lower for marker in toolish_markers)


def wrap_tool_with_display(tool_name: str, tool_func: Callable, state: ShellState) -> Callable:
    """Wrap a tool function to add visual feedback when it executes.

    Respects state.log_level:
    - "silent": no output
    - "normal": show tool name and success/failure
    - "verbose": show tool name, args, success/failure, and result preview
    """
    def wrapped(**kwargs):
        log_level = getattr(state, 'log_level', 'normal')

        # Display tool call (skip in silent mode)
        if log_level != "silent":
            try:
                args_display = ""
                # Always show a preview of args for key tools
                if kwargs:
                    # For sh/python/sql, show the code/command being run
                    if tool_name in ('sh', 'python', 'sql', 'cmd') and 'code' in kwargs:
                        code_preview = str(kwargs['code']).strip().split('\n')[0]  # First line
                        if len(code_preview) > 80:
                            code_preview = code_preview[:80] + "…"
                        args_display = code_preview
                    elif tool_name == 'sh' and 'bash_command' in kwargs:
                        cmd_preview = str(kwargs['bash_command']).strip().split('\n')[0]
                        if len(cmd_preview) > 80:
                            cmd_preview = cmd_preview[:80] + "…"
                        args_display = cmd_preview
                    elif tool_name in ('sh', 'cmd') and 'command' in kwargs:
                        cmd_preview = str(kwargs['command']).strip().split('\n')[0]
                        if len(cmd_preview) > 80:
                            cmd_preview = cmd_preview[:80] + "…"
                        args_display = cmd_preview
                    elif tool_name == 'python' and 'python_code' in kwargs:
                        code_preview = str(kwargs['python_code']).strip().split('\n')[0]
                        if len(code_preview) > 80:
                            code_preview = code_preview[:80] + "…"
                        args_display = code_preview
                    elif tool_name == 'agent' and 'npc_name' in kwargs:
                        args_display = f"@{kwargs['npc_name']}"
                        if 'request' in kwargs:
                            req = str(kwargs['request'])[:50]
                            args_display += f": {req}…" if len(str(kwargs['request'])) > 50 else f": {req}"
                    elif tool_name == 'agent' and 'query' in kwargs:
                        query_preview = str(kwargs['query']).strip()[:60]
                        args_display = query_preview + ("…" if len(str(kwargs['query'])) > 60 else "")
                    elif log_level == "verbose":
                        arg_parts = []
                        for _, v in kwargs.items():
                            v_str = str(v)
                            if len(v_str) > 40:
                                v_str = v_str[:40] + "…"
                            arg_parts.append(f"{v_str}")
                        args_display = " ".join(arg_parts)
                        if len(args_display) > 60:
                            args_display = args_display[:60] + "…"

                if args_display:
                    print(colored(f"  ⚡ {tool_name}", "cyan") + colored(f" {args_display}", "white", attrs=["dark"]), end="", flush=True)
                else:
                    print(colored(f"  ⚡ {tool_name}", "cyan"), end="", flush=True)
            except Exception:
                pass

        # Execute tool
        try:
            result = tool_func(**kwargs)
            if log_level != "silent":
                try:
                    print(colored(" ✓", "green"), flush=True)
                    # Show preview of result only in verbose mode
                    if log_level == "verbose":
                        result_preview = str(result)
                        if len(result_preview) > 200:
                            result_preview = result_preview[:200] + "..."
                        if result_preview and result_preview not in ('None', '', '{}', '[]'):
                            print(colored(f"    → {result_preview}", "white", attrs=["dark"]), flush=True)
                except Exception:
                    pass
            return result
        except Exception as e:
            if log_level != "silent":
                try:
                    print(colored(f" ✗ {str(e)[:100]}", "red"), flush=True)
                except Exception as e:
                    pass
            raise
    return wrapped


def collect_llm_tools(state: ShellState) -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """
    Assemble tool definitions + executable map from NPC tools, Jinxs, and MCP servers.
    This mirrors the auto-translation used in the Flask server path.
    """
    tools: List[Dict[str, Any]] = []
    tool_map: Dict[str, Callable] = {}

    # NPC-defined Python tools
    npc_obj = state.npc if isinstance(state.npc, NPC) else None
    if npc_obj and getattr(npc_obj, "tools", None):
        if isinstance(npc_obj.tools, list) and npc_obj.tools and callable(npc_obj.tools[0]):
            tools_schema, auto_map = auto_tools(npc_obj.tools)
            tools.extend(tools_schema or [])
            tool_map.update(auto_map or {})
        else:
            tools.extend(npc_obj.tools or [])
            if getattr(npc_obj, "tool_map", None):
                tool_map.update(npc_obj.tool_map)
    elif npc_obj and getattr(npc_obj, "tool_map", None):
        tool_map.update(npc_obj.tool_map)

    # Jinx tools from NPC and Team
    aggregated_jinxs: Dict[str, Any] = {}
    if npc_obj and getattr(npc_obj, "jinxs_dict", None):
        aggregated_jinxs.update(npc_obj.jinxs_dict)
    if state.team and isinstance(state.team, Team) and getattr(state.team, "jinxs_dict", None):
        aggregated_jinxs.update({k: v for k, v in state.team.jinxs_dict.items() if k not in aggregated_jinxs})

    if aggregated_jinxs:
        jinx_catalog: Dict[str, Dict[str, Any]] = {}
        if npc_obj and getattr(npc_obj, "jinx_tool_catalog", None):
            jinx_catalog.update(npc_obj.jinx_tool_catalog or {})
        if state.team and isinstance(state.team, Team) and getattr(state.team, "jinx_tool_catalog", None):
            jinx_catalog.update(state.team.jinx_tool_catalog or {})
        if not jinx_catalog:
            jinx_catalog = build_jinx_tool_catalog(aggregated_jinxs)

        tools.extend(list(jinx_catalog.values()))

        jinja_env_for_jinx = getattr(npc_obj, "jinja_env", None)
        if not jinja_env_for_jinx and state.team and isinstance(state.team, Team):
            jinja_env_for_jinx = getattr(state.team, "jinja_env", None)

        jinx_globals = {
            "state": state,
            "CommandHistory": CommandHistory,
            "load_kg_from_db": load_kg_from_db,
            "execute_rag_command": execute_rag_command,
            "execute_brainblast_command": execute_brainblast_command,
            "load_file_contents": load_file_contents,
            "search_web": search_web,
            "get_relevant_memories": get_relevant_memories,
        }

        for name, jinx_obj in aggregated_jinxs.items():
            def _make_runner(jinx=jinx_obj, jinja_env=jinja_env_for_jinx, tool_name=name, extras=jinx_globals):
                def runner(**kwargs):
                    input_values = kwargs if isinstance(kwargs, dict) else {}
                    try:
                        ctx = jinx.execute(
                            input_values=input_values,
                            npc=npc_obj,
                            messages=state.messages,
                            extra_globals=extras,
                            jinja_env=jinja_env
                        )
                        return ctx.get("output", ctx)
                    except Exception as exc:
                        return f"Jinx '{tool_name}' failed: {exc}"
                return runner
            tool_map[name] = _make_runner()

    # MCP tools via npcsh.corca client
    try:
        from npcsh.corca import MCPClientNPC, _resolve_and_copy_mcp_server_path  # type: ignore

        team_ctx_mcp_servers = None
        if state.team and isinstance(state.team, Team) and hasattr(state.team, "team_ctx"):
            team_ctx_mcp_servers = state.team.team_ctx.get("mcp_servers", [])

        mcp_server_path = _resolve_and_copy_mcp_server_path(
            explicit_path=None,
            current_path=state.current_path,
            team_ctx_mcp_servers=team_ctx_mcp_servers,
            interactive=False,
            auto_copy_bypass=True
        )

        if mcp_server_path:
            reuse_client = (
                state.mcp_client
                if state.mcp_client and getattr(state.mcp_client, "server_script_path", None) == mcp_server_path
                else None
            )
            mcp_client = reuse_client or MCPClientNPC()
            if reuse_client is None:
                try:
                    connected = mcp_client.connect_sync(mcp_server_path)
                except Exception:
                    connected = False
                if connected:
                    state.mcp_client = mcp_client
            if mcp_client and getattr(mcp_client, "available_tools_llm", None):
                for tool_def in mcp_client.available_tools_llm:
                    name = tool_def.get("function", {}).get("name")
                    if name and name not in tool_map:
                        tools.append(tool_def)
                tool_map.update(getattr(mcp_client, "tool_map", {}) or {})
    except Exception:
        pass  # MCP is optional; ignore failures

    # Deduplicate tools by name to avoid confusing the LLM
    deduped = {}
    for tool_def in tools:
        name = tool_def.get("function", {}).get("name")
        if name:
            deduped[name] = tool_def

    # Wrap all tools with display feedback for npcsh
    wrapped_tool_map = {name: wrap_tool_with_display(name, func, state) for name, func in tool_map.items()}

    return list(deduped.values()), wrapped_tool_map


def should_skip_kg_processing(user_input: str, assistant_output: str) -> bool:
    """Determine if this interaction is too trivial for KG processing"""
    
  
    if len(user_input.strip()) < 10:
        return True
    
    simple_bash = {'ls', 'pwd', 'cd', 'mkdir', 'touch', 'rm', 'mv', 'cp'}
    first_word = user_input.strip().split()[0] if user_input.strip() else ""
    if first_word in simple_bash:
        return True
    
    if len(assistant_output.strip()) < 20:
        return True
    
    if "exiting" in assistant_output.lower() or "exited" in assistant_output.lower():
        return True
    
    return False

def execute_slash_command(command: str,
                          stdin_input: Optional[str],
                          state: ShellState,
                          stream: bool,
                          router) -> Tuple[ShellState, Any]:
    """Executes slash commands using the router."""
    try:
        all_command_parts = shlex.split(command)
    except ValueError:
        all_command_parts = command.split()
    command_name = all_command_parts[0].lstrip('/')

    # --- QUIT/EXIT HANDLING ---
    if command_name in ['quit', 'exit', 'q']:
    
        print("Goodbye!")
        sys.exit(0)

    # --- NPC SWITCHING LOGIC ---
    if command_name in ['n', 'npc']:
        npc_to_switch_to = all_command_parts[1] if len(all_command_parts) > 1 else None
        if npc_to_switch_to and state.team and npc_to_switch_to in state.team.npcs:
            state.npc = state.team.npcs[npc_to_switch_to]
            return state, {"output": f"Switched to NPC: {npc_to_switch_to}", "messages": state.messages}
        else:
            available_npcs = list(state.team.npcs.keys()) if state.team else []
            return state, {"output": colored(f"NPC '{npc_to_switch_to}' not found. Available NPCs: {', '.join(available_npcs)}", "red"), "messages": state.messages}
    
    # --- ROUTER LOGIC ---
    handler = router.get_route(command_name)
    if handler:
        handler_kwargs = {
            'stream': stream, 'team': state.team, 'messages': state.messages, 'api_url': state.api_url,
            'api_key': state.api_key, 'stdin_input': stdin_input,
            'model': state.npc.model if isinstance(state.npc, NPC) and state.npc.model else state.chat_model,
            'provider': state.npc.provider if isinstance(state.npc, NPC) and state.npc.provider else state.chat_provider,
            'npc': state.npc, 'sprovider': state.search_provider, 'emodel': state.embedding_model,
            'eprovider': state.embedding_provider, 'igmodel': state.image_gen_model, 'igprovider': state.image_gen_provider,
            'vmodel': state.vision_model, 'vprovider': state.vision_provider, 'rmodel': state.reasoning_model, 
            'rprovider': state.reasoning_provider, 'state': state
        }
        try:
            result = handler(command=command, **handler_kwargs)
            if isinstance(result, dict): 
                state.messages = result.get("messages", state.messages)
            return state, result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return state, {"output": colored(f"Error executing slash command '{command_name}': {e}", "red"), "messages": state.messages}
    
    # Fallback for switching NPC by name
    if state.team and command_name in state.team.npcs:
        state.npc = state.team.npcs[command_name]
        return state, {"output": f"Switched to NPC: {state.npc.name}", "messages": state.messages}

    return state, {"output": colored(f"Unknown slash command or NPC: {command_name}", "red"), "messages": state.messages}


def process_pipeline_command(
    cmd_segment: str,
    stdin_input: Optional[str],
    state: ShellState,
    stream_final: bool, 
    review = False, 
    router = None,
    ) -> Tuple[ShellState, Any]:

    if not cmd_segment:
        return state, stdin_input

    available_models_all = get_locally_available_models(state.current_path)
    available_models_all_list = [
        item for key, item in available_models_all.items()
    ]

    model_override, provider_override, cmd_cleaned = get_model_and_provider(
        cmd_segment, available_models_all_list
    )
    cmd_to_process = cmd_cleaned.strip()
    if not cmd_to_process:
         return state, stdin_input

    npc_model = (
        state.npc.model 
        if isinstance(state.npc, NPC) and state.npc.model 
        else None
    )
    npc_provider = (
        state.npc.provider 
        if isinstance(state.npc, NPC) and state.npc.provider 
        else None
    )

    exec_model = model_override or npc_model or state.chat_model
    exec_provider = provider_override or npc_provider or state.chat_provider

    if cmd_to_process.startswith("/"):
        command_name = cmd_to_process.split()[0].lstrip('/')
        
        # Check if this is an interactive mode
        is_interactive_mode = False

        # Check if the jinx declares interactive: true
        if router.is_interactive(command_name):
            is_interactive_mode = True

        # Also check modes/ directory (legacy)
        if not is_interactive_mode:
            global_modes_jinx = os.path.expanduser(f'~/.npcsh/npc_team/jinxs/modes/{command_name}.jinx')
            if os.path.exists(global_modes_jinx):
                is_interactive_mode = True

        if not is_interactive_mode and state.team and state.team.team_path:
            team_modes_jinx = os.path.join(state.team.team_path, 'jinxs', 'modes', f'{command_name}.jinx')
            if os.path.exists(team_modes_jinx):
                is_interactive_mode = True
        
        if is_interactive_mode:
            result = execute_slash_command(
                cmd_to_process, 
                stdin_input, 
                state, 
                stream_final, 
                router
            )
        else:
            with SpinnerContext(
                f"Routing to {cmd_to_process.split()[0]}", 
                style="arrow"
            ):
                result = execute_slash_command(
                    cmd_to_process, 
                    stdin_input, 
                    state, 
                    stream_final, 
                    router
                )
        return result
    cmd_parts = parse_command_safely(cmd_to_process)
    if not cmd_parts:
        return state, stdin_input

    command_name = cmd_parts[0]

    if command_name == "cd":
        return handle_cd_command(cmd_parts, state)
    
    if command_name in interactive_commands:
        return handle_interactive_command(cmd_parts, state)
        
    if command_name in TERMINAL_EDITORS:
        print(f"Starting interactive editor: {command_name}...")
        full_command_str = " ".join(cmd_parts)
        output = open_terminal_editor(full_command_str)
        return state, output

    if validate_bash_command(cmd_parts):
        with SpinnerContext(f"Executing {command_name}", style="line"):
            try: # Added try-except for KeyboardInterrupt here
                success, result = handle_bash_command(
                    cmd_parts, 
                    cmd_to_process, 
                    stdin_input, 
                    state
                )
            except KeyboardInterrupt:
                print(colored("\nBash command interrupted by user.", "yellow"))
                return state, colored("Command interrupted.", "red")
        
        if success:
            return state, result
        else:
            print(
                colored(
                    f"Command failed. Consulting {exec_model}...", 
                    "yellow"
                ), 
                file=sys.stderr
            )
            fixer_prompt = (
                f"The command '{cmd_to_process}' failed with error: "
                f"'{result}'. Provide the correct command."
            )
            
            with SpinnerContext(
                f"{exec_model} analyzing error", 
                style="brain"
            ):
                try: # Added try-except for KeyboardInterrupt here
                    response = execute_llm_command(
                        fixer_prompt, 
                        model=exec_model,
                        provider=exec_provider,
                        npc=state.npc, 
                        stream=stream_final, 
                        messages=state.messages
                    )
                except KeyboardInterrupt:
                    print(colored("\nLLM analysis interrupted by user.", "yellow"))
                    return state, colored("LLM analysis interrupted.", "red")
            
            state.messages = response['messages']     
            return state, response['response']
    else:
        full_llm_cmd = (
            f"{cmd_to_process} {stdin_input}" 
            if stdin_input 
            else cmd_to_process
        )
        path_cmd = 'The current working directory is: ' + state.current_path
        if os.path.exists(state.current_path):
            all_files = os.listdir(state.current_path)
            # Limit to first 100 files to avoid token explosion
            limited_files = all_files[:100]
            file_list = "\n".join([
                os.path.join(state.current_path, f)
                for f in limited_files
            ])
            if len(all_files) > 100:
                file_list += f"\n... and {len(all_files) - 100} more files"
            ls_files = 'Files in the current directory (full paths):\n' + file_list
        else:
            ls_files = 'No files found in the current directory.'
        platform_info = (
            f"Platform: {platform.system()} {platform.release()} "
            f"({platform.machine()})"
        )
        info = path_cmd + '\n' + ls_files + '\n' + platform_info + '\n'
        # Note: Don't append user message here - get_llm_response/check_llm_command handle it

        tools_for_llm: List[Dict[str, Any]] = []
        tool_exec_map: Dict[str, Callable] = {}
        tool_capable = model_supports_tool_calls(exec_model, exec_provider)
        if tool_capable:
            tools_for_llm, tool_exec_map = collect_llm_tools(state)
            if not tools_for_llm:
                tool_capable = False

        npc_name = (
            state.npc.name 
            if isinstance(state.npc, NPC) 
            else "Assistant"
        )
        
        with SpinnerContext(
            f"{npc_name} processing with {exec_model}", 
            style="dots_pulse"
        ):
            # Build extra_globals for jinx execution
            application_globals_for_jinx = {
                "CommandHistory": CommandHistory, 
                "load_kg_from_db": load_kg_from_db,
                "execute_rag_command": execute_rag_command, 
                "execute_brainblast_command": execute_brainblast_command,
                "load_file_contents": load_file_contents, 
                "search_web": search_web,
                "get_relevant_memories": get_relevant_memories,

                'state': state
            }
            current_module = sys.modules[__name__]
            for name, func in inspect.getmembers(current_module, inspect.isfunction):
                application_globals_for_jinx[name] = func

            # Log messages before LLM call
            logger = logging.getLogger("npcsh.state")
            logger.debug(f"[process_pipeline_command] Before LLM call: {len(state.messages)} messages, tool_capable={tool_capable}")
            for i, msg in enumerate(state.messages[-3:]):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                preview = content[:80] if isinstance(content, str) else str(type(content))
                logger.debug(f"  msg[{len(state.messages)-3+i}] role={role}: {preview}...")

            try: # Added try-except for KeyboardInterrupt here
                if tool_capable:
                    # Build kwargs - don't pass tool_choice for gemini as it doesn't support it
                    llm_kwargs = {
                        "auto_process_tool_calls": True,
                        "tools": tools_for_llm,
                        "tool_map": tool_exec_map,
                    }
                    llm_kwargs["tool_choice"] = 'auto'

                    # Agent loop: keep calling LLM until it stops making tool calls
                    # The LLM decides when it's done - npcsh just facilitates
                    iteration = 0
                    max_iterations = 50  # Safety limit to prevent infinite loops
                    total_usage = {"input_tokens": 0, "output_tokens": 0}

                    while iteration < max_iterations:
                        iteration += 1

                        llm_result = get_llm_response(
                            full_llm_cmd if iteration == 1 else None,  # Only pass prompt on first call
                            model=exec_model,
                            provider=exec_provider,
                            npc=state.npc,
                            team=state.team,
                            messages=state.messages,
                            stream=False,  # Don't stream intermediate calls
                            attachments=state.attachments if iteration == 1 else None,
                            context=info if iteration == 1 else None,
                            **llm_kwargs,
                        )

                        # Accumulate usage
                        if isinstance(llm_result, dict) and llm_result.get('usage'):
                            total_usage["input_tokens"] += llm_result['usage'].get('input_tokens', 0)
                            total_usage["output_tokens"] += llm_result['usage'].get('output_tokens', 0)

                        # Update state messages
                        old_msg_count = len(state.messages) if state.messages else 0
                        if isinstance(llm_result, dict):
                            state.messages = llm_result.get("messages", state.messages)

                        # Display tool outputs from this iteration
                        for msg in state.messages[old_msg_count:]:
                            if msg.get("role") == "tool":
                                tool_name = msg.get("name", "tool")
                                tool_content = msg.get("content", "")
                                if tool_content and tool_content.strip():
                                    # Decode escaped newlines if present
                                    if isinstance(tool_content, str):
                                        tool_content = tool_content.replace('\\n', '\n').replace('\\t', '\t')
                                    print(colored(f"\n⚡ {tool_name}:", "cyan"))
                                    lines = tool_content.split('\n')
                                    if len(lines) > 50:
                                        render_markdown('\n'.join(lines[:25]))
                                        print(colored(f"\n... ({len(lines) - 50} lines hidden) ...\n", "white", attrs=["dark"]))
                                        render_markdown('\n'.join(lines[-25:]))
                                    else:
                                        render_markdown(tool_content)

                        # Check if LLM made tool calls - if not, it's done
                        tool_calls_made = isinstance(llm_result, dict) and llm_result.get("tool_calls")
                        if not tool_calls_made:
                            # LLM is done - no more tool calls
                            break

                        # Clear the prompt for continuation calls - context is in messages
                        full_llm_cmd = None

                    # Store accumulated usage
                    if isinstance(llm_result, dict):
                        llm_result['usage'] = total_usage

                else:
                    llm_result = check_llm_command(
                        full_llm_cmd,
                        model=exec_model,
                        provider=exec_provider,
                        api_url=state.api_url,
                        api_key=state.api_key,
                        npc=state.npc,
                        team=state.team,
                        messages=state.messages,
                        images=state.attachments,
                        stream=stream_final,
                        context=info,
                        extra_globals=application_globals_for_jinx,
                        tool_capable=tool_capable,
                    )
            except KeyboardInterrupt:
                print(colored("\nLLM processing interrupted by user.", "yellow"))
                return state, colored("LLM processing interrupted.", "red")

        # Extract output and messages from llm_result
        # get_llm_response uses 'response', check_llm_command uses 'output'
        if isinstance(llm_result, dict):
            new_messages = llm_result.get("messages", state.messages)
            logger.debug(f"[process_pipeline_command] After LLM call: received {len(new_messages)} messages (was {len(state.messages)})")
            state.messages = new_messages
            output_text = llm_result.get("output") or llm_result.get("response")

            # Preserve usage info for process_result to accumulate
            output = {
                'output': output_text,
                'usage': llm_result.get('usage'),
                'model': exec_model,
                'provider': exec_provider,
            }
        else:
            output = llm_result

        if tool_capable or not review:
            return state, output
        else:
            return review_and_iterate_command(
                original_command=full_llm_cmd,
                initial_result=llm_result,
                state=state,
                exec_model=exec_model,
                exec_provider=exec_provider,
                stream_final=stream_final,
                info=info
            )


def review_and_iterate_command(
    original_command: str,
    initial_result: Any,
    state: ShellState,
    exec_model: str,
    exec_provider: str,
    stream_final: bool,
    info: str,
    max_iterations: int = 2
) -> Tuple[ShellState, Any]:
    """
    Simple iteration on LLM command result to improve quality.
    """
    
  
    if isinstance(initial_result, dict):
        current_output = initial_result.get("output")
        current_messages = initial_result.get("messages", state.messages)
    else:
        current_output = initial_result
        current_messages = state.messages
    
  
    refinement_prompt = f"""
The previous response to "{original_command}" was:
{current_output}

Please review and improve this response if needed. Provide a better, more complete answer.
"""
    
  
    refined_result = check_llm_command(
        refinement_prompt,
        model=exec_model,      
        provider=exec_provider, 
        api_url=state.api_url,
        api_key=state.api_key,
        npc=state.npc,
        team=state.team,
        messages=current_messages,
        images=state.attachments,
        stream=stream_final,
        context=info,
    )
    
  
    if isinstance(refined_result, dict):
        state.messages = refined_result.get("messages", current_messages)
        return state, refined_result.get("output", current_output)
    else:
        state.messages = current_messages
        return state, refined_result
def check_mode_switch(command:str , state: ShellState):
    if command in ['/cmd', '/agent', '/chat']:
        state.current_mode = command[1:]
        return True, state
    return False, state


def _delegate_to_npc(state: ShellState, npc_name: str, command: str, delegation_depth: int = 0) -> Tuple[ShellState, Any]:
    """
    Delegate a command to a specific NPC.

    Specialists just receive the task directly - no mention of delegation.
    Only forenpc can delegate (depth 0), and we catch @mentions in forenpc responses.
    """
    import re

    MAX_DELEGATION_DEPTH = 1  # Only allow one level of delegation

    if delegation_depth > MAX_DELEGATION_DEPTH:
        return state, {'output': "⚠ Maximum delegation depth reached."}

    if not state.team or not hasattr(state.team, 'npcs') or npc_name not in state.team.npcs:
        return state, {'output': f"⚠ NPC '{npc_name}' not found in team"}

    target_npc = state.team.npcs[npc_name]
    model_name = target_npc.model if hasattr(target_npc, 'model') else 'unknown'

    try:
        # Build tools from the NPC's jinx catalog
        tools_for_npc = None
        tool_map_for_npc = None
        if hasattr(target_npc, 'jinx_tool_catalog') and target_npc.jinx_tool_catalog:
            tools_for_npc = list(target_npc.jinx_tool_catalog.values())
            # Build tool_map that executes jinxs
            tool_map_for_npc = {}
            for jinx_name, jinx_obj in target_npc.jinxs_dict.items():
                def make_executor(jname, jobj, npc):
                    # Get expected input names from jinx
                    expected_inputs = []
                    for inp in (jobj.inputs or []):
                        if isinstance(inp, str):
                            expected_inputs.append(inp)
                        elif isinstance(inp, dict):
                            expected_inputs.append(list(inp.keys())[0])

                    def executor(**received):
                        # Map received args to expected jinx inputs
                        mapped = {}
                        if expected_inputs:
                            # If we got unexpected keys, map first value to first expected input
                            received_keys = list(received.keys())
                            for i, expected in enumerate(expected_inputs):
                                if expected in received:
                                    mapped[expected] = received[expected]
                                elif i < len(received_keys):
                                    # Map positionally
                                    mapped[expected] = received[received_keys[i]]
                        else:
                            mapped = received

                        result = npc.execute_jinx(jname, mapped)
                        return result.get('output', str(result))
                    executor.__name__ = jname
                    return executor
                tool_map_for_npc[jinx_name] = make_executor(jinx_name, jinx_obj, target_npc)

        with SpinnerContext(
            f"{npc_name} processing with {model_name}",
            style="dots_pulse"
        ):
            # Just send the command directly - don't pass team context so they don't know about other NPCs
            result = target_npc.get_llm_response(
                command,
                messages=[],  # Fresh messages - don't leak conversation history
                context={},   # No team context - they shouldn't know about teammates
                tools=tools_for_npc,
                tool_map=tool_map_for_npc,
                auto_process_tool_calls=True
            )

        output = result.get("response") or result.get("output", "")
        if result.get("messages"):
            state.messages = result["messages"]

        # Only forenpc/sibiji (depth 0) can have @mentions processed
        if delegation_depth == 0 and output and isinstance(output, str):
            # Look for @npc_name patterns in the response
            at_mention_pattern = r'@(\w+)\s*,?\s*(?:could you|can you|please|would you)?[^.!?\n]*[.!?\n]?'
            matches = re.findall(at_mention_pattern, output, re.IGNORECASE)

            for mentioned_npc in matches:
                mentioned_npc = mentioned_npc.lower()
                if mentioned_npc in state.team.npcs and mentioned_npc != npc_name:
                    # Extract what they're asking the other NPC to do
                    delegation_match = re.search(
                        rf'@{mentioned_npc}\s*,?\s*(.*?)(?:\n|$)',
                        output,
                        re.IGNORECASE
                    )
                    if delegation_match:
                        sub_request = delegation_match.group(1).strip()
                        if sub_request:
                            # Recursive delegation will show its own spinner
                            state, sub_output = _delegate_to_npc(
                                state, mentioned_npc, sub_request, delegation_depth + 1
                            )
                            # Append the sub-NPC's response
                            if isinstance(sub_output, dict):
                                sub_text = sub_output.get('output', '')
                            else:
                                sub_text = str(sub_output)
                            if sub_text:
                                output += f"\n\n--- Response from {mentioned_npc} ---\n{sub_text}"

        return state, {'output': output}

    except KeyboardInterrupt:
        print(colored(f"\n{npc_name} interrupted.", "yellow"))
        return state, {'output': colored("Interrupted.", "red")}


def execute_command(
    command: str,
    state: ShellState,
    review = False,
    router = None,
    command_history = None,
    ) -> Tuple[ShellState, Any]:
    """
    Execute a command in npcsh.

    Routes commands based on:
    1. Mode switch commands (/agent, /chat, /cmd, etc.)
    2. Slash commands (/jinx_name) -> execute via router
    3. Default mode behavior -> pipeline processing in agent mode, or jinx execution for other modes
    """
    if not command.strip():
        return state, ""

    # Unescape @ and / at start of command that were escaped to prevent misinterpretation
    # (e.g., from pasted content that starts with @ or /)
    if command.startswith('\\@') or command.startswith('\\/'):
        command = command[1:]  # Remove the escape backslash

    # Check for mode switch commands
    mode_change, state = check_mode_switch(command, state)
    if mode_change:
        print(colored(f"⚡ Switched to {state.current_mode} mode", "green"))
        return state, 'Mode changed.'

    # Check for @npc delegation syntax: @sibiji do something
    if command.startswith('@') and ' ' in command:
        npc_name = command.split()[0][1:]  # Remove @ prefix
        delegated_command = command[len(npc_name) + 2:]  # Rest of command

        # Check if NPC exists in team
        if state.team and hasattr(state.team, 'npcs') and npc_name in state.team.npcs:
            state, output = _delegate_to_npc(state, npc_name, delegated_command)
            return state, output
        else:
            print(colored(f"⚠ NPC '{npc_name}' not found in team", "yellow"))
            # Fall through to normal processing

    original_command_for_embedding = command
    commands = split_by_pipes(command)
    # Unescape pipe characters that were escaped to prevent splitting (e.g., from pasted content)
    commands = [cmd.replace('\\|', '|') for cmd in commands]

    stdin_for_next = None
    final_output = None
    current_state = state

    # Agent mode uses pipeline processing (the original behavior)
    # Other modes route to their respective jinxs
    if state.current_mode == 'agent':
        total_stages = len(commands)

        for i, cmd_segment in enumerate(commands):
            stage_num = i + 1
            stage_emoji = ["🎯", "⚙️", "🔧", "✨", "🚀"][i % 5]

            if total_stages > 1:
                print(colored(
                    f"\n{stage_emoji} Pipeline Stage {stage_num}/{total_stages}",
                    "cyan",
                    attrs=["bold"]
                ))

            is_last_command = (i == len(commands) - 1)
            stream_this_segment = state.stream_output and not is_last_command

            try:
                current_state, output = process_pipeline_command(
                    cmd_segment.strip(),
                    stdin_for_next,
                    current_state,
                    stream_final=stream_this_segment,
                    review=review,
                    router=router
                )

                # For last command, preserve full dict with usage info
                if is_last_command:
                    if total_stages > 1:
                        print(colored("✅ Pipeline complete", "green"))
                    return current_state, output

                # For intermediate stages, extract output text for piping
                if isinstance(output, dict) and 'output' in output:
                    output = output['output']

                if isinstance(output, str):
                    stdin_for_next = output
                else:
                    try:
                        if stream_this_segment:
                            full_stream_output = (
                                print_and_process_stream_with_markdown(
                                    output,
                                    state.npc.model if isinstance(state.npc, NPC) else state.chat_model,
                                    state.npc.provider if isinstance(state.npc, NPC) else state.chat_provider,
                                    show=True
                                )
                            )
                            stdin_for_next = full_stream_output
                    except Exception:
                        if output is not None:
                            try:
                                stdin_for_next = str(output)
                            except Exception:
                                stdin_for_next = None
                        else:
                            stdin_for_next = None

                if total_stages > 1:
                    print(colored(f"  → Passing to stage {stage_num + 1}", "blue"))

            except KeyboardInterrupt:
                print(colored("\nOperation interrupted by user.", "yellow"))
                return current_state, colored("Command interrupted.", "red")
            except RateLimitError:
                print(colored('Rate Limit Exceeded', 'yellow'))
                messages = current_state.messages[0:1] + current_state.messages[-2:]
                current_state.messages = messages
                import time
                print('Waiting 30s before retry...')
                time.sleep(30)
                return execute_command(command, current_state, review=review, router=router)
            except Exception as pipeline_error:
                import traceback
                traceback.print_exc()
                error_msg = colored(
                    f"❌ Error in stage {stage_num} ('{cmd_segment[:50]}...'): {pipeline_error}",
                    "red"
                )
                return current_state, error_msg

        if final_output is not None and isinstance(final_output, str):
            store_command_embeddings(original_command_for_embedding, final_output, current_state)

        return current_state, final_output

    else:
        # For non-agent modes (chat, cmd, or any custom mode), route through the jinx
        mode_jinx_name = state.current_mode

        # Check if mode jinx exists in team or router
        mode_jinx = None
        if state.team and hasattr(state.team, 'jinxs_dict') and mode_jinx_name in state.team.jinxs_dict:
            mode_jinx = state.team.jinxs_dict[mode_jinx_name]
        elif router and mode_jinx_name in router.jinx_routes:
            # Execute via router
            try:
                result = router.execute(f"/{mode_jinx_name} {command}",
                                        state=state, npc=state.npc, messages=state.messages)
                if isinstance(result, dict):
                    state.messages = result.get('messages', state.messages)
                    return state, result.get('output', '')
                return state, str(result) if result else ''
            except KeyboardInterrupt:
                print(colored(f"\n{mode_jinx_name} interrupted.", "yellow"))
                return state, colored("Interrupted.", "red")

        if mode_jinx:
            # Execute the mode jinx directly
            try:
                result = mode_jinx.execute(
                    input_values={'query': command, 'stream': state.stream_output},
                    npc=state.npc,
                    messages=state.messages,
                    extra_globals={'state': state}
                )
                if isinstance(result, dict):
                    state.messages = result.get('messages', state.messages)
                    return state, result.get('output', '')
                return state, str(result) if result else ''
            except KeyboardInterrupt:
                print(colored(f"\n{mode_jinx_name} interrupted.", "yellow"))
                return state, colored("Interrupted.", "red")

        # Fallback: if mode jinx not found, use basic LLM response
        npc_model = state.npc.model if isinstance(state.npc, NPC) and state.npc.model else None
        npc_provider = state.npc.provider if isinstance(state.npc, NPC) and state.npc.provider else None
        active_model = npc_model or state.chat_model
        active_provider = npc_provider or state.chat_provider

        with SpinnerContext(f"Processing with {active_model}", style="brain"):
            try:
                response = get_llm_response(
                    command,
                    model=active_model,
                    provider=active_provider,
                    npc=state.npc,
                    stream=state.stream_output,
                    messages=state.messages
                )
            except KeyboardInterrupt:
                print(colored("\nInterrupted.", "yellow"))
                return state, colored("Interrupted.", "red")

        state.messages = response.get('messages', state.messages)
        return state, response.get('response', '')


def setup_shell() -> Tuple[CommandHistory, Team, Optional[NPC]]:
    setup_npcsh_config()

    db_path = NPCSH_DB_PATH
    db_path = os.path.expanduser(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    command_history = CommandHistory(db_path)

    if not is_npcsh_initialized():
        print("Setting up npcsh for first use...")
        initialize_base_npcs_if_needed(db_path)
        print("Setup complete.")

    try:
        setup_readline()
        atexit.register(save_readline_history)
        atexit.register(command_history.close)
    except OSError as e:
        print(f"Warning: Failed to setup readline history: {e}", file=sys.stderr)

    project_team_path = os.path.abspath(PROJECT_NPC_TEAM_PATH)
    global_team_path = os.path.expanduser(DEFAULT_NPC_TEAM_PATH)

    if not os.path.exists(global_team_path):
        initialize_base_npcs_if_needed(db_path)
    if os.path.exists(project_team_path):
        team_dir = project_team_path
        default_forenpc_name = "forenpc"
    else:
        # No project team in this directory - use global team.
        # To create a project team, use /init from within npcsh or `npc init`.
        team_dir = global_team_path
        default_forenpc_name = "sibiji"

    if not os.path.exists(team_dir):
        print(f"Creating team directory: {team_dir}")
        os.makedirs(team_dir, exist_ok=True)
        
    team_ctx = {}
    team_ctx_path = get_team_ctx_path(team_dir)
    if team_ctx_path:
        try:
            with open(team_ctx_path, "r") as f:
                team_ctx = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load context file {os.path.basename(team_ctx_path)}: {e}")
    
    forenpc_name = team_ctx.get("forenpc", default_forenpc_name)
    if forenpc_name is None:
        forenpc_name = "sibiji"

    forenpc_path = os.path.join(team_dir, f"{forenpc_name}.npc")

    team = Team(team_path=team_dir, db_conn=command_history.engine)
    
    forenpc_obj = team.forenpc if hasattr(team, 'forenpc') and team.forenpc else None

    for npc_name, npc_obj in team.npcs.items():
        if not npc_obj.model:
            npc_obj.model = initial_state.chat_model
        if not npc_obj.provider:
            npc_obj.provider = initial_state.chat_provider

    if team.forenpc and isinstance(team.forenpc, NPC):
        if not team.forenpc.model:
            team.forenpc.model = initial_state.chat_model
        if not team.forenpc.provider:
            team.forenpc.provider = initial_state.chat_provider
    
    team_name_from_ctx = team_ctx.get("name")
    if team_name_from_ctx:
        team.name = team_name_from_ctx
    elif team_dir:
        normalized_dir = os.path.normpath(team_dir)
        basename = os.path.basename(normalized_dir)
        if basename and basename != 'npc_team':
            team.name = basename
        else:
            team.name = "npcsh"
    else:
        team.name = "npcsh"

    return command_history, team, forenpc_obj
def initialize_router_with_jinxs(team, router):
    """Load global and team Jinxs into router"""
    global_jinxs_dir = os.path.expanduser("~/.npcsh/npc_team/jinxs")
    router.load_jinx_routes(global_jinxs_dir)
    
    if team and team.team_path:
        team_jinxs_dir = os.path.join(team.team_path, "jinxs")
        if os.path.exists(team_jinxs_dir):
            router.load_jinx_routes(team_jinxs_dir)
    
    return router
                

def process_memory_approvals(command_history, memory_queue):
    pending_memories = memory_queue.get_approval_batch(max_items=5)
    
    if not pending_memories:
        return
        
    print(f"\n🧠 Processing {len(pending_memories)} memories...")
    
    try:
        trainer = MemoryTrainer()
        auto_processed = []
        need_human_review = []
        
        for memory in pending_memories:
            result = trainer.auto_approve_memory(
                memory['content'], 
                memory['context'],
                confidence_threshold=0.85
            )
            
            if result['auto_processed']:
                auto_processed.append((memory, result))
            else:
                need_human_review.append(memory)
        
        for memory, result in auto_processed:
            command_history.update_memory_status(
                memory['memory_id'], 
                result['action']
            )
            print(f"  Auto-{result['action']}: {memory['content'][:50]}... (confidence: {result['confidence']:.2f})")
        
        if need_human_review:
            approvals = memory_approval_ui(need_human_review)
            
            for approval in approvals:
                command_history.update_memory_status(
                    approval['memory_id'],
                    approval['decision'],
                    approval.get('final_memory')
                )
    
    except Exception as e:
        print(f"Auto-approval failed: {e}")
        approvals = memory_approval_ui(pending_memories)
        
        for approval in approvals:
            command_history.update_memory_status(
                approval['memory_id'],
                approval['decision'], 
                approval.get('final_memory')
            )
def process_result(
    user_input: str,
    result_state: ShellState,
    output: Any,
    command_history: CommandHistory,
):
    team_name = result_state.team.name if result_state.team else "npcsh"
    npc_name = result_state.npc.name if isinstance(result_state.npc, NPC) else "npcsh"
    
    active_npc = result_state.npc if isinstance(result_state.npc, NPC) else NPC(
        name="default", 
        model=result_state.chat_model, 
        provider=result_state.chat_provider, 
        db_conn=command_history.engine
    )
    
    save_conversation_message(
        command_history,
        result_state.conversation_id,
        "user",
        user_input,
        wd=result_state.current_path,
        model=active_npc.model,
        provider=active_npc.provider,
        npc=npc_name,
        team=team_name,
        attachments=result_state.attachments,
    )
    result_state.attachments = None

    final_output_str = None

    # FIX: Handle dict output properly
    msg_input_tokens = None
    msg_output_tokens = None
    msg_cost = None

    if isinstance(output, dict):
        # Use None-safe check to not skip empty strings
        output_content = output.get('output') if 'output' in output else output.get('response')
        model_for_stream = output.get('model', active_npc.model)
        provider_for_stream = output.get('provider', active_npc.provider)

        # Accumulate token usage if available
        if 'usage' in output:
            usage = output['usage']
            msg_input_tokens = usage.get('input_tokens', 0)
            msg_output_tokens = usage.get('output_tokens', 0)
            result_state.session_input_tokens += msg_input_tokens
            result_state.session_output_tokens += msg_output_tokens
            # Calculate cost
            from npcpy.gen.response import calculate_cost
            msg_cost = calculate_cost(
                model_for_stream,
                msg_input_tokens,
                msg_output_tokens
            )
            result_state.session_cost_usd += msg_cost

        # If output_content is still a dict, convert to string
        if isinstance(output_content, dict):
            output_content = str(output_content)
        elif output_content is None or output_content == '':
            # No output from the agent - this is fine, don't show annoying message
            output_content = None
    else:
        output_content = output
        model_for_stream = active_npc.model
        provider_for_stream = active_npc.provider

    print('\n')
    if output_content is None:
        # No output to display - tool results already shown during execution
        pass
    elif user_input == '/help':
        if isinstance(output_content, str):
            render_markdown(output_content)
        else:
            render_markdown(str(output_content))
    elif result_state.stream_output:
        # FIX: Only stream if output_content is a generator, not a string
        if isinstance(output_content, str):
            final_output_str = output_content
            render_markdown(final_output_str)
        else:
            final_output_str = print_and_process_stream_with_markdown(
                output_content,
                model_for_stream,
                provider_for_stream,
                show=True
            )
    else:
        final_output_str = str(output_content)
        render_markdown(final_output_str)
        

    # Log message state after processing
    logger = logging.getLogger("npcsh.state")
    logger.debug(f"[process_result] Before final append: {len(result_state.messages)} messages, final_output_str={'set' if final_output_str else 'None'}")

    if final_output_str:
        if result_state.messages:
            if not result_state.messages or result_state.messages[-1].get("role") != "assistant":
                result_state.messages.append({
                    "role": "assistant",
                    "content": final_output_str
                })
                logger.debug(f"[process_result] Appended assistant message, now {len(result_state.messages)} messages")

        save_conversation_message(
            command_history,
            result_state.conversation_id,
            "assistant",
            final_output_str,
            wd=result_state.current_path,
            model=active_npc.model,
            provider=active_npc.provider,
            npc=npc_name,
            team=team_name,
            input_tokens=msg_input_tokens,
            output_tokens=msg_output_tokens,
            cost=msg_cost,
        )

        result_state.turn_count += 1

        if result_state.turn_count % 10 == 0:
            approved_facts = []
            
            conversation_turn_text = f"User: {user_input}\nAssistant: {final_output_str}"
            engine = command_history.engine

            memory_examples = command_history.get_memory_examples_for_context(
                npc=npc_name,
                team=team_name, 
                directory_path=result_state.current_path
            )
            
            memory_context = format_memory_context(memory_examples)
            
            try:
                facts = get_facts(
                    conversation_turn_text,
                    model=active_npc.model,
                    provider=active_npc.provider,
                    npc=active_npc,
                    context=memory_context + 'Memories should be fully self contained. They should not use vague pronouns or words like that or this or it.  Do not generate more than 1-2 memories at a time.'
                )
                
                if facts:
                    num_memories = len(facts)
                    print(colored(
                        f"\nThere are {num_memories} potential memories. Do you want to review them now?", 
                        "cyan"
                    ))
                    review_choice = input("[y/N]: ").strip().lower()
                    
                    if review_choice == 'y':
                        memories_for_approval = []
                        for i, fact in enumerate(facts):
                            memories_for_approval.append({
                                "memory_id": f"temp_{i}",
                                "content": fact['statement'],
                                "context": f"Type: {fact.get('type', 'unknown')}, Source: {fact.get('source_text', '')}",
                                "npc": npc_name,
                                "fact_data": fact
                            })
                        
                        approvals = memory_approval_ui(memories_for_approval)
                        
                        for approval in approvals:
                            fact_data = next(
                                m['fact_data'] for m in memories_for_approval 
                                if m['memory_id'] == approval['memory_id']
                            )
                            
                            command_history.add_memory_to_database(
                                message_id=f"{result_state.conversation_id}_{len(result_state.messages)}",
                                conversation_id=result_state.conversation_id,
                                npc=npc_name,
                                team=team_name,
                                directory_path=result_state.current_path,
                                initial_memory=fact_data['statement'],
                                status=approval['decision'],
                                model=active_npc.model,
                                provider=active_npc.provider,
                                final_memory=approval.get('final_memory')
                            )
                            
                            if approval['decision'] in ['human-approved', 'human-edited']:
                                approved_fact = {
                                    'statement': approval.get('final_memory') or fact_data['statement'],
                                    'source_text': fact_data.get('source_text', ''),
                                    'type': fact_data.get('type', 'explicit'),
                                    'generation': 0
                                }
                                approved_facts.append(approved_fact)
                    else:
                        for i, fact in enumerate(facts):
                            command_history.add_memory_to_database(
                                message_id=f"{result_state.conversation_id}_{len(result_state.messages)}",
                                conversation_id=result_state.conversation_id,
                                npc=npc_name,
                                team=team_name,
                                directory_path=result_state.current_path,
                                initial_memory=fact['statement'],
                                status='skipped',
                                model=active_npc.model,
                                provider=active_npc.provider,
                                final_memory=None
                            )
                        
                        print(colored(
                            f"Marked {num_memories} memories as skipped.", 
                            "yellow"
                        ))
                    
            except Exception as e:
                print(colored(f"Memory generation error: {e}", "yellow"))

            if result_state.build_kg and approved_facts:
                try:
                    if not should_skip_kg_processing(user_input, final_output_str):
                        npc_kg = load_kg_from_db(
                            engine, 
                            team_name, 
                            npc_name, 
                            result_state.current_path
                        )
                        evolved_npc_kg, _ = kg_evolve_incremental(
                            existing_kg=npc_kg,
                            new_facts=approved_facts,
                            model=active_npc.model,
                            provider=active_npc.provider,
                            npc=active_npc,
                            get_concepts=True,
                            link_concepts_facts=result_state.kg_link_facts,
                            link_concepts_concepts=result_state.kg_link_concepts,
                            link_facts_facts=result_state.kg_link_facts_facts,
                        )
                        save_kg_to_db(
                            engine,
                            evolved_npc_kg, 
                            team_name, 
                            npc_name, 
                            result_state.current_path
                        )
                except Exception as e:
                    print(colored(
                        f"Error during real-time KG evolution: {e}", 
                        "red"
                    ))

            print(colored(
                "\nChecking for potential team improvements...", 
                "cyan"
            ))
            try:
                summary = breathe(
                    messages=result_state.messages[-20:], 
                    npc=active_npc
                )
                characterization = summary.get('output')

                if characterization and result_state.team:
                    team_ctx_path = get_team_ctx_path(
                        result_state.team.team_path
                    )
                    if not team_ctx_path:
                        team_ctx_path = os.path.join(
                            result_state.team.team_path, 
                            "team.ctx"
                        )
                    
                    ctx_data = {}
                    if os.path.exists(team_ctx_path):
                        with open(team_ctx_path, 'r') as f:
                            ctx_data = yaml.safe_load(f) or {}
                    
                    current_context = ctx_data.get('context', '')

                    prompt = f"""Based on this characterization: {characterization},
                    suggest changes (additions, deletions, edits) to the team's context. 
                    Additions need not be fully formed sentences and can simply be equations, relationships, or other plain clear items.
                    
                    Current Context: "{current_context}". 
                    
                    Respond with JSON: {{"suggestion": "Your sentence."}}"""
                    
                    response = get_llm_response(
                        prompt, 
                        npc=active_npc, 
                        format="json"
                    )
                    suggestion = response.get("response", {}).get("suggestion")

                    if suggestion:
                        new_context = (
                            current_context + " " + suggestion
                        ).strip()
                        print(colored(
                            f"{npc_name} suggests updating team context:", 
                            "yellow"
                        ))
                        print(
                            f"  - OLD: {current_context}\n  + NEW: {new_context}"
                        )
                        
                        choice = input(
                            "Apply? [y/N/e(dit)]: "
                        ).strip().lower()
                        
                        if choice == 'y':
                            ctx_data['context'] = new_context
                            with open(team_ctx_path, 'w') as f:
                                yaml.dump(ctx_data, f)
                            print(colored("Team context updated.", "green"))
                        elif choice == 'e':
                            edited_context = input(
                                f"Edit context [{new_context}]: "
                            ).strip()
                            if edited_context:
                                ctx_data['context'] = edited_context
                            else:
                                ctx_data['context'] = new_context
                            with open(team_ctx_path, 'w') as f:
                                yaml.dump(ctx_data, f)
                            print(colored(
                                "Team context updated with edits.", 
                                "green"
                            ))
                        else:
                            print("Suggestion declined.")
            except Exception as e:
                import traceback
                print(colored(
                    f"Could not generate team suggestions: {e}", 
                    "yellow"
                ))
                traceback.print_exc()
                
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
    build_kg=NPCSH_BUILD_KG, 
    api_url=NPCSH_API_URL,
)
