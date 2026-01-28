"""
npcsh configuration management
"""
import os
import importlib.metadata

# Version
try:
    VERSION = importlib.metadata.version("npcsh")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"

# Default paths
DEFAULT_NPC_TEAM_PATH = "~/.npcsh/npc_team"
PROJECT_NPC_TEAM_PATH = "./npc_team"
READLINE_HISTORY_FILE = os.path.expanduser("~/.npcsh_history")

# Environment defaults
NPCSH_CHAT_MODEL = os.environ.get("NPCSH_CHAT_MODEL", "gemma3:4b")
NPCSH_CHAT_PROVIDER = os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
NPCSH_DB_PATH = os.path.expanduser(
    os.environ.get("NPCSH_DB_PATH", "~/npcsh_history.db")
)
NPCSH_VECTOR_DB_PATH = os.path.expanduser(
    os.environ.get("NPCSH_VECTOR_DB_PATH", "~/npcsh_chroma.db")
)
NPCSH_DEFAULT_MODE = os.environ.get("NPCSH_DEFAULT_MODE", "agent")
NPCSH_VISION_MODEL = os.environ.get("NPCSH_VISION_MODEL", "gemma3:4b")
NPCSH_VISION_PROVIDER = os.environ.get("NPCSH_VISION_PROVIDER", "ollama")
NPCSH_IMAGE_GEN_MODEL = os.environ.get(
    "NPCSH_IMAGE_GEN_MODEL", "runwayml/stable-diffusion-v1-5"
)
NPCSH_IMAGE_GEN_PROVIDER = os.environ.get("NPCSH_IMAGE_GEN_PROVIDER", "diffusers")
NPCSH_VIDEO_GEN_MODEL = os.environ.get(
    "NPCSH_VIDEO_GEN_MODEL", "damo-vilab/text-to-video-ms-1.7b"
)
NPCSH_VIDEO_GEN_PROVIDER = os.environ.get("NPCSH_VIDEO_GEN_PROVIDER", "diffusers")
NPCSH_EMBEDDING_MODEL = os.environ.get("NPCSH_EMBEDDING_MODEL", "nomic-embed-text")
NPCSH_EMBEDDING_PROVIDER = os.environ.get("NPCSH_EMBEDDING_PROVIDER", "ollama")
NPCSH_REASONING_MODEL = os.environ.get("NPCSH_REASONING_MODEL", "deepseek-r1")
NPCSH_REASONING_PROVIDER = os.environ.get("NPCSH_REASONING_PROVIDER", "ollama")
NPCSH_STREAM_OUTPUT = os.environ.get("NPCSH_STREAM_OUTPUT", "0") == "1"
NPCSH_API_URL = os.environ.get("NPCSH_API_URL", None)
NPCSH_SEARCH_PROVIDER = os.environ.get("NPCSH_SEARCH_PROVIDER", "duckduckgo")
NPCSH_BUILD_KG = os.environ.get("NPCSH_BUILD_KG", "1") != "0"
NPCSH_EDIT_APPROVAL = os.environ.get("NPCSH_EDIT_APPROVAL", "off")  # off, interactive, auto


def get_shell_config_file() -> str:
    """Get the path to the user's shell config file"""
    shell = os.environ.get("SHELL", "/bin/bash")

    if "zsh" in shell:
        return os.path.expanduser("~/.zshrc")
    elif "fish" in shell:
        return os.path.expanduser("~/.config/fish/config.fish")
    else:
        return os.path.expanduser("~/.bashrc")


def get_npcshrc_path() -> str:
    """Get path to npcshrc file"""
    return os.path.expanduser("~/.npcshrc")


def get_npcshrc_path_windows():
    """Get npcshrc path on Windows"""
    return os.path.expanduser("~/.npcshrc")


def ensure_npcshrc_exists() -> str:
    """Ensure npcshrc file exists and return its path"""
    npcshrc_path = get_npcshrc_path()

    if not os.path.exists(npcshrc_path):
        default_content = f"""# npcsh configuration file
export NPCSH_CHAT_MODEL="{NPCSH_CHAT_MODEL}"
export NPCSH_CHAT_PROVIDER="{NPCSH_CHAT_PROVIDER}"
export NPCSH_VISION_MODEL="{NPCSH_VISION_MODEL}"
export NPCSH_VISION_PROVIDER="{NPCSH_VISION_PROVIDER}"
export NPCSH_EMBEDDING_MODEL="{NPCSH_EMBEDDING_MODEL}"
export NPCSH_EMBEDDING_PROVIDER="{NPCSH_EMBEDDING_PROVIDER}"
export NPCSH_SEARCH_PROVIDER="{NPCSH_SEARCH_PROVIDER}"
export NPCSH_DEFAULT_MODE="{NPCSH_DEFAULT_MODE}"
export NPCSH_STREAM_OUTPUT="0"
"""
        with open(npcshrc_path, 'w') as f:
            f.write(default_content)

    return npcshrc_path


def add_npcshrc_to_shell_config() -> None:
    """Add sourcing of npcshrc to shell config if not present"""
    shell_config = get_shell_config_file()
    npcshrc_path = get_npcshrc_path()

    source_line = f'source "{npcshrc_path}"'

    if os.path.exists(shell_config):
        with open(shell_config, 'r') as f:
            content = f.read()
        if npcshrc_path not in content and '.npcshrc' not in content:
            with open(shell_config, 'a') as f:
                f.write(f"\n# Source npcsh configuration\n{source_line}\n")


def setup_npcsh_config() -> None:
    """Set up npcsh configuration"""
    ensure_npcshrc_exists()
    add_npcshrc_to_shell_config()


def is_npcsh_initialized() -> bool:
    """Check if npcsh has been initialized"""
    marker = os.path.expanduser("~/.npcsh/.initialized")
    return os.path.exists(marker)


def set_npcsh_initialized() -> None:
    """Mark npcsh as initialized"""
    npcsh_dir = os.path.expanduser("~/.npcsh")
    os.makedirs(npcsh_dir, exist_ok=True)
    marker = os.path.join(npcsh_dir, ".initialized")
    with open(marker, 'w') as f:
        f.write("1")


def set_npcsh_config_value(key: str, value: str) -> None:
    """Set a value in npcshrc"""
    npcshrc_path = ensure_npcshrc_exists()

    with open(npcshrc_path, 'r') as f:
        lines = f.readlines()

    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f'export {key}='):
            lines[i] = f'export {key}="{value}"\n'
            found = True
            break

    if not found:
        lines.append(f'export {key}="{value}"\n')

    with open(npcshrc_path, 'w') as f:
        f.writelines(lines)

    # Also set in current environment
    os.environ[key] = value


def get_setting_windows(key, default=None):
    """Get setting on Windows"""
    npcshrc_path = get_npcshrc_path_windows()
    if os.path.exists(npcshrc_path):
        with open(npcshrc_path, 'r') as f:
            for line in f:
                if line.strip().startswith(f'export {key}='):
                    value = line.split('=', 1)[1].strip().strip('"').strip("'")
                    return value
    return default
