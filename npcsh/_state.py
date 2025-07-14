
from colorama import Fore, Back, Style


from datetime import datetime
from dotenv import load_dotenv

import re
import os
from termcolor import colored



from typing import Dict, List
import re
import sqlite3
from datetime import datetime
import logging
import textwrap
from termcolor import colored
import sys
import platform


def get_npc_path(npc_name: str, db_path: str) -> str:
    # First, check in project npc_team directory
    project_npc_team_dir = os.path.abspath("./npc_team")
    project_npc_path = os.path.join(project_npc_team_dir, f"{npc_name}.npc")

    # Then, check in global npc_team directory
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")
    global_npc_path = os.path.join(user_npc_team_dir, f"{npc_name}.npc")

    # Check database for compiled NPCs
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            query = f"SELECT source_path FROM compiled_npcs WHERE name = '{npc_name}'"
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                return result[0]

    except Exception as e:
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

    # Fallback to file paths
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

    if is_npcsh_initialized():
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the compiled_npcs table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_npcs (
            name TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            compiled_content TEXT
        )
        """
    )

    # Get the path to the npc_team directory in the package
    package_dir = os.path.dirname(__file__)
    package_npc_team_dir = os.path.join(package_dir, "npc_team")

    

    # User's global npc_team directory
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")

    user_jinxs_dir = os.path.join(user_npc_team_dir, "jinxs")
    user_templates_dir = os.path.join(user_npc_team_dir, "templates")
    os.makedirs(user_npc_team_dir, exist_ok=True)
    os.makedirs(user_jinxs_dir, exist_ok=True)
    os.makedirs(user_templates_dir, exist_ok=True)
    # Copy NPCs from package to user directory
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

    # Copy jinxs from package to user directory
    package_jinxs_dir = os.path.join(package_npc_team_dir, "jinxs")
    if os.path.exists(package_jinxs_dir):
        for filename in os.listdir(package_jinxs_dir):
            if filename.endswith(".jinx"):
                source_jinx_path = os.path.join(package_jinxs_dir, filename)
                destination_jinx_path = os.path.join(user_jinxs_dir, filename)
                if (not os.path.exists(destination_jinx_path)) or file_has_changed(
                    source_jinx_path, destination_jinx_path
                ):
                    shutil.copy2(source_jinx_path, destination_jinx_path)
                    print(f"Copied jinx {filename} to {destination_jinx_path}")

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
    # Check the current shell
    shell = os.environ.get("SHELL", "")

    if "zsh" in shell:
        return os.path.expanduser("~/.zshrc")
    elif "bash" in shell:
        # On macOS, use .bash_profile for login shells
        if platform.system() == "Darwin":
            return os.path.expanduser("~/.bash_profile")
        else:
            return os.path.expanduser("~/.bashrc")
    else:
        # Default to .bashrc if we can't determine the shell
        return os.path.expanduser("~/.bashrc")



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
            npcshrc.write("export NPCSH_CHAT_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_CHAT_MODEL='llama3.2'\n")
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
    "continue",
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
    "help",
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
    # Common Unix commands
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
    "w",
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


interactive_commands = {
    "ipython": ["ipython"],
    "python": ["python", "-i"],
    "sqlite3": ["sqlite3"],
    "r": ["R", "--interactive"],
}


def start_interactive_session(command: list) -> int:
    """
    Starts an interactive session. Only works on Unix. On Windows, print a message and return 1.
    """
    if ON_WINDOWS or termios is None or tty is None or pty is None or select is None or signal is None:
        print("Interactive terminal sessions are not supported on Windows.")
        return 1
    # Save the current terminal settings
    old_tty = termios.tcgetattr(sys.stdin)
    try:
        # Create a pseudo-terminal
        master_fd, slave_fd = pty.openpty()

        # Start the process
        p = subprocess.Popen(
            command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            shell=True,
            preexec_fn=os.setsid,  # Create a new process group
        )

        # Set the terminal to raw mode
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

        # Wait for the process to terminate with a timeout
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(5)  # 5 second timeout
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
        # Restore the terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_tty)

    return p.returncode

def validate_bash_command(command_parts: list) -> bool:
    """
    Function Description:
        Validate if the command sequence is a valid bash command with proper arguments/flags.
    Args:
        command_parts : list : Command parts
    Keyword Args:
        None
    Returns:
        bool : bool : Boolean
    """
    if not command_parts:
        return False

    COMMAND_PATTERNS = {
        "cat": {
            "flags": ["-n", "-b", "-E", "-T", "-s", "--number", "-A", "--show-all"],
            "requires_arg": True,
        },
        "find": {
            "flags": [
                "-name",
                "-type",
                "-size",
                "-mtime",
                "-exec",
                "-print",
                "-delete",
                "-maxdepth",
                "-mindepth",
                "-perm",
                "-user",
                "-group",
            ],
            "requires_arg": True,
        },
        "who": {
            "flags": [
                "-a",
                "-b",
                "-d",
                "-H",
                "-l",
                "-p",
                "-q",
                "-r",
                "-s",
                "-t",
                "-u",
                "--all",
                "--count",
                "--heading",
            ],
            "requires_arg": False,
        },
        "open": {
            "flags": ["-a", "-e", "-t", "-f", "-F", "-W", "-n", "-g", "-h"],
            "requires_arg": True,
        },

    }

    base_command = command_parts[0]

    if base_command == 'which':
        return False # disable which arbitrarily cause the command parsing for it is too finnicky.


    if base_command not in COMMAND_PATTERNS:
        return True  # Allow other commands to pass through

    pattern = COMMAND_PATTERNS[base_command]
    args = []
    flags = []

    for i in range(1, len(command_parts)):
        part = command_parts[i]
        if part.startswith("-"):
            flags.append(part)
            if part not in pattern["flags"]:
                return False  # Invalid flag
        else:
            args.append(part)

    # Check if 'who' has any arguments (it shouldn't)
    if base_command == "who" and args:
        return False
    # Check if any required arguments are missing
    if pattern.get("requires_arg", False) and not args:
        return False

    return True


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

    # Map command to environment variable name
    var_map = {
        "model": "NPCSH_CHAT_MODEL",
        "provider": "NPCSH_CHAT_PROVIDER",
        "db_path": "NPCSH_DB_PATH",
    }

    if command not in var_map:
        return f"Unknown setting: {command}"

    env_var = var_map[command]

    # Read the current configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Check if the property exists and update it, or add it if it doesn't exist
    property_exists = False
    for i, line in enumerate(lines):
        if line.startswith(f"export {env_var}="):
            lines[i] = f"export {env_var}='{value}'\n"
            property_exists = True
            break

    if not property_exists:
        lines.append(f"export {env_var}='{value}'\n")

    # Save the updated configuration
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

    # Also set it for the current session
    os.environ["NPCSH_INITIALIZED"] = "1"
    print("NPCSH initialization flag set in .npcshrc")



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

    # Compare file modification times or contents to decide whether to update the file
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
                # Match KEY='value' or KEY="value" format
                match = re.match(r'^([A-Z_]+)\s*=\s*[\'"](.*?)[\'"]$', line)
                if match:
                    key, value = match.groups()
                    config[key] = value
    return config


def get_setting_windows(key, default=None):
    # Try environment variable first
    if env_value := os.getenv(key):
        return env_value

    # Fall back to .npcshrc file
    config = read_rc_file_windows(get_npcshrc_path_windows())
    return config.get(key, default)

NPCSH_CHAT_MODEL = os.environ.get("NPCSH_CHAT_MODEL", "llama3.2")
# print("NPCSH_CHAT_MODEL", NPCSH_CHAT_MODEL)
NPCSH_CHAT_PROVIDER = os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
# print("NPCSH_CHAT_PROVIDER", NPCSH_CHAT_PROVIDER)
NPCSH_DB_PATH = os.path.expanduser(
    os.environ.get("NPCSH_DB_PATH", "~/npcsh_history.db")
)
NPCSH_VECTOR_DB_PATH = os.path.expanduser(
    os.environ.get("NPCSH_VECTOR_DB_PATH", "~/npcsh_chroma.db")
)
#DEFAULT MODES = ['CHAT', 'AGENT', 'CODE', ]

NPCSH_DEFAULT_MODE = os.path.expanduser(os.environ.get("NPCSH_DEFAULT_MODE", "agent"))
NPCSH_VISION_MODEL = os.environ.get("NPCSH_VISION_MODEL", "llava:7b")
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
NPCSH_STREAM_OUTPUT = eval(os.environ.get("NPCSH_STREAM_OUTPUT", "0")) == 1
NPCSH_API_URL = os.environ.get("NPCSH_API_URL", None)
NPCSH_SEARCH_PROVIDER = os.environ.get("NPCSH_SEARCH_PROVIDER", "duckduckgo")

READLINE_HISTORY_FILE = os.path.expanduser("~/.npcsh_history")



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
