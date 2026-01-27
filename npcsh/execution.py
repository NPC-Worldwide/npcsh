"""
Command execution utilities for npcsh
"""
import os
import shutil
import subprocess
from typing import List, Tuple, Any, Optional

from termcolor import colored


# Commands that require interactive terminal handling
TERMINAL_EDITORS = ['vim', 'nvim', 'nano', 'vi', 'emacs', 'less', 'more', 'man']

# Interactive commands that need special handling (command -> args)
INTERACTIVE_COMMANDS = {
    'ipython': ['ipython'],
    'python': ['python', '-i'],
    'python3': ['python3', '-i'],
    'node': ['node'],
    'irb': ['irb'],
    'ghci': ['ghci'],
    'mysql': ['mysql'],
    'psql': ['psql'],
    'sqlite3': ['sqlite3'],
    'redis-cli': ['redis-cli'],
    'mongo': ['mongo'],
    'ssh': ['ssh'],
    'telnet': ['telnet'],
    'ftp': ['ftp'],
    'sftp': ['sftp'],
    'top': ['top'],
    'htop': ['htop'],
    'watch': ['watch'],
    'r': ['R', '--interactive'],
}


def validate_bash_command(command_parts: List[str]) -> bool:
    """
    Check if the command is a valid bash command.

    Returns True if the command exists in PATH or is a shell builtin.
    """
    if not command_parts:
        return False

    cmd = command_parts[0]

    # Check shell builtins
    builtins = {'cd', 'pwd', 'echo', 'export', 'source', 'alias', 'unalias',
                'history', 'set', 'unset', 'read', 'eval', 'exec', 'exit',
                'return', 'shift', 'trap', 'wait', 'jobs', 'fg', 'bg',
                'kill', 'ulimit', 'umask', 'type', 'hash', 'true', 'false'}

    if cmd in builtins:
        return True

    # Check if command exists in PATH
    return shutil.which(cmd) is not None


def handle_bash_command(
    cmd_parts: List[str],
    full_cmd: str,
    stdin_input: Optional[str],
    state: Any
) -> Tuple[bool, str]:
    """
    Execute a bash command and return the result.

    Args:
        cmd_parts: Parsed command parts
        full_cmd: Full command string
        stdin_input: Input to pipe to command
        state: Shell state

    Returns:
        Tuple of (success, output)
    """
    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=state.current_path,
            input=stdin_input,
            timeout=300
        )

        if result.returncode == 0:
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"
            return True, output.strip()
        else:
            error = result.stderr or result.stdout or f"Command exited with code {result.returncode}"
            return False, error.strip()

    except subprocess.TimeoutExpired:
        return False, "Command timed out after 5 minutes"
    except Exception as e:
        return False, str(e)


def open_terminal_editor(command: str) -> str:
    """Open a terminal editor command interactively"""
    try:
        subprocess.run(command, shell=True)
        return "Editor session completed"
    except Exception as e:
        return f"Editor error: {e}"


def handle_cd_command(cmd_parts: List[str], state: Any) -> Tuple[Any, str]:
    """Handle the cd command"""
    if len(cmd_parts) < 2:
        new_path = os.path.expanduser("~")
    else:
        new_path = os.path.expanduser(cmd_parts[1])

    if not os.path.isabs(new_path):
        new_path = os.path.join(state.current_path, new_path)

    new_path = os.path.normpath(new_path)

    if os.path.isdir(new_path):
        state.current_path = new_path
        os.chdir(new_path)
        return state, f"Changed to: {new_path}"
    else:
        return state, colored(f"Directory not found: {new_path}", "red")


def handle_interactive_command(cmd_parts: List[str], state: Any) -> Tuple[Any, str]:
    """Handle interactive commands by running them in a subprocess"""
    command = ' '.join(cmd_parts)
    try:
        subprocess.run(command, shell=True, cwd=state.current_path)
        return state, f"Interactive session ({cmd_parts[0]}) completed"
    except KeyboardInterrupt:
        return state, colored("Session interrupted", "yellow")
    except Exception as e:
        return state, colored(f"Error: {e}", "red")


def start_interactive_session(command: str) -> int:
    """Start an interactive shell session"""
    try:
        return subprocess.call(command, shell=True)
    except Exception as e:
        print(colored(f"Error starting session: {e}", "red"))
        return 1


def list_directory(args: List[str]) -> None:
    """List directory contents with formatting"""
    path = args[0] if args else "."
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        print(colored(f"Path not found: {path}", "red"))
        return

    if os.path.isfile(path):
        print(path)
        return

    try:
        entries = os.listdir(path)
        entries.sort()

        for entry in entries:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                print(colored(entry + "/", "blue", attrs=["bold"]))
            elif os.access(full_path, os.X_OK):
                print(colored(entry, "green", attrs=["bold"]))
            else:
                print(entry)

    except PermissionError:
        print(colored(f"Permission denied: {path}", "red"))
