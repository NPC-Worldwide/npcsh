"""
Readline and tab completion for npcsh
"""
import os
from typing import List, Any

try:
    import readline
except ImportError:
    readline = None

from .config import READLINE_HISTORY_FILE


def setup_readline() -> str:
    """Set up readline with history and completion"""
    if readline is None:
        return ""

    history_file = READLINE_HISTORY_FILE

    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass

    readline.set_history_length(10000)
    readline.parse_and_bind("tab: complete")

    return history_file


def save_readline_history():
    """Save readline history to file"""
    if readline is None:
        return

    try:
        readline.write_history_file(READLINE_HISTORY_FILE)
    except Exception:
        pass


def get_path_executables() -> List[str]:
    """Get list of executables in PATH"""
    executables = set()

    path_dirs = os.environ.get("PATH", "").split(os.pathsep)

    for path_dir in path_dirs:
        if os.path.isdir(path_dir):
            try:
                for entry in os.listdir(path_dir):
                    full_path = os.path.join(path_dir, entry)
                    if os.access(full_path, os.X_OK):
                        executables.add(entry)
            except PermissionError:
                pass

    return sorted(executables)


def get_file_completions(text: str) -> List[str]:
    """Get file/directory completions for text"""
    completions = []

    if text.startswith("~"):
        search_path = os.path.expanduser(text)
    else:
        search_path = text

    # Get directory to search
    if os.path.isdir(search_path):
        dir_path = search_path
        name_prefix = ""
    else:
        dir_path = os.path.dirname(search_path) or "."
        name_prefix = os.path.basename(search_path)

    if not os.path.isdir(dir_path):
        return completions

    try:
        for entry in os.listdir(dir_path):
            if entry.startswith(name_prefix):
                full_path = os.path.join(dir_path, entry)
                if os.path.isdir(full_path):
                    completions.append(entry + "/")
                else:
                    completions.append(entry)
    except PermissionError:
        pass

    return completions


def get_slash_commands(state: Any, router: Any) -> List[str]:
    """Get list of available slash commands"""
    commands = set()

    # Built-in commands and modes
    commands.update([
        '/help', '/set', '/agent', '/chat', '/cmd',
        '/sq', '/quit', '/exit', '/clear',
    ])

    # Team jinxs
    if state.team and hasattr(state.team, 'jinxs_dict'):
        for name in state.team.jinxs_dict:
            commands.add(f'/{name}')

    # Router jinxs
    if router and hasattr(router, 'jinx_routes'):
        for name in router.jinx_routes:
            commands.add(f'/{name}')

    return sorted(commands)


def get_npc_mentions(state: Any) -> List[str]:
    """Get list of available @npc mentions"""
    npcs = set()

    # Team NPCs
    if state.team and hasattr(state.team, 'npcs'):
        for name in state.team.npcs:
            npcs.add(f'@{name}')

    # Also add forenpc if available
    if state.team and hasattr(state.team, 'forenpc') and state.team.forenpc:
        npcs.add(f'@{state.team.forenpc.name}')

    # Default NPCs if team not loaded yet
    if not npcs:
        npcs.update(['@sibiji', '@guac', '@corca', '@kadiefa', '@plonk', '@forenpc'])

    return sorted(npcs)


def is_command_position(buffer: str, begidx: int) -> bool:
    """Check if we're completing a command (vs argument)"""
    # If we're at the start or after a pipe, it's command position
    before = buffer[:begidx].strip()
    return not before or before.endswith('|')


def make_completer(shell_state: Any, router: Any):
    """Create a completer function for readline"""

    executables = get_path_executables()

    def completer(text: str, state: int):
        if readline is None:
            return None

        try:
            buffer = readline.get_line_buffer()
            begidx = readline.get_begidx()

            # Build completion options
            options = []

            # Refresh slash commands and NPC mentions each time (they may change)
            slash_commands = get_slash_commands(shell_state, router)
            npc_mentions = get_npc_mentions(shell_state)

            if text.startswith('/'):
                # Slash command completion
                options = [c for c in slash_commands if c.startswith(text)]

            elif text.startswith('@'):
                # @npc mention completion
                options = [n for n in npc_mentions if n.startswith(text)]

            elif text.startswith('~') or '/' in text or text.startswith('.'):
                # File path completion
                options = get_file_completions(text)

            elif is_command_position(buffer, begidx):
                # Command completion
                options = [e for e in executables if e.startswith(text)]

            else:
                # Default to file completion
                options = get_file_completions(text)

            if state < len(options):
                return options[state]
            return None

        except Exception:
            return None

    return completer


def readline_safe_prompt(prompt: str) -> str:
    """Make prompt safe for readline (escape ANSI codes)"""
    if readline is None:
        return prompt
    # Wrap non-printing characters
    return prompt.replace('\x1b[', '\x01\x1b[').replace('m', 'm\x02')
