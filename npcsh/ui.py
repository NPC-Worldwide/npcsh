"""
UI helpers for npcsh - spinners, colors, formatting
"""
import sys
import threading
import time
from termcolor import colored


class SpinnerContext:
    """Context manager for showing a spinner during long operations"""

    SPINNER_CHARS = {
        "dots": "⣾⣽⣻⢿⡿⣟⣯⣷",
        "dots_pulse": "⣾⣽⣻⢿⡿⣟⣯⣷",
        "line": "-\\|/",
        "arrow": "←↖↑↗→↘↓↙",
        "brain": "🧠💭💡✨",
    }

    def __init__(self, message: str, style: str = "dots", delay: float = 0.1):
        self.message = message
        self.style = style
        self.delay = delay
        self.spinner = self.SPINNER_CHARS.get(style, self.SPINNER_CHARS["dots"])
        self._stop = False
        self._thread = None

    def __enter__(self):
        self._stop = False
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=0.5)
        # Clear spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 20) + '\r')
        sys.stdout.flush()

    def _spin(self):
        idx = 0
        while not self._stop:
            char = self.spinner[idx % len(self.spinner)]
            sys.stdout.write(f'\r{char} {self.message}...')
            sys.stdout.flush()
            idx += 1
            time.sleep(self.delay)


def show_thinking_animation(message="Thinking", duration=None):
    """Show a thinking animation for a fixed duration or until interrupted"""
    spinner = SpinnerContext(message)
    with spinner:
        if duration:
            time.sleep(duration)
        else:
            # Run until interrupted
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass


def orange(text: str) -> str:
    """Return text colored orange using colorama"""
    from colorama import Fore, Style
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"


def get_file_color(filepath: str) -> tuple:
    """Get color for file listing based on file type"""
    import os
    from colorama import Fore, Style

    if os.path.isdir(filepath):
        return Fore.BLUE, Style.BRIGHT
    elif os.path.islink(filepath):
        return Fore.CYAN, ""
    elif os.access(filepath, os.X_OK):
        return Fore.GREEN, Style.BRIGHT
    elif filepath.endswith(('.py', '.sh', '.bash', '.zsh')):
        return Fore.GREEN, ""
    elif filepath.endswith(('.md', '.txt', '.rst')):
        return Fore.WHITE, ""
    elif filepath.endswith(('.json', '.yaml', '.yml', '.toml')):
        return Fore.YELLOW, ""
    elif filepath.endswith(('.jpg', '.png', '.gif', '.svg', '.ico')):
        return Fore.MAGENTA, ""
    else:
        return "", ""


def format_file_listing(output: str) -> str:
    """Format file listing output with colors"""
    import os
    from colorama import Style

    lines = output.strip().split('\n')
    formatted = []

    for line in lines:
        if not line.strip():
            formatted.append(line)
            continue

        # Try to color the file part
        parts = line.rsplit('/', 1)
        if len(parts) == 2:
            path, filename = parts
            fg, style = get_file_color(line)
            formatted.append(f"{path}/{fg}{style}{filename}{Style.RESET_ALL}")
        else:
            formatted.append(line)

    return '\n'.join(formatted)


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width"""
    import textwrap
    return textwrap.fill(text, width=width)
