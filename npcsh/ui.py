"""
UI helpers for npcsh - spinners, colors, formatting
"""
import sys
import threading
import time
import collections
import shutil
import os
import signal
from termcolor import colored

# Global reference to current active spinner for sub-agent updates
_current_spinner = None

def get_current_spinner():
    """Get the currently active spinner, if any."""
    return _current_spinner

# Lock that all threads must hold before writing to stdout so that spinner
# updates and BottomBar redraws never interleave.
_stdout_lock = threading.Lock()

# Set to the active BottomBar instance while a processing phase is running.
_active_bottom_bar = None


class BottomBar:
    """Invisible input buffer that captures keystrokes during processing.

    While a command is running the user can type their next message.
    Pressing Enter queues the message; Ctrl-C / ESC interrupts the running
    command.  No scroll region or visual bar — just keystroke capture so
    output is never eaten.  The spinner shows the queued count instead.
    """

    def __init__(self):
        self.queue = collections.deque()
        self._stop = threading.Event()
        self._pause_req = threading.Event()
        self._pause_ack = threading.Event()
        self._thread = None
        self._buf = ""

    # ── public API ──────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self._pause_req.clear()
        self._pause_ack.clear()
        self._buf = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._pause_req.clear()
        if self._thread:
            self._thread.join(timeout=0.5)

    def pause(self):
        """Temporarily restore the terminal to cooked mode for input() calls."""
        self._pause_req.set()
        self._pause_ack.wait(timeout=1.0)

    def resume(self):
        """Re-enter cbreak mode after an input() call."""
        self._pause_ack.clear()
        self._pause_req.clear()

    # ── internal ────────────────────────────────────────────────────────────

    def _run(self):
        try:
            import termios
            import tty
            import select as _sel

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            def _enter_cbreak():
                tty.setcbreak(fd)
                s = termios.tcgetattr(fd)
                s[3] &= ~termios.ISIG        # Ctrl-C → \x03, not SIGINT
                termios.tcsetattr(fd, termios.TCSADRAIN, s)

            try:
                _enter_cbreak()

                while not self._stop.is_set():
                    if self._pause_req.is_set():
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        self._pause_ack.set()
                        while self._pause_req.is_set() and not self._stop.is_set():
                            time.sleep(0.05)
                        if not self._stop.is_set():
                            _enter_cbreak()
                        continue

                    ready, _, _ = _sel.select([sys.stdin], [], [], 0.15)
                    if not ready:
                        continue

                    c = sys.stdin.read(1)
                    if not c:
                        break

                    if c in ('\r', '\n'):
                        if self._buf.strip():
                            self.queue.append(self._buf.strip())
                            # Update spinner to show queued count
                            spinner = get_current_spinner()
                            if spinner:
                                spinner.set_status(f"[{len(self.queue)} queued]")
                        self._buf = ""
                    elif c in ('\x03', '\x1b'):     # Ctrl-C or ESC → interrupt
                        self._stop.set()
                        os.kill(os.getpid(), signal.SIGINT)
                        break
                    elif c in ('\x7f', '\x08'):     # Backspace
                        if self._buf:
                            self._buf = self._buf[:-1]
                    elif ord(c) >= 32:              # Printable character
                        self._buf += c

            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        except Exception:
            pass

class SpinnerContext:
    """Context manager for showing a spinner during long operations.

    Supports ESC key to interrupt (raises KeyboardInterrupt).
    Tracks elapsed time and token counts.
    """

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
        self._key_thread = None
        self._interrupted = False
        self._old_settings = None
        self._start_time = None
        self._tokens_in = 0
        self._tokens_out = 0
        self._status_msg = ""

    def update_tokens(self, tokens_in: int = 0, tokens_out: int = 0):
        """Update token counts displayed in spinner."""
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out

    def set_status(self, msg: str):
        """Set additional status message."""
        self._status_msg = msg

    def set_message(self, msg: str):
        """Update the main spinner message (e.g., when delegating to sub-agent)."""
        self.message = msg

    def __enter__(self):
        global _current_spinner
        _current_spinner = self
        self._stop = False
        self._interrupted = False
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        # Only start the ESC listener when BottomBar is not managing stdin.
        # (BottomBar forwards Ctrl-C/ESC as SIGINT itself.)
        if _active_bottom_bar is None:
            self._key_thread = threading.Thread(target=self._listen_for_esc, daemon=True)
            self._key_thread.start()
        return self

    def __exit__(self, *args):
        global _current_spinner
        _current_spinner = None
        self._stop = True
        if self._thread:
            self._thread.join(timeout=0.5)
        # Wait for key listener to restore terminal settings
        if self._key_thread:
            self._key_thread.join(timeout=0.5)
        # Clear spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 60) + '\r')
        sys.stdout.flush()
        # Check if we were interrupted by ESC
        if self._interrupted:
            raise KeyboardInterrupt("ESC pressed")

    def _listen_for_esc(self):
        """Listen for ESC key press to interrupt processing."""
        try:
            import termios
            import tty
            import select
            import signal
            import os

            fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self._stop:
                    # Check if input is available (non-blocking)
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        ch = sys.stdin.read(1)
                        if ch == '\x1b':  # ESC key
                            self._interrupted = True
                            self._stop = True
                            # Send SIGINT to main thread to interrupt blocking calls
                            os.kill(os.getpid(), signal.SIGINT)
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)
        except Exception:
            # If we can't set up terminal raw mode (e.g., not a tty), just skip ESC detection
            pass

    def _spin(self):
        idx = 0
        while not self._stop:
            char = self.spinner[idx % len(self.spinner)]

            # Build status line with timer
            elapsed = time.time() - self._start_time if self._start_time else 0
            mins, secs = divmod(int(elapsed), 60)
            timer_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"

            # Token info if available
            token_str = ""
            if self._tokens_in or self._tokens_out:
                token_str = colored(f" [{self._tokens_in}→{self._tokens_out} tok]", "cyan")

            # Additional status
            status_str = ""
            if self._status_msg:
                status_str = colored(f" {self._status_msg}", "yellow")

            hint = colored(" (ESC to cancel)", "white", attrs=["dark"])
            timer_display = colored(f" [{timer_str}]", "blue")

            line = f'\r{char} {self.message}...{timer_display}{token_str}{status_str}{hint}'
            # Clear rest of line; hold the lock so BottomBar redraws don't interleave.
            with _stdout_lock:
                sys.stdout.write(line + ' ' * 10)
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
