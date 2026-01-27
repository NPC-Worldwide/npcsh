"""
Interactive conversation viewer for /reattach command.
Provides a TUI for browsing and selecting previous conversations.
"""
import os
import sys
import tty
import termios
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from termcolor import colored


def get_terminal_size() -> Tuple[int, int]:
    """Get terminal width and height."""
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except:
        return 80, 24


def clear_screen():
    """Clear the terminal screen."""
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


def move_cursor(row: int, col: int):
    """Move cursor to specific position."""
    sys.stdout.write(f'\033[{row};{col}H')
    sys.stdout.flush()


def hide_cursor():
    """Hide the cursor."""
    sys.stdout.write('\033[?25l')
    sys.stdout.flush()


def show_cursor():
    """Show the cursor."""
    sys.stdout.write('\033[?25h')
    sys.stdout.flush()


def getch() -> str:
    """Read a single character from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle escape sequences
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                return f'\x1b[{ch3}'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + '...'


def format_timestamp(ts: str) -> str:
    """Format timestamp for display."""
    if not ts:
        return 'unknown'
    try:
        # Try parsing ISO format
        if 'T' in ts:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = datetime.strptime(ts[:19], '%Y-%m-%d %H:%M:%S')

        now = datetime.now()
        diff = now - dt.replace(tzinfo=None)

        if diff.days == 0:
            return f"Today {dt.strftime('%H:%M')}"
        elif diff.days == 1:
            return f"Yesterday {dt.strftime('%H:%M')}"
        elif diff.days < 7:
            return dt.strftime('%a %H:%M')
        else:
            return dt.strftime('%b %d')
    except:
        return ts[:16] if len(ts) > 16 else ts


class ConversationViewer:
    """Interactive TUI for browsing conversations."""

    def __init__(self, conversations: List[Dict], current_path: str):
        self.conversations = conversations
        self.current_path = current_path
        self.selected = 0
        self.scroll_offset = 0
        self.preview_conversation = None
        self.mode = 'list'  # 'list' or 'preview'
        self.preview_scroll = 0
        self.width, self.height = get_terminal_size()

    def draw_header(self):
        """Draw the header bar."""
        move_cursor(1, 1)
        header = f" CONVERSATIONS: {truncate(self.current_path, self.width - 20)} "
        header = header.ljust(self.width)
        sys.stdout.write(colored(header, 'white', 'on_blue', attrs=['bold']))

    def draw_help(self):
        """Draw the help bar at bottom."""
        move_cursor(self.height, 1)
        if self.mode == 'list':
            help_text = " ↑/↓:Navigate  Enter:Select  p:Preview  q:Quit "
        else:
            help_text = " ↑/↓:Scroll  b:Back  Enter:Select  q:Quit "
        help_text = help_text.ljust(self.width)
        sys.stdout.write(colored(help_text, 'white', 'on_blue'))

    def draw_conversation_list(self):
        """Draw the conversation list."""
        list_height = self.height - 4  # Header, separator, status, help

        # Calculate visible range
        if self.selected < self.scroll_offset:
            self.scroll_offset = self.selected
        elif self.selected >= self.scroll_offset + list_height:
            self.scroll_offset = self.selected - list_height + 1

        for i in range(list_height):
            row = 3 + i
            move_cursor(row, 1)

            idx = self.scroll_offset + i
            if idx >= len(self.conversations):
                sys.stdout.write(' ' * self.width)
                continue

            conv = self.conversations[idx]
            is_selected = idx == self.selected

            # Format conversation line
            convo_id = conv.get('conversation_id', '')[:12]
            msg_count = conv.get('msg_count', 0)
            last_msg = format_timestamp(conv.get('last_msg', ''))
            npcs = conv.get('npcs', 'default')
            if npcs and len(npcs) > 15:
                npcs = npcs[:12] + '...'

            # Build line
            prefix = '>' if is_selected else ' '
            line = f"{prefix} {convo_id:<14} {msg_count:>4} msgs  {last_msg:<15}  {npcs}"
            line = truncate(line, self.width - 1)
            line = line.ljust(self.width - 1)

            if is_selected:
                sys.stdout.write(colored(line, 'black', 'on_white', attrs=['bold']))
            else:
                sys.stdout.write(line)

        # Draw separator
        move_cursor(self.height - 2, 1)
        sys.stdout.write(colored('─' * self.width, 'grey'))

        # Draw status
        move_cursor(self.height - 1, 1)
        if self.conversations:
            conv = self.conversations[self.selected]
            full_id = conv.get('conversation_id', '')
            status = f" ID: {full_id}"
        else:
            status = " No conversations found"
        sys.stdout.write(truncate(status, self.width).ljust(self.width))

    def draw_preview(self):
        """Draw conversation preview."""
        if not self.preview_conversation:
            return

        preview_height = self.height - 4
        messages = self.preview_conversation

        # Draw messages
        line_num = 0
        for msg in messages:
            if line_num >= self.preview_scroll + preview_height:
                break
            if line_num < self.preview_scroll:
                line_num += 1
                continue

            row = 3 + (line_num - self.preview_scroll)
            move_cursor(row, 1)

            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200].replace('\n', ' ')

            if role == 'user':
                prefix = colored('You: ', 'green', attrs=['bold'])
            elif role == 'assistant':
                prefix = colored('AI:  ', 'blue', attrs=['bold'])
            else:
                prefix = colored(f'{role}: ', 'grey')

            line = truncate(content, self.width - 8)
            sys.stdout.write(prefix + line.ljust(self.width - 6))
            line_num += 1

        # Clear remaining lines
        for i in range(line_num - self.preview_scroll, preview_height):
            move_cursor(3 + i, 1)
            sys.stdout.write(' ' * self.width)

        # Draw separator and status
        move_cursor(self.height - 2, 1)
        sys.stdout.write(colored('─' * self.width, 'grey'))
        move_cursor(self.height - 1, 1)
        status = f" Preview: {len(messages)} messages (scroll: {self.preview_scroll})"
        sys.stdout.write(truncate(status, self.width).ljust(self.width))

    def draw(self):
        """Draw the full interface."""
        self.draw_header()
        if self.mode == 'list':
            self.draw_conversation_list()
        else:
            self.draw_preview()
        self.draw_help()
        sys.stdout.flush()

    def load_preview(self, fetch_messages_func):
        """Load messages for preview."""
        if not self.conversations:
            return
        conv = self.conversations[self.selected]
        convo_id = conv.get('conversation_id')
        if convo_id and fetch_messages_func:
            self.preview_conversation = fetch_messages_func(convo_id)
            self.preview_scroll = 0

    def run(self, fetch_messages_func=None) -> Optional[str]:
        """
        Run the interactive viewer.

        Args:
            fetch_messages_func: Function to fetch messages for a conversation_id

        Returns:
            Selected conversation_id or None if cancelled
        """
        if not self.conversations:
            print(colored("No conversations found for this path.", 'yellow'))
            return None

        old_settings = None
        try:
            # Setup terminal
            old_settings = termios.tcgetattr(sys.stdin.fileno())
            hide_cursor()
            clear_screen()

            while True:
                self.draw()

                key = getch()

                if key == 'q' or key == '\x03':  # q or Ctrl+C
                    return None

                elif key == '\x1b[A':  # Up
                    if self.mode == 'list':
                        if self.selected > 0:
                            self.selected -= 1
                    else:
                        if self.preview_scroll > 0:
                            self.preview_scroll -= 1

                elif key == '\x1b[B':  # Down
                    if self.mode == 'list':
                        if self.selected < len(self.conversations) - 1:
                            self.selected += 1
                    else:
                        self.preview_scroll += 1

                elif key == '\r' or key == '\n':  # Enter
                    if self.conversations:
                        return self.conversations[self.selected].get('conversation_id')

                elif key == 'p' and self.mode == 'list':
                    self.load_preview(fetch_messages_func)
                    if self.preview_conversation:
                        self.mode = 'preview'
                        clear_screen()

                elif key == 'b' and self.mode == 'preview':
                    self.mode = 'list'
                    clear_screen()

                elif key == 'j':  # vim-style down
                    if self.mode == 'list' and self.selected < len(self.conversations) - 1:
                        self.selected += 1
                    elif self.mode == 'preview':
                        self.preview_scroll += 1

                elif key == 'k':  # vim-style up
                    if self.mode == 'list' and self.selected > 0:
                        self.selected -= 1
                    elif self.mode == 'preview' and self.preview_scroll > 0:
                        self.preview_scroll -= 1

        except Exception:
            return None
        finally:
            # Restore terminal
            show_cursor()
            clear_screen()
            if old_settings:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)


def launch_conversation_viewer(
    db_path: str,
    target_path: str,
    limit: int = 50
) -> Optional[str]:
    """
    Launch the conversation viewer and return selected conversation_id.

    Args:
        db_path: Path to the npcsh database
        target_path: Directory path to filter conversations
        limit: Maximum number of conversations to show

    Returns:
        Selected conversation_id or None
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(f'sqlite:///{db_path}')

    # Fetch conversations
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT conversation_id, directory_path,
                   MIN(timestamp) as started,
                   MAX(timestamp) as last_msg,
                   COUNT(*) as msg_count,
                   GROUP_CONCAT(DISTINCT npc) as npcs
            FROM conversation_history
            WHERE directory_path = :path OR directory_path LIKE :path_pattern
            GROUP BY conversation_id
            ORDER BY last_msg DESC
            LIMIT :limit
        """), {"path": target_path, "path_pattern": target_path + "/%", "limit": limit})

        conversations = []
        for row in result.fetchall():
            conversations.append({
                'conversation_id': row[0],
                'directory_path': row[1],
                'started': row[2],
                'last_msg': row[3],
                'msg_count': row[4],
                'npcs': row[5]
            })

    def fetch_messages(convo_id: str) -> List[Dict]:
        """Fetch messages for a conversation."""
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT role, content, timestamp, npc
                FROM conversation_history
                WHERE conversation_id = :convo_id
                ORDER BY timestamp ASC
                LIMIT 100
            """), {"convo_id": convo_id})
            return [dict(row._mapping) for row in result.fetchall()]

    viewer = ConversationViewer(conversations, target_path)
    return viewer.run(fetch_messages_func=fetch_messages)
