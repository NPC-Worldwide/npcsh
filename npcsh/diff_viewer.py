"""
Git-diff approval TUI for npcsh.
Provides interactive diff viewing with approve/reject functionality.
"""
import os
import sys
import difflib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

# Platform-specific imports
try:
    import tty
    import termios
    import select
    HAS_TTY = True
except ImportError:
    HAS_TTY = False


class HunkDecision(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class DiffHunk:
    """Represents a single diff hunk"""
    start_original: int
    count_original: int
    start_modified: int
    count_modified: int
    lines: List[str]
    header: str


@dataclass
class DiffViewerState:
    """State for the diff viewer TUI"""
    file_path: str
    original: str
    modified: str
    hunks: List[DiffHunk] = field(default_factory=list)
    decisions: Dict[int, HunkDecision] = field(default_factory=dict)
    selected_hunk: int = 0
    scroll_offset: int = 0
    mode: str = "normal"  # normal, help


def compute_diff_hunks(original: str, modified: str) -> List[DiffHunk]:
    """Compute diff hunks between original and modified content."""
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        lineterm=''
    ))

    hunks = []
    current_hunk_lines = []
    current_header = ""
    start_orig = 0
    count_orig = 0
    start_mod = 0
    count_mod = 0

    for line in diff[2:]:  # Skip the --- and +++ headers
        if line.startswith('@@'):
            # Save previous hunk if exists
            if current_hunk_lines:
                hunks.append(DiffHunk(
                    start_original=start_orig,
                    count_original=count_orig,
                    start_modified=start_mod,
                    count_modified=count_mod,
                    lines=current_hunk_lines,
                    header=current_header
                ))

            # Parse new hunk header
            current_header = line.strip()
            current_hunk_lines = []

            # Parse @@ -start,count +start,count @@
            try:
                parts = line.split('@@')[1].strip().split()
                orig_part = parts[0]  # -start,count
                mod_part = parts[1]   # +start,count

                if ',' in orig_part:
                    start_orig, count_orig = map(int, orig_part[1:].split(','))
                else:
                    start_orig = int(orig_part[1:])
                    count_orig = 1

                if ',' in mod_part:
                    start_mod, count_mod = map(int, mod_part[1:].split(','))
                else:
                    start_mod = int(mod_part[1:])
                    count_mod = 1
            except (IndexError, ValueError):
                start_orig = count_orig = start_mod = count_mod = 0
        else:
            current_hunk_lines.append(line)

    # Save last hunk
    if current_hunk_lines:
        hunks.append(DiffHunk(
            start_original=start_orig,
            count_original=count_orig,
            start_modified=start_mod,
            count_modified=count_mod,
            lines=current_hunk_lines,
            header=current_header
        ))

    return hunks


def get_terminal_size() -> Tuple[int, int]:
    """Get terminal size (width, height)."""
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except:
        return 80, 24


class DiffViewer:
    """Interactive diff viewer with approve/reject functionality."""

    def __init__(self, file_path: str, original: str, modified: str):
        self.state = DiffViewerState(
            file_path=file_path,
            original=original,
            modified=modified
        )
        self.state.hunks = compute_diff_hunks(original, modified)

        # Initialize all hunks as pending
        for i in range(len(self.state.hunks)):
            self.state.decisions[i] = HunkDecision.PENDING

    def render_screen(self):
        """Render the diff viewer screen."""
        width, height = get_terminal_size()
        out = []

        # Clear screen and move to top
        out.append("\033[2J\033[H")

        # Header
        header = f" File Edit: {self.state.file_path} "
        if len(header) > width - 4:
            header = f" ...{self.state.file_path[-width+15:]} "
        out.append(f"\033[1;1H\033[44;37;1m{'=' * width}\033[0m")
        out.append(f"\033[1;2H\033[44;37;1m{header}\033[0m")

        # Stats line
        approved = sum(1 for d in self.state.decisions.values() if d == HunkDecision.APPROVED)
        rejected = sum(1 for d in self.state.decisions.values() if d == HunkDecision.REJECTED)
        pending = sum(1 for d in self.state.decisions.values() if d == HunkDecision.PENDING)
        stats = f"Hunks: {len(self.state.hunks)} | Approved: {approved} | Rejected: {rejected} | Pending: {pending}"
        out.append(f"\033[2;1H\033[90m{stats.center(width)}\033[0m")

        # Diff content area
        content_start = 4
        content_height = height - 6

        if not self.state.hunks:
            out.append(f"\033[{content_start};2H\033[33mNo differences found.\033[0m")
        else:
            # Render current hunk
            hunk = self.state.hunks[self.state.selected_hunk]
            decision = self.state.decisions[self.state.selected_hunk]

            # Hunk header with decision indicator
            decision_indicator = {
                HunkDecision.PENDING: "\033[33m[?]\033[0m",
                HunkDecision.APPROVED: "\033[32m[+]\033[0m",
                HunkDecision.REJECTED: "\033[31m[-]\033[0m"
            }[decision]

            hunk_header = f"{decision_indicator} Hunk {self.state.selected_hunk + 1}/{len(self.state.hunks)}: {hunk.header}"
            out.append(f"\033[3;1H\033[90m{'-' * width}\033[0m")
            out.append(f"\033[3;2H{hunk_header[:width-4]}")

            # Render diff lines
            visible_lines = hunk.lines[self.state.scroll_offset:self.state.scroll_offset + content_height]

            for i, line in enumerate(visible_lines):
                row = content_start + i
                if row >= height - 2:
                    break

                # Color based on line type
                if line.startswith('+'):
                    color = "\033[32m"  # Green for additions
                elif line.startswith('-'):
                    color = "\033[31m"  # Red for deletions
                else:
                    color = "\033[0m"   # Default for context

                # Truncate long lines
                display_line = line.rstrip()[:width-2]
                out.append(f"\033[{row};1H{color}{display_line}\033[0m")

            # Scroll indicator
            if len(hunk.lines) > content_height:
                scroll_pct = (self.state.scroll_offset / (len(hunk.lines) - content_height)) * 100
                scroll_info = f"[{int(scroll_pct)}%]"
                out.append(f"\033[{content_start};{width-len(scroll_info)-1}H\033[90m{scroll_info}\033[0m")

        # Footer with keybindings
        footer_y = height - 1
        out.append(f"\033[{footer_y};1H\033[90m{'-' * width}\033[0m")

        keys = "[a] Approve  [r] Reject  [A] Approve All  [R] Reject All  [j/k] Hunks  [q] Done  [?] Help"
        out.append(f"\033[{height};1H\033[90m{keys[:width]}\033[0m")

        sys.stdout.write(''.join(out))
        sys.stdout.flush()

    def handle_input(self, c: str) -> bool:
        """Handle input character. Returns False to exit."""
        if c == 'q':
            return False

        elif c == 'a':  # Approve current hunk
            self.state.decisions[self.state.selected_hunk] = HunkDecision.APPROVED
            if self.state.selected_hunk < len(self.state.hunks) - 1:
                self.state.selected_hunk += 1
                self.state.scroll_offset = 0

        elif c == 'r':  # Reject current hunk
            self.state.decisions[self.state.selected_hunk] = HunkDecision.REJECTED
            if self.state.selected_hunk < len(self.state.hunks) - 1:
                self.state.selected_hunk += 1
                self.state.scroll_offset = 0

        elif c == 'A':  # Approve all
            for i in range(len(self.state.hunks)):
                self.state.decisions[i] = HunkDecision.APPROVED

        elif c == 'R':  # Reject all
            for i in range(len(self.state.hunks)):
                self.state.decisions[i] = HunkDecision.REJECTED

        elif c == 'j' or c == '\x1b':  # Down/next hunk (or escape sequence)
            if c == '\x1b':
                # Handle escape sequences
                if HAS_TTY and select.select([sys.stdin], [], [], 0.05)[0]:
                    c2 = sys.stdin.read(1)
                    if c2 == '[':
                        c3 = sys.stdin.read(1)
                        if c3 == 'B':  # Down arrow
                            if self.state.selected_hunk < len(self.state.hunks) - 1:
                                self.state.selected_hunk += 1
                                self.state.scroll_offset = 0
                        elif c3 == 'A':  # Up arrow
                            if self.state.selected_hunk > 0:
                                self.state.selected_hunk -= 1
                                self.state.scroll_offset = 0
            else:
                if self.state.selected_hunk < len(self.state.hunks) - 1:
                    self.state.selected_hunk += 1
                    self.state.scroll_offset = 0

        elif c == 'k':  # Up/previous hunk
            if self.state.selected_hunk > 0:
                self.state.selected_hunk -= 1
                self.state.scroll_offset = 0

        elif c == 'n':  # Next hunk (same as j)
            if self.state.selected_hunk < len(self.state.hunks) - 1:
                self.state.selected_hunk += 1
                self.state.scroll_offset = 0

        elif c == 'p':  # Previous hunk (same as k)
            if self.state.selected_hunk > 0:
                self.state.selected_hunk -= 1
                self.state.scroll_offset = 0

        elif c == ' ':  # Scroll down within hunk
            if self.state.hunks:
                hunk = self.state.hunks[self.state.selected_hunk]
                _, height = get_terminal_size()
                content_height = height - 6
                max_scroll = max(0, len(hunk.lines) - content_height)
                self.state.scroll_offset = min(self.state.scroll_offset + 5, max_scroll)

        elif c == 'b':  # Scroll up within hunk
            self.state.scroll_offset = max(0, self.state.scroll_offset - 5)

        return True

    def apply_decisions(self) -> str:
        """Apply decisions and return the resulting content."""
        if not self.state.hunks:
            return self.state.modified

        # If all hunks approved, return modified
        if all(d == HunkDecision.APPROVED for d in self.state.decisions.values()):
            return self.state.modified

        # If all hunks rejected, return original
        if all(d == HunkDecision.REJECTED for d in self.state.decisions.values()):
            return self.state.original

        # Partial application - reconstruct from decisions
        # This is complex - for now, we'll use a simple approach:
        # If any hunk is rejected, we need to carefully reconstruct

        result_lines = self.state.original.splitlines(keepends=True)
        offset = 0  # Track line number offset from applied changes

        for i, hunk in enumerate(self.state.hunks):
            if self.state.decisions[i] == HunkDecision.APPROVED:
                # Apply this hunk
                start = hunk.start_original - 1 + offset

                # Count removals and additions in this hunk
                removals = [ln[1:] for ln in hunk.lines if ln.startswith('-')]
                additions = [ln[1:] for ln in hunk.lines if ln.startswith('+')]

                # Remove old lines
                del result_lines[start:start + len(removals)]

                # Insert new lines
                for j, line in enumerate(additions):
                    if not line.endswith('\n'):
                        line += '\n'
                    result_lines.insert(start + j, line)

                # Update offset
                offset += len(additions) - len(removals)

        return ''.join(result_lines)

    def run(self) -> Dict[str, any]:
        """Run the interactive diff viewer. Returns approval decisions."""
        if not HAS_TTY:
            print("TTY not available - cannot run interactive diff viewer")
            return {
                "approved": False,
                "decisions": {},
                "content": self.state.original
            }

        if not self.state.hunks:
            return {
                "approved": True,
                "decisions": {},
                "content": self.state.modified
            }

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            sys.stdout.write('\033[?25l')  # Hide cursor

            self.render_screen()

            while True:
                c = sys.stdin.read(1)
                if not self.handle_input(c):
                    break
                self.render_screen()

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write('\033[?25h')  # Show cursor
            sys.stdout.write('\033[2J\033[H')  # Clear screen
            sys.stdout.flush()

        # Determine if approved
        all_approved = all(d == HunkDecision.APPROVED for d in self.state.decisions.values())
        any_approved = any(d == HunkDecision.APPROVED for d in self.state.decisions.values())

        return {
            "approved": any_approved,
            "all_approved": all_approved,
            "decisions": {i: d.value for i, d in self.state.decisions.items()},
            "content": self.apply_decisions()
        }


def show_diff_approval(file_path: str, original: str, modified: str) -> Dict[str, any]:
    """
    Show an interactive diff approval dialog.

    Args:
        file_path: Path to the file being edited
        original: Original file content
        modified: Modified file content

    Returns:
        Dict with:
        - approved: bool - whether any changes were approved
        - all_approved: bool - whether all changes were approved
        - content: str - the resulting content after applying decisions
        - decisions: dict - per-hunk decisions
    """
    viewer = DiffViewer(file_path, original, modified)
    return viewer.run()


def quick_diff_preview(original: str, modified: str, max_lines: int = 20) -> str:
    """
    Generate a quick text-based diff preview (non-interactive).

    Args:
        original: Original content
        modified: Modified content
        max_lines: Maximum lines to show

    Returns:
        Colored diff string
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        lineterm=''
    ))

    if not diff:
        return "No changes"

    result = []
    for line in diff[:max_lines]:
        if line.startswith('+') and not line.startswith('+++'):
            result.append(f"\033[32m{line.rstrip()}\033[0m")
        elif line.startswith('-') and not line.startswith('---'):
            result.append(f"\033[31m{line.rstrip()}\033[0m")
        elif line.startswith('@@'):
            result.append(f"\033[36m{line.rstrip()}\033[0m")
        else:
            result.append(line.rstrip())

    if len(diff) > max_lines:
        result.append(f"\033[90m... ({len(diff) - max_lines} more lines)\033[0m")

    return '\n'.join(result)
