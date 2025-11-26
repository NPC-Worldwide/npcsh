"""
Command parsing utilities for npcsh
"""
import shlex
from typing import List


def split_by_pipes(command: str) -> List[str]:
    """
    Split a command by pipes, preserving quoted strings.

    Examples:
        'foo | bar' -> ['foo', 'bar']
        'foo "hello|world" | bar' -> ['foo "hello|world"', 'bar']
    """
    result = []
    current = []
    in_single_quote = False
    in_double_quote = False
    i = 0

    while i < len(command):
        char = command[i]

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(char)
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(char)
        elif char == '|' and not in_single_quote and not in_double_quote:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(char)

        i += 1

    # Add final segment
    if current:
        result.append(''.join(current).strip())

    return [s for s in result if s]


def parse_command_safely(cmd: str) -> List[str]:
    """
    Safely parse a command string into parts using shlex.

    Returns an empty list on parse errors.
    """
    try:
        return shlex.split(cmd)
    except ValueError:
        # Handle unmatched quotes, etc
        return cmd.split()


def parse_generic_command_flags(parts: List[str]) -> tuple:
    """
    Parse command flags in a generic way.

    Returns:
        Tuple of (parsed_flags dict, remaining_parts list)
    """
    parsed_flags = {}
    remaining = []

    i = 0
    while i < len(parts):
        part = parts[i]

        if part.startswith('--'):
            key = part[2:]
            if '=' in key:
                key, value = key.split('=', 1)
                parsed_flags[key] = _try_convert_type(value)
            elif i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                parsed_flags[key] = _try_convert_type(parts[i + 1])
                i += 1
            else:
                parsed_flags[key] = True
        elif part.startswith('-') and len(part) == 2:
            key = part[1]
            if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                parsed_flags[key] = _try_convert_type(parts[i + 1])
                i += 1
            else:
                parsed_flags[key] = True
        else:
            remaining.append(part)

        i += 1

    return parsed_flags, remaining


def _try_convert_type(value: str):
    """Try to convert string value to appropriate Python type"""
    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try bool
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False

    return value
