"""Launcher for the `npc` CLI.

Tries to exec the Rust npc binary, and falls back to the Python jinx/NPC
executor when the Rust binary is unavailable or fails.
"""
import os
import sys

from npcsh.launcher import (
    _start_server,
    _ensure_teams_yaml,
    DEFAULT_HOST,
    DEFAULT_PORT,
)


def _find_npc_binary():
    """Find the Rust `npc` binary without recursion risk.

    If the active venv or PATH front has a Python script named `npc`, it is
    ignored: we need the compiled Rust binary. We scan PATH from the back
    (system directories) forward, and also check ~/.npcsh/bin.
    """
    import platform

    ext = ".exe" if platform.system() == "Windows" else ""
    local_bin = os.path.expanduser(f"~/.npcsh/bin/npc{ext}")
    if os.path.isfile(local_bin) and _looks_native_binary(local_bin):
        return local_bin

    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for path_dir in reversed(path_dirs):
        candidate = os.path.join(path_dir, f"npcru{ext}")
        if os.path.isfile(candidate) and _looks_native_binary(candidate):
            return candidate
    return None


def _looks_native_binary(path: str) -> bool:
    import platform

    system = platform.system()
    try:
        with open(path, "rb") as f:
            header = f.read(4)
    except OSError:
        return False
    if system == "Linux":
        return header.startswith(b"\x7fELF")
    if system == "Darwin":
        return header in {
            b"\xcf\xfa\xed\xfe",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xfe\xed\xfa\xce",
        }
    if system == "Windows":
        return header.startswith(b"MZ")
    return True


def main():
    rust_bin = _find_npc_binary()

    if rust_bin:
        teams_yaml = _ensure_teams_yaml()
        if _start_server(DEFAULT_HOST, DEFAULT_PORT, teams_yaml=teams_yaml):
            env = os.environ.copy()
            env["NPCSH_SERVER_URL"] = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
            try:
                os.execvpe(rust_bin, [rust_bin] + sys.argv[1:], env)
            except OSError as e:
                print(
                    f"Warning: failed to exec {rust_bin} ({e}) — falling back to Python",
                    file=sys.stderr,
                )
        else:
            print(
                "Warning: Could not start the NPCSH server for Rust runner; "
                "falling back to Python.",
                file=sys.stderr,
            )

    raise RuntimeError("Rust npcru binary not found")


if __name__ == "__main__":
    main()
