import os
import sys


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
        candidate = os.path.join(path_dir, f"npc{ext}")
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
    if not rust_bin:
        print("ERROR: Rust npc binary not found.", file=sys.stderr)
        sys.exit(1)

    try:
        os.execvp(rust_bin, [rust_bin] + sys.argv[1:])
    except OSError as e:
        print(f"ERROR: failed to exec {rust_bin} ({e})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
