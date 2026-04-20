"""Launcher that delegates to the Rust npcrs binary if available, falls back to Python npcsh."""
import os
import sys
import shutil
import platform


_MAGIC_ELF = b"\x7fELF"
_MAGIC_MACHO = {
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xfe\xed\xfa\xce",
}
_MAGIC_PE = b"MZ"


def _host_binary_kind():
    system = platform.system()
    if system == "Linux":
        return "elf"
    if system == "Darwin":
        return "macho"
    if system == "Windows":
        return "pe"
    return None


def _binary_matches_host(path: str) -> bool:
    """Check the file's magic bytes match the host OS. Returns False for any mismatch."""
    kind = _host_binary_kind()
    if kind is None:
        return True
    try:
        with open(path, "rb") as f:
            header = f.read(4)
    except OSError:
        return False
    if kind == "elf":
        return header.startswith(_MAGIC_ELF)
    if kind == "macho":
        return header in _MAGIC_MACHO
    if kind == "pe":
        return header.startswith(_MAGIC_PE)
    return True


def _find_rust_binary():
    """Find a host-compatible npcrs binary. Skip any binary whose arch doesn't match."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(pkg_dir, "bin")

    ext = ".exe" if platform.system() == "Windows" else ""

    for name in (f"npcrs{ext}", f"npcsh{ext}"):
        pkg_bin = os.path.join(bin_dir, name)
        if os.path.isfile(pkg_bin) and os.access(pkg_bin, os.X_OK):
            if _binary_matches_host(pkg_bin):
                return pkg_bin
            print(
                f"Warning: {pkg_bin} exists but is not a {platform.system()} binary — skipping",
                file=sys.stderr,
            )

    found = shutil.which("npcrs")
    if found and _binary_matches_host(found):
        return found

    return None


def _load_npcshrc_engine():
    """Read NPCSH_ENGINE from ~/.npcshrc if not already in env."""
    if 'NPCSH_ENGINE' in os.environ:
        return os.environ['NPCSH_ENGINE']
    npcshrc = os.path.expanduser('~/.npcshrc')
    if os.path.exists(npcshrc):
        with open(npcshrc) as f:
            for line in f:
                line = line.strip()
                if line.startswith('export NPCSH_ENGINE='):
                    val = line.split('=', 1)[1].strip("'\"")
                    os.environ['NPCSH_ENGINE'] = val
                    return val
    return None


def _ask_engine():
    """Ask the user which engine to use and save the choice to ~/.npcshrc."""
    print("Which engine would you like to use?")
    print("  [1] Python (stable)")
    print("  [2] Rust   (experimental, requires npcrs binary)")
    try:
        choice = input("Enter 1 or 2 [default: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = '1'
    engine = 'rust' if choice == '2' else 'python'
    npcshrc = os.path.expanduser('~/.npcshrc')
    entry = f"export NPCSH_ENGINE='{engine}'\n"
    if os.path.exists(npcshrc):
        with open(npcshrc, 'a') as f:
            f.write(entry)
    else:
        with open(npcshrc, 'w') as f:
            f.write(entry)
    os.environ['NPCSH_ENGINE'] = engine
    return engine


def _fallback_to_python():
    from npcsh.npcsh import main as python_main
    python_main()


def main():
    engine = _load_npcshrc_engine()
    if engine is None:
        engine = _ask_engine()

    if engine == 'rust':
        rust_bin = _find_rust_binary()
        if rust_bin:
            try:
                os.execvp(rust_bin, [rust_bin] + sys.argv[1:])
            except OSError as e:
                print(
                    f"Warning: failed to exec {rust_bin} ({e}) — falling back to Python",
                    file=sys.stderr,
                )
        else:
            print("Warning: Rust engine selected but no compatible npcrs binary found. Falling back to Python.", file=sys.stderr)

    _fallback_to_python()
