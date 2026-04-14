"""Launcher that delegates to the Rust npcrs binary if available, falls back to Python npcsh."""
import os
import sys
import shutil
import platform


def _find_rust_binary():
    """Find the npcrs binary — check package dir first, then PATH."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(pkg_dir, "bin")

    name = "npcsh.exe" if platform.system() == "Windows" else "npcsh"

    # Check inside the package
    pkg_bin = os.path.join(bin_dir, name)
    if os.path.isfile(pkg_bin) and os.access(pkg_bin, os.X_OK):
        return pkg_bin

    # Check PATH
    found = shutil.which("npcrs")
    if found:
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


def main():
    engine = _load_npcshrc_engine()
    if engine is None:
        engine = _ask_engine()

    if engine == 'rust':
        rust_bin = _find_rust_binary()
        if rust_bin:
            os.execvp(rust_bin, [rust_bin] + sys.argv[1:])
        else:
            print("Warning: Rust engine selected but npcrs binary not found. Falling back to Python.")

    from npcsh.npcsh import main as python_main
    python_main()
