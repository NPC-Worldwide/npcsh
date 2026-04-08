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


def main():
    rust_bin = _find_rust_binary()
    if rust_bin:
        os.execvp(rust_bin, [rust_bin] + sys.argv[1:])
    else:
        from npcsh.npcsh import main as python_main
        python_main()
