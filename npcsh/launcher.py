"""Launcher that finds and execs the Rust npcrsh binary. Falls back to Python only if unavailable."""
import os
import sys
import shutil
import platform
import subprocess


def _host_binary_kind():
    system = platform.system()
    if system == "Linux":
        return "elf"
    if system == "Darwin":
        return "macho"
    if system == "Windows":
        return "pe"
    return None


_MAGIC_ELF = b"\x7fELF"
_MAGIC_MACHO = {
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xfe\xed\xfa\xce",
}
_MAGIC_PE = b"MZ"


def _binary_matches_host(path: str) -> bool:
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
    """Find a host-compatible npcrsh binary."""
    # 1. PATH lookup
    for name in ("npcrsh", "npcrs"):
        found = shutil.which(name)
        if found and _binary_matches_host(found):
            return found

    # 2. Installed package binary (wheel / pip install)
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ext = ".exe" if platform.system() == "Windows" else ""
    installed = os.path.join(pkg_dir, "bin", f"npcrs{ext}")
    if os.path.isfile(installed) and _binary_matches_host(installed):
        return installed

    # 3. Package-local build artifact (dev workflow)
    local_release = os.path.join(pkg_dir, "rust", "target", "release", "npcrsh")
    if os.path.isfile(local_release) and _binary_matches_host(local_release):
        return local_release

    local_debug = os.path.join(pkg_dir, "rust", "target", "debug", "npcrsh")
    if os.path.isfile(local_debug) and _binary_matches_host(local_debug):
        return local_debug

    return None


def _try_build_rust():
    """Try to build the Rust binary from source if cargo is available."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rust_dir = os.path.join(pkg_dir, "rust")
    if not os.path.isdir(rust_dir):
        return None
    cargo = shutil.which("cargo")
    if not cargo:
        return None
    try:
        print("Building Rust npcrsh binary...", file=sys.stderr)
        subprocess.run(
            [cargo, "build", "--release"],
            cwd=rust_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        binary = os.path.join(rust_dir, "target", "release", "npcrsh")
        if os.path.isfile(binary):
            return binary
    except Exception:
        pass
    return None


def _fallback_to_python():
    print(
        "WARNING: Rust npcrsh binary not found. Falling back to Python runner.\n"
        "The Python runner is deprecated. Build the Rust binary with:\n"
        "  cd npcsh/rust && cargo build --release",
        file=sys.stderr,
    )
    from npcsh.npcsh import main as python_main
    python_main()


def main():
    # Remove stale NPCSH_ENGINE lines from ~/.npcshrc so users don't get stuck
    npcshrc = os.path.expanduser("~/.npcshrc")
    if os.path.exists(npcshrc):
        try:
            with open(npcshrc, "r") as f:
                lines = f.readlines()
            filtered = [line for line in lines if "NPCSH_ENGINE" not in line]
            if len(filtered) != len(lines):
                with open(npcshrc, "w") as f:
                    f.writelines(filtered)
        except Exception:
            pass

    rust_bin = _find_rust_binary()
    if rust_bin is None:
        rust_bin = _try_build_rust()

    if rust_bin:
        try:
            os.execvp(rust_bin, [rust_bin] + sys.argv[1:])
        except OSError as e:
            print(
                f"Warning: failed to exec {rust_bin} ({e}) — falling back to Python",
                file=sys.stderr,
            )

    _fallback_to_python()
