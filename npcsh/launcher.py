"""Launcher that finds and execs the Rust npcrsh binary.

The Rust shell is a thin HTTP/SSE client of a running npcpy server. This
launcher starts the npcpy server using its own CLI, waits for its /health
endpoint, then execs the Rust binary.
"""
import os
import shutil
import platform
import subprocess
import sys
import time
import urllib.request
from typing import Optional


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5237
DEFAULT_SERVER_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
SERVER_LOG_PATH = os.path.expanduser("~/.npcsh/server.log")
NPCSH_TEAM_PATH = os.path.expanduser("~/.npcsh/npc_team")
NPCSH_TEAMS_YAML = os.path.expanduser("~/.npcsh/teams.yaml")


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
    here = os.path.dirname(os.path.abspath(__file__))
    ext = ".exe" if platform.system() == "Windows" else ""

    installed = os.path.join(here, "bin", f"npcrs{ext}")
    if os.path.isfile(installed) and _binary_matches_host(installed):
        return installed

    repo_dir = os.path.dirname(here)
    local_release = os.path.join(repo_dir, "rust", "target", "release", "npcrsh")
    if os.path.isfile(local_release) and _binary_matches_host(local_release):
        return local_release

    local_debug = os.path.join(repo_dir, "rust", "target", "debug", "npcrsh")
    if os.path.isfile(local_debug) and _binary_matches_host(local_debug):
        return local_debug

    return None


def _try_build_rust():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rust_dir = os.path.join(repo_dir, "rust")
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


def _server_alive(url: str) -> bool:
    """Return True if the npcpy server /health endpoint responds."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=0.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _start_server(host: str, port: int, teams_yaml: Optional[str] = None) -> bool:
    """Start the npcpy server via its own CLI and wait for /health."""
    url = f"http://{host}:{port}"
    if _server_alive(url):
        print(f"Using existing npcpy server at {url}", file=sys.stderr)
        return True

    log_dir = os.path.dirname(SERVER_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(SERVER_LOG_PATH, "a")

    cmd = [
        sys.executable,
        "-m",
        "npcpy.serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if teams_yaml:
        cmd.extend(["--teams-yaml", teams_yaml])

    print(f"Starting npcpy server on {host}:{port}...", file=sys.stderr)
    subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()

    for _ in range(600):
        if _server_alive(url):
            print(f"npcpy server ready at {url}.", file=sys.stderr)
            return True
        time.sleep(0.1)

    print(
        f"ERROR: npcpy server at {url} did not become healthy within 60 seconds. "
        f"See log: {SERVER_LOG_PATH}",
        file=sys.stderr,
    )
    return False


def _ensure_teams_yaml() -> Optional[str]:
    """Ensure a teams.yaml exists pointing to the default npcsh team."""
    if not os.path.isdir(NPCSH_TEAM_PATH):
        return None
    if os.path.isfile(NPCSH_TEAMS_YAML):
        return NPCSH_TEAMS_YAML
    try:
        os.makedirs(os.path.dirname(NPCSH_TEAMS_YAML), exist_ok=True)
        with open(NPCSH_TEAMS_YAML, "w") as f:
            f.write(f"teams:\n  npcsh: {NPCSH_TEAM_PATH}\n")
        return NPCSH_TEAMS_YAML
    except Exception as e:
        print(f"Warning: could not write {NPCSH_TEAMS_YAML}: {e}", file=sys.stderr)
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
    rust_bin = _find_rust_binary()
    if rust_bin is None:
        rust_bin = _try_build_rust()

    if rust_bin:
        teams_yaml = _ensure_teams_yaml()
        if not _start_server(DEFAULT_HOST, DEFAULT_PORT, teams_yaml=teams_yaml):
            print(
                "ERROR: Could not start the npcpy server; the Rust shell requires it. "
                "Start it manually with: python3 -m npcpy.serve",
                file=sys.stderr,
            )
            sys.exit(1)
        env = os.environ.copy()
        env["NPCPY_SERVER_URL"] = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
        try:
            os.execvpe(rust_bin, [rust_bin] + sys.argv[1:], env)
        except OSError as e:
            print(
                f"Warning: failed to exec {rust_bin} ({e}) — falling back to Python",
                file=sys.stderr,
            )

    _fallback_to_python()
