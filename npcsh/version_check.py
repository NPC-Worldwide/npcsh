import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from npcsh.npcsh import VERSION

_CACHE_PATH = Path.home() / ".npcsh" / ".version_check"
_CACHE_TTL_SECONDS = 86400


class VersionCheck:
    """Check for newer versions of npcsh across pip, cargo, and brew."""

    def __init__(self):
        self.current = VERSION
        self.available = {}
        self.detected_source = self._detect_source()

    def _detect_source(self):
        """Figure out how npcsh was installed."""
        try:
            import npcsh
            pkg_dir = os.path.dirname(os.path.abspath(npcsh.__file__))
            if "site-packages" in pkg_dir or "dist-packages" in pkg_dir:
                return "pip"
        except Exception:
            pass

        binary = shutil.which("npcrs") or shutil.which("npcsh")

        brew_paths = [
            "/opt/homebrew",
            "/usr/local",
            "/home/linuxbrew",
        ]
        if binary:
            for bp in brew_paths:
                if binary.startswith(bp):
                    return "brew"
            return "cargo"

        return "pip"

    def _cached_result(self):
        """Return cached (source, latest, checked_at) if still fresh."""
        if not _CACHE_PATH.exists():
            return None
        try:
            data = json.loads(_CACHE_PATH.read_text())
            checked_at = data.get("checked_at", 0)
            if time.time() - checked_at < _CACHE_TTL_SECONDS:
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _save_cache(self, data):
        """Write check result to cache file."""
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_text(json.dumps(data))
        except OSError:
            pass

    def _pip_latest(self):
        """Fetch latest version from PyPI."""
        req = Request(
            "https://pypi.org/pypi/npcsh/json",
            headers={"User-Agent": "npcsh-version-check"},
        )
        try:
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return data["info"]["version"]
        except (URLError, HTTPError, json.JSONDecodeError, KeyError):
            return None

    def _cargo_latest(self):
        """Fetch latest version from crates.io."""
        req = Request(
            "https://crates.io/api/v1/crates/npcsh",
            headers={"User-Agent": "npcsh-version-check"},
        )
        try:
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return data["crate"]["max_stable_version"]
        except (URLError, HTTPError, json.JSONDecodeError, KeyError):
            return None

    def _brew_latest(self):
        """Fetch latest version from Homebrew formula."""
        try:
            result = subprocess.run(
                ["brew", "info", "npcsh"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                first_line = result.stdout.splitlines()[0]
                if "stable" in first_line:
                    parts = first_line.split()
                    for i, p in enumerate(parts):
                        if p == "stable" and i + 1 < len(parts):
                            return parts[i + 1]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        req = Request(
            "https://raw.githubusercontent.com/NPC-Worldwide/homebrew-npcsh/main/Formula/npcsh.rb",
            headers={"User-Agent": "npcsh-version-check"},
        )
        try:
            with urlopen(req, timeout=5) as resp:
                text = resp.read().decode()
                for line in text.splitlines():
                    if "version" in line and '"' in line:
                        start = line.find('"') + 1
                        end = line.find('"', start)
                        if end > start:
                            return line[start:end]
        except (URLError, HTTPError):
            pass

        return None

    def _parse_version(self, v):
        """Simple tuple version parser for comparison."""
        if not v:
            return (0, 0, 0)
        parts = v.lstrip("v").split(".")
        try:
            return tuple(int(p) for p in parts[:3])
        except ValueError:
            return (0, 0, 0)

    def check(self):
        """Run version checks and return update info if any."""
        cached = self._cached_result()
        if cached:
            self.available = cached.get("available", {})
        else:
            latest = {
                "pip": self._pip_latest(),
                "cargo": self._cargo_latest(),
                "brew": self._brew_latest(),
            }
            self.available = latest
            self._save_cache({
                "available": latest,
                "checked_at": time.time(),
            })

        source = self.detected_source
        latest = self.available.get(source)
        if not latest:
            return None

        if self._parse_version(latest) > self._parse_version(self.current):
            return {
                "source": source,
                "current": self.current,
                "latest": latest,
                "command": self._update_command(source),
            }
        return None

    def _update_command(self, source):
        if source == "pip":
            return "pip install --upgrade npcsh"
        if source == "cargo":
            return "cargo install npcsh --force"
        if source == "brew":
            return "brew upgrade npcsh"
        return ""

    def check_async(self, callback):
        """Run check in background and call callback with result."""
        def _run():
            result = self.check()
            if result:
                callback(result)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
