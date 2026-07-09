from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


NPCSH_REPO = "NPC-Worldwide/npcsh"
CRATE_NAME = "npcsh"


_NPCSH_ARTIFACT = {
    ("Linux", "x86_64"): "npcsh-linux-x86_64",
    ("Darwin", "arm64"): "npcsh-macos-aarch64",
    ("Darwin", "x86_64"): "npcsh-macos-x86_64",
    ("Windows", "AMD64"): "npcsh-windows-x86_64.exe",
}

_NPCRU_ARTIFACT = {
    ("Linux", "x86_64"): "npcru-linux-x86_64",
    ("Darwin", "arm64"): "npcru-macos-aarch64",
    ("Darwin", "x86_64"): "npcru-macos-x86_64",
    ("Windows", "AMD64"): "npcru-windows-x86_64.exe",
}


def _install_bin_dir() -> Path:
    """Return the NPCSH bin directory where the binary should live."""
    return Path(os.path.expanduser("~/.npcsh/bin"))


def _remove_old_binaries() -> None:
    env_bin = os.path.dirname(os.path.realpath(sys.executable))
    for dir_path in os.environ.get("PATH", "").split(os.pathsep):
        if not dir_path:
            continue
        try:
            resolved = os.path.realpath(dir_path)
        except (OSError, ValueError):
            continue
        if resolved == env_bin:
            continue
        for name in ("npcrsh", "npcru"):
            p = Path(dir_path) / name
            if p.exists():
                try:
                    p.unlink()
                    print(f"Removed old binary: {p}")
                except OSError:
                    pass


def _latest_crate_version(crate: str) -> str | None:
    """Return the latest version of `crate` published on crates.io."""
    import urllib.request
    import json

    url = f"https://crates.io/api/v1/crates/{crate}/versions"
    headers = {"User-Agent": "npcsh-setup.py/1.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        versions = data.get("versions", [])
        for v in versions:
            num = v.get("num")
            yanked = v.get("yanked", False)
            if num and not yanked:
                return num
    except Exception as e:
        print(f"Warning: failed to query crates.io for latest version ({e})")
    return None


def _download_binary(bin_dir: Path, name: str, artifact_map: dict, version: str | None = None) -> bool:
    """Download a named release binary into bin_dir."""
    import urllib.request
    import json

    system = platform.system()
    machine = platform.machine()
    artifact = artifact_map.get((system, machine))
    if not artifact:
        print(f"Warning: no {name} binary for {system}/{machine}")
        return False

    ext = ".exe" if system == "Windows" else ""
    dst = bin_dir / f"{name}{ext}"

    if version:
        tag = f"v{version}"
        api_url = f"https://api.github.com/repos/{NPCSH_REPO}/releases/tags/{tag}"
    else:
        api_url = f"https://api.github.com/repos/{NPCSH_REPO}/releases/latest"

    try:
        with urllib.request.urlopen(api_url, timeout=15) as resp:
            release = json.loads(resp.read())

        asset_url = next(
            (a["browser_download_url"] for a in release.get("assets", []) if a["name"] == artifact),
            None,
        )
        if not asset_url:
            tag_name = release.get("tag_name", "unknown")
            print(f"Warning: {name} artifact '{artifact}' not found in release {tag_name}")
            return False

        print(f"Downloading {name} binary from {asset_url} ...")
        with urllib.request.urlopen(asset_url, timeout=120) as resp, open(dst, "wb") as f:
            shutil.copyfileobj(resp, f)

        os.chmod(str(dst), 0o755)
        print(f"{name} binary installed to {dst}")
        return True
    except Exception as e:
        print(f"Warning: failed to download {name} binary ({e})")
        return False


class BuildWithRust(build_py):
    def run(self):
        bin_dir = _install_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)

        _remove_old_binaries()

        latest_version = _latest_crate_version(CRATE_NAME)
        if not latest_version:
            raise RuntimeError("Could not determine latest npcsh crate version")

        print(f"Latest published npcsh crate version is {latest_version}")
        _download_binary(bin_dir, "npcrsh", _NPCSH_ARTIFACT, version=latest_version)
        _download_binary(bin_dir, "npcru", _NPCRU_ARTIFACT, version=latest_version)
        super().run()


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


def conflicts_with_system(name):
    """Return True if `name` would shadow a system executable outside this Python env."""
    env_bin = os.path.dirname(os.path.realpath(sys.executable))
    for dir_path in os.environ.get("PATH", "").split(os.pathsep):
        if not dir_path:
            continue
        try:
            resolved = os.path.realpath(dir_path)
        except (OSError, ValueError):
            continue
        if resolved == env_bin:
            continue
        candidate = os.path.join(dir_path, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return True
    return False


npc_team_dir = Path(__file__).parent / "npcsh" / "npc_team"
npc_entries = [f.stem for f in npc_team_dir.glob("*.npc")] if npc_team_dir.exists() else []
jinx_bin_dir = npc_team_dir / "jinxes" / "bin"
jinx_entries = [f.stem for f in jinx_bin_dir.glob("*.jinx")] if jinx_bin_dir.exists() else []

jinx_entries = [name for name in jinx_entries if not conflicts_with_system(name)]

dynamic_entries = []

base_requirements = [
    'npcpy>=2.1.3',
    "jinja2",
    "litellm",
    "docx",
    "scipy",
    "numpy",
    "thefuzz",
    "imagehash",
    "requests",
    "chroptiks",
    "matplotlib",
    "markdown",
    "networkx",
    "PyYAML",
    "PyMuPDF",
    "pyautogui",
    "pydantic",
    "pygments",
    "sqlalchemy",
    "termcolor",
    "rich",
    "colorama",
    "Pillow",
    "python-dotenv",
    "pandas",
    "beautifulsoup4",
    "duckduckgo-search",
    "flask",
    "flask_cors",
    "redis",
    "psycopg2-binary",
    "flask_sse",
    "wikipedia",
    "mcp",
]


_VERSION = (Path(__file__).parent / "VERSION").read_text().strip()

setup(
    name="npcsh",
    version=_VERSION,
    author="NPC Worldwide",
    author_email="info@npcworldwide.com",
    description="The composable multi-agent shell",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NPC-Worldwide/npcsh",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "npcsh": ["**/*"],
    },
    cmdclass={"build_py": BuildWithRust},
    entry_points={
        "console_scripts": [
            "npcsh=npcsh.launcher:main",
            "npc=npcsh.npc_launcher:main",
            *dynamic_entries,
        ],
    },
    install_requires=base_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
