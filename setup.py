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


_NPCSH_ARTIFACT = {
    ("Linux", "x86_64"): "npcsh-linux-x86_64",
    ("Darwin", "arm64"): "npcsh-macos-aarch64",
    ("Darwin", "x86_64"): "npcsh-macos-x86_64",
    ("Windows", "AMD64"): "npcsh-windows-x86_64.exe",
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
        for name in ("npcrsh",):
            p = Path(dir_path) / name
            if p.exists():
                try:
                    p.unlink()
                    print(f"Removed old binary: {p}")
                except OSError:
                    pass


def _download_npcrsh(bin_dir: Path) -> bool:
    """Download the latest npcrsh release binary into bin_dir."""
    import urllib.request
    import json

    system = platform.system()
    machine = platform.machine()
    artifact = _NPCSH_ARTIFACT.get((system, machine))
    if not artifact:
        print(f"Warning: no npcrsh binary for {system}/{machine}")
        return False

    ext = ".exe" if system == "Windows" else ""
    dst = bin_dir / f"npcrsh{ext}"

    if dst.exists() and _binary_matches_host(str(dst)):
        print(f"Valid npcrsh binary already present at {dst}")
        return True

    try:
        api_url = f"https://api.github.com/repos/{NPCSH_REPO}/releases/latest"
        with urllib.request.urlopen(api_url, timeout=15) as resp:
            release = json.loads(resp.read())

        asset_url = next(
            (a["browser_download_url"] for a in release.get("assets", []) if a["name"] == artifact),
            None,
        )
        if not asset_url:
            print(f"Warning: npcrsh artifact '{artifact}' not found in latest release")
            return False

        print(f"Downloading npcrsh binary from {asset_url} ...")
        with urllib.request.urlopen(asset_url, timeout=120) as resp, open(dst, "wb") as f:
            shutil.copyfileobj(resp, f)

        os.chmod(str(dst), 0o755)
        print(f"npcrsh binary installed to {dst}")
        return True
    except Exception as e:
        print(f"Warning: failed to download npcrsh binary ({e})")
        return False


class BuildWithRust(build_py):
    def run(self):
        bin_dir = _install_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)

        _remove_old_binaries()

        if _download_npcrsh(bin_dir):
            super().run()
            return

        print("Warning: npcrsh binary unavailable — falling back to Python-only mode")
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

npc_dynamic = [f"{name}=npcsh.npcsh:main" for name in npc_entries]
jinx_dynamic = [f"{name}=npcsh.npc:jinx_main" for name in jinx_entries]
dynamic_entries = npc_dynamic + jinx_dynamic

base_requirements = [
    'npcpy>=2.1.1',
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


setup(
    name="npcsh",
    version="1.2.30",
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
