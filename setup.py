from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import os
import platform
import subprocess
import sys
from pathlib import Path


NPCRS_REPO = "NPC-Worldwide/npcsh"

_NPCRS_ARTIFACT = {
    ("Linux", "x86_64"): "npcsh-linux-x86_64",
    ("Darwin", "arm64"): "npcsh-macos-aarch64",
    ("Darwin", "x86_64"): "npcsh-macos-x86_64",
    ("Windows", "AMD64"): "npcsh-windows-x86_64.exe",
}


def _purge_stale_binaries(bin_dir: Path) -> None:
    """Remove any pre-existing binary from bin_dir so we never ship a stale arch."""
    for p in bin_dir.iterdir():
        if p.name == ".gitkeep":
            continue
        try:
            p.unlink()
        except OSError:
            pass


def _download_npcrs(bin_dir: Path) -> bool:
    """Download the latest npcrs release binary into bin_dir. Returns True on success."""
    import urllib.request
    import json
    import shutil

    system = platform.system()
    machine = platform.machine()
    artifact = _NPCRS_ARTIFACT.get((system, machine))
    if not artifact:
        print(f"Warning: no npcrs binary for {system}/{machine}")
        return False

    ext = ".exe" if system == "Windows" else ""
    dst = bin_dir / f"npcrs{ext}"

    try:
        api_url = f"https://api.github.com/repos/{NPCRS_REPO}/releases/latest"
        with urllib.request.urlopen(api_url, timeout=15) as resp:
            release = json.loads(resp.read())

        asset_url = next(
            (a["browser_download_url"] for a in release.get("assets", []) if a["name"] == artifact),
            None,
        )
        if not asset_url:
            print(f"Warning: npcrs artifact '{artifact}' not found in latest release")
            return False

        print(f"Downloading npcrs binary from {asset_url} ...")
        with urllib.request.urlopen(asset_url, timeout=120) as resp, open(dst, "wb") as f:
            shutil.copyfileobj(resp, f)

        os.chmod(str(dst), 0o755)
        print(f"npcrs binary installed to {dst}")
        return True
    except Exception as e:
        print(f"Warning: failed to download npcrs binary ({e})")
        return False


class BuildWithRust(build_py):
    """Download the pre-built npcrs binary matching the host arch before packaging."""

    def run(self):
        bin_dir = Path(__file__).parent / "npcsh" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        _purge_stale_binaries(bin_dir)

        if not _download_npcrs(bin_dir):
            print("Warning: npcrs binary unavailable — falling back to Python-only mode")

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



# Auto-discover NPCs and bin jinxes for console_scripts entry points
npc_team_dir = Path(__file__).parent / "npcsh" / "npc_team"
npc_entries = [f.stem for f in npc_team_dir.glob("*.npc")] if npc_team_dir.exists() else []
jinx_bin_dir = npc_team_dir / "jinxes" / "bin"
jinx_entries = [f.stem for f in jinx_bin_dir.glob("*.jinx")] if jinx_bin_dir.exists() else []

# Filter out jinx names that would shadow system executables
jinx_entries = [name for name in jinx_entries if not conflicts_with_system(name)]

# NPC entries use npcsh:main, bin jinx entries use npc:jinx_main
npc_dynamic = [f"{name}=npcsh.npcsh:main" for name in npc_entries]
jinx_dynamic = [f"{name}=npcsh.npc:jinx_main" for name in jinx_entries]
dynamic_entries = npc_dynamic + jinx_dynamic

base_requirements = [
    'npcpy>=1.4.20',
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
    "mcp"
]

# API integration requirements
api_requirements = [
    "anthropic",
    "openai",
    "ollama", 
    "google-generativeai",
    "google-genai",
]

# Local ML/AI requirements
local_requirements = [
    "sentence_transformers",
    "opencv-python",
    "ollama",
    "kuzu",
    "chromadb",
    "diffusers",
    "nltk",
    "torch",
    "darts",
]

# Voice/Audio requirements
voice_requirements = [
    "pyaudio",
    "gtts",
    "playsound==1.2.2",
    "pygame",
    "faster_whisper",
    "pyttsx3",
]

# Benchmark requirements (Terminal-Bench integration)
benchmark_requirements = [
    "harbor",
    "terminal-bench",
]

extra_files = package_files("npcsh/npc_team/")

# Build package_data dict for npc_team files
def get_package_data_patterns():
    """Get patterns for all files in npc_team directory."""
    patterns = []
    npc_team_path = Path(__file__).parent / "npcsh" / "npc_team"
    if npc_team_path.exists():
        for root, dirs, files in os.walk(npc_team_path):
            rel_root = os.path.relpath(root, Path(__file__).parent / "npcsh")
            for f in files:
                patterns.append(os.path.join(rel_root, f))
    return patterns

_version = (Path(__file__).parent / "VERSION").read_text().strip()

setup(
    name="npcsh",
    version=_version,
    packages=find_packages(exclude=["tests*"]),
    install_requires=base_requirements,  # Only install base requirements by default
    extras_require={
        "lite": api_requirements,
        "local": local_requirements,
        "yap": voice_requirements,
        "bench": benchmark_requirements,
        "all": api_requirements + local_requirements + voice_requirements,
    },
    entry_points={
        "console_scripts": [
            # Main entry points
            "npcsh=npcsh.launcher:main",
            "npc=npcsh.npc:main",
            # Benchmark runner
            "npcsh-bench=npcsh.benchmark.runner:main",
            # Dynamic entry points from data files (NPCs and bin/ jinxes)
        ] + dynamic_entries,
    },
    author="Christopher Agostino",
    author_email="info@npcworldwi.de",
    description="npcsh is a command-line toolkit for using AI agents in novel ways.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NPC-Worldwide/npcsh",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    cmdclass={"build_py": BuildWithRust},
    include_package_data=True,
    package_data={
        "npcsh": [
            "bin/*",
            "npc_team/*.npc",
            "npc_team/*.ctx",
            "npc_team/jinxes/**/*.jinx",
            "npc_team/jinxes/**/*",
            "npc_team/templates/*",
            "benchmark/templates/*.j2",
        ],
    },
    data_files=[("npcsh/npc_team", extra_files)],
    python_requires=">=3.10",
)

