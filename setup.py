from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import os
import platform
import sys
from pathlib import Path


NPCRS_REPO = "NPC-Worldwide/npcsh"

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

_NPCRS_ARTIFACT = {
    ("Linux", "x86_64"): "npcsh-linux-x86_64",
    ("Darwin", "arm64"): "npcsh-macos-aarch64",
    ("Darwin", "x86_64"): "npcsh-macos-x86_64",
    ("Windows", "AMD64"): "npcsh-windows-x86_64.exe",
}


def _purge_stale_binaries(bin_dir: Path) -> None:
    for p in bin_dir.iterdir():
        if p.name == ".gitkeep":
            continue
        if _binary_matches_host(str(p)):
            continue
        try:
            p.unlink()
        except OSError:
            pass


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


def _download_npcrs(bin_dir: Path) -> bool:
    """Download the latest npcrs release binary into bin_dir. Returns True on success or if valid binary already present."""
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

    if dst.exists() and _binary_matches_host(str(dst)):
        print(f"Valid npcrs binary already present at {dst}")
        return True

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
    def run(self):
        bin_dir = Path(__file__).parent / "npcsh" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        _remove_old_binaries()
        _purge_stale_binaries(bin_dir)

        if _download_npcrs(bin_dir):
            super().run()
            return

        cargo = shutil.which("cargo")
        rust_dir = Path(__file__).parent / "rust"
        if cargo and rust_dir.exists():
            print("Downloading npcrs failed — building from source with cargo...")
            try:
                subprocess.run(
                    [cargo, "build", "--release"],
                    cwd=str(rust_dir),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                src = rust_dir / "target" / "release" / "npcrsh"
                if src.exists():
                    dst = bin_dir / "npcrs"
                    shutil.copy2(str(src), str(dst))
                    os.chmod(str(dst), 0o755)
                    print(f"Built npcrs binary at {dst}")
                    super().run()
                    return
            except Exception as e:
                print(f"Warning: cargo build failed ({e})")

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



npc_team_dir = Path(__file__).parent / "npcsh" / "npc_team"
npc_entries = [f.stem for f in npc_team_dir.glob("*.npc")] if npc_team_dir.exists() else []
jinx_bin_dir = npc_team_dir / "jinxes" / "bin"
jinx_entries = [f.stem for f in jinx_bin_dir.glob("*.jinx")] if jinx_bin_dir.exists() else []

jinx_entries = [name for name in jinx_entries if not conflicts_with_system(name)]

npc_dynamic = [f"{name}=npcsh.npcsh:main" for name in npc_entries]
jinx_dynamic = [f"{name}=npcsh.npc:jinx_main" for name in jinx_entries]
dynamic_entries = npc_dynamic + jinx_dynamic

base_requirements = [
    'npcpy>=1.4.29',
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

api_requirements = [
    "anthropic",
    "openai",
    "ollama",
    "google-generativeai",
    "google-genai",
]

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

voice_requirements = [
    "pyaudio",
    "gtts",
    "playsound==1.2.2",
    "pygame",
    "faster_whisper",
    "pyttsx3",
]

benchmark_requirements = [
    "harbor",
    "terminal-bench",
]

extra_files = package_files("npcsh/npc_team/")

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
    install_requires=base_requirements,
    extras_require={
        "lite": api_requirements,
        "local": local_requirements,
        "yap": voice_requirements,
        "bench": benchmark_requirements,
        "all": api_requirements + local_requirements + voice_requirements,
    },
    entry_points={
        "console_scripts": [
            "npcsh=npcsh.launcher:main",
            "npc=npcsh.npc:main",
            "npcsh-bench=npcsh.benchmark.runner:main",
            "npcsh-job=npcsh.job_runner:main",
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

