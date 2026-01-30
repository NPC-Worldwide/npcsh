from setuptools import setup, find_packages
import os
from pathlib import Path


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


# Auto-discover NPCs and bin jinxs for console_scripts entry points
npc_team_dir = Path(__file__).parent / "npcsh" / "npc_team"
npc_entries = [f.stem for f in npc_team_dir.glob("*.npc")] if npc_team_dir.exists() else []
jinx_bin_dir = npc_team_dir / "jinxs" / "bin"
jinx_entries = [f.stem for f in jinx_bin_dir.glob("*.jinx")] if jinx_bin_dir.exists() else []

# NPC entries use npcsh:main, bin jinx entries use npc:jinx_main
npc_dynamic = [f"{name}=npcsh.npcsh:main" for name in npc_entries]
jinx_dynamic = [f"{name}=npcsh.npc:jinx_main" for name in jinx_entries]
dynamic_entries = npc_dynamic + jinx_dynamic

base_requirements = [
    'npcpy', 
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

setup(
    name="npcsh",
    version="1.1.21",
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
            "npcsh=npcsh.npcsh:main",
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
    include_package_data=True,
    package_data={
        "npcsh": [
            "npc_team/*.npc",
            "npc_team/*.ctx",
            "npc_team/jinxs/**/*.jinx",
            "npc_team/jinxs/**/*",
            "npc_team/templates/*",
            "benchmark/templates/*.j2",
        ],
    },
    data_files=[("npcsh/npc_team", extra_files)],
    python_requires=">=3.10",
)

