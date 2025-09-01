from setuptools import setup, find_packages
import os
def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths
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

extra_files = package_files("npcsh/npc_team/")

setup(
    name="npcsh",
    version="1.0.26",
    packages=find_packages(exclude=["tests*"]),
    install_requires=base_requirements,  # Only install base requirements by default
    extras_require={
        "lite": api_requirements,
        "local": local_requirements,
        "yap": voice_requirements,
        "all": api_requirements + local_requirements + voice_requirements ,  
    },
    entry_points={
        "console_scripts": [
            "corca=npcsh.corca:main",
            "npcsh=npcsh.npcsh:main",
            "npc=npcsh.npc:main",
            "yap=npcsh.yap:main",
            "pti=npcsh.pti:main",
            "guac=npcsh.guac:main",
            "wander=npcsh.wander:main",
            "spool=npcsh.spool:main",
        ],
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
    data_files=[("npcsh/npc_team", extra_files)],
    python_requires=">=3.10",
)

