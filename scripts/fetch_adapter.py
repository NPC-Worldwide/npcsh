#!/usr/bin/env python3
"""
fetch_adapter.py

Download a published adapter from HuggingFace Hub.

Usage:
    python scripts/fetch_adapter.py --repo npc-worldwide/enpisi-coder --name npcsh-sft-toolcalls-all

    python scripts/fetch_adapter.py --repo npc-worldwide/enpisi-coder --name npcsh-sft \
        --output adapters/my-npcsh-sft

    python scripts/fetch_adapter.py --repo npc-worldwide/enpisi-coder --name full-model \
        --subfolder full
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Fetch adapter from HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="HF repo ID (e.g. npc-worldwide/enpisi-coder)")
    parser.add_argument("--name", required=True, help="Adapter name / subfolder in repo")
    parser.add_argument("--output", help="Local output path (default: adapters/<name>)")
    parser.add_argument("--subfolder", default="adapters", help="Repo subfolder (default: adapters)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token")
    args = parser.parse_args()

    output = args.output or f"adapters/{args.name}"
    path_in_repo = f"{args.subfolder}/{args.name}"

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from npcpy.ft import download_from_hub

    print(f"Fetching {args.repo}/{path_in_repo} → {output}")
    download_from_hub(
        repo_id=args.repo,
        local_path=output,
        path_in_repo=path_in_repo,
        token=args.token,
    )
    print(f"Done. Adapter saved to: {os.path.abspath(output)}")
    print("\nTo use it:")
    print("  export NPCSH_CHAT_MODEL='mlx-community/Qwen3-4B-4bit'")
    print("  export NPCSH_CHAT_PROVIDER='omlx'")
    print(f"  Place adapter at ~/.npcsh/adapters/{args.name}/")


if __name__ == "__main__":
    main()
