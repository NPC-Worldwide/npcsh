#!/usr/bin/env python3
"""Upload the npcsh_dpo_0.8b_smoke MLX LoRA adapter into the existing
npc-worldwide/enpisi-coder model repo under a non-destructive subfolder,
via npcpy's native upload_to_hub. Run from the npcsh repo root.

    python scripts/upload_enpisi_adapter.py
    python scripts/upload_enpisi_adapter.py --subfolder 0.8b-smoke --private
"""
import argparse
from npcpy.ft.export import upload_to_hub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="adapters/npcsh_dpo_0.8b_smoke",
                   help="Local adapter directory.")
    p.add_argument("--repo-id", default="npc-worldwide/enpisi-coder",
                   help="HF model repo id.")
    p.add_argument("--subfolder", default="0.8b-smoke",
                   help="Path in repo (non-destructive; existing 2b/, 2b-v2/, mlx/ left untouched).")
    p.add_argument("--token", default=None, help="HF token (reads HF_TOKEN env if omitted).")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    url = upload_to_hub(
        model_path=args.adapter,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        path_in_repo=args.subfolder,
        commit_message=f"Add 0.8B smoke MLX LoRA adapter (enpisi-coder, npcsh judge-rated traces) -> {args.subfolder}/",
    )
    print(f"adapter uploaded -> {url}/tree/main/{args.subfolder}")


if __name__ == "__main__":
    main()