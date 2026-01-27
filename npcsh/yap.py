"""
yap - Voice chat mode CLI entry point

This is a thin wrapper that executes the yap.jinx through the jinx mechanism.
"""
import argparse
import sys

from npcsh._state import setup_shell


def main():
    parser = argparse.ArgumentParser(description="yap - Voice chat mode")
    parser.add_argument("--model", "-m", type=str, help="LLM model to use")
    parser.add_argument("--provider", "-p", type=str, help="LLM provider to use")
    parser.add_argument("--files", "-f", nargs="*", help="Files to load for RAG context")
    parser.add_argument("--tts-model", type=str, default="kokoro", help="TTS model to use")
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice for TTS")
    args = parser.parse_args()

    # Setup shell to get team and default NPC
    command_history, team, default_npc = setup_shell()

    if not team or "yap" not in team.jinxs_dict:
        print("Error: yap jinx not found. Ensure npc_team/jinxs/modes/yap.jinx exists.")
        sys.exit(1)

    # Build context for jinx execution
    context = {
        "npc": default_npc,
        "team": team,
        "messages": [],
        "model": args.model,
        "provider": args.provider,
        "files": ",".join(args.files) if args.files else None,
        "tts_model": args.tts_model,
        "voice": args.voice,
    }

    # Execute the jinx
    yap_jinx = team.jinxs_dict["yap"]
    result = yap_jinx.execute(context=context, npc=default_npc)

    if isinstance(result, dict) and result.get("output"):
        print(result["output"])


if __name__ == "__main__":
    main()
