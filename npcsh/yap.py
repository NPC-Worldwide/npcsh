"""
yap - Voice chat mode CLI entry point

This is a thin wrapper that executes the yap.jinx through the jinx mechanism.
"""
import argparse
import os
import sys

from npcsh._state import setup_shell


def main():
    parser = argparse.ArgumentParser(description="yap - Voice chat mode")
    parser.add_argument("--model", "-m", type=str, help="LLM model to use")
    parser.add_argument("--provider", "-p", type=str, help="LLM provider to use")
    parser.add_argument("--files", "-f", nargs="*", help="Files to load for RAG context")
    parser.add_argument("--tts-model", type=str, default=None,
                        help="TTS engine (kokoro, qwen3, elevenlabs, openai, gemini, gtts)")
    parser.add_argument("--voice", type=str, default=None,
                        help="Voice ID (engine-specific, e.g. af_heart for kokoro, ryan for qwen3)")
    args = parser.parse_args()

    command_history, team, default_npc = setup_shell()

    if not team or "yap" not in team.jinxes_dict:
        print("Error: yap jinx not found. Ensure npc_team/jinxes/modes/yap.jinx exists.")
        sys.exit(1)

    saved_engine = os.environ.get("NPCSH_TTS_ENGINE", "")
    saved_voice = os.environ.get("NPCSH_TTS_VOICE", "")
    setup_done = os.environ.get("NPCSH_YAP_SETUP_DONE", "0") == "1"

    tts_model = args.tts_model or saved_engine or "kokoro"
    voice = args.voice or saved_voice or None

    has_explicit_args = args.tts_model is not None or args.voice is not None
    has_saved_prefs = bool(saved_engine)
    show_setup = not has_explicit_args and not has_saved_prefs and not setup_done

    context = {
        "npc": default_npc,
        "team": team,
        "messages": [],
        "model": args.model,
        "provider": args.provider,
        "files": ",".join(args.files) if args.files else None,
        "tts_model": tts_model,
        "voice": voice,
        "show_setup": show_setup,
    }

    yap_jinx = team.jinxes_dict["yap"]
    result = yap_jinx.execute(context=context, npc=default_npc)

    if isinstance(result, dict) and result.get("output"):
        print(result["output"])


if __name__ == "__main__":
    main()
