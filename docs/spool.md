# Spool Mode

Spool mode is a chat loop with a specified NPC. You can load files (PDF, CSV) into context, and interact with the NPC using text input. You can also analyze images or screenshots with vision models, and use RAG (retrieval augmented generation) on loaded content. Type `/sq` to exit spool mode.

## Usage

From the command line:
```
npc spool --npc ~/.npcsh/npc_team/sibiji.npc --model MODEL --provider PROVIDER --files file1.pdf file2.csv --stream true
```

Arguments:
- --npc: path to NPC file (default: sibiji.npc)
- --model: LLM model
- --provider: LLM provider
- --files: list of files to load (PDF, CSV supported)
- --stream: true/false (default: true)

## Features
- Chat with an NPC in a loop
- Load PDF/CSV files for context
- Use `/ots` to analyze images or screenshots with a vision model
- Use `/whisper` to enter voice mode (yap)
- RAG search on loaded content
- Conversation history is saved

No other features are present.
