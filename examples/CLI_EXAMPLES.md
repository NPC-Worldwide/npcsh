# NPC CLI Examples - All Available Commands

# Deep research exploration agent
npc alicanto "What are the implications of quantum computing for cybersecurity?"
npc alicanto "How might climate change impact global food security?" --num-npcs 8 --depth 5
npc alicanto "What ethical considerations should guide AI development?" --max_facts_per_chain 0.5 --max_thematic_groups 3 --format report

# Search through past messages
npc brainblast 'subtle summer winds' --top_k 10
npc brainblast 'executing a mirror in the wonderous moon'

# Compile NPCs for use without re-loading
npc compile /path/to/npc

# Show help for commands, NPCs, or Jinxs
npc help
npc help alicanto
npc help search

# Initialize NPC project
npc init

# Show available jinxs for current NPC/Team
npc jinxs

# Take screenshot and analyze with vision model
npc ots
npc ots --output_filename analysis.txt

# Execute a plan command (cron jobs)
npc plan 'backup my documents folder every day at 2am' -m gemma3:27b -p ollama
npc plan 'send me a daily weather report at 8am'

# Computer use with vision model GUI interaction
npc plonk --task 'open a web browser and navigate to github.com'
npc plonk --npc assistant --task 'take a screenshot and analyze what applications are running'

# Reasoning REPL loop with interruptions
npc pti -n frederic -m qwen3:latest -p ollama
npc pti --model deepseek-reasoner --provider deepseek

# Embedding search through ChromaDB with optional file input
npc rag "find documents about machine learning" --top_k 5
npc rag "search for python tutorials" --file ./documents/notes.txt

# Video generation
npc roll "generate a video of a hat riding a dog"
npc roll --provider ollama --model llama3 "a cat dancing in the rain"

# One-shot sampling from LLMs with specific parameters
npc sample 'prompt'
npc sample -m gemini-1.5-flash "Summarize the plot of 'The Matrix' in three sentences."
npc sample --model claude-3-5-haiku-latest "Translate 'good morning' to Japanese."
npc sample --model qwen3:latest "tell me about the last time you went shopping."
npc sample -p ollama -m gemma3:12b --temp 1.8 --top_k 50 "Write a haiku about the command line."
npc sample --model gpt-4o-mini "What are the primary colors?" --provider openai

# Internet search with various providers
npc search 'when is the moon gonna go away from the earth'
npc search --sprovider perplexity 'cal bears football schedule'
npc search --sprovider duckduckgo 'beef tongue'

# Knowledge graph search (in npcsh shell)
# /kg_search python                       # Keyword search
# /kg_search python mode=embedding        # Semantic similarity search
# /kg_search python mode=link depth=3     # Traverse graph links
# /kg_search python mode=all              # All methods combined
# /kg_search type=concepts                # List all concepts
# /kg_search concept="Mao Biography"      # Explore a specific concept

# Memory search and review (in npcsh shell)
# /mem_search python                      # Search all memories
# /mem_search python status=approved      # Search approved only
# /mem_review                             # Review pending memories
# /mem_review limit=50                    # Review more at once

# Serve an NPC team as a web service
npc serve
npc serve --port 5337 --cors 'http://localhost:5137/'

# Set configuration values
npc set --model gpt-4o-mini
npc set --provider openai
npc set --api_url https://localhost:1937

# Evolve knowledge graph with dreaming options
npc sleep
npc sleep --dream
npc sleep --backfill                     # Import approved memories into KG first
npc sleep --backfill --dream             # Backfill then dream
npc sleep --ops prune,deepen,abstract    # Specific operations

# Enter isolated chat with attachments and specified models
npc spool -n alicanto
npc spool --attachments ./test_data/port5337.png,./test_data/yuan2004.pdf,./test_data/books.csv
npc spool --provider ollama --model llama3
npc spool -p deepseek -m deepseek-reasoner

# Schedule listeners and daemons
npc trigger 'watch for new files in Downloads folder and organize them by type' -m gemma3:27b -p ollama
npc trigger 'send me a notification when CPU usage exceeds 80%'

# Image generation and editing
npc vixynt 'an image of a dog eating a hat'
npc vixynt --output_file ~/Desktop/dragon.png "A terrifying dragon"
npc vixynt "A photorealistic portrait of a cat wearing a wizard hat" -w 1024 --height 1024
npc vixynt -igp ollama --igmodel Qwen/QwenImage --output_file /tmp/sub.png --width 1024 --height 512 "A detailed steampunk submarine"
npc vixynt --attachments ./test_data/rabbit.PNG "Turn this rabbit into a fierce warrior in a snowy winter scene" -igp openai -igm gpt-image
npc vixynt --igmodel CompVis/stable-diffusion-v1-4 --igprovider diffusers "sticker of a red tree"

# Daydreaming for LLMs - thinking outside the box
npc wander "how does the bar of a galaxy influence the surrounding IGM?" --model gemini-2.0-flash --provider gemini --n-high-temp-streams 10 --high-temp 1.95 --low-temp 0.4
npc wander "how does the goos-hanchen effect impact neutron scattering?" --model gpt-4o-mini --provider openai --environment 'a ships library in the south.' --num-events 3
npc wander "what is the goos hanchen effect and does it affect water refraction?" --provider ollama --model deepseek-r1:32b --environment "a vast, dark ocean" --interruption-likelihood 0.1

# Voice chat mode (agentic voice control loop)
npc yap
npc yap --npc assistant --model llama3.2

# Basic prompting (non-slash commands)
npc "What is the weather like today?"
npc "Help me write a Python script to sort a list" --model gpt-4o --provider openai
npc "Explain quantum computing in simple terms" --npc sibiji
npc "How do I install Docker on Ubuntu?" --model claude-3-5-sonnet --provider anthropic