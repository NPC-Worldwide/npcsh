# Wander Mode

Wander mode lets you:
- Provide a problem or question.
- Optionally provide or generate a metaphorical environment.
- Alternate between focused and high-temperature LLM output.
- Optionally inject random events (encounter, discovery, obstacle, insight, shift, memory).
- After each session, continue with the same or a new problem/environment, or exit.

## Example Usage

From the command line:
```
npc --model "gemini-2.0-flash" --provider "gemini" wander "how does the bar of a galaxy influence the the surrounding IGM?" \
  environment='a ships library in the south.' \
  num-events=3 \
  n-high-temp-streams=10 \
  high-temp=1.95 \
  low-temp=0.4 \
  sample-rate=0.5 \
  interruption-likelihood=.1

npc --model "gpt-4o-mini" --provider "openai" wander "how does the goos-hanchen effect impact neutron scattering?" \
  environment='a ships library in the south.' \
  num-events=3 \
  n-high-temp-streams=10 \
  high-temp=1.95 \
  low-temp=0.4 \
  sample-rate=0.5 \
  interruption-likelihood=.1

npc wander "what is the goos hacnehn effect and how does it affect the water refraction" \
  --provider "ollama" \
  --model "deepseek-r1:32b" \
  environment="a vast, dark ocean ." \
  interruption-likelihood=.1

npc wander "what is the goos hacnehn effect and how does it affect the water refraction" \
  --provider "ollama" \
  --model "qwen3:latest" \
  environment="a vast, dark ocean ." \
  interruption-likelihood=.1
```

Arguments:
- problem: string (required)
- --model: string
- --provider: string
- --environment: string (optional)
- --no-events: disables random events
- --num-events: int
- --npc: path to NPC file

Requirements:
- LLM provider and at least one NPC file.

No other features are present.
