#!/usr/bin/env npcsh
# example_multi_step_research.nsh
# Demonstrates multi-step variable chaining and $_ usage.

# Step 1: Search for a topic
/search quantum computing latest developments

# Capture the search result automatically with $_
$search_result = $_

# Step 2: Extract key topics using an NPC
$topics = @default List 5 key topics from this search: $search_result

# Step 3: Research each topic in depth
# (In a real script you might loop, here we show manual chaining)
$deep_dive_1 = @default Research the first topic in depth: $topics
$deep_dive_2 = @default Research the second topic in depth: $topics

# Step 4: Synthesize everything
$synthesis = @default Synthesize these findings into a coherent summary: $deep_dive_1 $deep_dive_2

# Step 5: Save to file with timestamp
$timestamp = !date +%Y-%m-%d
!echo "# Research Report ($timestamp)\n\n$synthesis" > ~/research_report.md

!echo "Research saved to ~/research_report.md"
