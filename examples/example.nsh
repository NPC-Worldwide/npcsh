# Example npcsh script (.nsh file)
# This script demonstrates the capabilities of .nsh job files

# Set a variable
$count = 0

# Use a jinx to search the web
/search latest AI news

# Save the result to a variable  
$news_result = $_

# Process with an NPC
@analyzer summarize the news: $news_result

# Run a shell command
!echo "Job completed at $(date)"

# Increment counter
$count = $count + 1

# Log completion
!echo "Run #$count completed" >> ~/.npcsh/npc_team/logs/example_counter.log
