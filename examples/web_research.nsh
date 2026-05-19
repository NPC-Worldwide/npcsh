#!/usr/bin/env npcsh
# example_web_research.nsh
# Demonstrates variable assignment and chaining.

# Search for latest AI news
$search_query = "latest AI breakthroughs 2026"
search $search_query

# Capture result
$raw_results = $_

# Save raw results for inspection
echo "$raw_results" > /tmp/ai_news_raw.txt

# Summarize with the default NPC
summarize these AI news in 3 bullet points: $raw_results

# Capture summary
$summary = $_

# Write the summary to a report
echo "## AI News Report\n\n$summary" > ~/ai_news_report.md

# Display the final report path
echo "Report saved to ~/ai_news_report.md"

# Increment a counter for tracking
$run_count = 1
echo "Run #$run_count completed"
