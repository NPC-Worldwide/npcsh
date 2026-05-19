#!/usr/bin/env npcsh
# example.nsh
# Basic demonstration of .nsh script capabilities.

# Set variables
$project_name = "npcsh"
$version = "1.2.14"

# Run shell commands
echo "Building $project_name v$version"

# Use a jinx to search the web
search latest $project_name news

# Capture the result
$news = $_

# Process with an NPC
summarize this in one sentence: $news

# Capture NPC response
$summary = $_

# Combine variables
$report = "Project: $project_name\nVersion: $version\nNews: $summary"

# Save report
echo "$report" > /tmp/npcsh_status.txt

# Display
cat /tmp/npcsh_status.txt
