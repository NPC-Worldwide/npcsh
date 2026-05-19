#!/usr/bin/env npcsh
# example_batch_file_analysis.nsh
# Demonstrates iterating over files and calling jinxes on each.

# Get list of markdown files
$files = !find . -name "*.md" -type f | head -5

# Create output directory
!mkdir -p /tmp/analysis

# Process each file
# Note: In a real implementation you'd use bash loops.
# This shows the pattern of variable assignment and jinx calls.

$first_file = !echo "$files" | head -1
$content_1 = !cat "$first_file"

# Analyze first file
$analysis_1 = /edit_file Analyze this markdown for clarity: $content_1

# Store result
!echo "$analysis_1" > /tmp/analysis/file1.txt

$second_file = !echo "$files" | sed -n '2p'
$content_2 = !cat "$second_file"

# Analyze second file
$analysis_2 = /edit_file Analyze this markdown for clarity: $content_2

# Store result
!echo "$analysis_2" > /tmp/analysis/file2.txt

# Combine all analyses
$combined = "Analysis 1:\n$analysis_1\n\nAnalysis 2:\n$analysis_2"
!echo "$combined" > /tmp/analysis/summary.txt

!echo "Batch analysis complete. Results in /tmp/analysis/"
