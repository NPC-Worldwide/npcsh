#!/usr/bin/env npcsh
# example_code_review.nsh
# Automated code review pipeline using jinxes and variables.

# Find all Python files changed recently
$changed_files = !git diff --name-only --diff-filter=ACM HEAD~1 | grep '\.py$'

# If no Python files, stop
!test -n "$changed_files" || echo "No Python files changed"

# Get the diff for the first file
$first_file = !echo "$changed_files" | head -1
$diff = !git diff HEAD~1 -- "$first_file"

# Run the pr_review jinx on the diff
$review = /pr_review $diff

# Ask an NPC to generate action items from the review
$action_items = @default Based on this review, list 3 concrete fixes: $review

# Log everything
!echo "=== Code Review for $first_file ===" > /tmp/code_review.log
!echo "$review" >> /tmp/code_review.log
!echo "" >> /tmp/code_review.log
!echo "=== Action Items ===" >> /tmp/code_review.log
!echo "$action_items" >> /tmp/code_review.log

!echo "Review complete. Log: /tmp/code_review.log"
