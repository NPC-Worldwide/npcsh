#!/usr/bin/env npcsh
# example_git_workflow.nsh
# Demonstrates git workflow automation.

# Get current branch
$branch = git branch --show-current

# Get recent commits
$commits = git log --oneline -5

# Generate release notes using a jinx
release_notes $commits

# Capture notes
$notes = $_

# Review with NPC
is this release note clear and accurate? $notes

# Capture review
$review = $_

# If review is positive, create a tag
echo "$review" | grep -qi "yes" && git tag -a v$version -m "$notes"

# Push
git push origin $branch

echo "Workflow complete for branch: $branch"
