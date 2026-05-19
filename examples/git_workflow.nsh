#!/usr/bin/env npcsh
# example_git_workflow.nsh
# Demonstrates git workflow automation with jinxes.

# Get current branch
$branch = !git branch --show-current

# Get recent commits
$commits = !git log --oneline -5

# Generate release notes using a jinx
$notes = /release_notes $commits

# Review with NPC
$review = @default Is this release note clear and accurate? $notes

# If review is positive, create a tag
!echo "$review" | grep -qi "yes" && git tag -a v$version -m "$notes"

# Push
!git push origin $branch

!echo "Workflow complete for branch: $branch"
