---
name: git-workflow
description: Git workflow best practices. Use when asked about git branching, commits, or merge strategy. [Sections: branching, commits, merging, hotfix]
source_jinx: npcsh/npc_team/jinxes/skills/git-workflow.jinx
engine: skill
---

# git-workflow

Git workflow best practices. Use when asked about git branching, commits, or merge strategy. [Sections: branching, commits, merging, hotfix]

## Branching

Use feature branches off main/develop:
  git checkout -b feature/my-feature develop
Name branches descriptively:
  feature/ — new functionality
  fix/ — bug fixes
  chore/ — maintenance, deps, config
Keep branches short-lived. Rebase onto target before merging.


## Commits

Write clear commit messages:
  Line 1: imperative summary under 72 chars
  Line 3+: explain WHY, not what (the diff shows what)
One logical change per commit. Don't mix refactors with features.
Use conventional commits if the project uses them:
  feat: add user search
  fix: handle null avatar URL
  chore: bump eslint to v9


## Merging

Prefer squash merges for feature branches (clean history).
Use merge commits for long-lived branches (preserves context).
Always pull/rebase before merging to avoid unnecessary merge commits.
Delete branches after merging.
Run CI before merging — never merge a red build.


## Hotfix

For urgent production fixes:
  1. Branch off main: git checkout -b hotfix/fix-name main
  2. Make the minimal fix
  3. Test thoroughly
  4. Merge to main AND develop/release
  5. Tag the release
Keep hotfix scope minimal — fix only the immediate issue.
