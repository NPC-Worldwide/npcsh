# Skills

Skills are jinxs that serve knowledge content instead of executing code. They use the `skill.jinx` sub-jinx (like `python.jinx` or `sh.jinx`) and return sections of instructional methodology on demand.

Skills are not a separate system. They live in `jinxs/skills/`, load through the same compiler, and end up in `jinxs_dict` alongside every other jinx. Agents get skills through the same `jinxs:` list in `.npc` files — no separate configuration needed.

## Using Skills in npcsh

Skills appear as slash commands like any jinx:

```bash
/debugging                         # Returns all sections
/debugging -s reproduce            # Returns just the reproduce section
/debugging -s list                 # Returns available section names
/code-review -s correctness        # Returns the correctness checklist
/git-workflow -s commits           # Returns the commits section
```

The `-s` / `--section` parameter controls what gets returned:

| Value | Result |
|-------|--------|
| *(omitted)* or `all` | All sections concatenated |
| `list` | Names of available sections |
| `reproduce` | Exact section match |
| `repro` | Fuzzy match — returns `reproduce` |

## Built-in Skills

npcsh ships with three example skills:

**code-review** (SKILL.md format) — approach, structure, correctness, style, testing, security

```bash
/code-review -s correctness
```
```
Review for correctness:
- Does the logic handle edge cases (null, empty, boundary values)?
- Are error paths handled properly?
- Is state management correct (no race conditions, stale data)?
- Do new functions have the right return types?
```

**debugging** (SKILL.md format) — reproduce, isolate, diagnose, fix, verify

```bash
/debugging -s isolate
```
```
Narrow down the cause:
- Binary search through the codebase (git bisect)
- Comment out / disable components to isolate
- Check recent changes (git log, blame)
- Add targeted logging at boundaries
- Test with minimal input/config
```

**git-workflow** (.jinx format) — branching, commits, merging, hotfix

```bash
/git-workflow -s commits
```
```
Write clear commit messages:
  Line 1: imperative summary under 72 chars
  Line 3+: explain WHY, not what (the diff shows what)
One logical change per commit. Don't mix refactors with features.
Use conventional commits if the project uses them:
  feat: add user search
  fix: handle null avatar URL
  chore: bump eslint to v9
```

## Authoring Skills

### SKILL.md folder (recommended)

Create a folder in `jinxs/skills/` with a `SKILL.md` file. The folder name becomes the skill name.

```
jinxs/skills/deployment/
  SKILL.md
  scripts/          # Optional
  references/       # Optional
  assets/           # Optional
```

The `SKILL.md` has YAML frontmatter and `##`-delimited sections:

```markdown
---
description: Deployment checklist. Use when asked about deploying or releasing.
---
# Deployment

## pre-deploy
- Run full test suite
- Check for env var changes
- Review database migrations
- Verify feature flags

## deploy
- Tag the release
- Deploy to staging first
- Run smoke tests
- Deploy to production

## rollback
If something breaks:
- Revert to previous tag
- Check logs for the failure point
- Notify the team
```

### .jinx format

A regular jinx with `engine: skill` steps. Use this when you want explicit control over the structured data.

```yaml
jinx_name: api-design
description: "API design principles. [Sections: naming, errors, versioning]"
inputs:
- section: all
steps:
  - engine: skill
    skill_name: api-design
    skill_description: API design principles.
    sections:
      naming: |
        Use nouns for resources: /users, /orders
        Use HTTP verbs for actions: GET, POST, PUT, DELETE
        Plural resource names: /users not /user
      errors: |
        Return appropriate HTTP status codes.
        Include error message and code in response body.
        400 for client errors, 500 for server errors.
      versioning: |
        Version in the URL: /v1/users
        Never break existing clients.
        Deprecate before removing.
    scripts_json: '[]'
    references_json: '[]'
    assets_json: '[]'
    section: '{{section}}'
```

## Assigning Skills to Agents

Skills are jinxs, so assign them through the `jinxs:` list in `.npc` files:

```yaml
# reviewer.npc
name: reviewer
primary_directive: |
  You review code and provide actionable feedback.
model: llama3.2
provider: ollama
jinxs:
  - lib/core/sh
  - lib/core/python
  - skills/code-review
  - skills/debugging
```

The agent sees `code-review` and `debugging` as tools alongside `sh` and `python`. When it encounters a review task, it can call `code-review(section=correctness)` to get methodology, then use `python` or `sh` to inspect the actual code.

Team-level skills work the same way: any skill in the team's `jinxs/skills/` directory is available to all NPCs on the team, just like any other team jinx.

## Importing External Skills

Add `SKILLS_DIRECTORY` to your `.ctx` file to load skills from an external directory:

```yaml
model: llama3.2
provider: ollama
forenpc: lead-dev
SKILLS_DIRECTORY: ~/shared-skills
```

The path can be absolute or relative to the team directory. All `SKILL.md` folders and `.jinx` files in that directory are loaded alongside the team's own jinxs.

This lets you maintain a shared skills library across multiple teams without copying files.

## How It Works

Skills flow through the same pipeline as every other jinx:

1. `SKILL.md` folders are compiled into Jinx objects with `engine: skill` steps
2. `.jinx` skill files are loaded directly (they already have `engine: skill` steps)
3. Both end up in `jinxs_dict` alongside regular jinxs
4. During first-pass rendering, `engine: skill` expands through `skill.jinx`
5. At execution time, `skill.jinx` receives the structured data and returns the requested section

The agent calls skills identically to any other jinx — `{"action": "jinx", "jinx_name": "debugging", "inputs": {"section": "reproduce"}}`. The only difference is what comes back: content instead of code execution output.
