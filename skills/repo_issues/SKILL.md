---
name: repo_issues
description: Fetch GitHub issues for a repo and run each through LLM loop analysis,
  producing a structured report.
---

# repo_issues

Fetch GitHub issues for a repo and run each through LLM loop analysis, producing a structured report.

## Inputs

- `repo` (default: `''`)
- `state` (default: `'open'`)
- `limit` (default: `'20'`)
- `label` (default: `''`)
- `analysis_prompt` (default: `''`)

## Steps

- `fetch_and_analyze_issues` → [`fetch_and_analyze_issues.py`](./fetch_and_analyze_issues.py)

## Usage

```
/run_jinx jinx_ref=repo_issues input_values={"repo": "", "state": "open", "limit": "20", "label": "", "analysis_prompt": ""}
```
