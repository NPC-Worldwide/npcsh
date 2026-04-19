---
name: pr_review
description: Review a pull request using LLM analysis. Fetches diff and description,
  then provides a structured code review.
---

# pr_review

Review a pull request using LLM analysis. Fetches diff and description, then provides a structured code review.

## Inputs

- `pr` (default: `''`)

## Steps

- `review_pr` → [`review_pr.py`](./review_pr.py)

## Usage

```
/run_jinx jinx_ref=pr_review input_values={"pr": ""}
```
