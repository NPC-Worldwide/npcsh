---
description: Conducts code reviews. Use when asked to review code, review a PR, or check code quality.
---
# Code Review

## approach
Start by reading the full diff to understand the scope of changes.
Identify the purpose of the change — is it a bug fix, new feature, refactor, or config change?
Note which files are modified and how they relate to each other.

## structure
Check the following structural concerns:
- Are changes in the right files/modules?
- Is new code placed in a logical location?
- Are there any circular dependencies introduced?
- Is the change minimal — does it avoid unnecessary modifications?

## correctness
Review for correctness:
- Does the logic handle edge cases (null, empty, boundary values)?
- Are error paths handled properly?
- Is state management correct (no race conditions, stale data)?
- Do new functions have the right return types?

## style
Check style and conventions:
- Naming: are variables, functions, classes named clearly?
- Consistency: does the code match the project's existing patterns?
- Comments: are complex sections explained? Are there unnecessary comments?
- Formatting: consistent indentation, line length, spacing?

## testing
Evaluate test coverage:
- Are new code paths tested?
- Are edge cases covered?
- Do tests actually assert meaningful behavior (not just "runs without error")?
- Are there integration tests where needed?

## security
Check for security issues:
- Input validation at system boundaries
- No hardcoded secrets or credentials
- SQL injection, XSS, command injection prevention
- Proper authentication/authorization checks
- Sensitive data not logged or exposed
