---
description: Systematic debugging methodology. Use when asked to debug, troubleshoot, or diagnose issues.
---
# Debugging

## reproduce
First, reproduce the bug consistently:
- Get exact steps to trigger the issue
- Note the environment (OS, versions, config)
- Determine if it's deterministic or intermittent
- Find the minimal reproduction case

If you can't reproduce it, you can't verify a fix.

## isolate
Narrow down the cause:
- Binary search through the codebase (git bisect)
- Comment out / disable components to isolate
- Check recent changes (git log, blame)
- Add targeted logging at boundaries
- Test with minimal input/config

## diagnose
Once isolated, understand the root cause:
- Read the code path carefully, don't assume
- Check data flow — what goes in, what comes out?
- Look for state mutations, race conditions, stale caches
- Check external dependencies (API changes, schema drift)
- Verify assumptions with print/log/debugger

## fix
Apply the minimal fix:
- Fix the root cause, not the symptom
- Keep the change as small as possible
- Add a test that fails without the fix and passes with it
- Check for similar patterns elsewhere in the codebase
- Document WHY the bug happened if it's non-obvious

## verify
Confirm the fix works:
- Run the reproduction steps — bug should be gone
- Run the full test suite — no regressions
- Test edge cases around the fix
- Review the fix yourself before submitting
