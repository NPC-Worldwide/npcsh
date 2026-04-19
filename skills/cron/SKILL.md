---
name: cron
description: Manage cron jobs. Add, remove, list, or check scheduled tasks. Works
  cross-platform (crontab on Linux, launchd on macOS).
---

# cron

Manage cron jobs. Add, remove, list, or check scheduled tasks. Works cross-platform (crontab on Linux, launchd on macOS).

## Inputs

- `action` (default: `'list'`)
- `schedule` (default: `''`)
- `command` (default: `''`)
- `name` (default: `''`)

## Steps

- `manage_cron` → [`manage_cron.py`](./manage_cron.py)

## Usage

```
/run_jinx jinx_ref=cron input_values={"action": "list", "schedule": "", "command": "", "name": ""}
```
