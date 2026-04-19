---
name: systemd
description: Manage background services/daemons. Works cross-platform (launchd on
  macOS, systemd on Linux).
---

# systemd

Manage background services/daemons. Works cross-platform (launchd on macOS, systemd on Linux).

## Inputs

- `action` (default: `'list'`)
- `command` (default: `''`)
- `name` (default: `''`)
- `restart_on_failure` (default: `'true'`)

## Steps

- `manage_service` → [`manage_service.py`](./manage_service.py)

## Usage

```
/run_jinx jinx_ref=systemd input_values={"action": "list", "command": "", "name": "", "restart_on_failure": "true"}
```
