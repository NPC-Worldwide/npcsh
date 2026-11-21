# Loops

A **loop** is a task that runs on a schedule, repeatedly. Loops are executed by a
background daemon (`npcshd`), not by the shell, so they keep running after you
close `npcsh` and after you log out.

## Creating a loop

```
/loop <npc> <interval> <task>
```

```
/loop corca 15s 'check htop output and note anything above 80% CPU'
/loop corca 2h  'look for open PRs on npcsh and summarize blockers'
/loop corca 6h  jinx:shell 'git status --short'
```

`interval` accepts `s`, `m`, `h`, or `d`. Prefix the task with `jinx:<name>` to
run a jinx directly instead of starting an agent turn.

Jobs are written to `~/.npcsh/loops.yaml`. You can edit that file by hand; the
daemon watches it and picks up changes without a restart. Your position in each
schedule is preserved across edits and restarts, so a 12h loop does not reset
every time you open the shell.

## Managing loops

| Command | Effect |
|---|---|
| `/loops` | TUI: list loops, runs, history, and details |
| `/loop <id>` | Attach to that loop's live output |
| `/detach` | Stop watching; the loop keeps running |

In the `/loops` TUI, `r` starts a run immediately, `Enter` attaches, `space`
toggles enabled, and `d` deletes.

## The daemon

Loops need a long-lived process to own the schedule and execute turns. That is
`npcshd`. `npcsh` starts one automatically the first time it needs to, and every
team gets its own daemon because a team determines which NPCs and jinxes exist.

```bash
npcshd --help
npcshd --install-service     # start at login, restart after a crash
npcshd --uninstall-service
```

`--install-service` registers with launchd on macOS or `systemd --user` on
Linux. It captures your shell's `PATH` and `NPCSH_*` overrides at install time,
because a service does not inherit them — without this, a jinx step would run
under system Python and fail to import `npcpy`.

A daemon with no loops exits after ten minutes of idleness. Set
`NPCSHD_IDLE_EXIT` to change that, or to `0` to keep it running. Add
`--install-service` if you want loops to survive a reboot.

### Inspecting the daemon

```bash
sqlite3 ~/npcsh_history.db "SELECT team_dir,pid,port,status FROM daemon_state;"
tail -f ~/.npcsh/logs/npcshd.log
```

Each run's output is written to
`~/.npcsh/loops/<npc>/<task-slug>/runs/`, and a per-loop `loop.log` records
start/finish and any failure.

## Safety

Loops run with permissions auto-approved and without you watching, so scope the
task text accordingly: name the files a loop may touch, and say "read-only" when
it should not write. A loop whose NPC or jinx is missing backs off exponentially
instead of retrying at full rate, and its errors go to the loop log rather than
your prompt.
