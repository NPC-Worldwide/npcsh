//! `npcshd` client/registry plumbing.
//!
//! Loops must outlive any single `npcsh` invocation, so a separate process owns
//! the kernel, the cron registry, and the ticker. `npcsh` is a client: it can
//! create loops, list them, run one now, stop one, and watch output.
//!
//! ## Identity is per-team
//!
//! A daemon is bound to one team directory, because a team determines which
//! NPCs and jinxes exist. `daemon_state` is therefore keyed by `team_dir`, not
//! by a single global row. An earlier version used `id = 1` with
//! `CHECK (id = 1)`, which allowed exactly one daemon for the whole machine:
//! whichever team started first owned the row, and a shell in any *other* team
//! would find that row, believe a daemon was up, and attach to a process
//! running someone else's team. That surfaced as loops failing with
//! "NPC '<name>' not found in registered teams".
//!
//! Discovery uses the `daemon_state` table in the history DB: each daemon binds
//! an ephemeral loopback port and records it, so no hardcoded port and no Unix
//! socket (unavailable on Windows).

use serde::{Deserialize, Serialize};

pub const DAEMON_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub pid: u32,
    pub port: u16,
    pub team_dir: String,
    pub started_at: String,
    pub last_heartbeat: String,
    pub status: String,
}

/// A request from a client to the daemon. One JSON object per line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    Ping,
    List,
    Run {
        id: u32,
    },
    Stop {
        id: u32,
    },
    Enable {
        id: u32,
        enabled: bool,
    },
    /// Re-read `~/.npcsh/loops.yaml` and jinx-declared crons.
    Reload,
    /// Stream the live events of a running loop until it finishes.
    Attach {
        id: u32,
    },
    Shutdown,
}

/// One scheduled job as reported by the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobInfo {
    pub id: u32,
    pub npc: String,
    pub interval_secs: u64,
    pub task: String,
    pub kind: String,
    pub enabled: bool,
    pub running: bool,
    pub next_in_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uptime_secs: Option<u64>,
    /// Protocol version. A mismatch means the client and daemon disagree about
    /// the wire format, so callers can refuse to interpret the reply.
    #[serde(default)]
    pub version: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub jobs: Vec<JobInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub running: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

impl Response {
    pub fn ok() -> Self {
        Self {
            ok: true,
            error: None,
            pid: None,
            team_dir: None,
            uptime_secs: None,
            version: DAEMON_VERSION,
            jobs: Vec::new(),
            running: Vec::new(),
            output: None,
        }
    }

    pub fn err(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            error: Some(msg.into()),
            pid: None,
            team_dir: None,
            uptime_secs: None,
            version: DAEMON_VERSION,
            jobs: Vec::new(),
            running: Vec::new(),
            output: None,
        }
    }
}

// ── daemon_state persistence ─────────────────────────────────────────────────

fn open(db_path: &str) -> Option<rusqlite::Connection> {
    let expanded = shellexpand::tilde(db_path).to_string();
    rusqlite::Connection::open(expanded).ok()
}

/// Canonical form of a team directory, so `/x/team` and `/x/team/` and a
/// relative path all map to the same daemon.
pub fn team_key(team_dir: &str) -> String {
    let path = std::path::Path::new(team_dir);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|c| c.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    // Canonicalize when possible; fall back to a lexical cleanup so a deleted
    // directory still produces a stable key.
    let resolved = std::fs::canonicalize(&abs).unwrap_or(abs);
    resolved.to_string_lossy().trim_end_matches('/').to_string()
}

/// Create the table in the per-team shape, migrating away from the old global
/// `id = 1` layout. `daemon_state` is a live cache of running processes, so
/// dropping it loses nothing: rows are re-published on the next daemon start.
fn ensure_schema(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    let has_team_col = conn
        .prepare("SELECT team_dir FROM daemon_state LIMIT 1")
        .is_ok();

    if !has_team_col {
        // Old global shape (or no table at all). Recreate.
        let _ = conn.execute("DROP TABLE IF EXISTS daemon_state", []);
    }

    conn.execute(
        "CREATE TABLE IF NOT EXISTS daemon_state (
            team_dir TEXT PRIMARY KEY,
            pid INTEGER,
            port INTEGER,
            started_at DATETIME,
            last_heartbeat DATETIME,
            status TEXT
        )",
        [],
    )?;
    Ok(())
}

fn ensure_schema_at(db_path: &str) -> Option<rusqlite::Connection> {
    let conn = open(db_path)?;
    ensure_schema(&conn).ok()?;
    Some(conn)
}

/// Is this pid a live process? Used to reject rows left by a crashed daemon.
fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // Signal 0 performs error checking without delivering a signal.
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

pub fn read_status(db_path: &str, team_dir: &str) -> Option<DaemonStatus> {
    let conn = ensure_schema_at(db_path)?;
    let key = team_key(team_dir);
    let mut stmt = conn
        .prepare(
            "SELECT pid, port, started_at, last_heartbeat, status
             FROM daemon_state WHERE team_dir = ?1",
        )
        .ok()?;
    let mut rows = stmt
        .query_map([&key], |row| {
            Ok(DaemonStatus {
                pid: row.get::<_, Option<i64>>(0)?.unwrap_or(0) as u32,
                port: row.get::<_, Option<i64>>(1)?.unwrap_or(0) as u16,
                team_dir: key.clone(),
                started_at: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                last_heartbeat: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                status: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            })
        })
        .ok()?;
    let st = rows.next()?.ok()?;
    // A crashed daemon leaves a row behind. Report it as absent so the caller
    // starts a replacement instead of dialling a dead port.
    if !pid_alive(st.pid) {
        return None;
    }
    Some(st)
}

pub fn write_status(db_path: &str, st: &DaemonStatus) -> Option<()> {
    let conn = ensure_schema_at(db_path)?;
    conn.execute(
        "INSERT INTO daemon_state (team_dir, pid, port, started_at, last_heartbeat, status)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(team_dir) DO UPDATE SET
             pid = excluded.pid,
             port = excluded.port,
             started_at = excluded.started_at,
             last_heartbeat = excluded.last_heartbeat,
             status = excluded.status",
        rusqlite::params![
            team_key(&st.team_dir),
            st.pid as i64,
            st.port as i64,
            st.started_at,
            st.last_heartbeat,
            st.status
        ],
    )
    .ok()?;
    Some(())
}

pub fn touch_heartbeat(db_path: &str, team_dir: &str, status: &str) -> Option<()> {
    let conn = open(db_path)?;
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE daemon_state SET last_heartbeat = ?1, status = ?2 WHERE team_dir = ?3",
        rusqlite::params![now, status, team_key(team_dir)],
    )
    .ok()?;
    Some(())
}

/// Remove this team's row. Only call when this process owns it.
pub fn clear_status(db_path: &str, team_dir: &str) {
    if let Some(conn) = open(db_path) {
        let _ = conn.execute(
            "DELETE FROM daemon_state WHERE team_dir = ?1 AND pid = ?2",
            rusqlite::params![team_key(team_dir), std::process::id() as i64],
        );
    }
}

/// Drop rows whose process is gone, so `/loops` and friends stop listing ghosts.
pub fn prune_dead(db_path: &str) {
    let Some(conn) = ensure_schema_at(db_path) else {
        return;
    };
    let Ok(mut stmt) = conn.prepare("SELECT team_dir, pid FROM daemon_state") else {
        return;
    };
    let dead: Vec<String> = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<i64>>(1)?.unwrap_or(0) as u32,
            ))
        })
        .map(|rows| {
            rows.flatten()
                .filter(|(_, pid)| !pid_alive(*pid))
                .map(|(team, _)| team)
                .collect()
        })
        .unwrap_or_default();
    for team in dead {
        let _ = conn.execute(
            "DELETE FROM daemon_state WHERE team_dir = ?1",
            rusqlite::params![team],
        );
    }
}

// ── client side ──────────────────────────────────────────────────────────────

/// Send one request to this team's daemon and read one response line.
///
/// Returns `None` when there is no reachable daemon for this team, so callers
/// can fall back or report "not running" without treating it as an error.
pub async fn request(db_path: &str, team_dir: &str, req: &Request) -> Option<Response> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let st = read_status(db_path, team_dir)?;
    if st.port == 0 {
        return None;
    }
    let stream = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        tokio::net::TcpStream::connect(("127.0.0.1", st.port)),
    )
    .await
    .ok()?
    .ok()?;

    let (rd, mut wr) = stream.into_split();
    let mut line = serde_json::to_string(req).ok()?;
    line.push('\n');
    wr.write_all(line.as_bytes()).await.ok()?;
    wr.flush().await.ok()?;

    let mut reader = BufReader::new(rd);
    let mut buf = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        reader.read_line(&mut buf),
    )
    .await
    .ok()?
    .ok()?;
    let resp: Response = serde_json::from_str(buf.trim()).ok()?;
    if resp.version != DAEMON_VERSION {
        // Keep the error text useful: this shows up as "no daemon" in most
        // callers, and a stale binary is the usual cause.
        eprintln!(
            "npcsh: loop daemon speaks protocol v{} but this client expects v{}; \
             restart the daemon (pkill npcshd) or reinstall",
            resp.version, DAEMON_VERSION
        );
        return None;
    }
    Some(resp)
}

/// Is there a live daemon for *this* team answering on its recorded port?
pub async fn is_alive(db_path: &str, team_dir: &str) -> bool {
    matches!(
        request(db_path, team_dir, &Request::Ping).await,
        Some(r) if r.ok
    )
}

/// Default history DB path, honouring `NPCSH_HISTORY_DB`.
pub fn default_db_path() -> String {
    std::env::var("NPCSH_HISTORY_DB")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| shellexpand::tilde("~/npcsh_history.db").to_string())
}

// ── convenience wrappers for the current process's team ──────────────────────
//
// A `npcsh` process serves exactly one team, resolved at startup. These wrap
// the team-keyed API so call sites do not thread `team_dir` through every
// layer, while still keeping the explicit form available for the daemon (which
// must be told its team) and for tests that target another team.

/// The team this process serves, resolved once at startup.
///
/// `find_team_dir()` may prompt the user to pick from registered teams when the
/// project layout is ambiguous. It must never be called from a request handler:
/// it would block a socket round-trip on stdin and dump a team picker into the
/// middle of the prompt. So the resolver runs once and the answer is cached.
static CURRENT_TEAM: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Record the team this process serves. Safe to call more than once; the first
/// call wins so a later accidental call cannot move the process to another team.
pub fn set_current_team(team_dir: &str) {
    let _ = CURRENT_TEAM.set(team_key(team_dir));
}

fn current_team() -> String {
    CURRENT_TEAM
        .get()
        .cloned()
        .unwrap_or_else(|| team_key(&crate::find_team_dir()))
}

pub async fn request_current(req: &Request) -> Option<Response> {
    request(&default_db_path(), &current_team(), req).await
}

pub async fn is_current_alive() -> bool {
    is_alive(&default_db_path(), &current_team()).await
}

pub fn read_current_status() -> Option<DaemonStatus> {
    read_status(&default_db_path(), &current_team())
}
