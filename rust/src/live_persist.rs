//! Live conversation persistence.
//!
//! A turn used to be written to `conversation_history` only after the whole
//! agent loop finished (and the final assistant row was written twice: once
//! without usage and once with it). If the user hit Ctrl+C, or the server
//! died, or the binary was killed mid-stream, everything that had happened so
//! far vanished.
//!
//! This module makes persistence incremental:
//!
//! * `LiveTurn` is created as soon as a turn starts. It owns the user row and
//!   then appends one row per assistant message / tool call / tool result as
//!   they arrive off the SSE stream.
//! * Each row is committed to SQLite immediately (`synchronous=FULL`), so a
//!   Ctrl+C in the middle of a 20 minute tool loop leaves rows 1..N on disk.
//! * The turn is *not* closed on interrupt: the last assistant row is marked
//!   `interrupted` and the row is left in place. `resume_context()` can then
//!   rebuild the message list for the next turn.
//!
//! The DB writes are deliberately best-effort: losing a log line must never
//! take down the agent loop. Every failure is reported on stderr once per
//! process so the user knows the transcript is not being recorded.

use crate::stream_client::{StreamEvent, StreamResponse};
use npcrs::Message;
use rusqlite::{Connection, OpenFlags, params};
use std::sync::OnceLock;

const DIM: &str = "\x1b[90m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";

fn warn_once(err: &str) {
    static WARNED: OnceLock<bool> = OnceLock::new();
    if WARNED.set(true).is_err() {
        return;
    }
    eprintln!("{YELLOW}[history] could not persist turn: {err}{RESET}");
}

/// Where the transcript lives. Mirrors the rest of npcsh so `npcsh /history`
/// and the memory tools keep seeing the same rows.
pub fn history_db_path() -> String {
    std::env::var("NPCSH_HISTORY_DB")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("NPCSH_DB_PATH")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .map(|p| shellexpand::tilde(&p).to_string())
        .unwrap_or_else(|| shellexpand::tilde("~/npcsh_history.db").to_string())
}

/// Open a private connection for streaming writes.
///
/// The kernel already holds its own connection; SQLite in WAL mode allows many
/// readers plus one writer, and we want the writer to be available the instant
/// a chunk lands rather than waiting for the kernel to hand over `&mut self`.
fn open_conn(path: &str) -> rusqlite::Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )?;
    // FULL makes every committed row durable across SIGINT / SIGKILL of the
    // CLI process. The rows are tiny; the cost is negligible next to an LLM
    // round-trip.
    let _ = conn.pragma_update(None, "synchronous", "FULL");
    let _ = conn.busy_timeout(std::time::Duration::from_secs(10));
    let _ = conn.execute(
        "CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT UNIQUE,
            timestamp TEXT,
            role TEXT,
            content TEXT,
            conversation_id TEXT,
            directory_path TEXT,
            model TEXT,
            provider TEXT,
            npc TEXT,
            team TEXT,
            reasoning_content TEXT,
            tool_calls TEXT,
            tool_results TEXT,
            parent_message_id TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost VARCHAR(50),
            interrupted INTEGER DEFAULT 0
        )",
        [],
    );
    // Older databases predate the interrupted marker; add it lazily.
    let _ = conn.execute(
        "ALTER TABLE conversation_history ADD COLUMN interrupted INTEGER DEFAULT 0",
        [],
    );
    Ok(conn)
}

fn now_ts() -> String {
    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// A turn whose rows are appended to disk as the agent works.
pub struct LiveTurn {
    conn: Option<Connection>,
    conversation_id: String,
    directory_path: String,
    model: String,
    provider: String,
    npc: String,
    team: String,
    /// message_id of the most recent row, used as `parent_message_id`.
    last_message_id: Option<String>,
    /// message_id of the assistant row currently being streamed into.
    pub assistant_message_id: Option<String>,
    /// Number of rows appended (including the user row).
    pub rows_written: usize,
    pub enabled: bool,
    /// Assistant text accumulated for the row currently being streamed.
    assistant_buffer: String,
    /// Tool calls seen for the current assistant row (serialized JSON).
    pending_tool_calls: Vec<serde_json::Value>,
    /// Throttle: last time the in-flight assistant row was flushed.
    last_flush: std::time::Instant,
    chars_since_flush: usize,
    /// Tool call ids that already have a `tool` row on disk. Results are
    /// matched by id rather than by position, so a failed tool (which the
    /// stream reports but the response summary omits) cannot shift the
    /// alignment and cause a real result to be skipped.
    written_tool_ids: std::collections::HashSet<String>,
    /// Number of `tool` rows written for this turn (used to avoid duplicates
    /// when the turn ends and we reconcile with `response.tool_results`).
    pub tool_rows_written: usize,
    /// Set once the turn completed normally; `Drop` only marks the row
    /// interrupted when this is still false.
    finalized: bool,
}

impl LiveTurn {
    /// Begin a turn and immediately durably write the user message.
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        conversation_id: &str,
        user_input: &str,
        directory_path: &str,
        model: &str,
        provider: &str,
        npc: &str,
        team: &str,
    ) -> Self {
        let mut turn = Self {
            conn: None,
            conversation_id: conversation_id.to_string(),
            directory_path: directory_path.to_string(),
            model: model.to_string(),
            provider: provider.to_string(),
            npc: npc.to_string(),
            team: team.to_string(),
            last_message_id: None,
            assistant_message_id: None,
            rows_written: 0,
            enabled: !conversation_id.is_empty(),
            assistant_buffer: String::new(),
            pending_tool_calls: Vec::new(),
            last_flush: std::time::Instant::now(),
            chars_since_flush: 0,
            tool_rows_written: 0,
            written_tool_ids: std::collections::HashSet::new(),
            finalized: false,
        };

        if !turn.enabled {
            return turn;
        }

        match open_conn(&history_db_path()) {
            Ok(conn) => turn.conn = Some(conn),
            Err(e) => {
                warn_once(&e.to_string());
                turn.enabled = false;
                return turn;
            }
        }

        // Idempotent: a retry after a server error re-enters with the same
        // input. If the last row is an unanswered user row with identical
        // content, reuse it instead of writing a duplicate turn.
        match turn.dangling_user_row(user_input) {
            Some(id) => turn.last_message_id = Some(id),
            None => {
                turn.append("user", user_input, None, None, None, None);
            }
        }
        turn
    }

    /// message_id of the trailing `user` row with this exact content, i.e. a
    /// user row that has not been followed by any assistant output yet.
    fn dangling_user_row(&self, content: &str) -> Option<String> {
        let conn = self.conn.as_ref()?;
        let mut stmt = conn
            .prepare(
                "SELECT message_id, role, content FROM conversation_history
                 WHERE conversation_id = ?1
                 ORDER BY id DESC LIMIT 1",
            )
            .ok()?;
        let row = stmt
            .query_row(params![self.conversation_id], |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                ))
            })
            .ok()?;
        let (message_id, role, last_content) = row;
        if role == "user" && last_content == content && !message_id.is_empty() {
            Some(message_id)
        } else {
            None
        }
    }

    /// Re-open persistence for an existing turn (used by resume paths).
    pub fn attach(conversation_id: &str, directory_path: &str) -> Self {
        let mut turn = Self {
            conn: None,
            conversation_id: conversation_id.to_string(),
            directory_path: directory_path.to_string(),
            model: String::new(),
            provider: String::new(),
            npc: String::new(),
            team: String::new(),
            last_message_id: None,
            assistant_message_id: None,
            rows_written: 0,
            enabled: !conversation_id.is_empty(),
            assistant_buffer: String::new(),
            pending_tool_calls: Vec::new(),
            last_flush: std::time::Instant::now(),
            chars_since_flush: 0,
            tool_rows_written: 0,
            written_tool_ids: std::collections::HashSet::new(),
            finalized: false,
        };
        if !turn.enabled {
            return turn;
        }
        match open_conn(&history_db_path()) {
            Ok(conn) => turn.conn = Some(conn),
            Err(e) => {
                warn_once(&e.to_string());
                turn.enabled = false;
            }
        }
        turn
    }

    fn append(
        &mut self,
        role: &str,
        content: &str,
        tool_calls: Option<&str>,
        tool_results: Option<&str>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) -> Option<String> {
        let Some(conn) = self.conn.as_ref() else {
            return None;
        };
        let message_id = new_id();
        let res = conn.execute(
            "INSERT INTO conversation_history
             (message_id, timestamp, role, content, conversation_id, directory_path,
              model, provider, npc, team, tool_calls, tool_results,
              parent_message_id, input_tokens, output_tokens, interrupted)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, 0)",
            params![
                message_id,
                now_ts(),
                role,
                content,
                self.conversation_id,
                self.directory_path,
                self.model,
                self.provider,
                self.npc,
                self.team,
                tool_calls,
                tool_results,
                self.last_message_id,
                input_tokens.map(|t| t as i64),
                output_tokens.map(|t| t as i64),
            ],
        );
        match res {
            Ok(_) => {
                self.rows_written += 1;
                self.last_message_id = Some(message_id.clone());
                Some(message_id)
            }
            Err(e) => {
                warn_once(&e.to_string());
                None
            }
        }
    }

    /// Record an assistant message. When `streaming` is true the row is written
    /// as soon as we have content, before the turn completes, so a Ctrl+C keeps
    /// whatever the model had already produced.
    pub fn assistant(
        &mut self,
        content: &str,
        tool_calls_json: Option<&str>,
        reasoning: Option<&str>,
    ) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let tool_calls = tool_calls_json.filter(|s| !s.is_empty() && *s != "[]");
        // A turn writes ONE assistant row per agent-loop step. If a row is
        // already in flight, update it in place rather than appending a
        // duplicate (the old code wrote the final assistant message twice).
        let id = match self.assistant_message_id.clone() {
            Some(id) => {
                if let Some(conn) = self.conn.as_ref() {
                    let _ = conn.execute(
                        "UPDATE conversation_history
                         SET content = ?1, tool_calls = COALESCE(?2, tool_calls)
                         WHERE message_id = ?3",
                        params![content, tool_calls, id],
                    );
                }
                id
            }
            None => self.append("assistant", content, tool_calls, None, None, None)?,
        };
        if let (Some(conn), Some(reasoning)) = (self.conn.as_ref(), reasoning) {
            if !reasoning.is_empty() {
                let _ = conn.execute(
                    "UPDATE conversation_history SET reasoning_content = ?1 WHERE message_id = ?2",
                    params![reasoning, id],
                );
            }
        }
        self.assistant_message_id = Some(id.clone());
        Some(id)
    }

    /// Record a tool result row. Returns the row id.
    pub fn tool_result(&mut self, name: &str, tool_call_id: &str, content: &str) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let payload = serde_json::json!({
            "name": name,
            "tool_call_id": tool_call_id,
            "content": content,
        })
        .to_string();
        let id = self.append("tool", content, None, Some(&payload), None, None);
        if id.is_some() {
            self.tool_rows_written += 1;
            if !tool_call_id.is_empty() {
                self.written_tool_ids.insert(tool_call_id.to_string());
            }
        }
        id
    }

    /// Record any already-built `npcrs` message verbatim.
    pub fn record_message(&mut self, msg: &Message) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let content = msg.content.as_deref().unwrap_or("");
        let tool_calls_json = msg
            .tool_calls
            .as_ref()
            .map(|tcs| serde_json::to_string(tcs).unwrap_or_default())
            .filter(|s| !s.is_empty());
        let tool_results_json = msg.name.as_ref().map(|name| {
            serde_json::json!({
                "name": name,
                "tool_call_id": msg.tool_call_id.as_deref().unwrap_or(""),
                "content": content,
            })
            .to_string()
        });
        let id = self.append(
            &msg.role,
            content,
            tool_calls_json.as_deref(),
            tool_results_json.as_deref(),
            None,
            None,
        )?;
        if let Some(reasoning) = msg.reasoning_content.as_deref().or(msg.thinking.as_deref()) {
            if !reasoning.is_empty() {
                if let Some(conn) = self.conn.as_ref() {
                    let _ = conn.execute(
                        "UPDATE conversation_history SET reasoning_content = ?1 WHERE message_id = ?2",
                        params![reasoning, id],
                    );
                }
            }
        }
        Some(id)
    }

    /// Feed streamed assistant text into the in-flight assistant row.
    ///
    /// Rows are written at most once every `NPCSH_LIVE_FLUSH_MS` (default 500ms)
    /// or every `NPCSH_LIVE_FLUSH_CHARS` characters (default 400), whichever
    /// comes first. That keeps a SIGINT from losing more than half a second of
    /// output without issuing one INSERT per token.
    pub fn stream_delta(&mut self, delta: &str) {
        if !self.enabled {
            return;
        }
        self.assistant_buffer.push_str(delta);
        self.chars_since_flush += delta.chars().count();

        let flush_ms = std::env::var("NPCSH_LIVE_FLUSH_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(500);
        let flush_chars = std::env::var("NPCSH_LIVE_FLUSH_CHARS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(400);

        // First chunk of a row: write it at once so the row id exists and a
        // later interrupt updates that row in place instead of appending a new
        // one out of order.
        let due = self.assistant_message_id.is_none()
            || self.chars_since_flush >= flush_chars
            || self.last_flush.elapsed().as_millis() as u64 >= flush_ms;
        if due {
            self.flush_assistant();
        }
    }

    /// Persist the buffered assistant text and any pending tool calls now.
    ///
    /// A tool-call-only row is legal and important: the API requires every
    /// `tool` message to be preceded by an assistant message that declares the
    /// call. So this must run even when no text was produced.
    pub fn flush_assistant(&mut self) {
        if !self.enabled {
            return;
        }
        if self.assistant_buffer.is_empty() && self.pending_tool_calls.is_empty() {
            return;
        }
        let content = self.assistant_buffer.clone();
        let tool_calls = if self.pending_tool_calls.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&self.pending_tool_calls).unwrap_or_default())
        };
        match self.assistant_message_id.clone() {
            Some(id) => {
                if let Some(conn) = self.conn.as_ref() {
                    let _ = conn.execute(
                        "UPDATE conversation_history
                         SET content = ?1, tool_calls = COALESCE(?2, tool_calls)
                         WHERE message_id = ?3",
                        params![content, tool_calls, id],
                    );
                }
            }
            None => {
                self.assistant(&content, tool_calls.as_deref(), None);
            }
        }
        self.last_flush = std::time::Instant::now();
        self.chars_since_flush = 0;
    }

    /// Start a new assistant row (new agent-loop iteration / final answer).
    pub fn begin_assistant(&mut self) {
        self.assistant_buffer.clear();
        self.pending_tool_calls.clear();
        self.assistant_message_id = None;
        self.chars_since_flush = 0;
        self.last_flush = std::time::Instant::now();
    }

    /// Note a tool call on the current assistant row before it is flushed.
    pub fn note_tool_call(&mut self, name: &str, id: &str, args: &str) {
        if !self.enabled {
            return;
        }
        // Must match `npcrs::ToolCall` exactly: `arguments` is a *string* on
        // the wire, and `load_messages` deserializes back into `ToolCall`. An
        // earlier version stored the parsed object here, which made every
        // reload silently drop the tool calls.
        self.pending_tool_calls.push(serde_json::json!({
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": args},
        }));
    }

    /// Attach token/cost usage to the assistant row that was written live,
    /// instead of writing a duplicate assistant row like the old code did.
    pub fn finalize_usage(&mut self, input_tokens: u64, output_tokens: u64, cost: f64) {
        if !self.enabled {
            return;
        }
        let cost_str = format!("{:.6}", cost);
        let Some(id) = self.assistant_message_id.clone() else {
            // No assistant row yet (e.g. interrupted before any content): write
            // one so usage/cost still lands somewhere.
            if input_tokens > 0 || output_tokens > 0 || cost > 0.0 {
                let written = self.append(
                    "assistant",
                    "",
                    None,
                    None,
                    Some(input_tokens),
                    Some(output_tokens),
                );
                if let (Some(conn), Some(written)) = (self.conn.as_ref(), written) {
                    let _ = conn.execute(
                        "UPDATE conversation_history SET cost = ?1 WHERE message_id = ?2",
                        params![cost_str, written],
                    );
                }
            }
            return;
        };
        if let Some(conn) = self.conn.as_ref() {
            let _ = conn.execute(
                "UPDATE conversation_history
                 SET input_tokens = ?1, output_tokens = ?2, cost = ?3
                 WHERE message_id = ?4",
                params![input_tokens as i64, output_tokens as i64, cost_str, id],
            );
        }
    }

    /// Mark the in-flight assistant row as interrupted so downstream tooling
    /// (viewer, resume) knows the turn was cut short rather than finished.
    pub fn mark_interrupted(&mut self, partial_content: &str) {
        if !self.enabled {
            return;
        }
        let Some(conn) = self.conn.as_ref() else {
            return;
        };
        match self.assistant_message_id.clone() {
            Some(id) => {
                let _ = conn.execute(
                    "UPDATE conversation_history
                     SET interrupted = 1,
                         content = CASE
                             WHEN content IS NULL OR content = '' THEN ?1
                             ELSE content
                         END
                     WHERE message_id = ?2",
                    params![partial_content, id],
                );
            }
            None => {
                if !partial_content.is_empty() {
                    let id = self.assistant(partial_content, None, None);
                    if let (Some(conn), Some(id)) = (self.conn.as_ref(), id) {
                        let _ = conn.execute(
                            "UPDATE conversation_history SET interrupted = 1 WHERE message_id = ?1",
                            params![id],
                        );
                    }
                }
            }
        }
    }

    /// Close a turn from the stream response: reconcile any tool rows the
    /// events did not carry, then either finish (with usage) or flag the
    /// partial turn as interrupted.
    pub fn close_from_response(&mut self, response: &StreamResponse) {
        if !self.enabled {
            return;
        }
        self.reconcile_tool_calls(&response.tool_calls, &response.tool_results);
        if response.interrupted {
            self.flush_and_interrupt();
            return;
        }
        let usage = response.usage.clone();
        let i = usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
        let o = usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0);
        let c = usage.as_ref().map(|u| u.cost_usd).unwrap_or(0.0);
        self.finish(i, o, c);
    }

    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    /// Text streamed into the in-flight assistant row so far.
    pub fn buffered_content(&self) -> &str {
        &self.assistant_buffer
    }

    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Flush whatever has been streamed and mark the row as `interrupted`.
    /// Safe to call at any point; it is what runs on Ctrl+C.
    pub fn flush_and_interrupt(&mut self) {
        if !self.enabled || self.finalized {
            return;
        }
        self.flush_assistant();
        self.mark_interrupted("");
        self.finalized = true;
    }

    /// Attach tool calls the sink never saw (older servers do not emit
    /// `tool_start` on the wire) and append any tool results that are missing.
    pub fn reconcile_tool_calls(
        &mut self,
        tool_calls: &[npcrs::ToolCall],
        tool_results: &[Message],
    ) {
        if !self.enabled {
            return;
        }
        // Only synthesize tool calls the stream never reported: that means
        // the server is too old to emit tool_start. Calls whose ids already
        // have a row on disk are skipped so the final assistant row is not
        // given duplicates of what earlier rows already declare.
        if self.pending_tool_calls.is_empty() && !tool_calls.is_empty() {
            for tc in tool_calls {
                if self.written_tool_ids.contains(&tc.id) {
                    continue;
                }
                self.pending_tool_calls
                    .push(serde_json::to_value(tc).unwrap_or_else(|_| serde_json::json!({})));
            }
            self.flush_assistant();
        }
        // Match results by tool_call_id: the response summary can omit a
        // failed tool that the stream did report, so positional skipping
        // would drop a genuine result and leave a dangling call.
        for tr in tool_results {
            let name = tr.name.clone().unwrap_or_default();
            let call_id = tr.tool_call_id.clone().unwrap_or_default();
            if !call_id.is_empty() && self.written_tool_ids.contains(&call_id) {
                continue;
            }
            let content = tr.content.clone().unwrap_or_default();
            self.tool_result(&name, &call_id, &content);
        }
    }

    /// Close the turn cleanly: flush the last assistant row and attach usage.
    /// After this the `Drop` impl will not mark the row interrupted.
    pub fn finish(&mut self, input_tokens: u64, output_tokens: u64, cost: f64) {
        if !self.enabled || self.finalized {
            return;
        }
        self.flush_assistant();
        self.finalize_usage(input_tokens, output_tokens, cost);
        self.finalized = true;
    }
}

/// Last-resort persistence.
///
/// Every early return, every `?`, and every panic unwind passes through here,
/// so a turn that never reached `finish()` still leaves its partial transcript
/// on disk and flags the dangling assistant row as interrupted.
impl Drop for LiveTurn {
    fn drop(&mut self) {
        if !self.enabled || self.finalized {
            return;
        }
        self.flush_and_interrupt();
    }
}

/// Rebuild the message history for a conversation so an interrupted turn is
/// not silently dropped when the user keeps going.
///
/// Tool rows are re-attached to their assistant turn via `tool_results` so the
/// next request keeps the tool call/result pairing the API requires.
fn tool_call_id_of(payload: &Option<String>) -> String {
    payload
        .as_deref()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| {
            v.get("tool_call_id")
                .and_then(|x| x.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

pub fn load_messages(conversation_id: &str) -> Vec<Message> {
    let mut out = Vec::new();
    if conversation_id.is_empty() {
        return out;
    }
    let Ok(conn) = open_conn(&history_db_path()) else {
        return out;
    };
    let Ok(mut stmt) = conn.prepare(
        "SELECT role, content, tool_calls, tool_results, reasoning_content
         FROM conversation_history
         WHERE conversation_id = ?1
         ORDER BY id ASC",
    ) else {
        return out;
    };

    let rows = stmt.query_map(params![conversation_id], |row| {
        Ok((
            row.get::<_, Option<String>>(0)?.unwrap_or_default(),
            row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<String>>(4)?,
        ))
    });

    let Ok(rows) = rows else {
        return out;
    };

    for row in rows.flatten() {
        let (role, content, tool_calls, tool_results, reasoning) = row;
        if role == "tool" {
            // A tool message is only valid directly after an assistant message
            // that declares the matching call. An interrupt can leave an
            // orphan; drop those rather than build a transcript the provider
            // will reject.
            let wanted = tool_call_id_of(&tool_results);
            let declared = match out.last() {
                Some(last) => last
                    .tool_calls
                    .as_ref()
                    .map(|tcs| tcs.iter().any(|tc| tc.id == wanted))
                    .unwrap_or(false),
                None => false,
            };
            if !declared {
                continue;
            }
            // Prefer the structured tool result payload for the id/name.
            let (name, call_id) = tool_results
                .as_deref()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                .map(|v| {
                    (
                        v.get("name").and_then(|x| x.as_str()).map(str::to_string),
                        v.get("tool_call_id")
                            .and_then(|x| x.as_str())
                            .map(str::to_string),
                    )
                })
                .unwrap_or((None, None));
            out.push(Message {
                role: "tool".to_string(),
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                tool_calls: None,
                tool_call_id: call_id,
                name,
                thinking: None,
                reasoning_content: None,
            });
            continue;
        }

        if role != "user" && role != "assistant" {
            continue;
        }

        // Skip the empty trailing assistant placeholder an interrupt can leave
        // behind; it carries no information for the next request.
        if role == "assistant" && content.is_empty() && tool_calls.is_none() {
            continue;
        }

        let tool_calls_parsed = tool_calls
            .as_deref()
            .filter(|s| !s.is_empty() && *s != "[]")
            .and_then(|s| serde_json::from_str::<Vec<npcrs::ToolCall>>(s).ok());

        out.push(Message {
            role,
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls: tool_calls_parsed,
            tool_call_id: None,
            name: None,
            thinking: None,
            reasoning_content: reasoning.filter(|s| !s.is_empty()),
        });
    }

    // An interrupt lands between the assistant row that declares a tool call
    // and the row that records its result. Providers reject a transcript whose
    // last assistant message calls a tool that was never answered, so close the
    // dangling calls with a synthetic result before handing this back.
    close_dangling_tool_calls(&mut out);
    out
}

/// Append a placeholder `tool` message for every tool call the assistant
/// declared but no row ever answered. Keeps an interrupted turn replayable.
fn close_dangling_tool_calls(messages: &mut Vec<Message>) {
    let unanswered: Vec<npcrs::ToolCall> = match messages.last() {
        Some(last) if last.role == "assistant" => last
            .tool_calls
            .as_ref()
            .map(|tcs| tcs.iter().filter(|tc| !tc.id.is_empty()).cloned().collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    for tc in unanswered {
        messages.push(Message {
            role: "tool".to_string(),
            content: Some("[no result recorded: the turn was interrupted]".to_string()),
            tool_calls: None,
            tool_call_id: Some(tc.id.clone()),
            name: Some(tc.function.name.clone()),
            thinking: None,
            reasoning_content: None,
        });
    }
}

/// Build a `StreamEvent` sink that feeds a shared `LiveTurn`.
///
/// Returned as a `Box<dyn FnMut.. + Send>` so it can be handed to
/// `call_stream_with_sink` while the caller keeps its own handle for the final
/// reconcile / usage update.
pub fn event_sink(
    turn: &std::sync::Arc<std::sync::Mutex<LiveTurn>>,
) -> Option<Box<dyn FnMut(StreamEvent) + Send>> {
    use crate::stream_client::StreamEvent;
    let shared = std::sync::Arc::clone(turn);
    if !shared.lock().map(|t| t.enabled).unwrap_or(false) {
        return None;
    }
    Some(Box::new(move |ev: StreamEvent| {
        let Ok(mut t) = shared.lock() else {
            return;
        };
        match ev {
            StreamEvent::ContentDelta(text) => t.stream_delta(&text),
            StreamEvent::ToolStart { name, id, args } => {
                // Any text produced before the tool call belongs to this row.
                t.flush_assistant();
                t.note_tool_call(&name, &id, &args);
                t.flush_assistant();
            }
            StreamEvent::ToolResult { name, id, content } => {
                t.flush_assistant();
                t.tool_result(&name, &id, &content);
                // The agent loop starts a fresh assistant row after each tool.
                t.begin_assistant();
            }
            StreamEvent::ToolError { name, id, error } => {
                t.flush_assistant();
                t.tool_result(&name, &id, &error);
                t.begin_assistant();
            }
            StreamEvent::Interrupted => t.flush_and_interrupt(),
            StreamEvent::Finished => t.flush_assistant(),
        }
    }))
}

/// Print a short one-line notice about where the turn was persisted.
pub fn notice(enabled: bool, rows: usize) {
    if !enabled {
        eprintln!("{DIM}[history] persistence disabled for this turn{RESET}");
    } else if rows > 0 {
        eprintln!("{DIM}[history] {rows} row(s) saved live{RESET}");
    }
}
