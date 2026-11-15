//! Proves that a turn interrupted mid-way still leaves its transcript on disk,
//! and that resuming rebuilds a usable message list.
//!
//!   cargo run --example live_persist_demo
//!
//! The demo simulates a long tool loop -- assistant text, a tool call, a tool
//! result, more text -- then drops the turn without finishing, which is exactly
//! what Ctrl+C does.

use npcsh::live_persist::{self, LiveTurn};
use npcsh::stream_client::StreamEvent;
use std::sync::{Arc, Mutex};

fn main() {
    let dir = std::env::temp_dir().join(format!("npcsh_live_demo_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("history.db");
    // SAFETY: single-threaded setup, before any other thread reads the env.
    unsafe { std::env::set_var("NPCSH_HISTORY_DB", db.to_string_lossy().to_string()) };

    let conv = "demo-conversation";

    // ---- part 1: a turn that gets interrupted mid-tool-loop ----------------
    {
        let turn = Arc::new(Mutex::new(LiveTurn::begin(
            conv,
            "audit the repo and fix the flaky test",
            "/tmp/project",
            "gpt-4o-mini",
            "openai",
            "corca",
            "npc-core",
        )));
        let mut send = live_persist::event_sink(&turn).expect("sink should be enabled");

        send(StreamEvent::ContentDelta("Let me look at the ".into()));
        send(StreamEvent::ContentDelta("test files first.".into()));
        send(StreamEvent::ToolStart {
            name: "sh".into(),
            id: "call_1".into(),
            args: r#"{"bash_command":"ls"}"#.into(),
        });
        send(StreamEvent::ToolResult {
            name: "sh".into(),
            id: "call_1".into(),
            content: "test_a.py\ntest_b.py\n".into(),
        });
        send(StreamEvent::ContentDelta(
            "I see two test files, running them".into(),
        ));
        send(StreamEvent::Interrupted);

        // Turn goes out of scope here -> Drop. Nothing else is called on purpose.
    }

    // ---- part 2: what survived the interrupt? ------------------------------
    let rows = read_rows(&db, conv);
    println!("rows on disk after interrupt: {}", rows.len());
    for r in &rows {
        println!(
            "  {:<9} interrupted={} content={:?}",
            r.0,
            r.1,
            truncate(&r.2, 46)
        );
    }

    assert_eq!(rows.len(), 4, "user + assistant + tool result + assistant");
    assert_eq!(rows[0].0, "user");
    assert_eq!(rows[0].1, 0);
    assert_eq!(rows[1].0, "assistant");
    assert_eq!(rows[2].0, "tool");
    assert_eq!(rows[3].0, "assistant");
    assert_eq!(rows[3].1, 1, "last assistant row flagged interrupted");
    assert!(
        rows[3].2.contains("running them"),
        "partial streamed text was kept: {:?}",
        rows[3].2
    );

    // ---- part 3: resume rebuilds a valid message list ----------------------
    let msgs = live_persist::load_messages(conv);
    println!("\nrebuilt {} messages for resume:", msgs.len());
    for m in &msgs {
        println!(
            "  {:<9} tool_call_id={:?} content={:?}",
            m.role,
            m.tool_call_id,
            truncate(m.content.as_deref().unwrap_or(""), 40)
        );
    }
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[2].role, "tool");
    assert_eq!(msgs[2].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(msgs[2].name.as_deref(), Some("sh"));
    println!(
        "  assistant#1 tool_calls = {:?}",
        msgs[1].tool_calls.as_ref().map(|v| v.len())
    );
    assert!(
        msgs[1].tool_calls.is_some(),
        "the assistant row before a tool row must carry its tool_calls"
    );

    // ---- part 4: a completed turn is NOT marked interrupted ----------------
    {
        let mut turn = LiveTurn::begin(
            "demo-completed",
            "hello",
            "/tmp/project",
            "gpt-4o-mini",
            "openai",
            "corca",
            "npc-core",
        );
        turn.stream_delta("hi there");
        turn.finish(12, 34, 0.0004);
    }
    let done = read_rows(&db, "demo-completed");
    println!("\ncompleted turn rows: {}", done.len());
    for r in &done {
        println!(
            "  {:<9} interrupted={} content={:?}",
            r.0,
            r.1,
            truncate(&r.2, 40)
        );
    }
    assert_eq!(
        done.len(),
        2,
        "user + assistant, no duplicate assistant row"
    );
    assert!(done.iter().all(|r| r.1 == 0), "nothing flagged interrupted");

    // ---- part 5: retry semantics ------------------------------------------
    // (a) the failed attempt produced NOTHING (server unreachable before any
    //     byte arrived) -> the retry reuses the user row, no duplication.
    {
        let turn = LiveTurn::begin(
            "demo-retry-empty",
            "summarize the changelog",
            "/tmp/project",
            "gpt-4o-mini",
            "openai",
            "corca",
            "npc-core",
        );
        assert_eq!(turn.rows_written, 1, "just the user row");
        drop(turn);

        let mut retry = LiveTurn::begin(
            "demo-retry-empty",
            "summarize the changelog",
            "/tmp/project",
            "gpt-4o-mini",
            "openai",
            "corca",
            "npc-core",
        );
        retry.stream_delta("here is the summary");
        retry.finish(10, 5, 0.0);
    }
    let a = read_rows(&db, "demo-retry-empty");
    let a_users = a.iter().filter(|r| r.0 == "user").count();
    let a_asst = a.iter().filter(|r| r.0 == "assistant").count();
    println!("\n(a) silent failure then retry: {a_users} user, {a_asst} assistant");
    for r in &a {
        println!(
            "  {:<9} interrupted={} content={:?}",
            r.0,
            r.1,
            truncate(&r.2, 40)
        );
    }
    assert_eq!(
        a_users, 1,
        "retry after a silent failure must not duplicate the user row"
    );
    assert_eq!(a_asst, 1);

    // (b) the failed attempt produced PARTIAL output -> the retry is a new
    //     user row, because the first attempt already has its reply recorded.
    {
        let mut turn = LiveTurn::begin(
            "demo-retry-partial",
            "summarize the changelog",
            "/tmp/project",
            "gpt-4o-mini",
            "openai",
            "corca",
            "npc-core",
        );
        turn.stream_delta("partial ans");
        drop(turn);

        let mut retry = LiveTurn::begin(
            "demo-retry-partial",
            "summarize the changelog",
            "/tmp/project",
            "gpt-4o-mini",
            "openai",
            "corca",
            "npc-core",
        );
        retry.stream_delta("full answer");
        retry.finish(10, 5, 0.0);
    }
    let b = read_rows(&db, "demo-retry-partial");
    let b_users = b.iter().filter(|r| r.0 == "user").count();
    let b_asst = b.iter().filter(|r| r.0 == "assistant").count();
    println!("\n(b) partial failure then retry: {b_users} user, {b_asst} assistant");
    for r in &b {
        println!(
            "  {:<9} interrupted={} content={:?}",
            r.0,
            r.1,
            truncate(&r.2, 40)
        );
    }
    assert_eq!(
        b_users, 2,
        "a retry after partial output keeps both attempts"
    );
    assert_eq!(b_asst, 2);
    assert_eq!(
        b.iter().filter(|r| r.1 == 1).count(),
        1,
        "exactly one row interrupted"
    );

    part6(&db);
    println!("\nALL CHECKS PASSED");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Part 6: an interrupt can leave a tool row whose owning tool_calls were
/// never written. That transcript is invalid for a provider, so resume must
/// drop the orphan.
fn part6(db: &std::path::Path) {
    let payload = r#"{"name":"sh","tool_call_id":"call_missing"}"#;
    {
        let conn = rusqlite::Connection::open(db).unwrap();
        conn.execute(
            "INSERT INTO conversation_history
             (message_id, timestamp, role, content, conversation_id, tool_results)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                "orphan-1",
                "2026-01-01 00:00:00",
                "tool",
                "dangling output",
                "demo-orphan",
                payload
            ],
        )
        .unwrap();
    }
    let orphan = live_persist::load_messages("demo-orphan");
    println!(
        "\npart 6: {} message(s) survived the orphan filter",
        orphan.len()
    );
    for m in &orphan {
        println!(
            "  {:<9} content={:?}",
            m.role,
            truncate(m.content.as_deref().unwrap_or(""), 30)
        );
    }
    assert!(
        orphan.iter().all(|m| m.role != "tool"),
        "an unpaired tool row must not be replayed"
    );
}
/// (role, interrupted, content) ordered by insertion.
fn read_rows(db: &std::path::Path, conv: &str) -> Vec<(String, i64, String)> {
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT role, COALESCE(interrupted,0), COALESCE(content,'')
             FROM conversation_history WHERE conversation_id = ?1 ORDER BY id ASC",
        )
        .unwrap();
    let rows = stmt
        .query_map(rusqlite::params![conv], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    rows.flatten().collect()
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let t: String = s.chars().take(n).collect();
    format!("{t}...")
}
