use crate::cron::{CronJob, CronJobKind};
use crate::stream_client::{StreamEvent, StreamRequest};
use npcrs::Message;
use npcrs::error::Result;
use npcrs::kernel::Kernel;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, mpsc};

fn loop_auto_approve(_prompt: &str) -> String {
    "Yes (session)".to_string()
}

fn loop_conversation_id(job: &CronJob) -> String {
    format!("loop-{}-{}", job.npc, job.id)
}

fn prepare_loop_process(kernel: &mut Kernel, job: &CronJob, pid: u32) -> Result<()> {
    let conv_id = loop_conversation_id(job);
    // The DB is the source of truth for a loop conversation: a background run
    // may have persisted new messages while we were detached, and we must not
    // continue with stale in-memory state.
    let loaded = kernel
        .history
        .load_conversation_messages(&conv_id)
        .unwrap_or_default();

    let process = kernel
        .get_process_mut(pid)
        .ok_or_else(|| npcrs::NpcError::Other(format!("No process with pid {}", pid)))?;

    process.conversation_id = conv_id;
    process.messages.clear();

    for m in loaded {
        let role = m.role.as_str();
        let content = m.content.unwrap_or_default();
        let msg = match role {
            "assistant" => Message::assistant(content),
            "tool" => Message::tool_result(m.message_id, content),
            _ => Message::user(content),
        };
        process.messages.push(msg);
    }
    Ok(())
}

fn continuation_prompt(job: &CronJob) -> String {
    format!(
        "Continue the ongoing work for this loop.\n\nOriginal directive:\n{}\n\nPick the next concrete task, make progress, and call stop when you have finished that one task. Do not repeat work already done.",
        job.task
    )
}

/// Run a job once, right now, in a kernel owned by this call.
///
/// This backs `/cron run <id>` and any "run now" affordance. It shares the
/// single execution path with the ticker (`run_loop_detached`) so a manual run
/// and a scheduled run cannot diverge.
pub async fn execute_cron_job(
    job: &CronJob,
    cfg: DetachedLoopConfig,
    client: reqwest::Client,
) -> Option<String> {
    run_loop_detached(job.clone(), cfg, client).await
}

pub fn task_slug(task: &str, is_jinx: bool) -> String {
    let base = if is_jinx {
        format!("jinx_{}", task)
    } else {
        task.to_string()
    };
    base.to_lowercase()
        .replace(|c: char| !c.is_alphanumeric(), "_")
        .replace("__", "_")
        .trim_matches('_')
        .chars()
        .take(60)
        .collect::<String>()
}

#[derive(Clone, Debug)]
pub enum LoopEvent {
    ContentDelta(String),
    ToolStart {
        name: String,
        id: String,
        args: String,
    },
    ToolResult {
        name: String,
        id: String,
        content: String,
    },
    ToolError {
        name: String,
        id: String,
        error: String,
    },
    Error(String),
    Interrupted,
    Finished,
}

fn build_loop_stream_request(kernel: &Kernel, job: &CronJob, pid: u32) -> Option<StreamRequest> {
    let process = kernel.get_process(pid)?;
    let model = process.npc.resolved_model();
    let provider = process.npc.resolved_provider();
    let base_system = process.npc.system_prompt(kernel.team.context.as_deref());
    let system = if crate::memory_context_enabled() {
        if let Some(team_dir) = kernel.team.source_dir.as_deref() {
            let has_memory = kernel.jinxes.contains_key("memory");
            let has_knowledge = kernel.jinxes.contains_key("knowledge");
            let stores = crate::discover_knowledge_stores(team_dir);
            let appendix = crate::format_memory_context(&stores, has_memory, has_knowledge);
            if appendix.is_empty() {
                base_system
            } else {
                format!("{}\n{}", base_system, appendix)
            }
        } else {
            base_system
        }
    } else {
        base_system
    };

    let prompt = if process.messages.iter().any(|m| m.role == "user") {
        continuation_prompt(job)
    } else {
        job.task.clone()
    };

    let registered_teams = kernel
        .team
        .source_dir
        .as_ref()
        .map(|d| vec![d.clone()])
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|d| d.to_string_lossy().to_string())
                .map(|d| vec![d])
        });

    let mut messages = vec![Message::system(system)];
    messages.extend(process.messages.clone());
    messages.push(Message::user(prompt));

    Some(StreamRequest {
        model,
        provider,
        messages,
        commandstr: job.task.clone(),
        npc: Some(process.npc.name.clone()),
        registered_teams,
        conversation_id: Some(process.conversation_id.clone()),
        current_path: Some(
            std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string()),
        ),
        execution_mode: "tool_agent".to_string(),
        stream_id: Some(uuid::Uuid::new_v4().to_string()),
    })
}

pub fn save_loop_run(npc: &str, task: &str, is_jinx: bool, output: &str) -> std::io::Result<()> {
    let slug = task_slug(task, is_jinx);
    let base = PathBuf::from(shellexpand::tilde("~/.npcsh/loops").to_string())
        .join(npc)
        .join(slug)
        .join("runs");
    std::fs::create_dir_all(&base)?;
    let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let path = base.join(format!("{}.txt", ts));
    std::fs::write(&path, output)
}

pub fn spawn_npc_for_loop(kernel: &mut Kernel, name: &str, parent_pid: u32) -> Result<u32> {
    if let Some(proc) = kernel.find_by_name(name) {
        return Ok(proc.pid);
    }

    let teams = load_registered_teams();
    for (_team_name, team_dir) in teams {
        let path = std::path::Path::new(&team_dir).join(format!("{}.npc", name));
        if path.exists() {
            let npc = npcrs::npc_compiler::NPC::from_file(&path).map_err(|e| {
                npcrs::NpcError::Other(format!("Failed to load NPC {}: {}", name, e))
            })?;
            let pid = kernel.spawn(npc, parent_pid, npcrs::process::Capabilities::root());
            return Ok(pid);
        }
    }
    Err(npcrs::NpcError::Other(format!(
        "NPC '{}' not found in registered teams",
        name
    )))
}

fn load_registered_teams() -> Vec<(String, String)> {
    let path = shellexpand::tilde("~/.npcsh/teams.yaml").to_string();
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let parsed: serde_yaml::Value = match serde_yaml::from_str(&content) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    if let Some(teams) = parsed.get("teams").and_then(|t| t.as_mapping()) {
        for (name, path_value) in teams {
            let name = name.as_str().unwrap_or("").to_string();
            let path = path_value.as_str().unwrap_or("").to_string();
            if !name.is_empty() && !path.is_empty() {
                let expanded = shellexpand::tilde(&path).to_string();
                out.push((name, expanded));
            }
        }
    }
    out
}

// ── Detached (ownership-in-kernel) loop execution ────────────────────────────
//
// A loop must be able to run with no REPL attached. `prepare_loop_process`
// already reloads a loop's conversation from the DB (the DB is the source of
// truth), and `spawn_background_loop` drops its `&mut Kernel` before the task
// body ever runs. So a loop does not need the REPL's kernel: it can boot its
// own from the same team dir and DB, run one turn, persist, and go away.
//
// This is what makes loops detachable rather than in-session. The ticker owns
// execution; `cron_rx` is only a UI notification channel and can be dropped
// without losing a run.

/// Everything a detached run needs that is not in the `CronJob` itself.
#[derive(Clone, Debug)]
pub struct DetachedLoopConfig {
    pub team_dir: String,
    pub db_path: String,
    pub server_url: String,
}

impl DetachedLoopConfig {
    pub fn new(
        team_dir: impl Into<String>,
        db_path: impl Into<String>,
        server_url: impl Into<String>,
    ) -> Self {
        Self {
            team_dir: team_dir.into(),
            db_path: db_path.into(),
            server_url: server_url.into(),
        }
    }
}

/// Consecutive failures per loop key, used to back off a loop that cannot run.
///
/// A loop whose NPC does not exist will fail identically on every tick. Without
/// backoff it retries at its full interval forever, which fills the loop log and
/// burns nothing but noise. The streak resets on any successful run so a
/// transient failure (server down) does not permanently slow the loop.
fn failure_streak() -> &'static Mutex<HashMap<String, u32>> {
    static LOCK: std::sync::OnceLock<Mutex<HashMap<String, u32>>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record_failure(key: &str) -> u32 {
    let mut m = match failure_streak().lock() {
        Ok(m) => m,
        Err(_) => return 0,
    };
    let n = m.entry(key.to_string()).or_insert(0);
    *n += 1;
    *n
}

fn clear_failures(key: &str) {
    if let Ok(mut m) = failure_streak().lock() {
        m.remove(key);
    }
}

pub fn failure_count(key: &str) -> u32 {
    failure_streak()
        .lock()
        .ok()
        .and_then(|m| m.get(key).copied())
        .unwrap_or(0)
}

/// Keys of loops whose run is currently in flight. A tick for a key already in
/// this set is skipped rather than queued (same no-overlap policy as before),
/// but because the set lives here instead of in the REPL it keeps working when
/// no prompt is being drawn.
fn in_flight() -> &'static Mutex<std::collections::HashSet<String>> {
    static LOCK: std::sync::OnceLock<Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

pub fn is_in_flight(key: &str) -> bool {
    in_flight().lock().map(|s| s.contains(key)).unwrap_or(false)
}

pub fn in_flight_count() -> usize {
    in_flight().lock().map(|s| s.len()).unwrap_or(0)
}

fn loop_key_for(job: &CronJob) -> String {
    format!("loop-{}-{}", job.npc, job.id)
}

fn append_loop_log(npc: &str, task: &str, is_jinx: bool, line: &str) {
    let slug = task_slug(task, is_jinx);
    let dir = PathBuf::from(shellexpand::tilde("~/.npcsh/loops").to_string())
        .join(npc)
        .join(slug);
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("loop.log");
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(f, "[{}] {}", ts, line);
    }
}

/// Run exactly one turn of `job` in a kernel this task owns.
///
/// Returns the turn output, or `None` if the run could not be started. Errors
/// are reported into the loop's run output and log rather than propagated, so
/// one bad loop cannot take down the ticker.
pub async fn run_loop_detached(
    job: CronJob,
    cfg: DetachedLoopConfig,
    client: reqwest::Client,
) -> Option<String> {
    let is_jinx = job.kind == CronJobKind::Jinx;
    append_loop_log(&job.npc, &job.task, is_jinx, "run start");

    let mut kernel = match Kernel::boot(&cfg.team_dir, &cfg.db_path) {
        Ok(k) => k,
        Err(e) => {
            let msg = format!("[loop {}] kernel boot failed: {e}", job.id);
            append_loop_log(&job.npc, &job.task, is_jinx, &msg);
            let _ = save_loop_run(&job.npc, &job.task, is_jinx, &msg);
            return None;
        }
    };

    // Jinx loops are a plain syscall; the kernel is already booted so the jinx
    // table is live. They need no conversation state.
    if job.kind == CronJobKind::Jinx {
        let pid = match spawn_npc_for_loop(&mut kernel, &job.npc, 0) {
            Ok(p) => p,
            Err(e) => {
                let n = record_failure(&loop_key(&job));
                let msg = format!(
                    "[loop {}] npc {:?} unavailable: {e} (failure #{n})",
                    job.id, job.npc
                );
                append_loop_log(&job.npc, &job.task, is_jinx, &msg);
                let _ = save_loop_run(&job.npc, &job.task, is_jinx, &msg);
                return None;
            }
        };
        let mut args = HashMap::new();
        let mut pieces = job.task.splitn(2, ' ');
        let jinx_name = pieces.next().unwrap_or(&job.task).to_string();
        if let Some(rest) = pieces.next() {
            if let Some(first_input) = kernel.jinxes.get(&jinx_name).and_then(|j| j.inputs.first())
            {
                args.insert(first_input.name.clone(), rest.to_string());
            }
        }
        let out = match kernel.syscall(pid, &jinx_name, &args).await {
            Ok(o) => o,
            Err(e) => format!("[loop {}] /{} error: {e}", job.id, jinx_name),
        };
        let _ = save_loop_run(&job.npc, &job.task, is_jinx, &out);
        append_loop_log(
            &job.npc,
            &job.task,
            is_jinx,
            &format!("run end ({} bytes)", out.len()),
        );
        return Some(out);
    }

    let pid = match spawn_npc_for_loop(&mut kernel, &job.npc, 0) {
        Ok(p) => p,
        Err(e) => {
            let n = record_failure(&loop_key(&job));
            let msg = format!(
                "[loop {}] npc {:?} unavailable: {e} (failure #{n})",
                job.id, job.npc
            );
            append_loop_log(&job.npc, &job.task, is_jinx, &msg);
            let _ = save_loop_run(&job.npc, &job.task, is_jinx, &msg);
            return None;
        }
    };

    if let Err(e) = prepare_loop_process(&mut kernel, &job, pid) {
        let msg = format!("[loop {}] could not prepare loop process: {e}", job.id);
        append_loop_log(&job.npc, &job.task, is_jinx, &msg);
        let _ = save_loop_run(&job.npc, &job.task, is_jinx, &msg);
        return None;
    }

    let request = match build_loop_stream_request(&kernel, &job, pid) {
        Some(r) => r,
        None => {
            let msg = format!("[loop {}] could not build stream request", job.id);
            append_loop_log(&job.npc, &job.task, is_jinx, &msg);
            let _ = save_loop_run(&job.npc, &job.task, is_jinx, &msg);
            return None;
        }
    };

    // Persist live so a detach/reattach or a crash still leaves a transcript.
    let (npc_name, model, provider, conv_id) = match kernel.get_process(pid) {
        Some(p) => (
            p.npc.name.clone(),
            p.npc.resolved_model(),
            p.npc.resolved_provider(),
            p.conversation_id.clone(),
        ),
        None => {
            append_loop_log(
                &job.npc,
                &job.task,
                is_jinx,
                "process vanished before streaming",
            );
            return None;
        }
    };
    let team_name = kernel
        .team
        .source_dir
        .as_deref()
        .and_then(|d| std::path::Path::new(d).file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("npcsh")
        .to_string();
    let cwd = request
        .current_path
        .clone()
        .unwrap_or_else(|| ".".to_string());
    let prompt = request.commandstr.clone();
    let turn = Arc::new(Mutex::new(crate::live_persist::LiveTurn::begin(
        &conv_id, &prompt, &cwd, &model, &provider, &npc_name, &team_name,
    )));
    let persist_sink = crate::live_persist::event_sink(&turn);

    append_loop_log(
        &job.npc,
        &job.task,
        is_jinx,
        &format!("streaming (conv {conv_id}, model {model})"),
    );

    // Publish events so any attached REPL can render the run, and accept an
    // interrupt from `/detach`/Ctrl-C in the observing process. The persisted
    // sink and the bus listener both need the events, so wrap the sink.
    let (event_tx, _) = broadcast::channel::<LoopEvent>(256);
    let (interrupt_tx, interrupt_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let key = loop_key(&job);
    if let Ok(mut bus) = run_bus().lock() {
        bus.insert(
            key.clone(),
            RunHandle {
                events: event_tx.clone(),
                interrupt: interrupt_tx,
            },
        );
    }
    let bus_tx = event_tx.clone();
    let mut persist = persist_sink;
    let combined_sink: Box<dyn FnMut(StreamEvent) + Send> = Box::new(move |ev: StreamEvent| {
        if let Some(ref mut p) = persist {
            p(ev.clone());
        }
        let le = match ev {
            StreamEvent::ContentDelta(text) => LoopEvent::ContentDelta(text),
            StreamEvent::ToolStart { name, id, args } => LoopEvent::ToolStart { name, id, args },
            StreamEvent::ToolResult { name, id, content } => {
                LoopEvent::ToolResult { name, id, content }
            }
            StreamEvent::ToolError { name, id, error } => LoopEvent::ToolError { name, id, error },
            StreamEvent::Interrupted => LoopEvent::Interrupted,
            StreamEvent::Finished => LoopEvent::Finished,
        };
        let _ = bus_tx.send(le);
    });

    let res = crate::stream_client::call_stream_with_sink_no_render(
        &client,
        &cfg.server_url,
        &request,
        Some(&loop_auto_approve),
        Some(interrupt_rx),
        Some(combined_sink),
    )
    .await;

    // Retire the bus entry regardless of outcome.
    if let Ok(mut bus) = run_bus().lock() {
        bus.remove(&key);
    }
    let _ = event_tx.send(LoopEvent::Finished);

    match res {
        Ok(resp) => {
            if let Ok(mut t) = turn.lock() {
                t.close_from_response(&resp);
            }
            let out = resp.message.content.unwrap_or_default();
            clear_failures(&loop_key(&job));
            let _ = save_loop_run(&job.npc, &job.task, is_jinx, &out);
            append_loop_log(
                &job.npc,
                &job.task,
                is_jinx,
                &format!("run end ({} bytes)", out.len()),
            );
            Some(out)
        }
        Err(e) => {
            let msg = format!("[loop {}] stream error: {e}", job.id);
            append_loop_log(&job.npc, &job.task, is_jinx, &msg);
            let _ = save_loop_run(&job.npc, &job.task, is_jinx, &msg);
            Some(msg)
        }
    }
}

/// Spawn a detached run on `handle`, skipping (not queueing) if one is already
/// in flight. Safe to call from the cron ticker thread.
pub fn spawn_detached_loop(handle: &tokio::runtime::Handle, job: CronJob, cfg: DetachedLoopConfig) {
    let key = loop_key_for(&job);
    {
        let mut set = match in_flight().lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        if set.contains(&key) {
            return;
        }
        set.insert(key.clone());
    }
    let client = reqwest::Client::new();
    handle.spawn(async move {
        let _ = run_loop_detached(job, cfg, client).await;
        if let Ok(mut set) = in_flight().lock() {
            set.remove(&key);
        }
    });
}

// ── Detached run observation bus ─────────────────────────────────────────────
//
// A detached run is not tied to any REPL, so "entering" a loop can no longer
// mean "start a run in this process". It means "subscribe to whatever is
// running and render its events". This registry is how a fresh `npcsh` process
// (or a second terminal) can attach to work already in flight.

struct RunHandle {
    events: broadcast::Sender<LoopEvent>,
    interrupt: mpsc::UnboundedSender<()>,
}

fn run_bus() -> &'static Mutex<HashMap<String, RunHandle>> {
    static LOCK: std::sync::OnceLock<Mutex<HashMap<String, RunHandle>>> =
        std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Subscribe to a loop's live events. Returns `None` if nothing is running.
pub fn subscribe_loop(key: &str) -> Option<broadcast::Receiver<LoopEvent>> {
    run_bus()
        .lock()
        .ok()
        .and_then(|b| b.get(key).map(|h| h.events.subscribe()))
}

/// Ask a running loop to stop after the current turn.
pub fn interrupt_loop(key: &str) -> bool {
    run_bus()
        .lock()
        .ok()
        .and_then(|b| b.get(key).map(|h| h.interrupt.send(()).is_ok()))
        .unwrap_or(false)
}

/// Keys of loops currently executing, for `/loops` and `/ps`.
pub fn running_keys() -> Vec<String> {
    run_bus()
        .lock()
        .map(|b| b.keys().cloned().collect())
        .unwrap_or_default()
}

pub fn loop_key(job: &CronJob) -> String {
    format!("loop-{}-{}", job.npc, job.id)
}

/// Serialize a `LoopEvent` for the daemon's attach stream.
pub fn event_to_json(ev: &LoopEvent) -> String {
    let v = match ev {
        LoopEvent::ContentDelta(t) => serde_json::json!({"event": "content", "text": t}),
        LoopEvent::ToolStart { name, args, .. } => {
            serde_json::json!({"event": "tool_start", "name": name, "args": args})
        }
        LoopEvent::ToolResult { name, content, .. } => {
            serde_json::json!({"event": "tool_result", "name": name, "content": content})
        }
        LoopEvent::ToolError { name, error, .. } => {
            serde_json::json!({"event": "tool_error", "name": name, "error": error})
        }
        LoopEvent::Error(e) => serde_json::json!({"event": "error", "error": e}),
        LoopEvent::Interrupted => serde_json::json!({"event": "interrupted"}),
        LoopEvent::Finished => serde_json::json!({"event": "finished"}),
    };
    v.to_string()
}

/// Parse a line produced by `event_to_json` back into a `LoopEvent`.
pub fn event_from_json(line: &str) -> Option<LoopEvent> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    Some(match v.get("event")?.as_str()? {
        "content" => LoopEvent::ContentDelta(s("text")),
        "tool_start" => LoopEvent::ToolStart {
            name: s("name"),
            id: String::new(),
            args: s("args"),
        },
        "tool_result" => LoopEvent::ToolResult {
            name: s("name"),
            id: String::new(),
            content: s("content"),
        },
        "tool_error" => LoopEvent::ToolError {
            name: s("name"),
            id: String::new(),
            error: s("error"),
        },
        "error" => LoopEvent::Error(s("error")),
        "interrupted" => LoopEvent::Interrupted,
        "finished" => LoopEvent::Finished,
        _ => return None,
    })
}
