use crate::agent_turn::{ctrl_c_receiver, run_stream_turn, run_stream_turn_with_interrupt, Mode};
use crate::cron::{CronJob, CronJobKind};
use crate::stream_client::{self, StreamEvent, StreamRequest};
use npcrs::error::Result;
use npcrs::kernel::Kernel;
use npcrs::Message;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

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

pub async fn run_loop_interactive(
    kernel: &mut Kernel,
    job: &CronJob,
    client: &reqwest::Client,
    server_url: &str,
) -> Result<String> {
    let pid = if let Some(proc) = kernel.find_by_name(&job.npc) {
        proc.pid
    } else {
        spawn_npc_for_loop(kernel, &job.npc, 0)?
    };

    prepare_loop_process(kernel, job, pid)?;

    match job.kind {
        CronJobKind::Jinx => {
            let mut args = HashMap::new();
            let mut pieces = job.task.splitn(2, ' ');
            let jinx_name = pieces.next().unwrap_or(&job.task).to_string();
            if let Some(rest) = pieces.next() {
                if let Some(first_input) =
                    kernel.jinxes.get(&jinx_name).and_then(|j| j.inputs.first())
                {
                    args.insert(first_input.name.clone(), rest.to_string());
                }
            }
            let output = kernel.syscall(pid, &jinx_name, &args).await?;
            let _ = save_loop_run(&job.npc, &job.task, true, &output);
            Ok(output)
        }
        CronJobKind::Chat => {
            let process = kernel.get_process(pid).unwrap();
            let prompt = if process.messages.iter().any(|m| m.role == "user") {
                continuation_prompt(job)
            } else {
                job.task.clone()
            };
            let output = run_stream_turn_with_interrupt(
                kernel,
                pid,
                &prompt,
                Mode::Agent,
                client,
                server_url,
                true,
                Some(ctrl_c_receiver()),
                None,
                Some(&loop_auto_approve),
            )
            .await?;
            let _ = save_loop_run(&job.npc, &job.task, false, &output);
            Ok(output)
        }
    }
}

pub async fn execute_cron_job(
    kernel: &mut Kernel,
    job: &CronJob,
    client: &reqwest::Client,
    server_url: &str,
    save_history: bool,
) -> Option<String> {
    let pid = if let Some(proc) = kernel.find_by_name(&job.npc) {
        proc.pid
    } else {
        match spawn_npc_for_loop(kernel, &job.npc, 0) {
            Ok(pid) => pid,
            Err(e) => return Some(format!("[cron] could not spawn @{}: {}", job.npc, e)),
        }
    };
    if let Err(e) = prepare_loop_process(kernel, job, pid) {
        return Some(format!("[cron] could not prepare loop process: {e}"));
    }

    let mut out = Vec::new();
    let label = format!("[cron {}]", job.npc);

    match job.kind {
        CronJobKind::Jinx => {
            let mut args = HashMap::new();
            let mut pieces = job.task.splitn(2, ' ');
            let jinx_name = pieces.next().unwrap_or(&job.task).to_string();
            if let Some(rest) = pieces.next() {
                if let Some(first_input) =
                    kernel.jinxes.get(&jinx_name).and_then(|j| j.inputs.first())
                {
                    args.insert(first_input.name.clone(), rest.to_string());
                }
            }
            match kernel.syscall(pid, &jinx_name, &args).await {
                Ok(output) if !output.is_empty() => {
                    out.push(format!("{} /{}\n{}", label, jinx_name, output));
                }
                Ok(_) => {}
                Err(e) => out.push(format!("{} /{} error: {}", label, jinx_name, e)),
            }
        }
        CronJobKind::Chat => {
            let process = kernel.get_process(pid).unwrap();
            let prompt = if process.messages.iter().any(|m| m.role == "user") {
                continuation_prompt(job)
            } else {
                job.task.clone()
            };
            out.push(format!("{} {}", label, prompt));
            match run_stream_turn(
                kernel,
                pid,
                &prompt,
                Mode::Agent,
                client,
                server_url,
                save_history,
                None,
                Some(&loop_auto_approve),
            )
            .await
            {
                Ok(output) if !output.is_empty() => {
                    out.push(output);
                }
                Ok(_) => {}
                Err(e) => out.push(format!("{} error: {}", label, e)),
            }
        }
    }

    let full = if out.is_empty() {
        None
    } else {
        Some(out.join("\n"))
    };
    if let Some(ref text) = full {
        let _ = save_loop_run(&job.npc, &job.task, job.kind == CronJobKind::Jinx, text);
    }
    full
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
    ToolStart { name: String, id: String, args: String },
    ToolResult { name: String, id: String, content: String },
    ToolError { name: String, id: String, error: String },
    Error(String),
    Interrupted,
    Finished,
}

pub struct BackgroundLoop {
    pub job: CronJob,
    pub events: broadcast::Sender<LoopEvent>,
    pub interrupt: mpsc::UnboundedSender<()>,
    pub finished: Arc<AtomicBool>,
    pub task: Mutex<Option<JoinHandle<()>>>,
}

impl BackgroundLoop {
    pub fn subscribe(&self) -> broadcast::Receiver<LoopEvent> {
        self.events.subscribe()
    }

    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    pub fn interrupt(&self) {
        let _ = self.interrupt.send(());
    }
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

pub fn spawn_background_loop(
    kernel: &mut Kernel,
    job: CronJob,
    client: reqwest::Client,
    server_url: String,
) -> Result<Arc<BackgroundLoop>> {
    let pid = if let Some(proc) = kernel.find_by_name(&job.npc) {
        proc.pid
    } else {
        spawn_npc_for_loop(kernel, &job.npc, 0)?
    };
    prepare_loop_process(kernel, &job, pid)?;

    let request = build_loop_stream_request(kernel, &job, pid)
        .ok_or_else(|| npcrs::NpcError::Other("Failed to build stream request".to_string()))?;

    // Persist the loop turn to the same DB that prepare_loop_process reloads
    // from, so reattaching after a detached run sees the full conversation.
    let (npc_name, model, provider, conv_id) = {
        let process = kernel.get_process(pid).ok_or_else(|| {
            npcrs::NpcError::Other(format!("No process with pid {}", pid))
        })?;
        (
            process.npc.name.clone(),
            process.npc.resolved_model(),
            process.npc.resolved_provider(),
            process.conversation_id.clone(),
        )
    };
    let team_name_str = kernel
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
        &conv_id, &prompt, &cwd, &model, &provider, &npc_name, &team_name_str,
    )));
    let persist_sink = crate::live_persist::event_sink(&turn);

    let (event_tx, _event_rx) = broadcast::channel::<LoopEvent>(256);
    let (interrupt_tx, interrupt_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let finished = Arc::new(AtomicBool::new(false));

    let done = finished.clone();
    let job_for_task = job.clone();
    let event_tx_for_task = event_tx.clone();

    let task = tokio::spawn(async move {
        let res = stream_client::call_stream_with_sink_no_render(
            &client,
            &server_url,
            &request,
            Some(&loop_auto_approve),
            Some(interrupt_rx),
            Some(Box::new({
                let events = event_tx_for_task.clone();
                let mut persist = persist_sink;
                move |ev| {
                    if let Some(ref mut p) = persist {
                        p(ev.clone());
                    }
                    let le = match ev {
                        StreamEvent::ContentDelta(text) => LoopEvent::ContentDelta(text),
                        StreamEvent::ToolStart { name, id, args } => {
                            LoopEvent::ToolStart { name, id, args }
                        }
                        StreamEvent::ToolResult { name, id, content } => {
                            LoopEvent::ToolResult { name, id, content }
                        }
                        StreamEvent::ToolError { name, id, error } => {
                            LoopEvent::ToolError { name, id, error }
                        }
                        StreamEvent::Interrupted => LoopEvent::Interrupted,
                        StreamEvent::Finished => LoopEvent::Finished,
                    };
                    let _ = events.send(le);
                }
            })),
        )
        .await;

        match res {
            Ok(resp) => {
                {
                    if let Ok(mut t) = turn.lock() {
                        t.close_from_response(&resp);
                    }
                }
                let output = resp.message.content.unwrap_or_default();
                let _ = save_loop_run(
                    &job_for_task.npc,
                    &job_for_task.task,
                    job_for_task.kind == CronJobKind::Jinx,
                    &output,
                );
                let _ = event_tx_for_task.send(LoopEvent::Finished);
            }
            Err(e) => {
                let _ = event_tx_for_task.send(LoopEvent::Error(e));
            }
        }
        done.store(true, Ordering::SeqCst);
    });

    Ok(Arc::new(BackgroundLoop {
        job,
        events: event_tx,
        interrupt: interrupt_tx,
        finished,
        task: Mutex::new(Some(task)),
    }))
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

pub fn spawn_npc_for_loop(
    kernel: &mut Kernel,
    name: &str,
    parent_pid: u32,
) -> Result<u32> {
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
