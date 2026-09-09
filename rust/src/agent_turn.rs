use npcrs::error::Result;
use npcrs::kernel::Kernel;
use npcrs::process::{Capabilities, ProcessState};
use npcrs::{Message, NpcError};
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[90m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

pub fn cli_sessions() -> &'static Mutex<HashMap<u32, String>> {
    static LOCK: OnceLock<Mutex<HashMap<u32, String>>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(HashMap::new()))
}

pub async fn ensure_server_running(
    client: &reqwest::Client,
    server_url: &str,
) -> std::result::Result<(), String> {
    if client
        .get(server_url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .is_ok()
    {
        return Ok(());
    }

    let python = std::env::var("NPCSH_BACKEND_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let teams_yaml = std::env::var("NPCSH_TEAM_YAML")
        .unwrap_or_else(|_| shellexpand::tilde("~/.npcsh/teams.yaml").to_string());

    let host = std::env::var("NPCSH_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("NPCSH_SERVER_PORT").unwrap_or_else(|_| "5237".to_string());

    let mut cmd = tokio::process::Command::new(&python);
    cmd.arg("-m")
        .arg("npcpy.serve")
        .arg("--host")
        .arg(&host)
        .arg("--port")
        .arg(&port)
        .arg("--teams-yaml")
        .arg(&teams_yaml)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(false);

    cmd.spawn()
        .map_err(|e| format!("failed to spawn npcpy.serve: {e}"))?;

    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if client
            .get(server_url)
            .timeout(std::time::Duration::from_secs(1))
            .send()
            .await
            .is_ok()
        {
            return Ok(());
        }
    }

    Err("npcpy server did not become reachable after spawn".to_string())
}

pub async fn restart_server(
    client: &reqwest::Client,
    server_url: &str,
) -> std::result::Result<(), String> {
    let host = std::env::var("NPCSH_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("NPCSH_SERVER_PORT").unwrap_or_else(|_| "5237".to_string());

    match tokio::process::Command::new("lsof")
        .args(["-ti", &format!("tcp:{}", port)])
        .output()
        .await
    {
        Ok(output) if !output.stdout.is_empty() => {
            let pids: Vec<u32> = String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            for pid in pids {
                let _ = tokio::process::Command::new("kill")
                    .args(["-9", &pid.to_string()])
                    .output()
                    .await;
            }
        }
        _ => {}
    }

    for _ in 0..20 {
        if client
            .get(server_url)
            .timeout(std::time::Duration::from_millis(200))
            .send()
            .await
            .is_err()
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    ensure_server_running(client, server_url).await
}

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Agent,
    Chat,
    Cmd,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Agent => write!(f, "agent"),
            Mode::Chat => write!(f, "chat"),
            Mode::Cmd => write!(f, "cmd"),
        }
    }
}

pub async fn save_conversation_message(
    kernel: &mut Kernel,
    current_pid: u32,
    msg: &Message,
    cwd: &str,
) {
    let Some(process) = kernel.get_process(current_pid) else {
        return;
    };
    let model = process.npc.resolved_model();
    let provider = process.npc.resolved_provider();
    let npc_name = process.npc.name.clone();
    let conv_id = process.conversation_id.clone();
    if conv_id.is_empty() {
        return;
    }
    let team_name_str = kernel
        .team
        .source_dir
        .as_deref()
        .and_then(|d| std::path::Path::new(d).file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("npcsh")
        .to_string();

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

    let _ = kernel.history.save_conversation_message(
        &conv_id,
        &msg.role,
        content,
        cwd,
        Some(&model),
        Some(&provider),
        Some(&npc_name),
        Some(&team_name_str),
        tool_calls_json.as_deref(),
        tool_results_json.as_deref(),
        None,
        None,
        None,
        None,
    );
}

pub async fn save_conversation_turn(
    kernel: &mut Kernel,
    current_pid: u32,
    input: &str,
    output: &str,
    assistant_message: &Message,
    in_tok: u64,
    out_tok: u64,
    cost: f64,
    cwd: &str,
) {
    let Some(process) = kernel.get_process(current_pid) else {
        return;
    };
    let model = process.npc.resolved_model();
    let provider = process.npc.resolved_provider();
    let npc_name = process.npc.name.clone();
    let conv_id = process.conversation_id.clone();
    if conv_id.is_empty() {
        return;
    }
    let team_name_str = kernel
        .team
        .source_dir
        .as_deref()
        .and_then(|d| std::path::Path::new(d).file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("npcsh")
        .to_string();

    let _ = kernel.history.save_conversation_message(
        &conv_id,
        "user",
        input,
        cwd,
        Some(&model),
        Some(&provider),
        Some(&npc_name),
        Some(&team_name_str),
        None,
        None,
        None,
        Some(in_tok),
        None,
        None,
    );

    let tool_calls_json = assistant_message
        .tool_calls
        .as_ref()
        .map(|tcs| serde_json::to_string(tcs).unwrap_or_default())
        .filter(|s| !s.is_empty());

    let _ = kernel.history.save_conversation_message(
        &conv_id,
        "assistant",
        output,
        cwd,
        Some(&model),
        Some(&provider),
        Some(&npc_name),
        Some(&team_name_str),
        tool_calls_json.as_deref(),
        None,
        None,
        None,
        Some(out_tok),
        Some(cost),
    );

    for msg in process
        .messages
        .iter()
        .rev()
        .take_while(|m| m.role == "tool")
    {
        let tool_content = msg.content.as_deref().unwrap_or("");
        if tool_content.is_empty() {
            continue;
        }
        let tool_results_json = serde_json::json!({
            "name": msg.name.as_deref().unwrap_or("tool"),
            "tool_call_id": msg.tool_call_id.as_deref().unwrap_or(""),
            "content": tool_content,
        });
        let _ = kernel.history.save_conversation_message(
            &conv_id,
            "tool",
            tool_content,
            cwd,
            Some(&model),
            Some(&provider),
            Some(&npc_name),
            Some(&team_name_str),
            None,
            Some(&tool_results_json.to_string()),
            None,
            None,
            None,
            None,
        );
    }
}

pub struct MemoryScheduler {
    turn_count: u64,
    next_trigger: u64,
    lambda: f64,
    rng: rand::rngs::StdRng,
}

impl MemoryScheduler {
    pub fn new(lambda: f64) -> Self {
        let mut rng = rand::rngs::StdRng::from_entropy();
        let lambda = lambda.max(1.0);
        let first = Self::sample_interval(lambda, &mut rng);
        Self {
            turn_count: 0,
            next_trigger: first,
            lambda,
            rng,
        }
    }

    pub fn from_env() -> Option<Self> {
        if !crate::memory_context_enabled() {
            return None;
        }
        let lambda = std::env::var("NPCSH_MEMORY_LAMBDA")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(60.0);
        Some(Self::new(lambda))
    }

    fn sample_interval(lambda: f64, rng: &mut rand::rngs::StdRng) -> u64 {
        let u: f64 = rng.r#gen();
        let u = u.max(1e-10);
        let interval = (-lambda * u.ln()).ceil() as u64;
        interval.max(1)
    }

    pub fn should_trigger(&mut self) -> bool {
        self.turn_count += 1;
        self.turn_count >= self.next_trigger
    }

    pub fn defer(&mut self) {
        self.next_trigger = self.turn_count + 1;
    }

    pub fn reschedule(&mut self) {
        self.next_trigger = self.turn_count + Self::sample_interval(self.lambda, &mut self.rng);
    }
}

async fn extract_memory_candidates(
    client: &reqwest::Client,
    server_url: &str,
    conversation_text: &str,
    model: &str,
    provider: &str,
    npc_name: &str,
    team_name: &str,
    current_path: &str,
    conversation_id: &str,
) -> Result<Vec<String>> {
    let url = format!("{}/api/knowledge/extract", server_url);
    let body = serde_json::json!({
        "conversation_text": conversation_text,
        "conversation_id": conversation_id,
        "model": model,
        "provider": provider,
        "npc": npc_name,
        "team": team_name,
        "currentPath": current_path,
    });
    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| NpcError::Other(format!("memory extract request failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(NpcError::Other(format!(
            "memory extract returned {}: {}",
            status, text
        )));
    }
    let json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| NpcError::Other(format!("memory extract json failed: {e}")))?;
    let facts = json
        .get("facts")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut memories = Vec::new();
    for fact in facts {
        if let Some(statement) = fact
            .get("statement")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
        {
            if !statement.is_empty() {
                memories.push(statement);
            }
        }
    }
    Ok(memories)
}

pub async fn maybe_extract_memory(
    scheduler: &mut MemoryScheduler,
    kernel: &mut Kernel,
    current_pid: u32,
    user_input: &str,
    assistant_output: &str,
    client: &reqwest::Client,
    server_url: &str,
    cwd: &str,
) {
    if !scheduler.should_trigger() {
        return;
    }

    let Some(process) = kernel.get_process(current_pid) else {
        scheduler.reschedule();
        return;
    };
    let model = process.npc.resolved_model();
    let provider = process.npc.resolved_provider();
    let npc_name = process.npc.name.clone();
    let conv_id = process.conversation_id.clone();
    let team_name = kernel
        .team
        .source_dir
        .as_deref()
        .and_then(|d| std::path::Path::new(d).file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("npcsh")
        .to_string();
    let conversation_text = format!("User: {}\nAssistant: {}", user_input, assistant_output);

    eprint!("\n\x1b[90m🧠 thinking about memories...\x1b[0m");
    let _ = std::io::stderr().flush();

    let candidates = extract_memory_candidates(
        client,
        server_url,
        &conversation_text,
        &model,
        &provider,
        &npc_name,
        &team_name,
        cwd,
        &conv_id,
    )
    .await;

    match candidates {
        Ok(memories) if !memories.is_empty() => {
            let mut inserted = 0;
            for memory in memories {
                let message_id = format!("{}_auto_{}", conv_id, inserted);
                let _ = kernel.history.add_memory_to_database(
                    &message_id,
                    &conv_id,
                    &npc_name,
                    &team_name,
                    cwd,
                    &memory,
                    Some(&model),
                    Some(&provider),
                );
                inserted += 1;
            }
            if inserted > 0 {
                eprintln!(
                    "\r\x1b[90m🧠 {} memory candidate(s) queued for review\x1b[0m          ",
                    inserted
                );
            } else {
                eprint!("\r\x1b[2K");
            }
        }
        Ok(_) => {
            eprint!("\r\x1b[2K");
        }
        Err(e) => {
            eprintln!("\r\x1b[90mMemory extraction skipped: {}\x1b[0m", e);
        }
    }

    scheduler.reschedule();
}

pub fn ask_permission(prompt: &str) -> String {
    if std::env::var("NPCSH_ACCEPT_PERMISSIONS")
        .ok()
        .filter(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .is_some()
    {
        return "Yes (session)".to_string();
    }

    use crossterm::{
        ExecutableCommand,
        cursor::MoveTo,
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
        style::Print,
        terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use std::io::{self, Write};

    let options = vec!["Yes", "Yes (session)", "Yes (always)", "No", "No (never)"];

    let wrap_lines = |text: &str, width: usize| -> Vec<String> {
        let mut lines = Vec::new();
        for paragraph in text.split('\n') {
            let mut current = String::new();
            for word in paragraph.split_whitespace() {
                if current.is_empty() {
                    current.push_str(word);
                } else if current.len() + 1 + word.len() <= width.saturating_sub(2) {
                    current.push(' ');
                    current.push_str(word);
                } else {
                    lines.push(current);
                    current = word.to_string();
                }
            }
            if !current.is_empty() {
                lines.push(current);
            }
            if paragraph.is_empty() {
                lines.push(String::new());
            }
        }
        lines
    };

    struct PromptGuard;
    impl PromptGuard {
        fn new() -> io::Result<Self> {
            let mut stdout = io::stdout();
            stdout.execute(EnterAlternateScreen)?;
            terminal::enable_raw_mode()?;
            Ok(Self)
        }
    }
    impl Drop for PromptGuard {
        fn drop(&mut self) {
            let _ = terminal::disable_raw_mode();
            let _ = io::stdout().execute(LeaveAlternateScreen);
        }
    }

    let _guard = match PromptGuard::new() {
        Ok(g) => g,
        Err(_) => return "No".to_string(),
    };

    let mut stdout = io::stdout();
    let mut selected: usize = 0;
    let (_cols, rows) = terminal::size().unwrap_or((80, 24));

    loop {
        stdout.execute(Clear(ClearType::All)).ok();
        stdout.execute(MoveTo(0, 0)).ok();
        let wrapped = wrap_lines(prompt, _cols as usize);
        for line in &wrapped {
            stdout.execute(Print(line.clone())).ok();
            stdout.execute(Print("\n")).ok();
        }
        stdout.execute(Print("\n")).ok();
        for (i, opt) in options.iter().enumerate() {
            let marker = if i == selected { "> " } else { "  " };
            let line = format!("{}{}\n", marker, opt);
            stdout.execute(Print(line)).ok();
        }
        stdout.flush().ok();

        if let Ok(Event::Key(KeyEvent {
            code,
            kind: KeyEventKind::Press,
            ..
        })) = event::read()
        {
            match code {
                KeyCode::Up => {
                    if selected > 0 {
                        selected -= 1;
                    }
                }
                KeyCode::Down => {
                    if selected < options.len() - 1 {
                        selected += 1;
                    }
                }
                KeyCode::Enter => {
                    return options[selected].to_string();
                }
                KeyCode::Esc => {
                    return "No".to_string();
                }
                _ => {}
            }
        }
    }
}

fn is_server_error(e: &npcrs::NpcError) -> bool {
    let msg = e.to_string();
    msg.contains("HTTP stream returned 5")
        || msg.contains("Internal Server Error")
        || msg.contains("HTTP stream request failed")
}

pub async fn run_stream_turn_with_interrupt(
    kernel: &mut Kernel,
    current_pid: u32,
    input: &str,
    mode: Mode,
    client: &reqwest::Client,
    server_url: &str,
    save_history: bool,
    interrupt: Option<tokio::sync::mpsc::UnboundedReceiver<()>>,
    memory_scheduler: Option<&mut MemoryScheduler>,
    permission_prompt: Option<&dyn Fn(&str) -> String>,
) -> Result<String> {
    {
        let process = kernel
            .get_process_mut(current_pid)
            .ok_or_else(|| NpcError::Other(format!("No process with pid {}", current_pid)))?;
        if let Some(reason) = process.usage.exceeds(&process.limits) {
            process.kill(137);
            return Err(NpcError::Other(format!(
                "Process {} killed: {}",
                current_pid, reason
            )));
        }
        process.state = ProcessState::Running;
        process.new_turn();
    }

    {
        let process = kernel.get_process_mut(current_pid).unwrap();
        if process.conversation_id.is_empty() {
            process.conversation_id = std::env::var("NPCSH_CONVERSATION_ID")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        }
    }

    let (model, provider, system, npc_name, conv_id, team_name_str) = {
        let process = kernel
            .get_process(current_pid)
            .ok_or_else(|| NpcError::Other(format!("No process with pid {}", current_pid)))?;
        let model = process.npc.resolved_model();
        let provider = process.npc.resolved_provider();
        let base_system = process.npc.system_prompt(kernel.team.context.as_deref());
        let has_memory_jinx = kernel.jinxes.contains_key("memory");
        let has_knowledge_jinx = kernel.jinxes.contains_key("knowledge");
        let system = if crate::memory_context_enabled() && (has_memory_jinx || has_knowledge_jinx) {
            if let Some(team_dir) = kernel.team.source_dir.as_deref() {
                let stores = crate::discover_knowledge_stores(team_dir);
                let appendix =
                    crate::format_memory_context(&stores, has_memory_jinx, has_knowledge_jinx);
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
        let npc_name = process.npc.name.clone();
        let conv_id = process.conversation_id.clone();
        let team_name_str = kernel
            .team
            .source_dir
            .as_deref()
            .and_then(|d| std::path::Path::new(d).file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("npcsh")
            .to_string();
        (model, provider, system, npc_name, conv_id, team_name_str)
    };

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let cwd = cwd.replacen("/private/tmp", "/tmp", 1);
    let path_cmd = format!("The current working directory is: {}", cwd);
    let ls_files = if let Ok(entries) = std::fs::read_dir(&cwd) {
        let files: Vec<String> = entries
            .flatten()
            .take(100)
            .map(|e| {
                e.path()
                    .to_string_lossy()
                    .to_string()
                    .replacen("/private/tmp", "/tmp", 1)
            })
            .collect();
        let total = std::fs::read_dir(&cwd).map(|d| d.count()).unwrap_or(0);
        let mut listing = format!(
            "Files in the current directory (full paths):\n{}",
            files.join("\n")
        );
        if total > 100 {
            listing.push_str(&format!("\n... and {} more files", total - 100));
        }
        listing
    } else {
        "No files found in the current directory.".to_string()
    };
    let platform_info = format!(
        "Platform: {} {} ({})",
        std::env::consts::OS,
        "",
        std::env::consts::ARCH
    );
    let context_info = format!("{}\n{}\n{}", path_cmd, ls_files, platform_info);

    let tool_guidance = String::new();

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

    let execution_mode = if mode == Mode::Chat {
        "chat".to_string()
    } else {
        "tool_agent".to_string()
    };

    if crate::cli_providers::CLI_PROVIDERS.contains(&provider.as_str()) {
        let full_input = format!("{}\n\n{}", input, context_info);
        let session_id = cli_sessions().lock().unwrap().get(&current_pid).cloned();
        let cli_result = crate::cli_providers::run_cli_provider(
            &provider,
            &model,
            &full_input,
            &system,
            session_id.as_deref(),
        )
        .await;

        if let Some(result) = cli_result {
            let in_tok = result.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
            let out_tok = result
                .usage
                .as_ref()
                .map(|u| u.completion_tokens)
                .unwrap_or(0);
            let cost = result.cost_usd;

            {
                let process = kernel.get_process_mut(current_pid).unwrap();
                process.record_usage(in_tok, out_tok, cost);
                process.last_streamed = !result.text.is_empty();
                process.messages.push(Message::user(input));
                let msg = Message {
                    role: "assistant".to_string(),
                    content: if result.text.is_empty() {
                        None
                    } else {
                        Some(result.text.clone())
                    },
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    thinking: None,
                    reasoning_content: None,
                };
                process.messages.push(msg);
                process.state = ProcessState::Blocked;
            }

            if let Some(sid) = result.session_id {
                cli_sessions()
                    .lock()
                    .unwrap()
                    .insert(current_pid, sid.clone());
            }

            if save_history {
                let assistant_msg = Message {
                    role: "assistant".to_string(),
                    content: if result.text.is_empty() {
                        None
                    } else {
                        Some(result.text.clone())
                    },
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    thinking: None,
                    reasoning_content: None,
                };
                save_conversation_turn(
                    kernel,
                    current_pid,
                    input,
                    &result.text,
                    &assistant_msg,
                    in_tok,
                    out_tok,
                    cost,
                    &cwd,
                )
                .await;
            }

            return Ok(result.text);
        } else {
            return Err(NpcError::Other(format!(
                "CLI provider '{}' failed to produce a response; is the binary installed?",
                provider
            )));
        }
    }

    if save_history {
        save_conversation_message(kernel, current_pid, &Message::user(input), &cwd).await;
    }

    let request = crate::stream_client::StreamRequest {
        model,
        provider,
        messages: {
            let process = kernel.get_process(current_pid).unwrap();
            let mut msgs = vec![Message::system(system)];
            msgs.extend(process.messages.clone());
            msgs.push(Message::user(format!(
                "{}\n{}{}",
                input, context_info, tool_guidance
            )));
            msgs
        },
        commandstr: format!("{}\n{}{}", input, context_info, tool_guidance),
        npc: Some(npc_name.clone()),
        registered_teams,
        conversation_id: Some(conv_id.clone()),
        current_path: Some(cwd.clone()),
        execution_mode,
    };

    let response = crate::stream_client::call_stream_with_interrupt(
        client,
        server_url,
        &request,
        permission_prompt,
        interrupt,
    )
    .await
    .map_err(|e| NpcError::Other(e))?;
    let tool_results = response.tool_results;

    if let Some(ref usage) = response.usage {
        let process = kernel.get_process_mut(current_pid).unwrap();
        process.record_usage(usage.prompt_tokens, usage.completion_tokens, usage.cost_usd);
    }

    let mut assistant_message = response.message.clone();
    if !response.tool_calls.is_empty() {
        assistant_message.tool_calls = Some(response.tool_calls.clone());
    }

    let in_tok = response
        .usage
        .as_ref()
        .map(|u| u.prompt_tokens)
        .unwrap_or(0);
    let out_tok = response
        .usage
        .as_ref()
        .map(|u| u.completion_tokens)
        .unwrap_or(0);
    let cost = response.usage.as_ref().map(|u| u.cost_usd).unwrap_or(0.0);

    {
        let process = kernel.get_process_mut(current_pid).unwrap();
        process.last_streamed = response.streamed || response.message.content.is_some();
        process.last_thinking = response.message.thinking.clone();
        process.messages.push(Message::user(input));
        process.messages.push(assistant_message.clone());
        for tr in &tool_results {
            process.messages.push(tr.clone());
        }
    }

    if save_history {
        let assistant_for_save = response.message.clone();
        save_conversation_message(kernel, current_pid, &assistant_for_save, &cwd).await;
        for tr in &tool_results {
            save_conversation_message(kernel, current_pid, tr, &cwd).await;
        }

        let _ = kernel.history.save_conversation_message(
            &conv_id,
            "assistant",
            &response.message.content.clone().unwrap_or_default(),
            &cwd,
            Some(&request.model),
            Some(&request.provider),
            Some(&npc_name),
            Some(&team_name_str),
            assistant_message
                .tool_calls
                .as_ref()
                .map(|tcs| serde_json::to_string(tcs).unwrap_or_default())
                .as_deref(),
            None,
            None,
            Some(in_tok),
            Some(out_tok),
            Some(cost),
        );
    }

    let output = response.message.content.clone().unwrap_or_default();

    if save_history {
        if let Some(scheduler) = memory_scheduler {
            maybe_extract_memory(
                scheduler,
                kernel,
                current_pid,
                input,
                &output,
                client,
                server_url,
                &cwd,
            )
            .await;
        }
    }

    let process = kernel.get_process_mut(current_pid).unwrap();
    process.state = ProcessState::Blocked;
    Ok(output)
}

pub async fn run_stream_turn(
    kernel: &mut Kernel,
    current_pid: u32,
    input: &str,
    mode: Mode,
    client: &reqwest::Client,
    server_url: &str,
    save_history: bool,
    _memory_scheduler: Option<&mut MemoryScheduler>,
    permission_prompt: Option<&dyn Fn(&str) -> String>,
) -> Result<String> {
    const MAX_ATTEMPTS: usize = 4;
    let mut last_error: Option<NpcError> = None;

    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            eprintln!(
                "{YELLOW}npcpy server error; restarting server and retrying turn (attempt {}/{})...{RESET}",
                attempt,
                MAX_ATTEMPTS - 1
            );
            if let Err(e) = restart_server(client, server_url).await {
                last_error = Some(NpcError::Other(format!("failed to restart server: {e}")));
                continue;
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            if client
                .get(server_url)
                .timeout(std::time::Duration::from_secs(2))
                .send()
                .await
                .is_err()
            {
                last_error = Some(NpcError::Other(
                    "npcpy server smoke test failed after restart".to_string(),
                ));
                continue;
            }
            if let Some(process) = kernel.get_process_mut(current_pid) {
                process.new_turn();
            }
        }

        match run_stream_turn_with_interrupt(
            kernel,
            current_pid,
            input,
            mode,
            client,
            server_url,
            save_history,
            None,
            None,
            permission_prompt,
        )
        .await
        {
            Ok(output) => return Ok(output),
            Err(e) => {
                if is_server_error(&e) {
                    last_error = Some(e);
                    continue;
                }
                return Err(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        NpcError::Other("stream turn failed after server restart retries".to_string())
    }))
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

pub async fn spawn_npc_from_registered_teams(
    name: &str,
    kernel: &mut Kernel,
    current_pid: u32,
) -> Result<u32> {
    if kernel.find_by_name(name).is_some() {
        return Ok(0);
    }

    let teams = load_registered_teams();
    for (_team_name, team_dir) in teams {
        let path = std::path::Path::new(&team_dir).join(format!("{}.npc", name));
        if path.exists() {
            let npc = npcrs::npc_compiler::NPC::from_file(&path)
                .map_err(|e| NpcError::Other(format!("Failed to load NPC {}: {}", name, e)))?;
            let pid = kernel.spawn(npc, current_pid, Capabilities::root());
            return Ok(pid);
        }
    }
    Ok(0)
}

pub async fn run_command_loop(
    kernel: &mut Kernel,
    current_pid: u32,
    client: &reqwest::Client,
    server_url: &str,
    command: &str,
) -> Result<()> {
    let max_cmd_turns: usize = {
        let mut rng = rand::thread_rng();
        let z: f64 = rng.sample(rand_distr::StandardNormal);
        ((60.0 + 30.0 * z) as i64).clamp(10, 300) as usize
    };
    let initial_input = format!(
        "{}\n\n[one-shot mode] Solve this and then call the `stop` tool as soon as the task is complete.",
        command
    );
    let mut turn_input = initial_input.clone();
    let mut last_output = String::new();
    let auto_approve = |_prompt: &str| "Yes (session)".to_string();
    for turn in 0..max_cmd_turns {
        match run_stream_turn(
            kernel,
            current_pid,
            &turn_input,
            Mode::Agent,
            client,
            server_url,
            true,
            None,
            Some(&auto_approve),
        )
        .await
        {
            Ok(output) => {
                last_output = output;
                let tool_calls: Vec<npcrs::ToolCall> = kernel
                    .get_process(current_pid)
                    .and_then(|p| p.messages.iter().rev().find(|m| m.role == "assistant"))
                    .and_then(|m| m.tool_calls.as_ref())
                    .cloned()
                    .unwrap_or_default();
                let terminal = tool_calls
                    .iter()
                    .any(|tc| tc.r#type == "function" && tc.function.name == "stop");
                if tool_calls.is_empty() || terminal {
                    if !last_output.is_empty() {
                        println!("{}", last_output);
                    }
                    return Ok(());
                }
                turn_input = "The tool results are above. Call `stop` if the task is complete, otherwise take the next step.".to_string();
                if turn == max_cmd_turns - 1 {
                    eprintln!(
                        "{YELLOW}Warning: reached max agent turns for -c command; stopping.{RESET}"
                    );
                }
            }
            Err(e) => {
                eprintln!("{RED}Error: {e}{RESET}");
                std::process::exit(1);
            }
        }
    }
    if !last_output.is_empty() {
        println!("{}", last_output);
    }
    Ok(())
}

pub async fn run_command(
    command: &str,
    npc_name: Option<&str>,
    override_model: Option<&str>,
    override_provider: Option<&str>,
) -> Result<()> {
    let server_url =
        std::env::var("NPCSH_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:5237".to_string());
    let http_client = reqwest::Client::new();

    if let Err(e) = ensure_server_running(&http_client, &server_url).await {
        eprintln!("{RED}Error: unable to reach or start npcpy server: {e}{RESET}");
        std::process::exit(1);
    }

    let team_dir = crate::find_team_dir();
    let db_path = std::env::var("NPCSH_HISTORY_DB")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| shellexpand::tilde("~/npcsh_history.db").to_string());

    let mut current_pid: u32 = 0;
    let mut kernel = Kernel::boot(&team_dir, &db_path)?;

    if npc_name.is_none() {
        if let Some(lead) = kernel.team.forenpc.as_deref() {
            if let Some(proc) = kernel.find_by_name(lead) {
                current_pid = proc.pid;
            }
        }
    }

    if let Some(forced_conv_id) = std::env::var("NPCSH_CONVERSATION_ID")
        .ok()
        .filter(|s| !s.is_empty())
    {
        if let Some(process) = kernel.get_process_mut(current_pid) {
            process.conversation_id = forced_conv_id;
        }
    }

    if let Some(name) = npc_name {
        if let Some(proc) = kernel.find_by_name(name) {
            current_pid = proc.pid;
        } else {
            match spawn_npc_from_registered_teams(name, &mut kernel, current_pid).await {
                Ok(new_pid) if new_pid != 0 => {
                    current_pid = new_pid;
                }
                _ => eprintln!(
                    "{RED}Warning: NPC '{}' not found; using default.{RESET}",
                    name
                ),
            }
        }
    }
    {
        let process = kernel.get_process_mut(current_pid).unwrap();
        if let Some(m) = override_model {
            process.npc.model = Some(m.to_string());
        }
        if let Some(p) = override_provider {
            process.npc.provider = Some(p.to_string());
        }
    }

    run_command_loop(&mut kernel, current_pid, &http_client, &server_url, command).await
}
