
use npcrs::error::Result;
use npcrs::kernel::Kernel;
use npcrs::process::ProcessState;
use npcrs::{calculate_cost, Message};
use std::collections::HashMap;
use std::io::{self, Write};

mod stream_client;

const CYAN: &str = "\x1b[36m";
const PURPLE: &str = "\x1b[35m";
const DIM: &str = "\x1b[90m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

fn handle_paste_input(raw: &str) -> (String, Option<String>) {
    let bytes = raw.as_bytes();
    let is_binary = if bytes.len() > 4 {
        (bytes[0] == 0x89 && bytes[1] == b'P' && bytes[2] == b'N' && bytes[3] == b'G')
            || (bytes[0] == 0xFF && bytes[1] == 0xD8)
            || (bytes[0] == b'G' && bytes[1] == b'I' && bytes[2] == b'F' && bytes[3] == b'8')
            || (bytes[0] == b'B' && bytes[1] == b'M')
            || raw.starts_with("data:image/")
            || {
                let non_printable = bytes.iter().take(100).filter(|&&b| b < 32 && b != b'\n' && b != b'\r' && b != b'\t').count();
                non_printable > 10
            }
    } else {
        false
    };

    if is_binary {
        let ext = if bytes.len() > 4 && bytes[0] == 0x89 { ".png" }
            else if bytes.len() > 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 { ".jpg" }
            else if raw.starts_with("data:image/png") { ".png" }
            else if raw.starts_with("data:image/jpeg") || raw.starts_with("data:image/jpg") { ".jpg" }
            else { ".bin" };

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("npcsh_paste_{}{}", std::process::id(), ext));
        let write_data = if raw.starts_with("data:image/") {
            if let Some((_, data)) = raw.split_once(',') {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.decode(data).unwrap_or_default()
            } else {
                raw.as_bytes().to_vec()
            }
        } else {
            raw.as_bytes().to_vec()
        };
        let _ = std::fs::write(&temp_path, &write_data);
        let path_str = temp_path.to_string_lossy().to_string();
        eprintln!("\x1b[90m[pasted image: {}]\x1b[0m", path_str);
        return (format!("[pasted image: {}]", path_str), Some(path_str));
    }

    let line_count = raw.lines().count();
    let char_count = raw.len();
    if line_count > 3 || char_count > 500 {
        eprintln!("\x1b[90m[pasted: {} lines, {} chars]\x1b[0m", line_count, char_count);
        return (raw.to_string(), Some(raw.to_string()));
    }

    (raw.to_string(), None)
}

struct NpcHelper {
    npc_names: Vec<String>,
    commands: Vec<String>,
}

#[derive(Clone)]
struct Completion {
    display: String,
    replacement: String,
}

impl NpcHelper {
    fn new(npc_names: Vec<String>, jinx_names: Vec<String>) -> Self {
        let mut commands = vec![
            "/ps", "/stats", "/help", "/quit", "/exit", "/clear",
            "/agent", "/chat", "/cmd", "/switch", "/kill", "/jinxes",
            "/set", "/history",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();

        for j in jinx_names {
            commands.push(format!("/{}", j));
        }

        Self { npc_names, commands }
    }

    fn complete(&self, line: &str, pos: usize) -> (usize, Vec<Completion>) {
        let word_start = line[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
        let word = &line[word_start..pos];

        let mut matches = Vec::new();

        if word.starts_with('@') {
            let prefix = &word[1..];
            for name in &self.npc_names {
                if name.starts_with(prefix) {
                    matches.push(Completion {
                        display: format!("@{}", name),
                        replacement: format!("@{} ", name),
                    });
                }
            }
        } else if word.starts_with('/') {
            for cmd in &self.commands {
                if cmd.starts_with(word) {
                    matches.push(Completion {
                        display: cmd.clone(),
                        replacement: format!("{} ", cmd),
                    });
                }
            }
        }

        (word_start, matches)
    }

}

#[derive(Clone, PartialEq)]
enum Mode {
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

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--version" || a == "-v") {
        println!("npcsh {}", env!("NPCSH_VERSION"));
        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("npcrs=warn".parse().unwrap()),
        )
        .with_target(false)
        .without_time()
        .init();

    let _ = dotenvy::dotenv();
    load_npcshrc();

    let server_url = std::env::var("NPCPY_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:5237".to_string());
    let http_client = reqwest::Client::new();

    let invoked_as = std::env::args()
        .next()
        .and_then(|a| std::path::Path::new(&a).file_name().map(|f| f.to_string_lossy().to_string()))
        .unwrap_or_default();

    if invoked_as == "npc" {
        if let Some(file) = args.get(1) {
            if file == "init" {
                let dir = args.get(2).map(|s| s.as_str()).unwrap_or(".");
                return init_team(dir);
            } else if file.ends_with(".jinx") {
                let jinx_args: Vec<&str> = args[2..].iter().map(|s| s.as_str()).collect();
                return exec_jinx_file(file, &jinx_args).await;
            } else if file.ends_with(".npc") {
                return exec_npc_file(
                    file,
                    args.get(2).map(|s| s.as_str()),
                    &http_client,
                    &server_url,
                )
                .await;
            }
        }
    }

    if let Some(file) = args.get(1) {
        if file.ends_with(".nsh") && !file.starts_with('-') {
            return exec_nsh_file(file, &http_client, &server_url).await;
        }
    }

    if let Some(pos) = args.iter().position(|a| a == "-c" || a == "--command") {
        if let Some(command) = args.get(pos + 1) {
            let team_dir = find_team_dir();
            let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
            let mut kernel = Kernel::boot(&team_dir, &db_path)?;
            match run_stream_turn(
                &mut kernel,
                0,
                command,
                Mode::Agent,
                &http_client,
                &server_url,
            )
            .await
            {
                Ok(output) => {
                    if !output.is_empty() {
                        println!("{}", output);
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
            return Ok(());
        }
    }

    if args.iter().any(|a| a == "--refresh") {
        let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
        let user_npc_team = shellexpand::tilde("~/.npcsh/npc_team").to_string();
        let jinxes_dir = std::path::Path::new(&user_npc_team).join("jinxes");
        if jinxes_dir.exists() {
            let _ = std::fs::remove_dir_all(&jinxes_dir);
            eprintln!("Cleared existing jinxes directory");
        }
        if let Ok(entries) = std::fs::read_dir(&user_npc_team) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".npc") {
                        let _ = std::fs::remove_file(entry.path());
                        eprintln!("Removed {}", name);
                    }
                }
            }
        }
        run_python_initialization(&db_path);
        eprintln!("Refresh complete!");
        std::process::exit(0);
    }

    let team_dir = find_team_dir();
    let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();

    let team_path = std::path::Path::new(&team_dir);
    let jinxes_dir = team_path.join("jinxes");
    let has_npcs = match std::fs::read_dir(team_path) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .any(|e| {
                e.path()
                    .extension()
                    .map_or(false, |ext| ext == "npc")
            }),
        Err(_) => false,
    };
    if !jinxes_dir.exists() || !has_npcs {
        run_python_initialization(&db_path);
    }

    let mut kernel = Kernel::boot(&team_dir, &db_path)?;

    print_welcome(&kernel);

    eprintln!("{DIM}  connected to npcpy server{RESET}");

    let npc_names: Vec<String> = kernel.ps().iter().map(|p| p.npc.name.clone()).collect();
    let jinx_names: Vec<String> = kernel.jinx_names().into_iter().map(String::from).collect();
    let helper = NpcHelper::new(npc_names, jinx_names);

    let history_path = shellexpand::tilde("~/.npcsh_history").to_string();
    let mut history: Vec<String> = std::fs::read_to_string(&history_path)
        .unwrap_or_default()
        .lines()
        .map(|s| s.to_string())
        .collect();
    let mut history_index: Option<usize> = None;

    let mut current_pid: u32 = 0;
    let mut mode = Mode::Agent;
    let mut _turn_count: u64 = 0;
    let mut session_input_tokens: u64 = 0;
    let mut session_output_tokens: u64 = 0;
    let mut session_cost: f64 = 0.0;
    let session_start = std::time::Instant::now();

    loop {
        let npc_name = kernel
            .get_process(current_pid)
            .map(|p| p.npc.name.as_str())
            .unwrap_or("???");

        let cwd = std::env::current_dir()
            .map(|p| {
                let s = p.display().to_string();
                let home = shellexpand::tilde("~").to_string();
                let s = if let Some(rest) = s.strip_prefix(&home) {
                    format!("~{}", rest)
                } else {
                    s
                };
                const MAX_CWD: usize = 20;
                if s.len() > MAX_CWD {
                    let sep = std::path::MAIN_SEPARATOR;
                    let parts: Vec<&str> = s.split(sep).filter(|x| !x.is_empty()).collect();
                    if let Some(last) = parts.last() {
                        format!(".../{last}")
                    } else {
                        format!("...{}", &s[s.len().saturating_sub(MAX_CWD - 3)..])
                    }
                } else {
                    s
                }
            })
            .unwrap_or_else(|_| "?".to_string());

        let model = kernel
            .get_process(current_pid)
            .map(|p| p.npc.resolved_model())
            .unwrap_or_else(|| "?".to_string());

        let usage_hint = if session_input_tokens > 0 || session_output_tokens > 0 {
            let elapsed = session_start.elapsed().as_secs();
            let time_str = if elapsed >= 3600 {
                format!("{}h{}m", elapsed / 3600, (elapsed % 3600) / 60)
            } else if elapsed >= 60 {
                format!("{}m{}s", elapsed / 60, elapsed % 60)
            } else {
                format!("{}s", elapsed)
            };
            let cost_str = if session_cost > 0.0 {
                format!(" | ${:.4}", session_cost)
            } else {
                String::new()
            };
            format!(
                " {DIM}{},{} tok{} | {}{RESET}",
                session_input_tokens, session_output_tokens, cost_str, time_str
            )
        } else {
            String::new()
        };

        let prompt = if usage_hint.is_empty() {
            format!(
                "{CYAN}{BOLD}{npc_name}{RESET} {DIM}[{mode}|{model}]{RESET} {DIM}{cwd}{RESET} {PURPLE}>{RESET} "
            )
        } else {
            format!(
                "{CYAN}{BOLD}{npc_name}{RESET} {DIM}[{mode}|{model}]{RESET} {DIM}{cwd}{RESET}\n{DIM}{usage_hint}{RESET}\n{PURPLE}>{RESET} "
            )
        };

        let input = match readline_raw(
            &prompt,
            &mut history,
            &mut history_index,
            &helper,
            &mut kernel,
            current_pid,
        ) {
            Ok(ReadlineResult::Input(line)) => {
                history.push(line.clone());
                history_index = None;
                line
            }
            Ok(ReadlineResult::Cancel) => continue,
            Ok(ReadlineResult::Eof) => {
                println!();
                break;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        };

        let (input, _pasted_content) = handle_paste_input(&input);
        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }


        let handled = match input.as_str() {
            "exit" | "quit" | "/quit" | "/exit" => break,

            "/ps" => {
                for p in kernel.ps() {
                    let state_color = match p.state {
                        npcrs::process::ProcessState::Running => GREEN,
                        npcrs::process::ProcessState::Blocked => YELLOW,
                        npcrs::process::ProcessState::Dead => RED,
                        _ => DIM,
                    };
                    println!(
                        "  {CYAN}@{:<12}{RESET} pid:{:<3} {state_color}{:?}{RESET}  tokens:{}/{} cost:${:.4} turns:{}",
                        p.npc.name, p.pid, p.state,
                        p.usage.total_input_tokens, p.usage.total_output_tokens,
                        p.usage.total_cost_usd, p.usage.total_turns,
                    );
                }
                true
            }

            "/stats" => {
                let s = kernel.stats();
                println!(
                    "{BOLD}Kernel Stats{RESET}\n  uptime: {}s\n  processes: {} (run:{} blk:{} dead:{})\n  tokens: {} (in+out)\n  cost: ${:.4}\n  jinxes: {}",
                    s.uptime_secs, s.total_processes, s.running, s.blocked, s.dead,
                    s.total_tokens, s.total_cost_usd, s.jinx_count,
                );
                true
            }

            "/help" => {
                println!("{BOLD}npcsh-rs{RESET} — NPC OS Shell v{}\n", env!("NPCSH_VERSION"));
                println!("{BOLD}Modes:{RESET}");
                println!("  {CYAN}/agent{RESET}          Full agent mode (tools + bash + LLM)");
                println!("  {CYAN}/chat{RESET}           Chat-only mode (LLM, no tools)");
                println!("  {CYAN}/cmd{RESET}            Command mode (bash first, LLM fallback)");
                println!();
                println!("{BOLD}NPC Commands:{RESET}");
                println!("  {CYAN}@npc{RESET}            Switch to NPC process");
                println!("  {CYAN}@npc command{RESET}    Delegate command to NPC");
                println!("  {CYAN}/switch <npc>{RESET}   Switch to NPC process");
                println!("  {CYAN}/kill{RESET}           Kill current process");
                println!();
                println!("{BOLD}Info:{RESET}");
                println!("  {CYAN}/ps{RESET}             List processes");
                println!("  {CYAN}/stats{RESET}          Kernel stats");
                println!("  {CYAN}/jinxes{RESET}         List available tools");
                println!("  {CYAN}/history{RESET}        Show conversation history");
                println!();
                println!("{BOLD}Config:{RESET}");
                println!("  {CYAN}/set key=val{RESET}    Set model, provider, mode");
                println!("  {CYAN}/clear{RESET}          Clear conversation");
                println!();
                println!("{BOLD}Shell:{RESET}");
                println!("  Any text is sent to the current NPC.");
                println!("  In {CYAN}/cmd{RESET} mode, input runs as bash first.");
                println!("  Tab completes @npcs and /commands.");
                true
            }

            "/agent" => {
                mode = Mode::Agent;
                eprintln!("{GREEN}Switched to agent mode{RESET}");
                true
            }
            "/chat" => {
                mode = Mode::Chat;
                eprintln!("{GREEN}Switched to chat mode{RESET}");
                true
            }
            "/cmd" => {
                mode = Mode::Cmd;
                eprintln!("{GREEN}Switched to cmd mode{RESET}");
                true
            }

            "/jinxes" => {
                let names = kernel.jinx_names();
                let mut sorted: Vec<&str> = names;
                sorted.sort();
                println!("{BOLD}Available jinxes ({}):{RESET}", sorted.len());
                for chunk in sorted.chunks(6) {
                    println!("  {}", chunk.iter().map(|n| format!("{CYAN}/{n}{RESET}")).collect::<Vec<_>>().join("  "));
                }
                true
            }

            "/clear" => {
                if let Some(p) = kernel.get_process_mut(current_pid) {
                    p.messages.clear();
                    eprintln!("{GREEN}Conversation cleared{RESET}");
                }
                true
            }

            "/history" => {
                if let Some(p) = kernel.get_process(current_pid) {
                    if p.messages.is_empty() {
                        println!("{DIM}(no messages){RESET}");
                    } else {
                        for m in &p.messages {
                            let role_color = match m.role.as_str() {
                                "user" => CYAN,
                                "assistant" => GREEN,
                                _ => DIM,
                            };
                            let content = m.content.as_deref().unwrap_or("");
                            let preview = if content.len() > 80 {
                                format!("{}...", &content[..80])
                            } else {
                                content.to_string()
                            };
                            println!("  {role_color}{:<10}{RESET} {}", m.role, preview);
                        }
                    }
                }
                true
            }

            "/kill" => {
                if current_pid == 0 {
                    eprintln!("{RED}Cannot kill init (pid 0){RESET}");
                } else {
                    let name = kernel.get_process(current_pid).map(|p| p.npc.name.clone());
                    kernel.kill(current_pid, 0).ok();
                    current_pid = 0;
                    eprintln!("{YELLOW}Killed @{} — switched to init{RESET}", name.unwrap_or_default());
                }
                true
            }

            _ => false,
        };

        if handled {
            continue;
        }

        if input.starts_with("/set ") {
            let rest = input.strip_prefix("/set ").unwrap().trim();
            handle_set_command(rest, &mut kernel, current_pid, &mut mode);
            continue;
        }

        if input.starts_with('@') {
            let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
            let target = parts[0];

            if let Some(command) = parts.get(1) {
                eprintln!("{DIM}delegating to @{target}...{RESET}");
                match kernel.delegate(current_pid, target, command).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
                }
            } else {
                if let Some(proc) = kernel.find_by_name(target) {
                    current_pid = proc.pid;
                    eprintln!("{GREEN}Switched to @{target} (pid:{current_pid}){RESET}");
                } else {
                    eprintln!("{RED}NPC '{target}' not found.{RESET} Available:");
                    for p in kernel.ps() {
                        eprintln!("  {CYAN}@{}{RESET}", p.npc.name);
                    }
                }
            }
            continue;
        }

        if input.starts_with('/') {
            let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
            let cmd_name = parts[0];
            let args_str = parts.get(1).unwrap_or(&"");

            if kernel.jinxes.contains_key(cmd_name) {
                let mut args = std::collections::HashMap::new();

                if !args_str.is_empty() {
                    let mut has_kv = false;
                    for part in args_str.split_whitespace() {
                        if let Some((k, v)) = part.split_once('=') {
                            args.insert(k.to_string(), v.to_string());
                            has_kv = true;
                        }
                    }
                    if !has_kv {
                        if let Some(first_input) = kernel.jinxes[cmd_name].inputs.first() {
                            args.insert(first_input.name.clone(), args_str.to_string());
                        }
                    }
                }

                match kernel.syscall(current_pid, cmd_name, &args).await {
                    Ok(output) => {
                        if !output.is_empty() {
                            println!("{}", output);
                        }
                    }
                    Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
                }
            } else {
                eprintln!("{RED}Unknown command: /{cmd_name}{RESET}");
            }
            continue;
        }

        if input.starts_with("cd ") || input == "cd" {
            let target = input.strip_prefix("cd").unwrap().trim();
            let target = if target.is_empty() {
                shellexpand::tilde("~").to_string()
            } else {
                shellexpand::tilde(target).to_string()
            };
            let target = if std::path::Path::new(&target).is_relative() {
                let cwd = std::env::current_dir().unwrap_or_default();
                cwd.join(&target)
                    .canonicalize()
                    .unwrap_or_else(|_| cwd.join(&target))
                    .display()
                    .to_string()
            } else {
                target
            };
            match std::env::set_current_dir(&target) {
                Ok(_) => eprintln!("{DIM}Changed to: {target}{RESET}"),
                Err(e) => eprintln!("{RED}cd: {e}{RESET}"),
            }
            continue;
        }

        if is_terminal_editor(&input) {
            run_interactive(&input);
            continue;
        }

        if is_interactive(&input) {
            run_interactive(&input);
            continue;
        }

        _turn_count += 1;

        let (npc_name_str, team_name_str, model_str, provider_str, conv_id) = {
            let p = kernel.get_process(current_pid);
            let npc_name = p.map(|p| p.npc.name.clone()).unwrap_or_else(|| "npcsh".to_string());
            let team_name = kernel.team.source_dir.as_deref()
                .and_then(|d| std::path::Path::new(d).file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("npcsh")
                .to_string();
            let model = p.map(|p| p.npc.resolved_model()).unwrap_or_else(|| "qwen3.5:2b".to_string());
            let provider = p.map(|p| p.npc.resolved_provider()).unwrap_or_else(|| "ollama".to_string());
            let conv_id = p.map(|p| p.conversation_id.clone()).unwrap_or_default();
            (npc_name, team_name, model, provider, conv_id)
        };
        let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| ".".to_string());

        let cli_intercepted = false;

        let exec_result = if cli_intercepted {
            None
        } else {
            match mode {
                Mode::Agent => {
                    if is_bash_command(&input) {
                        run_bash(&input).await;
                        None
                    } else {
                        Some(
                            run_stream_turn(
                                &mut kernel,
                                current_pid,
                                &input,
                                mode.clone(),
                                &http_client,
                                &server_url,
                            )
                            .await,
                        )
                    }
                }
                Mode::Chat | Mode::Cmd => {
                    if matches!(mode, Mode::Cmd) && run_bash(&input).await {
                        None
                    } else {
                        Some(
                            run_stream_turn(
                                &mut kernel,
                                current_pid,
                                &input,
                                mode.clone(),
                                &http_client,
                                &server_url,
                            )
                            .await,
                        )
                    }
                }
            }
        };

        if let Some(result) = exec_result {
            match result {
                Ok(output) => {
                    if !output.is_empty() {
                        println!("\n{}", output);
                    }

                    let p = kernel.get_process(current_pid);
                    let (in_tok, out_tok, cost) = p.map(|p| {
                        (p.usage.total_input_tokens, p.usage.total_output_tokens, p.usage.total_cost_usd)
                    }).unwrap_or((0, 0, 0.0));

                    let _ = kernel.history.save_conversation_message(
                        &conv_id, "user", &input, &cwd,
                        Some(&model_str), Some(&provider_str),
                        Some(&npc_name_str), Some(&team_name_str),
                        None, None, None,
                        Some(in_tok), None, None,
                    );

                    let _ = kernel.history.save_conversation_message(
                        &conv_id, "assistant", &output, &cwd,
                        Some(&model_str), Some(&provider_str),
                        Some(&npc_name_str), Some(&team_name_str),
                        None, None, None,
                        Some(in_tok), Some(out_tok), Some(cost),
                    );
                }
                Err(e) => {
                    eprintln!("{RED}Error: {e}{RESET}");
                }
            }
        }

        if let Some(p) = kernel.get_process(current_pid) {
            session_input_tokens = p.usage.total_input_tokens;
            session_output_tokens = p.usage.total_output_tokens;
            session_cost = p.usage.total_cost_usd;
            if p.usage.total_turns > 0 {
                eprintln!(
                    "{DIM}[tokens:{}/{} | turn:{} | cost:${:.4}]{RESET}",
                    p.usage.total_input_tokens,
                    p.usage.total_output_tokens,
                    p.usage.total_turns,
                    p.usage.total_cost_usd,
                );
            }
        }
    }

    let _ = std::fs::write(&history_path, history.join("\n") + "\n");

    eprintln!("\n{DIM}Kernel shutting down.{RESET}");
    let s = kernel.stats();
    eprintln!(
        "{DIM}uptime: {}s | tokens: {} | cost: ${:.4}{RESET}",
        s.uptime_secs, s.total_tokens, s.total_cost_usd
    );
    Ok(())
}

async fn run_stream_turn(
    kernel: &mut Kernel,
    current_pid: u32,
    input: &str,
    mode: Mode,
    client: &reqwest::Client,
    server_url: &str,
) -> Result<String> {
    {
        let process = kernel.get_process_mut(current_pid).ok_or_else(|| {
            npcrs::NpcError::Other(format!("No process with pid {}", current_pid))
        })?;
        if let Some(reason) = process.usage.exceeds(&process.limits) {
            process.kill(137);
            return Err(npcrs::NpcError::Other(format!(
                "Process {} killed: {}",
                current_pid, reason
            )));
        }
        process.state = ProcessState::Running;
        process.new_turn();
    }

    let (
        model,
        provider,
        system,
        _api_url,
        _api_key,
        npc_name,
        active_npc,
        tool_defs,
        _think_mode,
        conv_id,
    ) = {
        let process = kernel.get_process(current_pid).ok_or_else(|| {
            npcrs::NpcError::Other(format!("No process with pid {}", current_pid))
        })?;
        let (td, _ex) = process.npc.resolve_tools(&kernel.jinxes);
        let model = process.npc.resolved_model();
        let provider = process.npc.resolved_provider();
        let system = process.npc.system_prompt(kernel.team.context.as_deref());
        let api_url = process.npc.api_url.clone();
        let api_key = process.npc.api_key.clone();
        let npc_name = process.npc.name.clone();
        let active_npc = process.npc.clone();
        let think_mode = process.think;
        let conv_id = process.conversation_id.clone();
        (
            model, provider, system, api_url, api_key, npc_name, active_npc, td,
            think_mode, conv_id,
        )
    };

    let tools = if tool_defs.is_empty() || mode == Mode::Chat {
        None
    } else {
        Some(tool_defs.clone())
    };

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let path_cmd = format!("The current working directory is: {}", cwd);
    let ls_files = if let Ok(entries) = std::fs::read_dir(&cwd) {
        let files: Vec<String> = entries
            .flatten()
            .take(100)
            .map(|e| e.path().to_string_lossy().to_string())
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

    let tool_guidance = if tools.is_some() {
        let tool_names: Vec<&str> =
            tool_defs.iter().map(|t| t.function.name.as_str()).collect();
        format!(
            "\nYou have access to these tools: {}. Call tools via the function calling interface.\n\n\
Use tools when you need to take action (run commands, search, edit files, etc.). Use chat to respond to the user.\n\
IMPORTANT: After at most 3-5 tool calls, you MUST call stop to finish. Do not keep reading files or running commands indefinitely — gather what you need, respond, and stop.\n\
Do not call stop without first calling chat to deliver a response to the user.\n\
The user can see tool outputs directly. Do not re-write or repeat them in your chat response — just reference the relevant parts.",
            tool_names.join(", ")
        )
    } else {
        String::new()
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

    {
        let process = kernel.get_process_mut(current_pid).unwrap();
        process.messages.push(Message::user(input));
    }

    let max_iterations = 12;
    let mut total_input_tokens: u64 = 0;
    let mut total_output_tokens: u64 = 0;
    let mut final_output = String::new();
    let mut tool_calls_count = 0;
    let mut stop_requested = false;

    for iteration in 0..max_iterations {
        if stop_requested {
            break;
        }

        let mut messages = vec![Message::system(&system)];
        {
            let process = kernel.get_process(current_pid).unwrap();
            messages.extend(process.messages.clone());
        }

        let iter_prompt = if iteration == 0 {
            format!("{}\n{}{}", input, context_info, tool_guidance)
        } else {
            "Continue. Call stop when done.".to_string()
        };
        messages.push(Message::user(&iter_prompt));

        eprintln!(
            "\x1b[90m  [iter {}] {} msgs\x1b[0m",
            iteration + 1,
            messages.len(),
        );

        let execution_mode = if tools.is_some() { "tool_agent" } else { "chat" }.to_string();

        let request = stream_client::StreamRequest {
            model: model.clone(),
            provider: provider.clone(),
            messages,
            tools: tools.clone(),
            commandstr: iter_prompt.clone(),
            npc: Some(npc_name.clone()),
            registered_teams: registered_teams.clone(),
            conversation_id: Some(conv_id.clone()),
            current_path: Some(cwd.clone()),
            execution_mode,
        };

        let response = stream_client::call_stream(client, server_url, &request)
            .await
            .map_err(|e| npcrs::NpcError::Other(e))?;

        if let Some(ref usage) = response.usage {
            total_input_tokens += usage.prompt_tokens;
            total_output_tokens += usage.completion_tokens;
            let cost = calculate_cost(&model, usage.prompt_tokens, usage.completion_tokens);
            let process = kernel.get_process_mut(current_pid).unwrap();
            process.record_usage(usage.prompt_tokens, usage.completion_tokens, cost);
        }

        {
            let process = kernel.get_process_mut(current_pid).unwrap();
            process.last_streamed = response.streamed;
            process.last_thinking = response.message.thinking.clone();
        }

        if let Some(ref tool_calls) = response.message.tool_calls {
            tool_calls_count += 1;

            {
                let process = kernel.get_process_mut(current_pid).unwrap();
                process.messages.push(response.message.clone());
            }

            let called: Vec<String> = tool_calls
                .iter()
                .map(|tc| {
                    let preview = if tc.function.arguments.len() > 200 {
                        format!("{}...", &tc.function.arguments[..200])
                    } else {
                        tc.function.arguments.clone()
                    };
                    format!("{}({})", tc.function.name, preview)
                })
                .collect();
            eprintln!(
                "\x1b[90m  [iter {}] tools: {}\x1b[0m",
                iteration + 1,
                called.join(", ")
            );

            for tc in tool_calls {
                let tc_id = tc.id.clone();
                let tc_name = tc.function.name.clone();
                let tc_args_str = tc.function.arguments.clone();

                let args: HashMap<String, String> =
                    serde_json::from_str(&tc_args_str).unwrap_or_default();

                let tool_result = kernel.run_tool(&tc_name, &args, &active_npc).await;

                if tc_name == "chat" {
                    final_output = args
                        .get("message")
                        .or_else(|| args.get("query"))
                        .cloned()
                        .unwrap_or_default();
                } else {
                    eprintln!("\x1b[36m\n⚡ {} [{}|{}]:\x1b[0m", tc_name, model, provider);
                    let preview = if tool_result.len() > 500 {
                        format!(
                            "{}...\n[{} chars total]",
                            &tool_result[..500],
                            tool_result.len()
                        )
                    } else {
                        tool_result.clone()
                    };
                    eprintln!("{}", preview);
                }

                if tc_name == "stop" {
                    stop_requested = true;
                }

                let process = kernel.get_process_mut(current_pid).unwrap();
                process
                    .messages
                    .push(Message::tool_result(&tc_id, &tool_result));
            }
        } else {
            final_output = response.message.content.clone().unwrap_or_default();
            let process = kernel.get_process_mut(current_pid).unwrap();
            process.messages.push(response.message);
            break;
        }
    }

    eprintln!(
        "\x1b[90m  [{} iterations, {} tool call rounds]\x1b[0m",
        std::cmp::min(max_iterations, tool_calls_count + 1),
        tool_calls_count,
    );

    let process = kernel.get_process_mut(current_pid).unwrap();
    process.state = ProcessState::Blocked;
    Ok(final_output)
}

fn handle_set_command(rest: &str, kernel: &mut Kernel, pid: u32, mode: &mut Mode) {
    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    if parts.len() != 2 {
        eprintln!("Usage: /set key=value");
        eprintln!("  model=gpt-4o  provider=openai  mode=chat");
        return;
    }
    let key = parts[0].trim();
    let value = parts[1].trim();

    match key {
        "model" => {
            if let Some(p) = kernel.get_process_mut(pid) {
                p.npc.model = Some(value.to_string());
                eprintln!("{GREEN}model = {value}{RESET}");
            }
        }
        "provider" => {
            if let Some(p) = kernel.get_process_mut(pid) {
                p.npc.provider = Some(value.to_string());
                eprintln!("{GREEN}provider = {value}{RESET}");
            }
        }
        "mode" => match value {
            "agent" => *mode = Mode::Agent,
            "chat" => *mode = Mode::Chat,
            "cmd" => *mode = Mode::Cmd,
            _ => eprintln!("{RED}Unknown mode: {value}{RESET}"),
        },
        _ => eprintln!("{RED}Unknown setting: {key}{RESET}"),
    }
}

fn print_welcome(kernel: &Kernel) {
    let s = kernel.stats();

    const BLUE: &str = "\x1b[1;94m";
    const RUST: &str = "\x1b[1;38;5;202m";

    eprintln!();
    eprintln!("  {BLUE}                         {RESET}{RUST}     ██╗     {RESET}");
    eprintln!("  {BLUE}                         {RESET}{RUST}     ██║     {RESET}");
    eprintln!("  {BLUE}█▀▀▀█╗ ██████╗  ██████╗ {RESET}{RUST}█████╗██████╗ {RESET}");
    eprintln!("  {BLUE}██╔═██║██╔══██╗██╔════╝ {RESET}{RUST}██╔══╝██╔═██║ {RESET}");
    eprintln!("  {BLUE}██║ ██║██║  ██║██║      {RESET}{RUST}█████╗██║ ██║ {RESET}");
    eprintln!("  {BLUE}██║ ██║██║  ██║██║      {RESET}{RUST}╚══██║██║ ██║ {RESET}");
    eprintln!("  {BLUE}██║ ██║██████╔╝╚██████╗ {RESET}{RUST}█████║██║ ██║ {RESET}");
    eprintln!("  {BLUE}╚═╝ ╚═╝██╔═══╝  ╚═════╝{RESET}{RUST}╚════╝╚═╝ ╚═╝ {RESET}");
    eprintln!("  {BLUE}       ██║              {RESET}");
    eprintln!("  {BLUE}       ╚═╝              {RESET}");
    eprintln!();
    eprintln!("  {BOLD}npcsh{RESET} v{} {DIM}(rust){RESET}", env!("NPCSH_VERSION"));
    eprintln!("  {DIM}{} processes | {} jinxes | /help for commands{RESET}", s.total_processes, s.jinx_count);
    eprintln!();

    eprintln!("  {DIM}mode:{RESET} {BOLD}agent{RESET}  {DIM}switch:{RESET} /agent  /cmd  /chat");
    eprint!("  {DIM}npcs:{RESET} ");
    let names: Vec<String> = kernel.ps().iter().map(|p| format!("{BLUE}@{}{RESET}", p.npc.name)).collect();
    eprintln!("{}", names.join("  "));
    eprintln!();

    let mut groups: std::collections::BTreeMap<String, std::collections::BTreeMap<Option<String>, Vec<String>>> =
        std::collections::BTreeMap::new();

    for (jname, jinx) in &kernel.jinxes {
        let (group, subdir) = if let Some(ref sp) = jinx.source_path {
            let parts: Vec<&str> = sp.split(std::path::MAIN_SEPARATOR).collect();
            if let Some(idx) = parts.iter().position(|&p| p == "jinxes") {
                let remaining = &parts[idx + 1..];
                if remaining.len() > 2 {
                    (remaining[0].to_string(), Some(remaining[1].to_string()))
                } else if remaining.len() > 1 {
                    (remaining[0].to_string(), None)
                } else {
                    ("root".to_string(), None)
                }
            } else {
                ("other".to_string(), None)
            }
        } else {
            ("other".to_string(), None)
        };
        groups.entry(group).or_default().entry(subdir).or_default().push(jname.clone());
    }

    let group_order = ["bin", "lib", "skills", "etc", "sys", "usr", "root", "other"];
    let mut sorted_groups: Vec<_> = groups.keys().cloned().collect();
    sorted_groups.sort_by_key(|g| {
        group_order.iter().position(|o| o == g).unwrap_or(99)
    });

    for group in &sorted_groups {
        if let Some(subdirs) = groups.get(group) {
            eprintln!("  {RUST}{group}/{RESET}");
            if let Some(names) = subdirs.get(&None) {
                let mut sorted = names.clone();
                sorted.sort();
                let line: Vec<String> = sorted.iter().map(|n| format!("/{}", n)).collect();
                let mut current = String::from("    ");
                for item in &line {
                    if current.len() + item.len() + 2 > 80 && current.trim().len() > 0 {
                        eprintln!("{}", current);
                        current = String::from("    ");
                    }
                    current.push_str(item);
                    current.push_str("  ");
                }
                if current.trim().len() > 0 {
                    eprintln!("{}", current);
                }
            }
            for (sd, names) in subdirs {
                if let Some(sd) = sd {
                    let mut sorted = names.clone();
                    sorted.sort();
                    let items: Vec<String> = sorted.iter().map(|n| format!("/{}", n)).collect();
                    eprintln!("      {DIM}{sd}:{RESET} {}", items.join("  "));
                }
            }
        }
    }

    eprintln!();
    eprintln!("  {DIM}/jinxes for full list{RESET}");
    eprintln!();
}

fn load_npcshrc() {
    let rc_path = shellexpand::tilde("~/.npcshrc").to_string();
    let path = std::path::Path::new(&rc_path);

    if !path.exists() {
        return;
    }

    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line = line.strip_prefix("export ").unwrap_or(line);

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"').trim_matches('\'');
                if std::env::var(key).is_err() {
                    unsafe { std::env::set_var(key, value) };
                }
            }
        }
    }
}


const TERMINAL_EDITORS: &[&str] = &[
    "vim", "nvim", "nano", "vi", "emacs", "less", "more", "man",
];

const INTERACTIVE_COMMANDS: &[&str] = &[
    "ipython", "python", "python3", "node", "irb", "ghci",
    "mysql", "psql", "sqlite3", "redis-cli", "mongo",
    "ssh", "telnet", "ftp", "sftp", "top", "htop", "watch", "r",
];

const SHELL_BUILTINS: &[&str] = &[
    "cd", "pwd", "echo", "export", "source", "alias", "unalias",
    "history", "set", "unset", "read", "eval", "exec", "exit",
    "return", "shift", "trap", "wait", "jobs", "fg", "bg",
    "kill", "ulimit", "umask", "type", "hash", "true", "false",
];

fn is_bash_command(input: &str) -> bool {
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.is_empty() {
        return false;
    }

    let cmd = parts[0];

    if SHELL_BUILTINS.contains(&cmd) {
        return true;
    }

    if let Ok(output) = std::process::Command::new("which")
        .arg(cmd)
        .output()
    {
        return output.status.success();
    }

    false
}

fn is_terminal_editor(input: &str) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    TERMINAL_EDITORS.contains(&cmd)
}

fn is_interactive(input: &str) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    INTERACTIVE_COMMANDS.contains(&cmd)
}

async fn run_bash(input: &str) -> bool {
    match tokio::process::Command::new("bash")
        .arg("-c")
        .arg(input)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .await
    {
        Ok(status) => status.success(),
        Err(e) => {
            eprintln!("{RED}bash: {e}{RESET}");
            false
        }
    }
}

fn run_interactive(input: &str) {
    let _ = std::process::Command::new("bash")
        .arg("-c")
        .arg(input)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();
}

fn run_python_initialization(db_path: &str) {
    eprintln!("Re-initializing npc_team from package...");
    let output = std::process::Command::new("python3")
        .args([
            "-c",
            &format!(
                "import os; os.environ['NPCSH_INITIALIZED'] = '0'; \
                 from npcsh._state import initialize_base_npcs_if_needed; \
                 initialize_base_npcs_if_needed('{db_path}')"
            ),
        ])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            eprintln!("Initialization complete.");
        }
        Ok(o) => {
            eprintln!(
                "Initialization output: {}",
                String::from_utf8_lossy(&o.stdout)
            );
            eprintln!(
                "Initialization errors: {}",
                String::from_utf8_lossy(&o.stderr)
            );
        }
        Err(e) => {
            eprintln!("Failed to run Python initialization: {}", e);
        }
    }
}

fn find_team_dir() -> String {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--team") {
        if let Some(dir) = args.get(pos + 1) {
            return dir.clone();
        }
    }

    if std::path::Path::new("./npc_team").exists() {
        return "./npc_team".to_string();
    }

    let global = shellexpand::tilde("~/.npcsh/npc_team").to_string();
    if std::path::Path::new(&global).exists() {
        return global;
    }

    ".".to_string()
}

async fn exec_npc_file(
    npc_file: &str,
    command: Option<&str>,
    client: &reqwest::Client,
    server_url: &str,
) -> Result<()> {
    use npcrs::npc_compiler::NPC;

    let npc = NPC::from_file(npc_file)?;

    let model = npc.resolved_model();
    let provider = npc.resolved_provider();

    if let Some(cmd) = command {
        let request = stream_client::StreamRequest {
            model,
            provider,
            messages: vec![
                npcrs::Message::system(npc.system_prompt(None)),
                npcrs::Message::user(cmd),
            ],
            tools: None,
            commandstr: cmd.to_string(),
            npc: Some(npc.name.clone()),
            registered_teams: None,
            conversation_id: None,
            current_path: Some(
                std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| ".".to_string())),
            execution_mode: "chat".to_string(),
        };
        let response = stream_client::call_stream(client, server_url, &request)
            .await
            .map_err(|e| npcrs::NpcError::Other(e))?;
        if let Some(text) = response.message.content {
            println!("{}", text);
        }
    } else {
        let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
        let team_dir = find_team_dir();
        let mut kernel = npcrs::Kernel::boot(&team_dir, &db_path)?;

        if let Some(p) = kernel.get_process_mut(0) {
            p.npc = npc;
        }

        eprintln!("\x1b[1;94m{}\x1b[0m", npc_file);
        eprintln!(
            "NPC: {} | model: {} | provider: {}",
            kernel.get_process(0).map(|p| p.npc.name.as_str()).unwrap_or("?"),
            model,
            provider
        );
        eprintln!();

        let mut rl = rustyline::DefaultEditor::new().unwrap();
        loop {
            let input = match rl.readline("\x1b[35m> \x1b[0m") {
                Ok(line) => line.trim().to_string(),
                Err(_) => break,
            };
            if input.is_empty() {
                continue;
            }
            if input == "exit" || input == "quit" {
                break;
            }
            rl.add_history_entry(&input).ok();

            match run_stream_turn(
                &mut kernel,
                0,
                &input,
                Mode::Agent,
                client,
                server_url,
            )
            .await
            {
                Ok(output) => {
                    if !output.is_empty() {
                        println!("\n{}", output);
                    }
                }
                Err(e) => eprintln!("\x1b[31mError: {}\x1b[0m", e),
            }
        }
    }

    Ok(())
}

async fn exec_jinx_file(jinx_file: &str, args: &[&str]) -> Result<()> {
    use npcrs::npc_compiler::{load_jinx_from_file, execute_jinx};

    let jinx = load_jinx_from_file(jinx_file)?;

    let mut input_values = std::collections::HashMap::new();
    let mut positional_idx = 0;

    for arg in args {
        if let Some((k, v)) = arg.split_once('=') {
            input_values.insert(k.to_string(), v.to_string());
        } else {
            if let Some(input) = jinx.inputs.get(positional_idx) {
                input_values.insert(input.name.clone(), arg.to_string());
                positional_idx += 1;
            }
        }
    }

    let empty_jinxes = std::collections::HashMap::new();
    let result = execute_jinx(&jinx, &input_values, &empty_jinxes).await?;

    if !result.output.is_empty() {
        println!("{}", result.output);
    }

    if !result.success {
        if let Some(err) = result.error {
            eprintln!("Error: {}", err);
        }
        std::process::exit(1);
    }

    Ok(())
}

async fn exec_nsh_file(
    script_path: &str,
    client: &reqwest::Client,
    server_url: &str,
) -> Result<()> {
    let team_dir = find_team_dir();
    let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
    let mut kernel = Kernel::boot(&team_dir, &db_path)?;

    let content = std::fs::read_to_string(script_path)?;

    let raw_lines: Vec<&str> = content.lines().collect();
    let lines: Vec<&str> = raw_lines.into_iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    let mut variables: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut last_output = String::new();

    for (i, line) in lines.iter().enumerate() {
        let line = line.to_string();

        let cmd_to_exec = if let Some((var_name, var_expr)) = line.trim().strip_prefix('$').and_then(|rest| {
            if let Some(eq_pos) = rest.find('=') {
                let vname = rest[..eq_pos].trim().to_string();
                let expr = rest[eq_pos + 1..].trim().to_string();
                if !vname.is_empty() && vname.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    return Some((vname, expr));
                }
            }
            None
        }) {
            Some((var_name, var_expr))
        } else {
            None
        };

        let mut substituted = if let Some((_, ref expr)) = cmd_to_exec {
            expr.clone()
        } else {
            line.clone()
        };
        for (k, v) in &variables {
            substituted = substituted.replace(&format!("${}", k), v);
            substituted = substituted.replace(&format!("${{{}}}", k), v);
        }
        substituted = substituted.replace("$_", &last_output);

        let cmd = if substituted.starts_with('!') {
            substituted[1..].trim().to_string()
        } else {
            substituted
        };

        match run_stream_turn(
            &mut kernel,
            0,
            &cmd,
            Mode::Agent,
            client,
            server_url,
        )
        .await
        {
            Ok(output) => {
                last_output = output.clone();
                if !output.is_empty() {
                    println!("{}", output);
                }
                if let Some((var_name, _)) = cmd_to_exec {
                    variables.insert(var_name, output);
                }
            }
            Err(e) => {
                eprintln!("Error on line {}: {}", i + 1, e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

fn init_team(dir: &str) -> Result<()> {
    let dir = std::path::Path::new(dir).canonicalize().unwrap_or_else(|_| std::path::PathBuf::from(dir));
    let team_dir = dir.join("npc_team");

    if team_dir.exists() && std::fs::read_dir(&team_dir).map(|mut d| d.next().is_some()).unwrap_or(false) {
        eprintln!("npc_team/ already exists at {}", team_dir.display());
        return Ok(());
    }

    std::fs::create_dir_all(team_dir.join("jinxes")).unwrap();

    let global_jinxes = shellexpand::tilde("~/.npcsh/npc_team/jinxes/lib").to_string();
    let dest_lib = team_dir.join("jinxes").join("lib");
    if std::path::Path::new(&global_jinxes).is_dir() && !dest_lib.exists() {
        fn copy_dir(src: &std::path::Path, dst: &std::path::Path) {
            std::fs::create_dir_all(dst).ok();
            if let Ok(entries) = std::fs::read_dir(src) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let dest = dst.join(entry.file_name());
                    if path.is_dir() {
                        copy_dir(&path, &dest);
                    } else {
                        std::fs::copy(&path, &dest).ok();
                    }
                }
            }
        }
        copy_dir(std::path::Path::new(&global_jinxes), &dest_lib);
    }

    let forenpc = "#!/usr/bin/env npc\n\
name: forenpc\n\
primary_directive: You are the team coordinator. Delegate tasks to specialists and synthesize results.\n\
model: qwen3.5:2b\n\
provider: ollama\n\
jinxes:\n\
  - sh\n\
  - python\n\
  - edit_file\n\
  - load_file\n\
  - web_search\n\
  - file_search\n\
  - delegate\n";
    let fp = team_dir.join("forenpc.npc");
    std::fs::write(&fp, forenpc).unwrap();
    #[cfg(unix)] {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fp, std::fs::Permissions::from_mode(0o755)).ok();
    }

    let coder = "#!/usr/bin/env npc\n\
name: coder\n\
primary_directive: You are a coding specialist. Write, debug, and refactor code. Run tests. Edit files.\n\
model: qwen3.5:2b\n\
provider: ollama\n\
jinxes:\n\
  - sh\n\
  - python\n\
  - edit_file\n\
  - load_file\n\
  - file_search\n";
    let cp = team_dir.join("coder.npc");
    std::fs::write(&cp, coder).unwrap();
    #[cfg(unix)] {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&cp, std::fs::Permissions::from_mode(0o755)).ok();
    }

    let ctx = "context: A development team.\nforenpc: forenpc\n";
    std::fs::write(team_dir.join("team.ctx"), ctx).unwrap();

    println!("Created npc_team/ at {}", team_dir.display());
    println!("  forenpc.npc — coordinator");
    println!("  coder.npc   — coding specialist");
    println!("  team.ctx    — team context");
    println!();
    println!("Run npcsh to start, or:");
    println!("  npc {} 'what can you do?'", fp.display());
    println!("  npc {} 'list all TODO comments in this project'", cp.display());

    Ok(())
}


enum ReadlineResult {
    Input(String),
    Cancel,
    Eof,
}

fn readline_raw(
    prompt: &str,
    history: &mut Vec<String>,
    history_index: &mut Option<usize>,
    helper: &NpcHelper,
    kernel: &mut Kernel,
    current_pid: u32,
) -> io::Result<ReadlineResult> {
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal;

    terminal::enable_raw_mode()?;

    print!("\x1b[?2004h");
    io::stdout().flush()?;

    let mut buf = String::new();
    let mut pos: usize = 0;

    print!("{}", prompt);
    io::stdout().flush()?;

    let mut tab_matches: Vec<Completion> = Vec::new();
    let mut tab_index: usize = 0;

    let result = loop {
        if crossterm::event::poll(std::time::Duration::from_millis(50))? {
            match event::read()? {
                Event::Paste(text) => {
                    tab_matches.clear();
                    let text = text.replace('\r', "\n");
                    if text.contains('\n') {
                        buf.insert_str(pos, &text);
                        pos += text.len();
                        redraw_prompt(prompt, &buf, pos);
                        io::stdout().flush()?;
                        print!("\r\n");
                        break Ok(ReadlineResult::Input(buf));
                    }
                    buf.insert_str(pos, &text);
                    pos += text.len();
                    redraw_prompt(prompt, &buf, pos);
                    io::stdout().flush()?;
                    continue;
                }
                Event::Key(key) => {
                    match key.code {
                    KeyCode::Char(c) => {
                        if key.modifiers.contains(KeyModifiers::CONTROL) {
                            match c {
                                'c' => {
                                    print!("\r\n");
                                    break Ok(ReadlineResult::Cancel);
                                }
                                'd' => {
                                    if buf.is_empty() {
                                        break Ok(ReadlineResult::Eof);
                                    }
                                }
                                'a' => {
                                    if pos > 0 {
                                        print!("{}", "\x1b[D".repeat(pos));
                                        pos = 0;
                                        io::stdout().flush()?;
                                    }
                                }
                                'e' => {
                                    print!("\r\n");
                                    if let Some(p) = kernel.get_process(current_pid) {
                                        if let Some(ref t) = p.last_thinking {
                                            println!("{BOLD}═══ Thinking ═══{RESET}");
                                            println!("{}", t);
                                            println!("{BOLD}═{RESET}");
                                        } else {
                                            println!("{DIM}(no thinking content available){RESET}");
                                        }
                                    }
                                    redraw_prompt(prompt, &buf, pos);
                                }
                                'o' => {
                                    print!("\r\n");
                                    if let Some(p) = kernel.get_process(current_pid) {
                                        let mut tool_calls: Vec<&npcrs::r#gen::ToolCall> = Vec::new();
                                        for m in p.messages.iter().rev().take(10) {
                                            if let Some(ref tc) = m.tool_calls {
                                                for t in tc.iter().rev() {
                                                    tool_calls.push(t);
                                                }
                                            }
                                        }
                                        if tool_calls.is_empty() {
                                            println!("{DIM}(no tool calls in recent messages){RESET}");
                                        } else {
                                            let total = tool_calls.len().min(5);
                                            println!("{BOLD}═══ Last {} tool call{} ═══{RESET}", total, if total > 1 { "s" } else { "" });
                                            for (i, tc) in tool_calls.iter().take(5).enumerate() {
                                                println!("  [{}/{}] {CYAN}{}{RESET}", i + 1, total, tc.function.name);
                                                let args = &tc.function.arguments;
                                                let preview = if args.len() > 200 {
                                                    format!("{}…", &args[..200])
                                                } else {
                                                    args.to_string()
                                                };
                                                println!("    {}", preview);
                                            }
                                            println!("{BOLD}═{RESET}");
                                        }
                                    }
                                    redraw_prompt(prompt, &buf, pos);
                                }
                                _ => {}
                            }
                        } else {
                            tab_matches.clear();
                            if pos == buf.len() {
                                buf.push(c);
                                print!("{}", c);
                                pos += 1;
                            } else {
                                buf.insert(pos, c);
                                pos += 1;
                                redraw_prompt(prompt, &buf, pos);
                            }
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Backspace => {
                        tab_matches.clear();
                        if pos > 0 {
                            buf.remove(pos - 1);
                            pos -= 1;
                            if pos == buf.len() {
                                print!("\x08 \x08");
                            } else {
                                redraw_prompt(prompt, &buf, pos);
                            }
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Delete => {
                        tab_matches.clear();
                        if pos < buf.len() {
                            buf.remove(pos);
                            redraw_prompt(prompt, &buf, pos);
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Enter => {
                        print!("\r\n");
                        break Ok(ReadlineResult::Input(buf));
                    }
                    KeyCode::Left => {
                        if pos > 0 {
                            pos -= 1;
                            print!("\x1b[D");
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Right => {
                        if pos < buf.len() {
                            pos += 1;
                            print!("\x1b[C");
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Home => {
                        if pos > 0 {
                            print!("{}", "\x1b[D".repeat(pos));
                            pos = 0;
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::End => {
                        if pos < buf.len() {
                            print!("{}", "\x1b[C".repeat(buf.len() - pos));
                            pos = buf.len();
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Up => {
                        if let Some(idx) = *history_index {
                            if idx > 0 {
                                let new_idx = idx - 1;
                                *history_index = Some(new_idx);
                                buf = history[new_idx].clone();
                                pos = buf.len();
                                redraw_prompt(prompt, &buf, pos);
                            }
                        } else if !history.is_empty() {
                            let new_idx = history.len() - 1;
                            *history_index = Some(new_idx);
                            buf = history[new_idx].clone();
                            pos = buf.len();
                            redraw_prompt(prompt, &buf, pos);
                        }
                    }
                    KeyCode::Down => {
                        if let Some(idx) = *history_index {
                            if idx + 1 < history.len() {
                                let new_idx = idx + 1;
                                *history_index = Some(new_idx);
                                buf = history[new_idx].clone();
                                pos = buf.len();
                                redraw_prompt(prompt, &buf, pos);
                            } else {
                                *history_index = None;
                                buf.clear();
                                pos = 0;
                                redraw_prompt(prompt, &buf, pos);
                            }
                        }
                    }
                    KeyCode::Tab => {
                        if tab_matches.is_empty() {
                            let (word_start, matches) = helper.complete(&buf, pos);
                            if matches.len() == 1 {
                                let replacement = &matches[0].replacement;
                                let new_buf = format!("{}{}{}", &buf[..word_start], replacement, &buf[pos..]);
                                pos = word_start + replacement.len();
                                buf = new_buf;
                                redraw_prompt(prompt, &buf, pos);
                                tab_matches.clear();
                            } else if !matches.is_empty() {
                                tab_matches = matches;
                                tab_index = 0;
                                print!("\r\n");
                                for m in &tab_matches {
                                    println!("  {}", m.display);
                                }
                                redraw_prompt(prompt, &buf, pos);
                            }
                        } else {
                            if !tab_matches.is_empty() {
                                tab_index = (tab_index + 1) % tab_matches.len();
                                let word_start = buf[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
                                let replacement = &tab_matches[tab_index].replacement;
                                let new_buf = format!("{}{}{}", &buf[..word_start], replacement, &buf[pos..]);
                                pos = word_start + replacement.len();
                                buf = new_buf;
                                redraw_prompt(prompt, &buf, pos);
                            }
                        }
                    }
                    _ => {}
                }
            }
                _ => {}
            }
        }
    };

    let _ = terminal::disable_raw_mode();

    print!("\x1b[?2004l");
    let _ = io::stdout().flush();

    result
}

fn redraw_prompt(prompt: &str, buf: &str, pos: usize) {
    let prompt_lines = prompt.chars().filter(|c| *c == '\n').count() + 1;

    print!("\x1b[2K\x1b[G");
    for _ in 1..prompt_lines {
        print!("\x1b[A\x1b[2K\x1b[G");
    }
    print!("{}", prompt);
    print!("{}", buf);
    if pos < buf.len() {
        print!("{}", "\x1b[D".repeat(buf.len() - pos));
    }
    let _ = io::stdout().flush();
}
