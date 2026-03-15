//! npcsh-rs — the NPC OS shell.
//!
//! Full-featured REPL with readline, tab completion, colored prompt,
//! mode system, .npcshrc config, and streaming output.

use npcrs::error::Result;
use npcrs::kernel::Kernel;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{CompletionType, Config, Editor, Helper};
use std::borrow::Cow;

// ── Colors ──
const CYAN: &str = "\x1b[36m";
const PURPLE: &str = "\x1b[35m";
const DIM: &str = "\x1b[90m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

// ── Tab Completion ──
struct NpcHelper {
    npc_names: Vec<String>,
    commands: Vec<String>,
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

        // Add jinxes as slash commands
        for j in jinx_names {
            commands.push(format!("/{}", j));
        }

        Self { npc_names, commands }
    }
}

impl Completer for NpcHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let word_start = line[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
        let word = &line[word_start..pos];

        let mut matches = Vec::new();

        if word.starts_with('@') {
            // NPC completion
            let prefix = &word[1..];
            for name in &self.npc_names {
                if name.starts_with(prefix) {
                    matches.push(Pair {
                        display: format!("@{}", name),
                        replacement: format!("@{} ", name),
                    });
                }
            }
        } else if word.starts_with('/') {
            // Command completion
            for cmd in &self.commands {
                if cmd.starts_with(word) {
                    matches.push(Pair {
                        display: cmd.clone(),
                        replacement: format!("{} ", cmd),
                    });
                }
            }
        }

        Ok((word_start, matches))
    }
}

impl Hinter for NpcHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        if pos != line.len() {
            return None; // Only hint at end of line
        }
        let word_start = line.rfind(' ').map(|i| i + 1).unwrap_or(0);
        let word = &line[word_start..];

        if word.starts_with('/') && word.len() > 1 {
            for cmd in &self.commands {
                if cmd.starts_with(word) && cmd.len() > word.len() {
                    return Some(cmd[word.len()..].to_string());
                }
            }
        } else if word.starts_with('@') && word.len() > 1 {
            let prefix = &word[1..];
            for name in &self.npc_names {
                if name.starts_with(prefix) && name.len() > prefix.len() {
                    return Some(name[prefix.len()..].to_string());
                }
            }
        }
        None
    }
}

impl Highlighter for NpcHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Borrowed(prompt)
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("\x1b[90m{}\x1b[0m", hint))
    }
}

impl Validator for NpcHelper {}
impl Helper for NpcHelper {}

// ── Mode ──
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
    // Init logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("npcrs=warn".parse().unwrap()),
        )
        .with_target(false)
        .without_time()
        .init();

    // Load .env and .npcshrc
    let _ = dotenvy::dotenv();
    load_npcshrc();

    // Check if invoked as `npc` or `npc-jinx` (shebang mode)
    let invoked_as = std::env::args()
        .next()
        .and_then(|a| std::path::Path::new(&a).file_name().map(|f| f.to_string_lossy().to_string()))
        .unwrap_or_default();

    let args: Vec<String> = std::env::args().collect();

    // npc <file.npc|file.jinx|init> [args...] — detect by extension or subcommand
    if invoked_as == "npc" {
        if let Some(file) = args.get(1) {
            if file == "init" {
                let dir = args.get(2).map(|s| s.as_str()).unwrap_or(".");
                return init_team(dir);
            } else if file.ends_with(".jinx") {
                let jinx_args: Vec<&str> = args[2..].iter().map(|s| s.as_str()).collect();
                return exec_jinx_file(file, &jinx_args).await;
            } else if file.ends_with(".npc") {
                return exec_npc_file(file, args.get(2).map(|s| s.as_str())).await;
            }
            // Not a file — fall through to REPL with --npc flag handling below
        }
    }

    // Find team directory
    let team_dir = find_team_dir();
    let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();

    // Boot the kernel
    let mut kernel = Kernel::boot(&team_dir, &db_path)?;

    // Print welcome
    print_welcome(&kernel);

    // Set up readline
    let config = Config::builder()
        .completion_type(CompletionType::List)
        .build();

    let npc_names: Vec<String> = kernel.ps().iter().map(|p| p.npc.name.clone()).collect();
    let jinx_names: Vec<String> = kernel.jinx_names().into_iter().map(String::from).collect();
    let helper = NpcHelper::new(npc_names, jinx_names);

    let history_path = shellexpand::tilde("~/.npcsh_history").to_string();
    let mut rl = Editor::with_config(config).unwrap();
    rl.set_helper(Some(helper));
    let _ = rl.load_history(&history_path);

    // REPL state
    let mut current_pid: u32 = 0;
    let mut mode = Mode::Agent;
    let mut _turn_count: u64 = 0;
    let mut session_input_tokens: u64 = 0;
    let mut session_output_tokens: u64 = 0;
    let mut session_cost: f64 = 0.0;
    let session_start = std::time::Instant::now();

    loop {
        // Build prompt
        let npc_name = kernel
            .get_process(current_pid)
            .map(|p| p.npc.name.as_str())
            .unwrap_or("???");

        let cwd = std::env::current_dir()
            .map(|p| {
                let s = p.display().to_string();
                // Shorten home dir
                let home = shellexpand::tilde("~").to_string();
                if let Some(rest) = s.strip_prefix(&home) {
                    format!("~{}", rest)
                } else {
                    s
                }
            })
            .unwrap_or_else(|_| "?".to_string());

        let model = kernel
            .get_process(current_pid)
            .map(|p| p.npc.resolved_model())
            .unwrap_or_else(|| "?".to_string());

        // Build usage hint (like Python's token_hint)
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

        let prompt = format!(
            "{DIM}{cwd}{RESET} {CYAN}{BOLD}{npc_name}{RESET} {DIM}[{mode}|{model}]{RESET}{usage_hint}\n{PURPLE}>{RESET} "
        );

        // Read input
        let input = match rl.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                eprintln!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        };

        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }

        rl.add_history_entry(&input).ok();

        // ── Built-in commands ──
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
                println!("{BOLD}npcsh-rs{RESET} — NPC OS Shell v{}\n", env!("CARGO_PKG_VERSION"));
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

        // /set key=value
        if input.starts_with("/set ") {
            let rest = input.strip_prefix("/set ").unwrap().trim();
            handle_set_command(rest, &mut kernel, current_pid, &mut mode);
            continue;
        }

        // @npc delegation or switch
        if input.starts_with('@') {
            let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
            let target = parts[0];

            if let Some(command) = parts.get(1) {
                // Delegate
                eprintln!("{DIM}delegating to @{target}...{RESET}");
                match kernel.delegate(current_pid, target, command).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
                }
            } else {
                // Switch
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

        // /slash commands → try as jinx
        if input.starts_with('/') {
            let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
            let cmd_name = parts[0];
            let args_str = parts.get(1).unwrap_or(&"");

            // Check if it's a known jinx
            if kernel.jinxes.contains_key(cmd_name) {
                let mut args = std::collections::HashMap::new();

                // Parse key=value args from the command line
                if !args_str.is_empty() {
                    // Try key=value pairs first
                    let mut has_kv = false;
                    for part in args_str.split_whitespace() {
                        if let Some((k, v)) = part.split_once('=') {
                            args.insert(k.to_string(), v.to_string());
                            has_kv = true;
                        }
                    }
                    // If no key=value, assign to first input
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

        // ── cd is special — changes working directory (like npcsh handle_cd_command) ──
        if input.starts_with("cd ") || input == "cd" {
            let target = input.strip_prefix("cd").unwrap().trim();
            let target = if target.is_empty() {
                shellexpand::tilde("~").to_string()
            } else {
                shellexpand::tilde(target).to_string()
            };
            // Resolve relative paths
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

        // ── Terminal editors — hand over full terminal (like npcsh open_terminal_editor) ──
        if is_terminal_editor(&input) {
            run_interactive(&input);
            continue;
        }

        // ── Interactive commands — hand over full terminal (like npcsh handle_interactive_command) ──
        if is_interactive(&input) {
            run_interactive(&input);
            continue;
        }

        // ── Mode-specific dispatch (mirrors npcsh process_pipeline_command) ──
        _turn_count += 1;

        // Get process metadata for DB saves
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

        let exec_result = match mode {
            Mode::Agent => {
                if is_bash_command(&input) {
                    run_bash(&input).await;
                    None // bash output already printed
                } else {
                    Some(kernel.exec(current_pid, &input).await)
                }
            }
            Mode::Chat => {
                Some(kernel.exec_chat(current_pid, &input).await)
            }
            Mode::Cmd => {
                if run_bash(&input).await {
                    None
                } else {
                    Some(kernel.exec(current_pid, &input).await)
                }
            }
        };

        // Process result (like Python's process_result)
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

                    // Save user message with input tokens
                    let _ = kernel.history.save_conversation_message(
                        &conv_id, "user", &input, &cwd,
                        Some(&model_str), Some(&provider_str),
                        Some(&npc_name_str), Some(&team_name_str),
                        None, None, None,
                        Some(in_tok), None, None,
                    );

                    // Save assistant message with output tokens + cost
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

        // Accumulate session totals and show usage
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

    // Save history
    let _ = rl.save_history(&history_path);

    eprintln!("\n{DIM}Kernel shutting down.{RESET}");
    let s = kernel.stats();
    eprintln!(
        "{DIM}uptime: {}s | tokens: {} | cost: ${:.4}{RESET}",
        s.uptime_secs, s.total_tokens, s.total_cost_usd
    );
    Ok(())
}

/// Handle /set key=value commands.
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

/// Print the welcome screen.
fn print_welcome(kernel: &Kernel) {
    let s = kernel.stats();

    const BLUE: &str = "\x1b[1;94m";
    const RUST: &str = "\x1b[1;38;5;202m";

    const SHAD: &str = "\x1b[38;5;238m";
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
    eprintln!("  {BOLD}npcsh{RESET} v{} {DIM}(rust){RESET}", env!("CARGO_PKG_VERSION"));
    eprintln!("  {DIM}{} processes | {} jinxes | /help for commands{RESET}", s.total_processes, s.jinx_count);
    eprintln!();

    // Modes + NPCs
    eprintln!("  {DIM}mode:{RESET} {BOLD}agent{RESET}  {DIM}switch:{RESET} /agent  /cmd  /chat");
    eprint!("  {DIM}npcs:{RESET} ");
    let names: Vec<String> = kernel.ps().iter().map(|p| format!("{BLUE}@{}{RESET}", p.npc.name)).collect();
    eprintln!("{}", names.join("  "));
    eprintln!();

    // Jinxes organized by directory group (mirrors Python's startup display)
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

    // Sort groups in a sensible order
    let group_order = ["bin", "lib", "skills", "etc", "sys", "usr", "root", "other"];
    let mut sorted_groups: Vec<_> = groups.keys().cloned().collect();
    sorted_groups.sort_by_key(|g| {
        group_order.iter().position(|o| o == g).unwrap_or(99)
    });

    for group in &sorted_groups {
        if let Some(subdirs) = groups.get(group) {
            eprintln!("  {RUST}{group}/{RESET}");
            // Top-level jinxes (no subdir)
            if let Some(names) = subdirs.get(&None) {
                let mut sorted = names.clone();
                sorted.sort();
                let line: Vec<String> = sorted.iter().map(|n| format!("/{}", n)).collect();
                // Wrap to terminal width
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
            // Sub-directory groups
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

/// Load ~/.npcshrc if it exists (sets env vars for model/provider config).
fn load_npcshrc() {
    let rc_path = shellexpand::tilde("~/.npcshrc").to_string();
    let path = std::path::Path::new(&rc_path);

    if !path.exists() {
        return;
    }

    // Parse simple KEY=VALUE and export KEY=VALUE lines
    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Strip "export " prefix
            let line = line.strip_prefix("export ").unwrap_or(line);

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"').trim_matches('\'');
                // Only set if not already set (env takes precedence)
                if std::env::var(key).is_err() {
                    // SAFETY: We only call this at startup before spawning threads
                    unsafe { std::env::set_var(key, value) };
                }
            }
        }
    }
}

// ── Terminal/Interactive command lists (from npcsh/execution.py) ──

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

/// Check if input is a bash command — the npcsh way.
/// Checks shell builtins first, then looks up the command in PATH via `which`.
fn is_bash_command(input: &str) -> bool {
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.is_empty() {
        return false;
    }

    let cmd = parts[0];

    // Shell builtins
    if SHELL_BUILTINS.contains(&cmd) {
        return true;
    }

    // Check PATH (equivalent to shutil.which)
    if let Ok(output) = std::process::Command::new("which")
        .arg(cmd)
        .output()
    {
        return output.status.success();
    }

    false
}

/// Check if input is a terminal editor.
fn is_terminal_editor(input: &str) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    TERMINAL_EDITORS.contains(&cmd)
}

/// Check if input is an interactive command.
fn is_interactive(input: &str) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    INTERACTIVE_COMMANDS.contains(&cmd)
}

/// Run a bash command directly, returning true if it succeeded.
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

/// Run an interactive/editor command (inherits full terminal).
fn run_interactive(input: &str) {
    let _ = std::process::Command::new("bash")
        .arg("-c")
        .arg(input)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();
}

/// Find the team directory (project-local or global).
fn find_team_dir() -> String {
    // CLI args
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--team") {
        if let Some(dir) = args.get(pos + 1) {
            return dir.clone();
        }
    }

    // Project-local
    if std::path::Path::new("./npc_team").exists() {
        return "./npc_team".to_string();
    }

    // Global
    let global = shellexpand::tilde("~/.npcsh/npc_team").to_string();
    if std::path::Path::new(&global).exists() {
        return global;
    }

    ".".to_string()
}

/// Execute a .npc file directly (shebang: #!/usr/bin/env npc)
async fn exec_npc_file(npc_file: &str, command: Option<&str>) -> Result<()> {
    use npcrs::npc::Npc;
    use npcrs::llm::LlmClient;
    use npcrs::memory::CommandHistory;

    // Use the proper loader which handles shebang stripping + Jinja2 preprocessing
    let npc = Npc::from_file(npc_file)?;

    let client = LlmClient::from_env();
    let model = npc.resolved_model();
    let provider = npc.resolved_provider();

    if let Some(cmd) = command {
        // One-shot: run command and exit
        let system = npc.system_prompt(None);
        let messages = vec![
            npcrs::llm::Message::system(system),
            npcrs::llm::Message::user(cmd),
        ];
        let response = client
            .chat_completion(&provider, &model, &messages, None, npc.api_url.as_deref())
            .await?;
        if let Some(text) = response.message.content {
            println!("{}", text);
        }
    } else {
        // Interactive: boot with this NPC as forenpc
        let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
        let team_dir = find_team_dir();
        let mut kernel = npcrs::Kernel::boot(&team_dir, &db_path)?;

        // Replace the init process NPC with this one
        if let Some(p) = kernel.get_process_mut(0) {
            p.npc = npc;
        }

        eprintln!("\x1b[1;94m{}\x1b[0m", npc_file);
        eprintln!("NPC: {} | model: {} | provider: {}", 
            kernel.get_process(0).map(|p| p.npc.name.as_str()).unwrap_or("?"),
            model, provider);
        eprintln!();

        // Simple REPL
        let mut rl = rustyline::DefaultEditor::new().unwrap();
        loop {
            let input = match rl.readline("\x1b[35m> \x1b[0m") {
                Ok(line) => line.trim().to_string(),
                Err(_) => break,
            };
            if input.is_empty() { continue; }
            if input == "exit" || input == "quit" { break; }
            rl.add_history_entry(&input).ok();

            match kernel.exec(0, &input).await {
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

/// Execute a .jinx file directly (shebang: #!/usr/bin/env npc-jinx)
async fn exec_jinx_file(jinx_file: &str, args: &[&str]) -> Result<()> {
    use npcrs::jinx::{self, load_jinx_from_file};

    // Use the proper loader which handles shebang stripping
    let jinx = load_jinx_from_file(jinx_file)?;

    // Build input values from positional args or key=value pairs
    let mut input_values = std::collections::HashMap::new();
    let mut positional_idx = 0;

    for arg in args {
        if let Some((k, v)) = arg.split_once('=') {
            input_values.insert(k.to_string(), v.to_string());
        } else {
            // Map positional args to jinx input names
            if let Some(input) = jinx.inputs.get(positional_idx) {
                input_values.insert(input.name.clone(), arg.to_string());
                positional_idx += 1;
            }
        }
    }

    let empty_jinxes = std::collections::HashMap::new();
    let result = jinx::execute_jinx(&jinx, &input_values, &empty_jinxes).await?;

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

/// npc init [dir] — create a working npc_team/ in the current directory
fn init_team(dir: &str) -> Result<()> {
    let dir = std::path::Path::new(dir).canonicalize().unwrap_or_else(|_| std::path::PathBuf::from(dir));
    let team_dir = dir.join("npc_team");

    if team_dir.exists() && std::fs::read_dir(&team_dir).map(|mut d| d.next().is_some()).unwrap_or(false) {
        eprintln!("npc_team/ already exists at {}", team_dir.display());
        return Ok(());
    }

    std::fs::create_dir_all(team_dir.join("jinxes")).unwrap();

    // Copy core jinxes from global install
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

    // forenpc.npc
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

    // coder.npc
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

    // team.ctx
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
