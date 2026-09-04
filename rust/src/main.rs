use npcrs::error::Result;
use npcrs::kernel::Kernel;
use npcrs::process::{Capabilities, ProcessState};
use npcrs::{Message, calculate_cost};
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod cli_providers;
mod cron;
mod interrupt;
mod tui;
mod tutorial;
mod version_check;

use crate::cli_providers::{CLI_PROVIDERS, run_cli_provider};
use crate::cron::CronRegistry;
use npcsh::agent_turn::{self, run_command_loop};
use npcsh::markdown::render_block;
use npcsh::{
    discover_knowledge_stores, exec_jinx_file, exec_npc_file, find_team_dir, format_memory_context,
    init_team, memory_context_enabled, real_user_home, resolve_team_layout, set_resolved_team_dir,
    stream_client, team_sync,
};

fn cli_sessions() -> &'static Mutex<HashMap<u32, String>> {
    static LOCK: OnceLock<Mutex<HashMap<u32, String>>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(HashMap::new()))
}

async fn ensure_server_running(
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

async fn restart_server(
    client: &reqwest::Client,
    server_url: &str,
) -> std::result::Result<(), String> {
    let host = std::env::var("NPCSH_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("NPCSH_SERVER_PORT").unwrap_or_else(|_| "5237".to_string());

    // Kill any existing process listening on the server port.
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

    // Wait for the port to be free.
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
                let non_printable = bytes
                    .iter()
                    .take(100)
                    .filter(|&&b| b < 32 && b != b'\n' && b != b'\r' && b != b'\t')
                    .count();
                non_printable > 10
            }
    } else {
        false
    };

    if is_binary {
        let ext = if bytes.len() > 4 && bytes[0] == 0x89 {
            ".png"
        } else if bytes.len() > 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
            ".jpg"
        } else if raw.starts_with("data:image/png") {
            ".png"
        } else if raw.starts_with("data:image/jpeg") || raw.starts_with("data:image/jpg") {
            ".jpg"
        } else {
            ".bin"
        };

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("npcsh_paste_{}{}", std::process::id(), ext));
        let write_data = if raw.starts_with("data:image/") {
            if let Some((_, data)) = raw.split_once(',') {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .unwrap_or_default()
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
        eprintln!(
            "\x1b[90m[pasted: {} lines, {} chars]\x1b[0m",
            line_count, char_count
        );
        return (raw.to_string(), Some(raw.to_string()));
    }

    (raw.to_string(), None)
}

struct NpcHelper {
    npc_names: Vec<String>,
    commands: Vec<String>,
    jinx_names: Vec<String>,
}

#[derive(Clone)]
struct Completion {
    display: String,
    replacement: String,
}

impl NpcHelper {
    fn new(npc_names: Vec<String>, jinx_names: Vec<String>) -> Self {
        let commands = CORE_COMMANDS
            .iter()
            .filter(|c| c.name.starts_with('/'))
            .map(|c| c.name.to_string())
            .collect();
        Self {
            npc_names,
            commands,
            jinx_names,
        }
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
            let mut had_cmd = false;
            for cmd in &self.commands {
                if cmd.starts_with(word) {
                    had_cmd = true;
                    matches.push(Completion {
                        display: cmd.clone(),
                        replacement: format!("{} ", cmd),
                    });
                }
            }
            if !had_cmd {
                matches.extend(complete_paths(word));
            }
        } else if !word.is_empty() {
            for name in &self.jinx_names {
                if name.starts_with(word) {
                    matches.push(Completion {
                        display: name.clone(),
                        replacement: format!("{} ", name),
                    });
                }
            }
            matches.extend(complete_paths(word));
        } else {
            matches.extend(complete_paths(word));
        }

        (word_start, matches)
    }
}

fn complete_paths(word: &str) -> Vec<Completion> {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());

    let expanded = shellexpand::tilde(word).to_string();

    let (search_dir, file_prefix, typed_dir_prefix): (String, String, String) =
        if word.ends_with('/') {
            let dir = if expanded.is_empty() {
                cwd.clone()
            } else {
                expanded.clone()
            };
            (dir, String::new(), word.to_string())
        } else if let Some(idx) = expanded.rfind('/') {
            let dir = expanded[..idx + 1].to_string();
            let file = expanded[idx + 1..].to_string();
            let typed_dir = word[..word.rfind('/').map(|i| i + 1).unwrap_or(0)].to_string();
            (dir, file, typed_dir)
        } else {
            (format!("{}/", cwd), expanded.clone(), String::new())
        };

    let mut out = Vec::new();
    let entries = match std::fs::read_dir(&search_dir) {
        Ok(e) => e,
        Err(_) => return out,
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with(&file_prefix) {
            continue;
        }
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let display = if is_dir {
            format!("{}/", name)
        } else {
            name.clone()
        };
        let replacement = if is_dir {
            format!("{}{}/", typed_dir_prefix, name)
        } else {
            format!("{}{} ", typed_dir_prefix, name)
        };
        out.push(Completion {
            display,
            replacement,
        });
    }

    for (dot, dot_display) in [(".", "."), ("..", "..")] {
        if dot.starts_with(&file_prefix) {
            out.push(Completion {
                display: dot_display.to_string(),
                replacement: format!("{}{}/", typed_dir_prefix, dot_display),
            });
        }
    }

    out.sort_by(|a, b| a.display.cmp(&b.display));
    out
}

async fn run_jinx_command(kernel: &mut Kernel, current_pid: u32, cmd_name: &str, args_str: &str) {
    let mut args = HashMap::new();

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
}

#[derive(Clone, Copy, PartialEq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoreCmd {
    Agent,
    Chat,
    CmdMode,
    Kill,
    Clear,
    Flush,
    Config,
    Ctx,
    History,
    Memories,
    Mems,
    Knowledge,
    Model,
    Reattach,
    Set,
    Setup,
    Team,
    Commit,
    Gitt,
    Cron,
    Loop,
    LoopDemo,
    LoopOff,
    LoopOn,
    LoopRm,
    Loops,
    Jinx(&'static str),
    Doctor,
    Init,
    Nsync,
    Reload,
    Shh,
    Update,
    Usage,
    Verbose,
    Exit,
    Help,
    Jinxes,
    Ps,
    Stats,
    Tutorial,
    ProposeJinxes,
}

struct CommandDef {
    name: &'static str,
    category: &'static str,
    description: &'static str,
    cmd: CoreCmd,
}

const CORE_COMMANDS: &[CommandDef] = &[
    CommandDef {
        name: "/exit",
        category: "Info",
        description: "Exit npcsh",
        cmd: CoreCmd::Exit,
    },
    CommandDef {
        name: "exit",
        category: "Info",
        description: "Exit npcsh",
        cmd: CoreCmd::Exit,
    },
    CommandDef {
        name: "/quit",
        category: "Info",
        description: "Exit npcsh",
        cmd: CoreCmd::Exit,
    },
    CommandDef {
        name: "quit",
        category: "Info",
        description: "Exit npcsh",
        cmd: CoreCmd::Exit,
    },
    CommandDef {
        name: "/help",
        category: "Info",
        description: "Show this help",
        cmd: CoreCmd::Help,
    },
    CommandDef {
        name: "help",
        category: "Info",
        description: "Show this help",
        cmd: CoreCmd::Help,
    },
    CommandDef {
        name: "/jinxes",
        category: "Info",
        description: "List available jinxes",
        cmd: CoreCmd::Jinxes,
    },
    CommandDef {
        name: "jinxes",
        category: "Info",
        description: "List available jinxes",
        cmd: CoreCmd::Jinxes,
    },
    CommandDef {
        name: "/ps",
        category: "Info",
        description: "List processes",
        cmd: CoreCmd::Ps,
    },
    CommandDef {
        name: "ps",
        category: "Info",
        description: "List processes",
        cmd: CoreCmd::Ps,
    },
    CommandDef {
        name: "/stats",
        category: "Info",
        description: "Kernel stats",
        cmd: CoreCmd::Stats,
    },
    CommandDef {
        name: "stats",
        category: "Info",
        description: "Kernel stats",
        cmd: CoreCmd::Stats,
    },
    CommandDef {
        name: "/tutorial",
        category: "Info",
        description: "Run interactive tutorial",
        cmd: CoreCmd::Tutorial,
    },
    CommandDef {
        name: "tutorial",
        category: "Info",
        description: "Run interactive tutorial",
        cmd: CoreCmd::Tutorial,
    },
    CommandDef {
        name: "/agent",
        category: "Modes",
        description: "Full agent mode (tools + bash + LLM)",
        cmd: CoreCmd::Agent,
    },
    CommandDef {
        name: "agent",
        category: "Modes",
        description: "Full agent mode (tools + bash + LLM)",
        cmd: CoreCmd::Agent,
    },
    CommandDef {
        name: "/chat",
        category: "Modes",
        description: "Chat-only mode (LLM, no tools)",
        cmd: CoreCmd::Chat,
    },
    CommandDef {
        name: "chat",
        category: "Modes",
        description: "Chat-only mode (LLM, no tools)",
        cmd: CoreCmd::Chat,
    },
    CommandDef {
        name: "/cmd",
        category: "Modes",
        description: "Command mode (bash first, LLM fallback)",
        cmd: CoreCmd::CmdMode,
    },
    CommandDef {
        name: "cmd",
        category: "Modes",
        description: "Command mode (bash first, LLM fallback)",
        cmd: CoreCmd::CmdMode,
    },
    CommandDef {
        name: "/kill",
        category: "NPCs",
        description: "Kill current process",
        cmd: CoreCmd::Kill,
    },
    CommandDef {
        name: "kill",
        category: "NPCs",
        description: "Kill current process",
        cmd: CoreCmd::Kill,
    },
    CommandDef {
        name: "/clear",
        category: "System / Config",
        description: "Clear conversation",
        cmd: CoreCmd::Clear,
    },
    CommandDef {
        name: "clear",
        category: "System / Config",
        description: "Clear conversation",
        cmd: CoreCmd::Clear,
    },
    CommandDef {
        name: "/flush",
        category: "System / Config",
        description: "Flush the last N messages from the conversation",
        cmd: CoreCmd::Flush,
    },
    CommandDef {
        name: "flush",
        category: "System / Config",
        description: "Flush the last N messages from the conversation",
        cmd: CoreCmd::Flush,
    },
    CommandDef {
        name: "/config",
        category: "System / Config",
        description: "Configuration TUI",
        cmd: CoreCmd::Config,
    },
    CommandDef {
        name: "config",
        category: "System / Config",
        description: "Configuration TUI",
        cmd: CoreCmd::Config,
    },
    CommandDef {
        name: "/ctx",
        category: "System / Config",
        description: "Browse and edit team context fields",
        cmd: CoreCmd::Ctx,
    },
    CommandDef {
        name: "ctx",
        category: "System / Config",
        description: "Browse and edit team context fields",
        cmd: CoreCmd::Ctx,
    },
    CommandDef {
        name: "/history",
        category: "System / Config",
        description: "Show conversation history",
        cmd: CoreCmd::History,
    },
    CommandDef {
        name: "history",
        category: "System / Config",
        description: "Show conversation history",
        cmd: CoreCmd::History,
    },
    CommandDef {
        name: "/memories",
        category: "System / Config",
        description: "Browse memory lifecycle TUI",
        cmd: CoreCmd::Memories,
    },
    CommandDef {
        name: "memories",
        category: "System / Config",
        description: "Browse memory lifecycle TUI",
        cmd: CoreCmd::Memories,
    },
    CommandDef {
        name: "/mems",
        category: "System / Config",
        description: "Browse and edit memory lifecycle TUI",
        cmd: CoreCmd::Mems,
    },
    CommandDef {
        name: "mems",
        category: "System / Config",
        description: "Browse and edit memory lifecycle TUI",
        cmd: CoreCmd::Mems,
    },
    CommandDef {
        name: "/kg",
        category: "System / Config",
        description: "Browse knowledge stores and entries",
        cmd: CoreCmd::Knowledge,
    },
    CommandDef {
        name: "kg",
        category: "System / Config",
        description: "Browse knowledge stores and entries",
        cmd: CoreCmd::Knowledge,
    },
    CommandDef {
        name: "/model",
        category: "System / Config",
        description: "Model selection TUI",
        cmd: CoreCmd::Model,
    },
    CommandDef {
        name: "model",
        category: "System / Config",
        description: "Model selection TUI",
        cmd: CoreCmd::Model,
    },
    CommandDef {
        name: "/reattach",
        category: "System / Config",
        description: "Reattach to files/sessions",
        cmd: CoreCmd::Reattach,
    },
    CommandDef {
        name: "reattach",
        category: "System / Config",
        description: "Reattach to files/sessions",
        cmd: CoreCmd::Reattach,
    },
    CommandDef {
        name: "/set",
        category: "System / Config",
        description: "Set model, provider, or mode",
        cmd: CoreCmd::Set,
    },
    CommandDef {
        name: "set",
        category: "System / Config",
        description: "Set model, provider, or mode",
        cmd: CoreCmd::Set,
    },
    CommandDef {
        name: "/setup",
        category: "System / Config",
        description: "First-time setup TUI",
        cmd: CoreCmd::Setup,
    },
    CommandDef {
        name: "setup",
        category: "System / Config",
        description: "First-time setup TUI",
        cmd: CoreCmd::Setup,
    },
    CommandDef {
        name: "/team",
        category: "System / Config",
        description: "Team management TUI",
        cmd: CoreCmd::Team,
    },
    CommandDef {
        name: "team",
        category: "System / Config",
        description: "Team management TUI",
        cmd: CoreCmd::Team,
    },
    CommandDef {
        name: "/commit",
        category: "Tools",
        description: "Commit helper TUI",
        cmd: CoreCmd::Commit,
    },
    CommandDef {
        name: "commit",
        category: "Tools",
        description: "Commit helper TUI",
        cmd: CoreCmd::Commit,
    },
    CommandDef {
        name: "/gitt",
        category: "Tools",
        description: "Git TUI",
        cmd: CoreCmd::Gitt,
    },
    CommandDef {
        name: "gitt",
        category: "Tools",
        description: "Git TUI",
        cmd: CoreCmd::Gitt,
    },
    CommandDef {
        name: "/cron",
        category: "Loops",
        description: "Cron management",
        cmd: CoreCmd::Cron,
    },
    CommandDef {
        name: "cron",
        category: "Loops",
        description: "Cron management",
        cmd: CoreCmd::Cron,
    },
    CommandDef {
        name: "/loop",
        category: "Loops",
        description: "Create a loop",
        cmd: CoreCmd::Loop,
    },
    CommandDef {
        name: "loop",
        category: "Loops",
        description: "Create a loop",
        cmd: CoreCmd::Loop,
    },
    CommandDef {
        name: "/loop_demo",
        category: "Loops",
        description: "Add a demo heartbeat loop",
        cmd: CoreCmd::LoopDemo,
    },
    CommandDef {
        name: "loop_demo",
        category: "Loops",
        description: "Add a demo heartbeat loop",
        cmd: CoreCmd::LoopDemo,
    },
    CommandDef {
        name: "/loopoff",
        category: "Loops",
        description: "Disable a loop",
        cmd: CoreCmd::LoopOff,
    },
    CommandDef {
        name: "loopoff",
        category: "Loops",
        description: "Disable a loop",
        cmd: CoreCmd::LoopOff,
    },
    CommandDef {
        name: "/loopon",
        category: "Loops",
        description: "Enable a loop",
        cmd: CoreCmd::LoopOn,
    },
    CommandDef {
        name: "loopon",
        category: "Loops",
        description: "Enable a loop",
        cmd: CoreCmd::LoopOn,
    },
    CommandDef {
        name: "/looprm",
        category: "Loops",
        description: "Remove a loop",
        cmd: CoreCmd::LoopRm,
    },
    CommandDef {
        name: "looprm",
        category: "Loops",
        description: "Remove a loop",
        cmd: CoreCmd::LoopRm,
    },
    CommandDef {
        name: "/loops",
        category: "Loops",
        description: "List loops",
        cmd: CoreCmd::Loops,
    },
    CommandDef {
        name: "loops",
        category: "Loops",
        description: "List loops",
        cmd: CoreCmd::Loops,
    },
    CommandDef {
        name: "/doctor",
        category: "System Commands",
        description: "Diagnose and auto-fix common issues",
        cmd: CoreCmd::Doctor,
    },
    CommandDef {
        name: "doctor",
        category: "System Commands",
        description: "Diagnose and auto-fix common issues",
        cmd: CoreCmd::Doctor,
    },
    CommandDef {
        name: "/init",
        category: "System Commands",
        description: "Initialize / reinitialize npcsh",
        cmd: CoreCmd::Init,
    },
    CommandDef {
        name: "init",
        category: "System Commands",
        description: "Initialize / reinitialize npcsh",
        cmd: CoreCmd::Init,
    },
    CommandDef {
        name: "/nsync",
        category: "System Commands",
        description: "Sync npcsh state",
        cmd: CoreCmd::Nsync,
    },
    CommandDef {
        name: "nsync",
        category: "System Commands",
        description: "Sync npcsh state",
        cmd: CoreCmd::Nsync,
    },
    CommandDef {
        name: "/refresh",
        category: "System Commands",
        description: "Refresh npcsh (alias of reload)",
        cmd: CoreCmd::Reload,
    },
    CommandDef {
        name: "refresh",
        category: "System Commands",
        description: "Refresh npcsh (alias of reload)",
        cmd: CoreCmd::Reload,
    },
    CommandDef {
        name: "/reload",
        category: "System Commands",
        description: "Reload npcsh state",
        cmd: CoreCmd::Reload,
    },
    CommandDef {
        name: "reload",
        category: "System Commands",
        description: "Reload npcsh state",
        cmd: CoreCmd::Reload,
    },
    CommandDef {
        name: "/shh",
        category: "System Commands",
        description: "Toggle quiet mode",
        cmd: CoreCmd::Shh,
    },
    CommandDef {
        name: "shh",
        category: "System Commands",
        description: "Toggle quiet mode",
        cmd: CoreCmd::Shh,
    },
    CommandDef {
        name: "/update",
        category: "System Commands",
        description: "Update npcsh",
        cmd: CoreCmd::Update,
    },
    CommandDef {
        name: "update",
        category: "System Commands",
        description: "Update npcsh",
        cmd: CoreCmd::Update,
    },
    CommandDef {
        name: "/usage",
        category: "System Commands",
        description: "Show usage info",
        cmd: CoreCmd::Usage,
    },
    CommandDef {
        name: "usage",
        category: "System Commands",
        description: "Show usage info",
        cmd: CoreCmd::Usage,
    },
    CommandDef {
        name: "/verbose",
        category: "System Commands",
        description: "Toggle verbose mode",
        cmd: CoreCmd::Verbose,
    },
    CommandDef {
        name: "verbose",
        category: "System Commands",
        description: "Toggle verbose mode",
        cmd: CoreCmd::Verbose,
    },
    CommandDef {
        name: "/propose-jinxes",
        category: "Tools",
        description: "Suggest new jinxes/skills or edits based on team context",
        cmd: CoreCmd::ProposeJinxes,
    },
    CommandDef {
        name: "propose-jinxes",
        category: "Tools",
        description: "Suggest new jinxes/skills or edits based on team context",
        cmd: CoreCmd::ProposeJinxes,
    },
];

const COMMAND_CATEGORIES: &[&str] = &[
    "Modes",
    "NPCs",
    "System / Config",
    "Tools",
    "Loops",
    "System Commands",
    "Info",
];

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
    ensure_memory_pref();

    resolve_team_layout();

    let server_url =
        std::env::var("NPCSH_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:5237".to_string());
    let http_client = reqwest::Client::new();

    if let Err(e) = ensure_server_running(&http_client, &server_url).await {
        eprintln!("{RED}Error: unable to reach or start npcpy server: {e}{RESET}");
        std::process::exit(1);
    }

    if let Some(file) = args.get(1) {
        if file.ends_with(".nsh") && !file.starts_with('-') {
            return exec_nsh_file(file, &http_client, &server_url).await;
        }
    }

    let cli_npc = arg_value(&args, &["-n", "--npc"]);
    let cli_model = arg_value(&args, &["-m", "--model"]);
    let cli_provider = arg_value(&args, &["-p", "--provider"]);
    let cli_command = arg_value(&args, &["-c", "--command"]);

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
        eprintln!("Refresh complete!");
        std::process::exit(0);
    }

    let team_dir = find_team_dir();
    let db_path = std::env::var("NPCSH_HISTORY_DB")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| shellexpand::tilde("~/npcsh_history.db").to_string());

    let team_path = std::path::Path::new(&team_dir);
    let jinxes_dir = team_path.join("jinxes");
    let has_npcs = match std::fs::read_dir(team_path) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().map_or(false, |ext| ext == "npc")),
        Err(_) => false,
    };
    let mut current_pid: u32 = 0;
    let mut kernel = Kernel::boot(&team_dir, &db_path)?;

    // Default the active NPC to the team's lead/forenpc instead of the init process.
    if cli_npc.is_none() {
        if let Some(lead) = kernel.team.forenpc.as_deref() {
            if let Some(proc) = kernel.find_by_name(lead) {
                current_pid = proc.pid;
            }
        }
    }

    // One-shot/benchmark mode: let the runner pin the conversation_id so it can
    // later retrieve the exact transcript from npcsh_history.db.
    if let Some(forced_conv_id) = std::env::var("NPCSH_CONVERSATION_ID")
        .ok()
        .filter(|s| !s.is_empty())
    {
        if let Some(process) = kernel.get_process_mut(current_pid) {
            process.conversation_id = forced_conv_id;
        }
    }

    if let Some(name) = cli_npc.as_deref() {
        if let Some(proc) = kernel.find_by_name(name) {
            current_pid = proc.pid;
        } else {
            match spawn_npc_from_registered_teams(name, &mut kernel, current_pid).await {
                Ok(new_pid) if new_pid != 0 => {
                    current_pid = new_pid;
                }
                _ => eprintln!("{RED}Warning: NPC '{name}' not found; using default.{RESET}"),
            }
        }
    }
    {
        let process = kernel.get_process_mut(current_pid).unwrap();
        if let Some(m) = cli_model.as_deref() {
            process.npc.model = Some(m.to_string());
        }
        if let Some(p) = cli_provider.as_deref() {
            process.npc.provider = Some(p.to_string());
        }
    }

    // One-shot command mode: run a full agent loop for the instruction and exit.
    if let Some(cmd) = cli_command {
        return run_command_loop(
            &mut kernel,
            current_pid,
            &http_client,
            &server_url,
            cmd.as_str(),
        )
        .await;
    }

    let cron_file = shellexpand::tilde("~/.npcsh/loops.yaml").to_string();
    let cron_registry = Arc::new(Mutex::new(CronRegistry::with_file(cron_file)));
    cron_registry.lock().unwrap().load_from_jinxes(&team_dir);
    let (cron_tx, mut cron_rx) = tokio::sync::mpsc::unbounded_channel();
    crate::cron::spawn_cron_ticker(cron_registry.clone(), cron_tx);

    print_welcome(&kernel, current_pid);

    let current_version = env!("NPCSH_VERSION").to_string();
    let http_client_for_update = http_client.clone();
    tokio::spawn(async move {
        if let Some(info) =
            version_check::check_version(&http_client_for_update, &current_version).await
        {
            eprintln!("{}", version_check::format_update_notice(&info));
        }
    });

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

    let mut mode = Mode::Agent;
    let mut _turn_count: u64 = 0;
    let mut session_input_tokens: u64 = 0;
    let mut session_output_tokens: u64 = 0;
    let mut session_cost: f64 = 0.0;
    let session_start = std::time::Instant::now();
    let mut input_queue: VecDeque<String> = VecDeque::new();
    let mut memory_scheduler = MemoryScheduler::from_env();

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

        let prompt = format!(
            "{CYAN}{BOLD}{npc_name}{RESET} {DIM}[{mode}|{model}]{RESET} {DIM}{cwd}{RESET}{usage_hint} {PURPLE}>{RESET} "
        );

        while let Ok(job) = cron_rx.try_recv() {
            let _ = execute_cron_job_and_capture(
                &mut kernel,
                current_pid,
                &job,
                &http_client,
                &server_url,
                &mut session_input_tokens,
                &mut session_output_tokens,
                &mut session_cost,
            )
            .await;
        }

        let input = if let Some(queued) = input_queue.pop_front() {
            println!("{}{}", prompt, queued);
            queued
        } else {
            match readline_raw(
                &prompt,
                &mut history,
                mode.clone(),
                &mut history_index,
                &helper,
                &mut kernel,
                current_pid,
            ) {
                Ok(ReadlineResult::Input(line)) => {
                    history.push(line.clone());
                    history_index = None;
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&history_path)
                        .and_then(|mut f| writeln!(f, "{}", line));
                    line
                }
                Ok(ReadlineResult::Cancel) => continue,
                Ok(ReadlineResult::Reattach) => {
                    let _ = run_reattach(
                        &mut kernel,
                        current_pid,
                        std::env::current_dir()
                            .ok()
                            .as_ref()
                            .map(|p| p.to_str().unwrap_or(".")),
                    );
                    continue;
                }
                Ok(ReadlineResult::Eof) => {
                    println!();
                    break;
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    break;
                }
            }
        };

        let (input, _pasted_content) = handle_paste_input(&input);
        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }

        let cmd_token = input.split_whitespace().next().unwrap_or("");

        let maybe_core = CORE_COMMANDS
            .iter()
            .find(|c| c.name == cmd_token)
            .map(|c| c.cmd);
        if let Some(cmd) = maybe_core {
            let rest = input.strip_prefix(cmd_token).unwrap_or("").trim();
            match dispatch_core_command(
                cmd,
                rest,
                &mut kernel,
                &mut current_pid,
                &mut mode,
                &cron_registry,
            )
            .await
            {
                CoreDispatch::Break => break,
                CoreDispatch::Handled => continue,
                CoreDispatch::NotHandled => {}
            }
        }

        if input == "/set" || input == "set" {
            eprintln!("Usage: set key=value");
            eprintln!("  model=<model-name>  provider=<provider-name>  mode=chat");
            continue;
        }
        if input.starts_with("/set ") || input.starts_with("set ") {
            let rest = input
                .strip_prefix("/set ")
                .or(input.strip_prefix("set "))
                .unwrap()
                .trim();
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
                    match spawn_npc_from_registered_teams(target, &mut kernel, current_pid).await {
                        Ok(new_pid) if new_pid != 0 => {
                            current_pid = new_pid;
                            eprintln!("{GREEN}Switched to @{target} (pid:{current_pid}){RESET}");
                        }
                        _ => {
                            eprintln!("{RED}NPC '{target}' not found.{RESET} Available:");
                            for p in kernel.ps() {
                                eprintln!("  {CYAN}@{}{RESET}", p.npc.name);
                            }
                        }
                    }
                }
            }
            continue;
        }

        if input.starts_with('/') && input.len() > 1 {
            let name = &input[1..];
            if kernel.find_by_name(name).is_some() {
                if let Err(e) = tui::run_agent_dashboard_tui(&mut kernel, name, &cron_registry) {
                    eprintln!("{RED}Error: {e}{RESET}");
                }
                continue;
            }
            match spawn_npc_from_registered_teams(name, &mut kernel, current_pid).await {
                Ok(new_pid) if new_pid != 0 => {
                    current_pid = new_pid;
                    eprintln!("{GREEN}Switched to @{name} (pid:{current_pid}){RESET}");
                    continue;
                }
                _ => {}
            }
        }

        if input.starts_with('/') {
            let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
            let cmd_name = parts[0];
            eprintln!("{RED}Unknown command: /{cmd_name}{RESET}");
            continue;
        }
        if !input.is_empty() {
            let first = input.split_whitespace().next().unwrap_or("");
            if CORE_COMMANDS.iter().any(|c| c.name == first) {
                // Core commands are dispatched above; reaching here means the
                // command matched CORE_COMMANDS but was not handled. Continue
                // so the user doesn't fall through to bash/LLM with a typo.
                continue;
            }
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
            let npc_name = p
                .map(|p| p.npc.name.clone())
                .unwrap_or_else(|| "npcsh".to_string());
            let team_name = kernel
                .team
                .source_dir
                .as_deref()
                .and_then(|d| std::path::Path::new(d).file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("npcsh")
                .to_string();
            let model = p
                .map(|p| p.npc.resolved_model())
                .unwrap_or_else(|| "qwen3.5:2b".to_string());
            let provider = p
                .map(|p| p.npc.resolved_provider())
                .unwrap_or_else(|| "ollama".to_string());
            let conv_id = p.map(|p| p.conversation_id.clone()).unwrap_or_default();
            (npc_name, team_name, model, provider, conv_id)
        };
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());

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
                        let (result, queued) = run_interactive_stream_turn(
                            &mut kernel,
                            current_pid,
                            &input,
                            mode.clone(),
                            &http_client,
                            &server_url,
                            &mut memory_scheduler,
                        )
                        .await;
                        input_queue.extend(queued);
                        Some(result)
                    }
                }
                Mode::Chat | Mode::Cmd => {
                    if matches!(mode, Mode::Cmd) && run_bash(&input).await {
                        None
                    } else {
                        let (result, queued) = run_interactive_stream_turn(
                            &mut kernel,
                            current_pid,
                            &input,
                            mode.clone(),
                            &http_client,
                            &server_url,
                            &mut memory_scheduler,
                        )
                        .await;
                        input_queue.extend(queued);
                        Some(result)
                    }
                }
            }
        };

        if let Some(result) = exec_result {
            match result {
                Ok(output) => {
                    let streamed = kernel
                        .get_process(current_pid)
                        .map(|p| p.last_streamed)
                        .unwrap_or(false);
                    if !streamed && !output.trim().is_empty() {
                        println!("\n{}", render_block(output.trim()));
                    }

                    let p = kernel.get_process(current_pid);
                    let (in_tok, out_tok, cost) = p
                        .map(|p| {
                            (
                                p.usage.total_input_tokens,
                                p.usage.total_output_tokens,
                                p.usage.total_cost_usd,
                            )
                        })
                        .unwrap_or((0, 0, 0.0));

                    let _ = kernel.history.save_conversation_message(
                        &conv_id,
                        "user",
                        &input,
                        &cwd,
                        Some(&model_str),
                        Some(&provider_str),
                        Some(&npc_name_str),
                        Some(&team_name_str),
                        None,
                        None,
                        None,
                        Some(in_tok),
                        None,
                        None,
                    );

                    let _ = kernel.history.save_conversation_message(
                        &conv_id,
                        "assistant",
                        &output,
                        &cwd,
                        Some(&model_str),
                        Some(&provider_str),
                        Some(&npc_name_str),
                        Some(&team_name_str),
                        None,
                        None,
                        None,
                        Some(in_tok),
                        Some(out_tok),
                        Some(cost),
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

async fn run_stream_turn_with_interrupt(
    kernel: &mut Kernel,
    current_pid: u32,
    input: &str,
    mode: Mode,
    client: &reqwest::Client,
    server_url: &str,
    save_history: bool,
    interrupt: Option<tokio::sync::mpsc::UnboundedReceiver<()>>,
    memory_scheduler: Option<&mut MemoryScheduler>,
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
        let process = kernel.get_process(current_pid).ok_or_else(|| {
            npcrs::NpcError::Other(format!("No process with pid {}", current_pid))
        })?;
        let model = process.npc.resolved_model();
        let provider = process.npc.resolved_provider();
        let base_system = process.npc.system_prompt(kernel.team.context.as_deref());
        let has_memory_jinx = kernel.jinxes.contains_key("memory");
        let has_knowledge_jinx = kernel.jinxes.contains_key("knowledge");
        let system = if memory_context_enabled() && (has_memory_jinx || has_knowledge_jinx) {
            if let Some(team_dir) = kernel.team.source_dir.as_deref() {
                let stores = discover_knowledge_stores(team_dir);
                let appendix = format_memory_context(&stores, has_memory_jinx, has_knowledge_jinx);
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
    // macOS /tmp is a symlink to /private/tmp. Task instructions reference /tmp
    // explicitly and warn against /private/tmp, so keep the model context consistent.
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

    if CLI_PROVIDERS.contains(&provider.as_str()) {
        let full_input = format!("{}\n\n{}", input, context_info);
        let session_id = cli_sessions().lock().unwrap().get(&current_pid).cloned();
        let cli_result = run_cli_provider(
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
            return Err(npcrs::NpcError::Other(format!(
                "CLI provider '{}' failed to produce a response; is the binary installed?",
                provider
            )));
        }
    }

    // Persist the user turn before we block on the LLM stream. If the parent
    // kills us during a long generation (benchmark timeout), the user message
    // is already in npcsh_history.db so the transcript is not lost.
    if save_history {
        save_conversation_message(kernel, current_pid, &Message::user(input), &cwd).await;
    }

    let request = stream_client::StreamRequest {
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

    let response = stream_client::call_stream_with_interrupt(
        client,
        server_url,
        &request,
        Some(&ask_permission),
        interrupt,
    )
    .await
    .map_err(|e| npcrs::NpcError::Other(e))?;
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

    // Save assistant and tool messages as they arrive so a timeout/kill does
    // not lose the partial conversation. The user message was already persisted
    // before the LLM stream started.
    if save_history {
        let assistant_for_save = response.message.clone();
        save_conversation_message(kernel, current_pid, &assistant_for_save, &cwd).await;
        for tr in &tool_results {
            save_conversation_message(kernel, current_pid, tr, &cwd).await;
        }

        // Update token/cost metadata on the assistant row after the stream is
        // complete and usage is known.
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

fn is_server_error(e: &npcrs::NpcError) -> bool {
    let msg = e.to_string();
    msg.contains("HTTP stream returned 5")
        || msg.contains("Internal Server Error")
        || msg.contains("HTTP stream request failed")
}

async fn run_stream_turn(
    kernel: &mut Kernel,
    current_pid: u32,
    input: &str,
    mode: Mode,
    client: &reqwest::Client,
    server_url: &str,
    save_history: bool,
    memory_scheduler: Option<&mut MemoryScheduler>,
) -> Result<String> {
    const MAX_ATTEMPTS: usize = 4;
    let mut last_error: Option<npcrs::NpcError> = None;

    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            eprintln!(
                "{YELLOW}npcpy server error; restarting server and retrying turn (attempt {}/{})...{RESET}",
                attempt,
                MAX_ATTEMPTS - 1
            );
            if let Err(e) = restart_server(client, server_url).await {
                last_error = Some(npcrs::NpcError::Other(format!(
                    "failed to restart server: {e}"
                )));
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
                last_error = Some(npcrs::NpcError::Other(
                    "npcpy server smoke test failed after restart".to_string(),
                ));
                continue;
            }
            // Reset turn state so the retry doesn't inherit a broken stream state.
            if let Some(process) = kernel.get_process_mut(current_pid) {
                process.new_turn();
            }
        }

        match run_stream_turn_with_interrupt(
            kernel,
            current_pid,
            input,
            mode.clone(),
            client,
            server_url,
            save_history,
            None,
            None,
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
        npcrs::NpcError::Other("stream turn failed after server restart retries".to_string())
    }))
}

async fn run_interactive_stream_turn_once(
    kernel: &mut Kernel,
    current_pid: u32,
    input: &str,
    mode: Mode,
    client: &reqwest::Client,
    server_url: &str,
    memory_scheduler: Option<&mut MemoryScheduler>,
) -> (Result<String>, Vec<String>) {
    let _raw_guard = {
        let _ = crossterm::terminal::enable_raw_mode();
        RawModeRestore
    };

    let (interrupt_tx, interrupt_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let (queue_tx, mut queue_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let running = Arc::new(AtomicBool::new(true));

    let listener = interrupt::spawn_listener(interrupt_tx, queue_tx, running.clone());

    let result = run_stream_turn_with_interrupt(
        kernel,
        current_pid,
        input,
        mode,
        client,
        server_url,
        true,
        Some(interrupt_rx),
        memory_scheduler,
    )
    .await;

    running.store(false, Ordering::Relaxed);
    let _ = listener.await;

    let mut queued = Vec::new();
    while let Ok(line) = queue_rx.try_recv() {
        queued.push(line);
    }

    (result, queued)
}

struct RawModeRestore;
impl Drop for RawModeRestore {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

async fn run_interactive_stream_turn(
    kernel: &mut Kernel,
    current_pid: u32,
    input: &str,
    mode: Mode,
    client: &reqwest::Client,
    server_url: &str,
    memory_scheduler: &mut Option<MemoryScheduler>,
) -> (Result<String>, Vec<String>) {
    const MAX_ATTEMPTS: usize = 4;
    let mut last_error: Option<npcrs::NpcError> = None;
    let mut all_queued = Vec::new();

    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            if attempt == 1 {
                eprintln!(
                    "{RED}npcpy server error: {}{RESET}",
                    last_error
                        .as_ref()
                        .map(|e| e.to_string())
                        .unwrap_or_default()
                );
                eprintln!("{YELLOW}Restart the server and retry? [Y/n]{RESET}");
                let mut line = String::new();
                let _ = std::io::stdin().read_line(&mut line);
                if line.trim().eq_ignore_ascii_case("n") || line.trim().eq_ignore_ascii_case("no") {
                    return (
                        Err(last_error.unwrap_or_else(|| {
                            npcrs::NpcError::Other("stream turn aborted".to_string())
                        })),
                        all_queued,
                    );
                }
            }

            eprintln!(
                "{YELLOW}Restarting npcpy server (interactive attempt {}/{})...{RESET}",
                attempt,
                MAX_ATTEMPTS - 1
            );
            if let Err(e) = restart_server(client, server_url).await {
                last_error = Some(npcrs::NpcError::Other(format!(
                    "failed to restart server: {e}"
                )));
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
                last_error = Some(npcrs::NpcError::Other(
                    "npcpy server smoke test failed after restart".to_string(),
                ));
                continue;
            }
            if let Some(process) = kernel.get_process_mut(current_pid) {
                process.new_turn();
            }
        }

        let (result, queued) = run_interactive_stream_turn_once(
            kernel,
            current_pid,
            input,
            mode.clone(),
            client,
            server_url,
            memory_scheduler.as_mut(),
        )
        .await;
        all_queued.extend(queued);

        match result {
            Ok(output) => return (Ok(output), all_queued),
            Err(e) => {
                if is_server_error(&e) {
                    last_error = Some(e);
                    continue;
                }
                return (Err(e), all_queued);
            }
        }
    }

    (
        Err(last_error.unwrap_or_else(|| {
            npcrs::NpcError::Other("stream turn failed after server restart retries".to_string())
        })),
        all_queued,
    )
}

async fn save_conversation_message(
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

async fn save_conversation_turn(
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

fn ask_permission(prompt: &str) -> String {
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
        event::{self, Event, KeyCode, KeyEventKind},
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

    let (cols, _rows) = terminal::size().unwrap_or((80, 24));
    let width = cols as usize;

    let _guard = match PromptGuard::new() {
        Ok(g) => g,
        Err(_) => return "No".to_string(),
    };

    let mut stdout = io::stdout();
    let _ = stdout.execute(Clear(ClearType::All));

    let mut selected: usize = 0;

    let draw = |sel: usize, out: &mut io::Stdout| -> io::Result<()> {
        out.execute(MoveTo(0, 0))?;
        out.execute(Clear(ClearType::All))?;

        let prompt_lines = wrap_lines(prompt, width);
        let total_rows = prompt_lines
            .len()
            .saturating_add(1)
            .saturating_add(options.len())
            .saturating_add(1);
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24) as usize;
        let start_row = rows.saturating_sub(total_rows).min(rows.saturating_sub(1)) as u16;

        let mut row: u16 = start_row;
        for line in prompt_lines {
            out.execute(MoveTo(0, row))?;
            out.execute(Print(format!("  {}", line)))?;
            row = row.saturating_add(1);
        }

        row = row.saturating_add(1);
        for (i, opt) in options.iter().enumerate() {
            out.execute(MoveTo(0, row))?;
            if i == sel {
                out.execute(Print(format!("  \x1b[7m {}\x1b[0m", opt)))?;
            } else {
                out.execute(Print(format!("  {}", opt)))?;
            }
            row = row.saturating_add(1);
        }

        out.execute(MoveTo(0, row.saturating_add(1)))?;
        out.execute(Print("  [↑/↓ or j/k] select, Enter confirm, q cancel"))?;
        out.flush()
    };

    let decision: String = loop {
        if draw(selected, &mut stdout).is_err() {
            break "No".to_string();
        }
        match event::read() {
            Ok(Event::Key(key)) if key.kind != KeyEventKind::Release => match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    selected = (selected + 1) % options.len();
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    selected = if selected == 0 {
                        options.len() - 1
                    } else {
                        selected - 1
                    };
                }
                KeyCode::Enter => break options[selected].to_string(),
                KeyCode::Char('q') | KeyCode::Esc => break "No".to_string(),
                _ => {}
            },
            _ => {}
        }
    };

    while event::poll(std::time::Duration::from_millis(0)).unwrap_or(false) {
        let _ = event::read();
    }

    decision
}

fn handle_set_command(rest: &str, kernel: &mut Kernel, pid: u32, mode: &mut Mode) {
    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    if parts.len() != 2 {
        eprintln!("Usage: set key=value");
        eprintln!("  model=<model-name>  provider=<provider-name>  mode=chat");
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

enum CoreDispatch {
    Break,
    Handled,
    NotHandled,
}

async fn dispatch_core_command(
    cmd: CoreCmd,
    rest: &str,
    kernel: &mut Kernel,
    current_pid: &mut u32,
    mode: &mut Mode,
    cron_registry: &Arc<Mutex<CronRegistry>>,
) -> CoreDispatch {
    match cmd {
        CoreCmd::Exit => CoreDispatch::Break,

        CoreCmd::Help => {
            print_core_help();
            CoreDispatch::Handled
        }

        CoreCmd::Agent => {
            *mode = Mode::Agent;
            eprintln!("{GREEN}Switched to agent mode{RESET}");
            CoreDispatch::Handled
        }
        CoreCmd::Chat => {
            *mode = Mode::Chat;
            eprintln!("{GREEN}Switched to chat mode{RESET}");
            CoreDispatch::Handled
        }
        CoreCmd::CmdMode => {
            *mode = Mode::Cmd;
            eprintln!("{GREEN}Switched to cmd mode{RESET}");
            CoreDispatch::Handled
        }

        CoreCmd::Kill => {
            if *current_pid == 0 {
                eprintln!("{RED}Cannot kill init (pid 0){RESET}");
            } else {
                let name = kernel.get_process(*current_pid).map(|p| p.npc.name.clone());
                kernel.kill(*current_pid, 0).ok();
                *current_pid = 0;
                eprintln!(
                    "{YELLOW}Killed @{} — switched to init{RESET}",
                    name.unwrap_or_default()
                );
            }
            CoreDispatch::Handled
        }

        CoreCmd::Clear => {
            if let Some(p) = kernel.get_process_mut(*current_pid) {
                p.messages.clear();
                eprintln!("{GREEN}Conversation cleared{RESET}");
            }
            CoreDispatch::Handled
        }
        CoreCmd::Flush => {
            let n: usize = rest.trim().parse().unwrap_or(1);
            if let Some(p) = kernel.get_process_mut(*current_pid) {
                let original = p.messages.len();
                let mut keep_system = false;
                let mut working: Vec<Message> = Vec::new();
                if let Some(first) = p.messages.first() {
                    if first.role == "system" {
                        keep_system = true;
                        working = p.messages.iter().skip(1).cloned().collect();
                    } else {
                        working = p.messages.clone();
                    }
                }
                let remove = n.min(working.len());
                if remove > 0 {
                    working.truncate(working.len() - remove);
                }
                p.messages.clear();
                if keep_system {
                    if let Some(first) = p.messages.first() {
                        p.messages.push(first.clone());
                    }
                }
                p.messages.extend(working);
                eprintln!(
                    "{GREEN}Flushed {remove} message(s). Context is now {} messages.{RESET}",
                    p.messages.len()
                );
            }
            CoreDispatch::Handled
        }

        CoreCmd::History => {
            if let Some(p) = kernel.get_process(*current_pid) {
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
            CoreDispatch::Handled
        }

        CoreCmd::Reattach => {
            let show_all = rest == "all";
            let filter = if show_all {
                None
            } else if rest.is_empty() {
                Some(
                    std::env::current_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| ".".to_string()),
                )
            } else {
                Some(shellexpand::tilde(rest).to_string())
            };
            if let Err(e) = run_reattach(kernel, *current_pid, filter.as_deref()) {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }

        CoreCmd::Config => {
            if let Err(e) = tui::run_config_tui() {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }
        CoreCmd::Ctx => {
            if let Err(e) = tui::run_ctx_tui(kernel, current_pid) {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }
        CoreCmd::Model => {
            if let Err(e) = tui::run_model_tui() {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }
        CoreCmd::Setup => {
            if let Err(e) = tui::run_setup_tui() {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }
        CoreCmd::Team => {
            match tui::run_team_tui(kernel) {
                Ok(Some(new_team_dir)) => {
                    let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
                    match Kernel::boot(&new_team_dir, &db_path) {
                        Ok(new_kernel) => {
                            *kernel = new_kernel;
                            *current_pid = 0;
                            if let Some(lead) = kernel.team.forenpc.as_deref() {
                                if let Some(proc) = kernel.find_by_name(lead) {
                                    *current_pid = proc.pid;
                                }
                            }
                            println!("{GREEN}Switched to team {new_team_dir}{RESET}");
                        }
                        Err(e) => {
                            eprintln!("{RED}Failed to switch team: {e}{RESET}");
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
            }
            CoreDispatch::Handled
        }
        CoreCmd::Commit => {
            if let Err(e) = tui::run_commit_tui() {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }
        CoreCmd::Gitt => {
            let path = if rest.is_empty() { None } else { Some(rest) };
            if let Err(e) = tui::run_gitt_tui(path) {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }

        CoreCmd::ProposeJinxes => {
            let Some(process) = kernel.get_process(*current_pid) else {
                eprintln!("{RED}No active NPC.{RESET}");
                return CoreDispatch::Handled;
            };
            let npc_name = process.npc.name.clone();

            // Gather existing jinxes
            let mut jinx_list = Vec::new();
            for (name, jinx) in &kernel.jinxes {
                let desc = if jinx.description.is_empty() {
                    "(no description)"
                } else {
                    jinx.description.as_str()
                };
                jinx_list.push(format!("- {name}: {desc}"));
            }
            jinx_list.sort();

            // Gather recent conversation turns
            let mut recent = String::new();
            if let Some(p) = kernel.get_process(*current_pid) {
                let mut count = 0;
                for m in p.messages.iter().rev() {
                    if count >= 10 {
                        break;
                    }
                    if m.role == "user" || m.role == "assistant" {
                        let content = m.content.as_deref().unwrap_or("").trim();
                        if !content.is_empty() {
                            recent.push_str(&format!(
                                "{}: {}\n",
                                m.role,
                                content.chars().take(500).collect::<String>()
                            ));
                            count += 1;
                        }
                    }
                }
            }

            // Team context
            let team_ctx = kernel.team.context.as_deref().unwrap_or("(none)");

            let prompt = format!(
                "You are helping improve an NPC team's tooling.\n\n\
                Team context:\n{team_ctx}\n\n\
                Existing jinxes/skills:\n{}\n\n\
                Recent conversation:\n{recent}\n\n\
                Based on the team context and recent work, propose 1-3 new jinxes/skills or concrete edits to existing ones. \
                For each proposal give: name, purpose, inputs, and a short draft of the jinx steps. \
                Write in plain text suitable for turning into a .jinx file later.",
                jinx_list.join("\n")
            );

            eprintln!("{DIM}Asking @{npc_name} for jinx proposals...{RESET}");
            match kernel.exec_chat(*current_pid, &prompt).await {
                Ok(output) => {
                    // Decide destination: local team dir if it has a jinxes folder, else global
                    let base_dir = kernel
                        .team
                        .source_dir
                        .as_deref()
                        .and_then(|d| {
                            let p = std::path::Path::new(d).join("jinxes");
                            if p.is_dir() { Some(p) } else { None }
                        })
                        .unwrap_or_else(|| {
                            shellexpand::tilde("~/.npcsh/npc_team/usr/jinxes")
                                .into_owned()
                                .into()
                        });
                    let _ = std::fs::create_dir_all(&base_dir);
                    let stamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let out_path = base_dir.join(format!("proposed-{stamp}.md"));
                    match std::fs::write(&out_path, &output) {
                        Ok(_) => {
                            eprintln!("{GREEN}Wrote proposals to {}{RESET}", out_path.display());
                        }
                        Err(e) => {
                            eprintln!("{RED}Failed to write proposals: {e}{RESET}");
                            println!("{output}");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("{RED}Proposal request failed: {e}{RESET}");
                }
            }
            CoreDispatch::Handled
        }

        CoreCmd::Memories => {
            if let Err(e) = tui::run_memories_tui() {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }
        CoreCmd::Mems => {
            if let Err(e) = tui::run_memories_tui() {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }
        CoreCmd::Knowledge => {
            if let Err(e) = tui::run_knowledge_tui() {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }

        CoreCmd::Jinxes => {
            if let Err(e) = tui::run_jinxes_tui(kernel) {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }

        CoreCmd::Ps => {
            for p in kernel.ps() {
                let state_color = match p.state {
                    npcrs::process::ProcessState::Running => GREEN,
                    npcrs::process::ProcessState::Blocked => YELLOW,
                    npcrs::process::ProcessState::Dead => RED,
                    _ => DIM,
                };
                println!(
                    "  {CYAN}@{:<12}{RESET} pid:{:<3} {state_color}{:?}{RESET}  tokens:{}/{} cost:${:.4} turns:{}",
                    p.npc.name,
                    p.pid,
                    p.state,
                    p.usage.total_input_tokens,
                    p.usage.total_output_tokens,
                    p.usage.total_cost_usd,
                    p.usage.total_turns,
                );
            }
            CoreDispatch::Handled
        }

        CoreCmd::Stats => {
            let s = kernel.stats();
            println!(
                "{BOLD}Kernel Stats{RESET}\n  uptime: {}s\n  processes: {} (run:{} blk:{} dead:{})\n  tokens: {} (in+out)\n  cost: ${:.4}\n  jinxes: {}",
                s.uptime_secs,
                s.total_processes,
                s.running,
                s.blocked,
                s.dead,
                s.total_tokens,
                s.total_cost_usd,
                s.jinx_count,
            );
            CoreDispatch::Handled
        }

        CoreCmd::Cron => {
            handle_cron_command(rest, kernel, cron_registry, *current_pid).await;
            CoreDispatch::Handled
        }
        CoreCmd::Loop => {
            handle_loop_command(rest, kernel, cron_registry, *current_pid).await;
            CoreDispatch::Handled
        }
        CoreCmd::Loops => {
            handle_jobs_command(rest, kernel, cron_registry, *current_pid).await;
            CoreDispatch::Handled
        }
        CoreCmd::LoopDemo => {
            let npc_name = kernel
                .get_process(*current_pid)
                .map(|p| p.npc.name.clone())
                .unwrap_or_default();
            if npc_name.is_empty() {
                eprintln!("{RED}No active NPC to attach the demo loop to.{RESET}");
            } else {
                let loop_rest = format!("{npc_name} 10s heartbeat demo: print the current time");
                handle_loop_command(&loop_rest, kernel, cron_registry, *current_pid).await;
            }
            CoreDispatch::Handled
        }
        CoreCmd::LoopRm => {
            if let Ok(id) = rest.parse::<u32>() {
                if cron_registry.lock().unwrap().remove(id) {
                    eprintln!("{GREEN}Removed loop {id}{RESET}");
                } else {
                    eprintln!("{RED}No loop with id {id}{RESET}");
                }
            } else {
                eprintln!("Usage: /looprm <id>");
            }
            CoreDispatch::Handled
        }
        CoreCmd::LoopOff => {
            if let Ok(id) = rest.parse::<u32>() {
                if cron_registry.lock().unwrap().enable(id, false) {
                    eprintln!("{GREEN}Disabled loop {id}{RESET}");
                } else {
                    eprintln!("{RED}No loop with id {id}{RESET}");
                }
            } else {
                eprintln!("Usage: /loopoff <id>");
            }
            CoreDispatch::Handled
        }
        CoreCmd::LoopOn => {
            if let Ok(id) = rest.parse::<u32>() {
                if cron_registry.lock().unwrap().enable(id, true) {
                    eprintln!("{GREEN}Enabled loop {id}{RESET}");
                } else {
                    eprintln!("{RED}No loop with id {id}{RESET}");
                }
            } else {
                eprintln!("Usage: /loopon <id>");
            }
            CoreDispatch::Handled
        }

        CoreCmd::Tutorial => {
            if let Err(e) = tutorial::run_tutorial_tui(cron_registry, kernel) {
                eprintln!("{RED}Error: {e}{RESET}");
            }
            CoreDispatch::Handled
        }

        CoreCmd::Doctor => {
            run_doctor_command(rest).await;
            CoreDispatch::Handled
        }
        CoreCmd::Init => {
            run_init_command(rest).await;
            CoreDispatch::Handled
        }
        CoreCmd::Nsync => {
            run_nsync_command(rest).await;
            CoreDispatch::Handled
        }
        CoreCmd::Reload => {
            run_reload_command(kernel, current_pid).await;
            CoreDispatch::Handled
        }
        CoreCmd::Shh => {
            run_shh_command();
            CoreDispatch::Handled
        }
        CoreCmd::Update => {
            run_update_command().await;
            CoreDispatch::Handled
        }
        CoreCmd::Usage => {
            run_usage_command(kernel, *current_pid);
            CoreDispatch::Handled
        }
        CoreCmd::Verbose => {
            run_verbose_command();
            CoreDispatch::Handled
        }

        CoreCmd::Jinx(jinx_name) => {
            run_jinx_command(kernel, *current_pid, jinx_name, rest).await;
            CoreDispatch::Handled
        }

        CoreCmd::Set => CoreDispatch::NotHandled,
    }
}

async fn run_doctor_command(_rest: &str) {
    let home = shellexpand::tilde("~").to_string();
    let npcsh_dir = std::path::Path::new(&home).join(".npcsh");
    let bin_dir = npcsh_dir.join("bin");
    let mut fixes: Vec<String> = Vec::new();

    if !npcsh_dir.exists() {
        if let Err(e) = tokio::fs::create_dir_all(&npcsh_dir).await {
            eprintln!("{RED}Failed to create ~/.npcsh: {e}{RESET}");
            return;
        }
        fixes.push("Created ~/.npcsh".to_string());
    }

    if !bin_dir.exists() {
        if let Err(e) = tokio::fs::create_dir_all(&bin_dir).await {
            eprintln!("{RED}Failed to create ~/.npcsh/bin: {e}{RESET}");
            return;
        }
        fixes.push("Created ~/.npcsh/bin".to_string());
    }

    let db_path = std::path::Path::new(&home).join("npcsh_history.db");
    if !db_path.exists() {
        fixes.push("No history database yet; it will be created on first use.".to_string());
    }

    let npc_team_dir = npcsh_dir.join("npc_team");
    if !npc_team_dir.exists() {
        if let Err(e) = tokio::fs::create_dir_all(&npc_team_dir).await {
            eprintln!("{RED}Failed to create ~/.npcsh/npc_team: {e}{RESET}");
            return;
        }
        fixes.push("Created ~/.npcsh/npc_team".to_string());
    }

    if fixes.is_empty() {
        println!("{GREEN}No common issues found. npcsh looks healthy.{RESET}");
    } else {
        println!("{BOLD}Doctor fixes:{RESET}");
        for f in fixes {
            println!("  {GREEN}{f}{RESET}");
        }
    }
}

async fn run_init_command(rest: &str) {
    let home = shellexpand::tilde("~").to_string();
    let npcsh_dir = std::path::Path::new(&home).join(".npcsh");
    let bin_dir = npcsh_dir.join("bin");
    let npc_team_dir = npcsh_dir.join("npc_team");

    let force = rest.trim() == "--force" || rest.trim() == "-f";

    if npcsh_dir.exists() && !force {
        println!("{YELLOW}npcsh is already initialized.{RESET}");
        println!("Run {CYAN}/init --force{RESET} to reset directories.");
        return;
    }

    for dir in [&bin_dir, &npc_team_dir] {
        if let Err(e) = tokio::fs::create_dir_all(dir).await {
            eprintln!("{RED}Failed to create {}: {e}{RESET}", dir.display());
            return;
        }
    }

    let rc_path = std::path::Path::new(&home).join(".npcshrc");
    if !rc_path.exists() || force {
        let rc = "export NPCSH_CHAT_MODEL=qwen3.5:2b\nexport NPCSH_CHAT_PROVIDER=ollama\nexport NPCSH_DEFAULT_MODE=agent\nexport NPCSH_EMBEDDING_MODEL=nomic-embed-text\nexport NPCSH_EMBEDDING_PROVIDER=ollama\n";
        if let Err(e) = tokio::fs::write(&rc_path, rc).await {
            eprintln!("{RED}Failed to write ~/.npcshrc: {e}{RESET}");
            return;
        }
        println!("{GREEN}Wrote ~/.npcshrc{RESET}");
    }

    println!("{GREEN}npcsh initialized.{RESET}");
    println!("  bin:      {}", bin_dir.display());
    println!("  npc_team: {}", npc_team_dir.display());
    println!("  rc:       {}", rc_path.display());
}

async fn run_nsync_command(_rest: &str) {
    let home = real_user_home().unwrap_or_else(|| shellexpand::tilde("~").to_string());
    let home_path = std::path::PathBuf::from(home);
    let user_team = home_path.join(".npcsh").join("npc_team");
    let tag = env!("CARGO_PKG_VERSION");
    let repo = "NPC-Worldwide/npcsh";
    let tarball_url = format!(
        "https://github.com/{}/archive/refs/tags/v{}.tar.gz",
        repo, tag
    );

    println!("{BOLD}Syncing npcsh state...{RESET}");

    let tmp_dir = std::env::temp_dir().join(format!("npcsh-sync-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    if let Err(e) = fs::create_dir_all(&tmp_dir) {
        eprintln!("  {RED}failed to create temp dir: {e}{RESET}");
        return;
    }

    let tarball_path = tmp_dir.join(format!("npcsh-v{}.tar.gz", tag));
    match reqwest::get(&tarball_url).await {
        Ok(resp) if resp.status().is_success() => match resp.bytes().await {
            Ok(bytes) => {
                if let Err(e) = fs::write(&tarball_path, &bytes) {
                    eprintln!("  {RED}failed to write tarball: {e}{RESET}");
                    return;
                }
            }
            Err(e) => {
                eprintln!("  {RED}failed to download tarball: {e}{RESET}");
                return;
            }
        },
        Ok(resp) => {
            eprintln!(
                "  {RED}failed to download tarball: {}{RESET}",
                resp.status()
            );
            return;
        }
        Err(e) => {
            eprintln!("  {RED}failed to download tarball: {e}{RESET}");
            return;
        }
    }

    let source = tmp_dir
        .join(format!("npcsh-{}", tag))
        .join("npcsh")
        .join("npc_team");
    let output = std::process::Command::new("tar")
        .arg("-xzf")
        .arg(&tarball_path)
        .arg("-C")
        .arg(&tmp_dir)
        .arg(format!("npcsh-{}/npcsh/npc_team", tag))
        .output();
    match output {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            eprintln!(
                "  {RED}failed to extract tarball: {}{RESET}",
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        Err(e) => {
            eprintln!("  {RED}failed to run tar: {e}{RESET}");
            return;
        }
    }

    match team_sync::sync_team(&source, &user_team) {
        Ok(_) => {
            println!("  synced base team from {}", source.display());
            let _ = set_resolved_team_dir(user_team.to_string_lossy().to_string());
        }
        Err(e) => eprintln!("  {RED}failed to sync base team: {e}{RESET}"),
    }

    let _ = fs::remove_dir_all(&tmp_dir);

    let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
    println!("  team dir: {}", user_team.display());
    println!("  db path:  {db_path}");
    println!("{GREEN}State synced.{RESET}");
}

async fn run_reload_command(kernel: &mut Kernel, current_pid: &mut u32) {
    let team_dir = find_team_dir();
    let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
    match Kernel::boot(&team_dir, &db_path) {
        Ok(new_kernel) => {
            *kernel = new_kernel;
            *current_pid = 0;
            println!("{GREEN}Reloaded team from {team_dir}{RESET}");
        }
        Err(e) => eprintln!("{RED}Failed to reload: {e}{RESET}"),
    }
}

fn run_shh_command() {
    static QUIET: AtomicBool = AtomicBool::new(false);
    let was = QUIET.load(Ordering::Relaxed);
    QUIET.store(!was, Ordering::Relaxed);
    if was {
        println!("{GREEN}Verbose mode on.{RESET}");
    } else {
        println!("{GREEN}Quiet mode on.{RESET}");
    }
}

fn run_verbose_command() {
    static VERBOSE: AtomicBool = AtomicBool::new(false);
    let was = VERBOSE.load(Ordering::Relaxed);
    VERBOSE.store(!was, Ordering::Relaxed);
    if was {
        println!("{GREEN}Verbose mode off.{RESET}");
    } else {
        println!("{GREEN}Verbose mode on.{RESET}");
    }
}

fn run_usage_command(kernel: &Kernel, current_pid: u32) {
    let mut inp = 0u64;
    let mut out = 0u64;
    let mut cost = 0.0f64;
    let mut turns = 0u64;

    if let Some(p) = kernel.get_process(current_pid) {
        inp = p.usage.total_input_tokens;
        out = p.usage.total_output_tokens;
        cost = p.usage.total_cost_usd;
        turns = p.usage.total_turns;
    }

    let total = inp + out;
    let fmt = |n: u64| {
        if n >= 1000 {
            format!("{:.1}k", n as f64 / 1000.0)
        } else {
            n.to_string()
        }
    };
    let cost_str = if cost == 0.0 {
        "free (local)".to_string()
    } else if cost < 0.01 {
        format!("${cost:.4}")
    } else {
        format!("${cost:.2}")
    };

    println!("{BOLD}Session Usage{RESET}");
    println!(
        "  Tokens: {} in / {} out ({} total)",
        fmt(inp),
        fmt(out),
        fmt(total)
    );
    println!("  Cost:   {cost_str}");
    println!("  Turns:  {turns}");
}

async fn run_update_command() {
    let client = reqwest::Client::new();
    let current = env!("NPCSH_VERSION").to_string();

    println!("{BOLD}Checking for updates...{RESET}");

    let info = match version_check::check_version(&client, &current).await {
        Some(i) => i,
        None => {
            println!("{GREEN}npcsh is up to date ({current}).{RESET}");
            return;
        }
    };

    println!(
        "{YELLOW}Update available: {current} → {}{RESET}",
        info.latest
    );

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{RED}Could not determine current executable path: {e}{RESET}");
            return;
        }
    };
    let exe_str = exe.display().to_string();

    if exe_str.starts_with("/opt/homebrew")
        || exe_str.starts_with("/usr/local")
        || exe_str.starts_with("/home/linuxbrew")
    {
        println!("Installed via Homebrew. Run:");
        println!("  {CYAN}brew upgrade npcsh{RESET}");
        return;
    }

    if exe_str.contains(".cargo") || exe_str.contains("cargo-install") {
        println!("Installed via cargo. Run:");
        println!("  {CYAN}cargo install npcsh --force{RESET}");
        return;
    }

    let install_dir = exe
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from(shellexpand::tilde("~/.npcsh/bin").as_ref()));

    println!(
        "Downloading latest release into {}{RESET}",
        install_dir.display()
    );
    println!("Run: {CYAN}curl -fsSL https://enpisi.com/install-npcsh.sh | sh{RESET}");
    println!("Or restart npcsh after running the install script.");
}

fn print_core_help() {
    println!(
        "{BOLD}npcsh-rs{RESET} — NPC OS Shell v{}\n",
        env!("NPCSH_VERSION")
    );
    println!("{BOLD}Modes:{RESET}");
    println!("  {CYAN}@npc{RESET}            Switch to NPC process (across all registered teams)");
    println!("  {CYAN}@npc command{RESET}    Delegate command to NPC");
    println!("  {CYAN}/agent{RESET}          Full agent mode (tools + bash + LLM)");
    println!("  {CYAN}/chat{RESET}           Chat-only mode (LLM, no tools)");
    println!("  {CYAN}/cmd{RESET}            Command mode (bash first, LLM fallback)");
    println!("  {CYAN}/kill{RESET}           Kill current process");
    println!();

    for category in COMMAND_CATEGORIES {
        let cmds: Vec<&CommandDef> = CORE_COMMANDS
            .iter()
            .filter(|c| c.category == *category && c.name.starts_with('/'))
            .collect();
        if cmds.is_empty() {
            continue;
        }
        println!("{BOLD}{}:{RESET}", category);
        for c in cmds {
            println!("  {CYAN}{:<14}{RESET} {}", c.name, c.description);
        }
        println!();
    }

    println!("{BOLD}Jinxes:{RESET}");
    println!("  Jinxes are invoked by name without a leading slash.");
    println!("  Use {CYAN}/jinxes{RESET} to browse them.");
    println!();
    println!("{BOLD}Shell:{RESET}");
    println!("  Any text is sent to the current NPC.");
    println!("  In {CYAN}/cmd{RESET} mode, input runs as bash first.");
    println!("  Tab completes @npcs, /commands, jinx names, and file paths.");
}

async fn handle_cron_command(
    rest: &str,
    kernel: &mut Kernel,
    registry: &Arc<Mutex<CronRegistry>>,
    current_pid: u32,
) {
    let parts: Vec<&str> = rest.split_whitespace().collect();
    let cmd = parts.first().copied().unwrap_or("list");
    match cmd {
        "list" | "" => {
            let reg = registry.lock().unwrap();
            let jobs = reg.list();
            if jobs.is_empty() {
                eprintln!(
                    "{DIM}No cron jobs. Use /loop <npc> <interval> <task> or /cron add ...{RESET}"
                );
            } else {
                eprintln!("{BOLD}Cron jobs:{RESET}");
                for j in jobs {
                    let status = if j.enabled { GREEN } else { RED };
                    let kind = if j.kind == crate::cron::CronJobKind::Jinx {
                        "jinx"
                    } else {
                        "chat"
                    };
                    let last = j.last_run.map(|_| "ran").unwrap_or("never");
                    eprintln!(
                        "  [{status}{:>3}{RESET}] @{:<12} every {:<5} [{:<4}] {} (last: {})",
                        j.id,
                        j.npc,
                        crate::cron::format_duration(j.interval_secs),
                        kind,
                        j.task,
                        last
                    );
                }
            }
        }
        "add" => {
            if parts.len() < 4 {
                eprintln!("Usage: /cron add <npc> <interval> <chat|jinx:...>");
                eprintln!("  /cron add sibiji 30s 'summarize recent commits'");
                eprintln!("  /cron add corca 5m jinx:shell 'git status'");
                return;
            }
            let npc = parts[1].to_string();
            let interval = parts[2];
            let task = parts[3..].join(" ");
            let (kind, task) = if let Some(t) = task.strip_prefix("jinx:") {
                (crate::cron::CronJobKind::Jinx, t.to_string())
            } else {
                (crate::cron::CronJobKind::Chat, task)
            };
            let secs = crate::cron::parse_duration(interval);
            let id = registry
                .lock()
                .unwrap()
                .add(npc.clone(), secs, task.clone(), kind);
            eprintln!("{GREEN}Added cron job {id}: @{npc} every {interval} -> {task}{RESET}");
        }
        "remove" | "rm" => {
            if let Some(id_str) = parts.get(1) {
                if let Ok(id) = id_str.parse::<u32>() {
                    if registry.lock().unwrap().remove(id) {
                        eprintln!("{GREEN}Removed cron job {id}{RESET}");
                    } else {
                        eprintln!("{RED}No cron job with id {id}{RESET}");
                    }
                } else {
                    eprintln!("{RED}Invalid id: {id_str}{RESET}");
                }
            } else {
                eprintln!("Usage: /cron remove <id>");
            }
        }
        "enable" | "on" => {
            if let Some(id_str) = parts.get(1) {
                if let Ok(id) = id_str.parse::<u32>() {
                    if registry.lock().unwrap().enable(id, true) {
                        eprintln!("{GREEN}Enabled cron job {id}{RESET}");
                    } else {
                        eprintln!("{RED}No cron job with id {id}{RESET}");
                    }
                }
            }
        }
        "disable" | "off" => {
            if let Some(id_str) = parts.get(1) {
                if let Ok(id) = id_str.parse::<u32>() {
                    if registry.lock().unwrap().enable(id, false) {
                        eprintln!("{GREEN}Disabled cron job {id}{RESET}");
                    } else {
                        eprintln!("{RED}No cron job with id {id}{RESET}");
                    }
                }
            }
        }
        "run" => {
            if let Some(id_str) = parts.get(1) {
                if let Ok(id) = id_str.parse::<u32>() {
                    let job = registry
                        .lock()
                        .unwrap()
                        .list()
                        .iter()
                        .find(|j| j.id == id)
                        .cloned();
                    if let Some(job) = job {
                        if let Some(out) = execute_cron_job_and_capture(
                            kernel,
                            current_pid,
                            &job,
                            &reqwest::Client::new(),
                            "http://127.0.0.1:5237",
                            &mut 0,
                            &mut 0,
                            &mut 0.0,
                        )
                        .await
                        {
                            println!("{}", out);
                        }
                    } else {
                        eprintln!("{RED}No cron job with id {id}{RESET}");
                    }
                }
            }
        }
        _ => eprintln!("{RED}Unknown /cron subcommand: {cmd}{RESET}"),
    }
}

async fn handle_jobs_command(
    target: &str,
    kernel: &mut Kernel,
    registry: &Arc<Mutex<CronRegistry>>,
    _current_pid: u32,
) {
    let reg = registry.lock().unwrap();
    let jobs = reg.list();
    let mut filtered: Vec<&crate::cron::CronJob> = jobs
        .iter()
        .filter(|j| {
            if target.is_empty() {
                true
            } else {
                j.npc == target || target == "all"
            }
        })
        .collect();
    if filtered.is_empty() {
        if target.is_empty() {
            eprintln!("{DIM}No loops. Use /loop <npc> <interval> <task>{RESET}");
        } else {
            eprintln!("{DIM}No loops for @{target}.{RESET}");
        }
        return;
    }
    filtered.sort_by(|a, b| a.npc.cmp(&b.npc).then(a.id.cmp(&b.id)));

    eprintln!(
        "{BOLD}Loops{RESET} {}",
        if target.is_empty() {
            "(all NPCs)".to_string()
        } else {
            format!("for @{target}")
        }
    );
    eprintln!("  {DIM}Use /looprm <id>, /loopoff <id>, /loopon <id>{RESET}");
    let mut last_npc = String::new();
    for j in filtered {
        if j.npc != last_npc {
            eprintln!("\n  {CYAN}@{}{RESET}", j.npc);
            last_npc = j.npc.clone();
        }
        let status = if j.enabled { GREEN } else { RED };
        let kind = if j.kind == crate::cron::CronJobKind::Jinx {
            "jinx"
        } else {
            "chat"
        };
        let last = j.last_run.map(|_| "ran").unwrap_or("never");
        let next_secs = j
            .next_run
            .duration_since(std::time::Instant::now())
            .as_secs();
        eprintln!(
            "    [{status}{:>3}{RESET}] every {:<6} [{:<4}] next in {:<6} | {} (last: {})",
            j.id,
            crate::cron::format_duration(j.interval_secs),
            kind,
            crate::cron::format_duration(next_secs),
            j.task,
            last
        );
    }
}

async fn handle_loop_command(
    rest: &str,
    kernel: &mut Kernel,
    registry: &Arc<Mutex<CronRegistry>>,
    current_pid: u32,
) {
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() < 3 {
        eprintln!("Usage: /loop <npc> <interval> <chat task|jinx:jinx_name>");
        eprintln!("  /loop sibiji 30s 'check for new emails and summarize'");
        eprintln!("  /loop corca 5m 'review open PRs'");
        eprintln!("  /loop sibiji 10s jinx:shell 'echo heartbeat'");
        return;
    }
    let npc = parts[0].to_string();
    let interval = parts[1];
    let task = parts[2..].join(" ");
    let secs = crate::cron::parse_duration(interval);
    let (kind, task) = if let Some(t) = task.strip_prefix("jinx:") {
        (crate::cron::CronJobKind::Jinx, t.to_string())
    } else {
        (crate::cron::CronJobKind::Chat, task)
    };
    let is_jinx = kind == crate::cron::CronJobKind::Jinx;
    let label = if is_jinx {
        format!("jinx:{task}")
    } else {
        task.clone()
    };
    let id = registry
        .lock()
        .unwrap()
        .add(npc.clone(), secs, task.clone(), kind);
    eprintln!("{GREEN}Loop {id} added: @{npc} every {interval} -> {label}{RESET}");
}

async fn execute_cron_job_and_capture(
    kernel: &mut Kernel,
    current_pid: u32,
    job: &crate::cron::CronJob,
    client: &reqwest::Client,
    server_url: &str,
    session_input_tokens: &mut u64,
    session_output_tokens: &mut u64,
    session_cost: &mut f64,
) -> Option<String> {
    let mut out = Vec::new();
    let Some(proc) = kernel.find_by_name(&job.npc) else {
        out.push(format!("{RED}[cron] unknown NPC: @{}{RESET}", job.npc));
        return Some(out.join("\n"));
    };
    let pid = proc.pid;
    let label = format!("[cron {}]", job.npc);
    match job.kind {
        crate::cron::CronJobKind::Jinx => {
            let mut args = std::collections::HashMap::new();
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
                Ok(output) => {
                    if !output.is_empty() {
                        out.push(format!("{CYAN}{label}{RESET} /{jinx_name}\n{output}"));
                    }
                }
                Err(e) => out.push(format!("{RED}{label} /{jinx_name} error: {e}{RESET}")),
            }
        }
        crate::cron::CronJobKind::Chat => {
            out.push(format!(
                "{CYAN}{label}{RESET} {DIM}{task}{RESET}",
                task = job.task
            ));
            match run_stream_turn(
                kernel,
                pid,
                &job.task,
                Mode::Agent,
                client,
                server_url,
                false,
                None,
            )
            .await
            {
                Ok(output) => {
                    if let Some(p) = kernel.get_process(pid) {
                        *session_input_tokens += p.usage.total_input_tokens;
                        *session_output_tokens += p.usage.total_output_tokens;
                        *session_cost += p.usage.total_cost_usd;
                    }
                    if !output.is_empty() {
                        out.push(output);
                    }
                }
                Err(e) => out.push(format!("{RED}{label} error: {e}{RESET}")),
            }
        }
    }
    let full = if out.is_empty() {
        None
    } else {
        Some(out.join("\n"))
    };
    if let Some(ref text) = full {
        let _ = save_loop_run(
            &job.npc,
            &job.task,
            job.kind == crate::cron::CronJobKind::Jinx,
            text,
        );
    }
    full
}

fn task_slug(task: &str, is_jinx: bool) -> String {
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

fn save_loop_run(npc: &str, task: &str, is_jinx: bool, output: &str) -> std::io::Result<()> {
    let slug = task_slug(task, is_jinx);
    let base = std::path::PathBuf::from(shellexpand::tilde("~/.npcsh/loops").to_string())
        .join(npc)
        .join(slug)
        .join("runs");
    std::fs::create_dir_all(&base)?;
    let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let path = base.join(format!("{}.txt", ts));
    std::fs::write(&path, output)
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

fn arg_value(args: &[String], flags: &[&str]) -> Option<String> {
    for window in args.windows(2) {
        if flags.contains(&window[0].as_str()) {
            return Some(window[1].clone());
        }
    }
    None
}

async fn spawn_npc_from_registered_teams(
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
            let npc = npcrs::npc_compiler::NPC::from_file(&path).map_err(|e| {
                npcrs::NpcError::Other(format!("Failed to load NPC {}: {}", name, e))
            })?;
            let pid = kernel.spawn(npc, current_pid, Capabilities::root());
            return Ok(pid);
        }
    }
    Ok(0)
}

fn current_npc_name(kernel: &Kernel, current_pid: u32) -> String {
    kernel
        .get_process(current_pid)
        .map(|p| p.npc.name.clone())
        .unwrap_or_default()
}

fn current_npc_jinxes(kernel: &Kernel, current_pid: u32) -> std::collections::HashSet<String> {
    kernel
        .get_process(current_pid)
        .map(|p| p.npc.jinx_names.iter().cloned().collect())
        .unwrap_or_default()
}

fn print_welcome(kernel: &Kernel, current_pid: u32) {
    let s = kernel.stats();

    const BLUE: &str = "\x1b[1;94m";
    const RUST: &str = "\x1b[1;38;5;202m";
    const GREEN: &str = "\x1b[1;32m";
    const BOLD_GREEN: &str = "\x1b[1;32m";

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
    eprintln!(
        "  {BOLD}npcsh{RESET} v{} {DIM}(rust){RESET}",
        env!("NPCSH_VERSION")
    );
    eprintln!(
        "  {DIM}{} processes | {} jinxes | /help for commands{RESET}",
        s.total_processes, s.jinx_count
    );
    eprintln!();

    eprintln!("  {DIM}mode:{RESET} {BOLD}agent{RESET}");
    eprint!("  {DIM}npcs:{RESET} ");
    let names: Vec<String> = kernel
        .ps()
        .iter()
        .map(|p| format!("{BLUE}@{}{RESET}", p.npc.name))
        .collect();
    eprintln!("{}", names.join("  "));
    eprintln!();

    let mut groups: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<Option<String>, Vec<String>>,
    > = std::collections::BTreeMap::new();

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
        groups
            .entry(group)
            .or_default()
            .entry(subdir)
            .or_default()
            .push(jname.clone());
    }

    let group_order = ["bin", "lib", "skills", "etc", "sys", "usr", "root", "other"];
    let mut sorted_groups: Vec<_> = groups.keys().cloned().collect();
    sorted_groups.sort_by_key(|g| group_order.iter().position(|o| o == g).unwrap_or(99));

    let active_jinxes = current_npc_jinxes(kernel, current_pid);

    for group in &sorted_groups {
        if let Some(subdirs) = groups.get(group) {
            eprintln!("  {RUST}{group}/{RESET}");
            if let Some(names) = subdirs.get(&None) {
                let mut sorted = names.clone();
                sorted.sort();
                let mut current = String::from("    ");
                for item in &sorted {
                    let colored = if active_jinxes.contains(item) {
                        format!("{BOLD_GREEN}{item}{RESET}")
                    } else {
                        item.clone()
                    };
                    let display_width = if active_jinxes.contains(item) {
                        item.len()
                    } else {
                        item.len()
                    };
                    if current.len() + display_width + 2 > 80 && current.trim().len() > 0 {
                        eprintln!("{}", current);
                        current = String::from("    ");
                    }
                    current.push_str(&colored);
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
                    let rendered: Vec<String> = sorted
                        .iter()
                        .map(|n| {
                            if active_jinxes.contains(n) {
                                format!("{BOLD_GREEN}{n}{RESET}")
                            } else {
                                n.clone()
                            }
                        })
                        .collect();
                    eprintln!("      {DIM}{sd}:{RESET} {}", rendered.join("  "));
                }
            }
        }
    }

    eprintln!();
    eprintln!(
        "  {DIM}jinxes in {BOLD_GREEN}bold{RESET}{DIM} are available to the current agent (@{}){RESET}",
        current_npc_name(kernel, current_pid)
    );
    eprintln!();
    eprintln!("  {DIM}core commands:{RESET}");
    for category in COMMAND_CATEGORIES {
        let cmds: Vec<&CommandDef> = CORE_COMMANDS
            .iter()
            .filter(|c| c.category == *category && c.name.starts_with('/'))
            .collect();
        if cmds.is_empty() {
            continue;
        }
        let mut line = format!("    {DIM}{}:{RESET} ", category);
        let joined = cmds
            .iter()
            .map(|c| format!("{CYAN}{}", c.name))
            .collect::<Vec<_>>()
            .join("\x1b[0m  ");
        line.push_str(&joined);
        line.push_str("\x1b[0m");
        eprintln!("{}", line);
    }
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

fn ensure_memory_pref() {
    if std::env::var("NPCSH_MEMORY").is_ok() {
        return;
    }
    let rc_path = shellexpand::tilde("~/.npcshrc").to_string();
    let mut has_entry = false;
    if let Ok(content) = std::fs::read_to_string(&rc_path) {
        has_entry = content
            .lines()
            .any(|l| l.trim_start_matches("export ").starts_with("NPCSH_MEMORY"));
    }
    if has_entry {
        return;
    }

    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        unsafe { std::env::set_var("NPCSH_MEMORY", "1") };
        let _ = append_npcshrc_line("NPCSH_MEMORY", "1");
        return;
    }

    eprintln!("{CYAN}Memory / knowledge extraction{RESET}");
    eprintln!("npcsh can automatically extract memories from your conversations.");
    eprintln!("You can change this later by editing NPCSH_MEMORY in ~/.npcshrc.");
    eprint!("Enable automatic memory extraction? [Y/n] {RESET}");
    let _ = std::io::Write::flush(&mut std::io::stderr());
    let mut answer = String::new();
    if std::io::stdin().read_line(&mut answer).is_err() {
        answer = "y".to_string();
    }
    let enabled = match answer.trim().to_lowercase().as_str() {
        "" | "y" | "yes" | "true" | "1" | "on" => true,
        _ => false,
    };
    let value = if enabled { "1" } else { "0" };
    unsafe { std::env::set_var("NPCSH_MEMORY", value) };
    let _ = append_npcshrc_line("NPCSH_MEMORY", value);
    eprintln!(
        "{}Memory extraction {}",
        if enabled { GREEN } else { YELLOW },
        if enabled { "enabled" } else { "disabled" }
    );
}

fn append_npcshrc_line(key: &str, value: &str) -> std::io::Result<()> {
    let rc_path = shellexpand::tilde("~/.npcshrc").to_string();
    let path = std::path::Path::new(&rc_path);
    let mut content = String::new();
    if path.exists() {
        content = std::fs::read_to_string(path)?;
        if !content.ends_with('\n') {
            content.push('\n');
        }
    }
    content.push_str(&format!("export {}={}\n", key, value));
    std::fs::write(path, content)
}

struct MemoryScheduler {
    turn_count: u64,
    next_trigger: u64,
    lambda: f64,
    rng: rand::rngs::StdRng,
}

impl MemoryScheduler {
    fn new(lambda: f64) -> Self {
        let mut rng = rand::rngs::StdRng::from_entropy();
        // lambda is the mean interval (in turns) between memory extractions.
        let lambda = lambda.max(1.0);
        let first = Self::sample_interval(lambda, &mut rng);
        Self {
            turn_count: 0,
            next_trigger: first,
            lambda,
            rng,
        }
    }

    fn from_env() -> Option<Self> {
        if !memory_context_enabled() {
            return None;
        }
        // Default to a mean interval of ~60 turns between memory extractions.
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

    fn should_trigger(&mut self) -> bool {
        self.turn_count += 1;
        self.turn_count >= self.next_trigger
    }

    fn defer(&mut self) {
        self.next_trigger = self.turn_count + 1;
    }

    fn reschedule(&mut self) {
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
        .map_err(|e| npcrs::NpcError::Other(format!("memory extract request failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(npcrs::NpcError::Other(format!(
            "memory extract returned {}: {}",
            status, text
        )));
    }
    let json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| npcrs::NpcError::Other(format!("memory extract json failed: {e}")))?;
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

async fn maybe_extract_memory(
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

const TERMINAL_EDITORS: &[&str] = &["vim", "nvim", "nano", "vi", "emacs", "less", "more", "man"];

const INTERACTIVE_COMMANDS: &[&str] = &[
    "ipython",
    "python",
    "python3",
    "node",
    "irb",
    "ghci",
    "mysql",
    "psql",
    "sqlite3",
    "redis-cli",
    "mongo",
    "ssh",
    "telnet",
    "ftp",
    "sftp",
    "top",
    "htop",
    "watch",
    "r",
];

const SHELL_BUILTINS: &[&str] = &[
    "cd", "pwd", "echo", "export", "source", "alias", "unalias", "history", "set", "unset", "read",
    "eval", "exec", "exit", "return", "shift", "trap", "wait", "jobs", "fg", "bg", "kill",
    "ulimit", "umask", "type", "hash", "true", "false",
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

    if let Ok(output) = std::process::Command::new("which").arg(cmd).output() {
        return output.status.success();
    }

    false
}

/// Fast, cache-backed bash detection used only for live input coloring.
fn looks_like_bash(input: &str) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    if cmd.is_empty() {
        return false;
    }
    if SHELL_BUILTINS.contains(&cmd) {
        return true;
    }
    if cmd.contains('/') {
        return std::path::Path::new(cmd).is_file();
    }
    cached_path_lookup(cmd)
}

static PATH_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, bool>>> =
    std::sync::OnceLock::new();

fn cached_path_lookup(cmd: &str) -> bool {
    if cmd.is_empty() || cmd.contains('/') {
        return false;
    }
    let cache = PATH_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    {
        let read = cache.lock().unwrap();
        if let Some(hit) = read.get(cmd) {
            return *hit;
        }
    }
    let found = std::env::var("PATH")
        .ok()
        .map(|path| {
            path.split(':')
                .any(|dir| std::path::Path::new(dir).join(cmd).is_file())
        })
        .unwrap_or(false);
    cache.lock().unwrap().insert(cmd.to_string(), found);
    found
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

fn format_ts(ts: &str) -> String {
    use std::time::{Duration, SystemTime};
    let now = SystemTime::now();
    let dt: Option<chrono::DateTime<chrono::Utc>> = if ts.contains('T') {
        ts.parse::<chrono::DateTime<chrono::Utc>>().ok()
    } else {
        chrono::NaiveDateTime::parse_from_str(&ts[..ts.len().min(19)], "%Y-%m-%d %H:%M:%S")
            .ok()
            .map(|ndt| chrono::DateTime::from_naive_utc_and_offset(ndt, chrono::Utc))
    };
    if let Some(dt) = dt {
        let local = dt.with_timezone(&chrono::Local);
        let utc_dt: chrono::DateTime<chrono::Utc> = dt;
        let diff = now
            .duration_since(SystemTime::from(utc_dt))
            .unwrap_or(Duration::ZERO);
        let days = diff.as_secs() / 86400;
        if days == 0 {
            format!("Today {}", local.format("%H:%M"))
        } else if days == 1 {
            format!("Yesterday {}", local.format("%H:%M"))
        } else if days < 7 {
            local.format("%a %H:%M").to_string()
        } else {
            local.format("%b %d").to_string()
        }
    } else {
        ts.chars().take(16).collect()
    }
}

fn run_reattach(kernel: &mut Kernel, current_pid: u32, filter: Option<&str>) -> Result<()> {
    use crossterm::event::{self, Event, KeyCode};
    use crossterm::terminal;
    use rusqlite::Connection;

    let db_path = std::env::var("NPCSH_DB_PATH")
        .map(|s| shellexpand::tilde(&s).to_string())
        .unwrap_or_else(|_| shellexpand::tilde("~/npcsh_history.db").to_string());
    let conn = Connection::open(&db_path)
        .map_err(|e| npcrs::NpcError::Other(format!("failed to open history db: {e}")))?;

    type ConvoRow = (
        std::string::String,
        std::string::String,
        std::string::String,
        std::string::String,
        i64,
        std::string::String,
        std::string::String,
        i64,
        i64,
        f64,
    );

    let convos: Vec<ConvoRow> = if let Some(path) = filter {
        let path_slash = format!("{path}/");
        let mut stmt = conn.prepare(
            "SELECT conversation_id, directory_path, MIN(timestamp) as started, MAX(timestamp) as last_msg, \
             COUNT(*) as msg_count, GROUP_CONCAT(DISTINCT npc) as npcs, GROUP_CONCAT(DISTINCT model) as models, \
             COALESCE(SUM(input_tokens), 0) as total_input_tokens, COALESCE(SUM(output_tokens), 0) as total_output_tokens, \
             COALESCE(SUM(CAST(cost AS REAL)), 0) as total_cost \
             FROM conversation_history \
             WHERE directory_path = ?1 OR directory_path = ?2 \
             GROUP BY conversation_id \
             ORDER BY last_msg DESC"
        ).map_err(|e| npcrs::NpcError::Other(format!("query failed: {e}")))?;
        stmt.query_map([path, &path_slash], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
                row.get(8)?,
                row.get(9)?,
            ))
        })
        .map_err(|e| npcrs::NpcError::Other(format!("query failed: {e}")))?
        .filter_map(|r| r.ok())
        .collect()
    } else {
        let mut stmt = conn.prepare(
            "SELECT conversation_id, directory_path, MIN(timestamp) as started, MAX(timestamp) as last_msg, \
             COUNT(*) as msg_count, GROUP_CONCAT(DISTINCT npc) as npcs, GROUP_CONCAT(DISTINCT model) as models, \
             COALESCE(SUM(input_tokens), 0) as total_input_tokens, COALESCE(SUM(output_tokens), 0) as total_output_tokens, \
             COALESCE(SUM(CAST(cost AS REAL)), 0) as total_cost \
             FROM conversation_history \
             GROUP BY conversation_id \
             ORDER BY last_msg DESC"
        ).map_err(|e| npcrs::NpcError::Other(format!("query failed: {e}")))?;
        stmt.query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
                row.get(8)?,
                row.get(9)?,
            ))
        })
        .map_err(|e| npcrs::NpcError::Other(format!("query failed: {e}")))?
        .filter_map(|r| r.ok())
        .collect()
    };

    if convos.is_empty() {
        let target = filter.unwrap_or("ALL PATHS");
        println!("{DIM}No conversations for: {target}{RESET}");
        return Ok(());
    }

    terminal::enable_raw_mode().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut stdout = io::stdout();
    let _ = write!(stdout, "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H");
    let _ = stdout.flush();

    let (cols, rows) = terminal::size().unwrap_or((80, 24));
    let cols = cols as usize;
    let rows = rows as usize;
    let list_height = rows.saturating_sub(5);

    let selected: std::cell::Cell<usize> = std::cell::Cell::new(0);
    let scroll: std::cell::Cell<usize> = std::cell::Cell::new(0);
    let mode: std::cell::Cell<char> = std::cell::Cell::new('l');
    let preview_scroll: std::cell::Cell<usize> = std::cell::Cell::new(0);
    let preview_msgs: std::cell::RefCell<
        Vec<(
            std::string::String,
            std::string::String,
            std::string::String,
            Option<i64>,
            Option<i64>,
        )>,
    > = std::cell::RefCell::new(Vec::new());
    let selected_conversation: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);

    fn short_model(model: &str) -> String {
        if model.is_empty() {
            return "-".to_string();
        }
        // Return the first whitespace-delimited token, truncated to 8 chars.
        let token = model.split_whitespace().next().unwrap_or(model);
        token[..token.len().min(8)].to_string()
    }

    let target = filter
        .map(|s| s.to_string())
        .unwrap_or_else(|| "ALL PATHS".to_string());
    let (rows_static, cols_static) = (rows, cols);

    let draw_list = |stdout: &mut std::io::Stdout| {
        let sel_idx = selected.get();
        let scr = scroll.get();
        let header = format!(
            " REATTACH ({} convos): {} ",
            convos.len(),
            &target[..target.len().min(cols_static.saturating_sub(30))]
        );
        let _ = write!(
            stdout,
            "\x1b[H\x1b[7;1m{}\x1b[0m\n",
            header
                .chars()
                .take(cols_static)
                .collect::<String>()
                .pad_right(cols_static)
        );
        let _ = write!(stdout, "\x1b[90m{}\x1b[0m\n", "─".repeat(cols_static));
        for i in 0..list_height {
            let idx = scr + i;
            let _ = write!(stdout, "\x1b[{};1H\x1b[K", 3 + i);
            if idx >= convos.len() {
                continue;
            }
            let c = &convos[idx];
            let cid = &c.0[..c.0.len().min(12)];
            let npcs = c.5.as_str();
            let npcs = &npcs[..npcs.len().min(10)];
            let models = short_model(c.6.as_str());
            let line = format!(
                " {cid:<14} {:>3} msgs  {} {npcs:<10} {models:<12}",
                c.4,
                format_ts(&c.3)
            );
            let line = &line[..line.len().min(cols_static.saturating_sub(2))];
            if idx == sel_idx {
                let _ = write!(
                    stdout,
                    "\x1b[7;1m>{}\x1b[0m",
                    line.pad_right(cols_static.saturating_sub(1))
                );
            } else {
                let _ = write!(stdout, " {}", line.pad_right(cols_static.saturating_sub(1)));
            }
        }
        let sel = &convos[sel_idx];
        let in_tok = sel.7;
        let out_tok = sel.8;
        let cost = sel.9;
        let cost_str = if cost > 0.0 {
            format!("${cost:.4}")
        } else {
            "-".to_string()
        };
        let tok_str = if in_tok > 0 || out_tok > 0 {
            format!("{in_tok}in/{out_tok}out")
        } else {
            "-".to_string()
        };
        let footer = format!(
            " {}  {}  tokens:{}  cost:{}",
            &sel.0[..sel.0.len().min(16)],
            short_model(sel.6.as_str()),
            tok_str,
            cost_str
        );
        let _ = write!(
            stdout,
            "\x1b[{};1H\x1b[K\x1b[90m{}\x1b[0m",
            rows_static - 2,
            "─".repeat(cols_static)
        );
        let _ = write!(
            stdout,
            "\x1b[{};1H\x1b[K{}",
            rows_static - 1,
            footer
                .chars()
                .take(cols_static)
                .collect::<String>()
                .pad_right(cols_static)
        );
        let _ = write!(
            stdout,
            "\x1b[{};1H\x1b[K\x1b[7m j/k:Nav  Enter:Select  p:Preview  q:Quit  [{}/{}] \x1b[0m",
            rows_static,
            sel_idx + 1,
            convos.len()
        );
        let _ = stdout.flush();
    };

    let draw_preview = |stdout: &mut std::io::Stdout| {
        let sel_idx = selected.get();
        let scr = preview_scroll.get();
        let cid = &convos[sel_idx].0;
        let header = format!(
            " PREVIEW: {} ",
            &cid[..cid.len().min(cols_static.saturating_sub(12))]
        );
        let _ = write!(
            stdout,
            "\x1b[H\x1b[7;1m{}\x1b[0m\n",
            header
                .chars()
                .take(cols_static)
                .collect::<String>()
                .pad_right(cols_static)
        );
        let _ = write!(stdout, "\x1b[90m{}\x1b[0m\n", "─".repeat(cols_static));
        let msgs = preview_msgs.borrow();
        for i in 0..list_height {
            let idx = scr + i;
            let _ = write!(stdout, "\x1b[{};1H\x1b[K", 3 + i);
            if idx >= msgs.len() {
                continue;
            }
            let (role, content, model, in_tok, out_tok) = &msgs[idx];
            let content = content
                .replace('\n', " ")
                .chars()
                .take(200)
                .collect::<String>();
            let prefix = match role.as_str() {
                "user" => format!("{GREEN};1mYou:\x1b[0m "),
                "assistant" => {
                    let m = short_model(model);
                    let tok_info = if in_tok.is_some() || out_tok.is_some() {
                        format!(" [{}|{}]", in_tok.unwrap_or(0), out_tok.unwrap_or(0))
                    } else {
                        String::new()
                    };
                    format!("\x1b[34;1mAI({m}{tok_info}):\x1b[0m ")
                }
                _ => format!("\x1b[90m{role}:\x1b[0m "),
            };
            let max_content = cols_static.saturating_sub(prefix.len());
            let _ = write!(
                stdout,
                "{}{}",
                prefix,
                content.chars().take(max_content).collect::<String>()
            );
        }
        let _ = write!(
            stdout,
            "\x1b[{};1H\x1b[K\x1b[90m{}\x1b[0m",
            rows_static - 2,
            "─".repeat(cols_static)
        );
        let _ = write!(
            stdout,
            "\x1b[{};1H\x1b[K {} messages",
            rows_static - 1,
            msgs.len()
        );
        let _ = write!(
            stdout,
            "\x1b[{};1H\x1b[K\x1b[7m j/k:Scroll  b:Back  Enter:Select  q:Quit \x1b[0m",
            rows_static
        );
        let _ = stdout.flush();
    };

    trait PadRight {
        fn pad_right(&self, n: usize) -> String;
    }
    impl PadRight for str {
        fn pad_right(&self, n: usize) -> String {
            if self.len() >= n {
                self.to_string()
            } else {
                format!("{}{}", self, " ".repeat(n - self.len()))
            }
        }
    }

    loop {
        let sel = selected.get();
        let scr = scroll.get();
        if sel < scr {
            scroll.set(sel);
        } else if sel >= scr + list_height {
            scroll.set(sel - list_height + 1);
        }

        if mode.get() == 'l' {
            draw_list(&mut stdout);
        } else {
            draw_preview(&mut stdout);
        }

        if let Ok(true) = event::poll(std::time::Duration::from_millis(50)) {
            if let Ok(Event::Key(key)) = event::read() {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q')
                        if key.modifiers == crossterm::event::KeyModifiers::CONTROL =>
                    {
                        selected_conversation.replace(None);
                        break;
                    }
                    KeyCode::Char('q') | KeyCode::Char('Q') => {
                        selected_conversation.replace(None);
                        break;
                    }
                    KeyCode::Esc => {
                        if mode.get() == 'p' {
                            mode.set('l');
                        } else {
                            selected_conversation.replace(None);
                            break;
                        }
                    }
                    KeyCode::Char('j') | KeyCode::Down => {
                        if mode.get() == 'l' && selected.get() < convos.len() - 1 {
                            selected.set(selected.get() + 1);
                        } else if mode.get() == 'p'
                            && preview_scroll.get() + list_height < preview_msgs.borrow().len()
                        {
                            preview_scroll.set(preview_scroll.get() + 1);
                        }
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        if mode.get() == 'l' && selected.get() > 0 {
                            selected.set(selected.get() - 1);
                        } else if mode.get() == 'p' && preview_scroll.get() > 0 {
                            preview_scroll.set(preview_scroll.get() - 1);
                        }
                    }
                    KeyCode::Char('p') if mode.get() == 'l' => {
                        let cid = convos[selected.get()].0.clone();
                        let mut stmt = conn.prepare(
                            "SELECT role, content, model, input_tokens, output_tokens FROM conversation_history \
                             WHERE conversation_id = ?1 ORDER BY timestamp ASC"
                        ).map_err(|e| npcrs::NpcError::Other(format!("preview query failed: {e}")))?;
                        let loaded: Vec<(String, String, String, Option<i64>, Option<i64>)> = stmt
                            .query_map([&cid], |row| {
                                Ok((
                                    row.get(0)?,
                                    row.get(1)?,
                                    row.get(2)?,
                                    row.get(3)?,
                                    row.get(4)?,
                                ))
                            })
                            .map_err(|e| {
                                npcrs::NpcError::Other(format!("preview query failed: {e}"))
                            })?
                            .filter_map(|r| r.ok())
                            .collect();
                        preview_msgs.replace(loaded);
                        preview_scroll.set(0);
                        mode.set('p');
                    }
                    KeyCode::Char('b') if mode.get() == 'p' => {
                        mode.set('l');
                    }
                    KeyCode::Enter => {
                        let cid = convos[selected.get()].0.clone();
                        let _ = terminal::disable_raw_mode();
                        let _ = write!(stdout, "\x1b[?1049l\x1b[?25h\x1b[999;1H\r\n");
                        let _ = stdout.flush();
                        if let Some(p) = kernel.get_process_mut(current_pid) {
                            p.conversation_id = cid.clone();
                            p.messages.clear();
                            let mut stmt = conn.prepare(
                                "SELECT role, content FROM conversation_history WHERE conversation_id = ?1 ORDER BY timestamp ASC"
                            ).map_err(|e| npcrs::NpcError::Other(format!("load query failed: {e}")))?;
                            let rows = stmt
                                .query_map([&cid], |row| {
                                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                                })
                                .map_err(|e| {
                                    npcrs::NpcError::Other(format!("load query failed: {e}"))
                                })?
                                .filter_map(|r| r.ok());
                            for (role, content) in rows {
                                let msg = match role.as_str() {
                                    "user" => Message::user(content.clone()),
                                    "assistant" => Message::assistant(content.clone()),
                                    "system" => Message::system(content.clone()),
                                    "tool" => Message::tool_result("", content.clone()),
                                    _ => Message::user(format!("[{role}] {content}")),
                                };
                                p.messages.push(msg);
                            }
                            selected_conversation.replace(Some(cid.clone()));
                            println!(
                                "{GREEN}Reattached to: {cid} ({} messages loaded)\x1b[0m",
                                p.messages.len()
                            );
                            for msg in &p.messages {
                                let role = msg.role.as_str();
                                let content = msg.content.as_deref().unwrap_or("");
                                if role == "user" {
                                    println!("{CYAN}> {content}\x1b[0m");
                                } else if role == "assistant" {
                                    println!("{}", content);
                                } else {
                                    println!("{DIM}[{role}] {content}\x1b[0m");
                                }
                            }
                            println!();
                        } else {
                            selected_conversation.replace(Some(cid.clone()));
                            println!("{YELLOW}Selected: {cid}\x1b[0m");
                        }
                        println!("{DIM}Press any key to continue...\x1b[0m");
                        let _ = stdout.flush();
                        loop {
                            if event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
                                if let Ok(Event::Key(_)) = event::read() {
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    let _ = terminal::disable_raw_mode();
    if selected_conversation.borrow().is_none() {
        let _ = write!(stdout, "\x1b[2J\x1b[H\x1b[?1049l\x1b[?25h");
    } else {
        let _ = write!(stdout, "\x1b[?1049l\x1b[?25h");
    }
    let _ = stdout.flush();
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
    let lines: Vec<&str> = raw_lines
        .into_iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    let mut variables: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut last_output = String::new();

    for (i, line) in lines.iter().enumerate() {
        let line = line.to_string();

        let cmd_to_exec = if let Some((var_name, var_expr)) =
            line.trim().strip_prefix('$').and_then(|rest| {
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
            true,
            None,
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum TokenClass {
    CoreCommand,
    JinxName,
    NpcRef,
    BashCommand,
    UnknownSlash,
    Text,
}

fn classify_input(buf: &str, jinx_names: &[String]) -> TokenClass {
    let first = buf.split_whitespace().next().unwrap_or("");
    if first.is_empty() {
        return TokenClass::Text;
    }
    if first.starts_with('/') {
        if CORE_COMMANDS.iter().any(|c| c.name == first) {
            TokenClass::CoreCommand
        } else {
            TokenClass::UnknownSlash
        }
    } else if CORE_COMMANDS.iter().any(|c| c.name == first) {
        TokenClass::CoreCommand
    } else if jinx_names.iter().any(|n| n == first) {
        TokenClass::JinxName
    } else if first.starts_with('@') {
        TokenClass::NpcRef
    } else if looks_like_bash(buf) {
        TokenClass::BashCommand
    } else {
        TokenClass::Text
    }
}

fn color_for_class(class: TokenClass) -> &'static str {
    match class {
        TokenClass::CoreCommand => CYAN,
        TokenClass::JinxName => CYAN,
        TokenClass::NpcRef => PURPLE,
        TokenClass::BashCommand => YELLOW,
        TokenClass::UnknownSlash => RED,
        TokenClass::Text => "",
    }
}

fn colorize_input(buf: &str, jinx_names: &[String]) -> String {
    let class = classify_input(buf, jinx_names);
    let color = color_for_class(class);
    if color.is_empty() {
        return buf.to_string();
    }
    let first = buf.split_whitespace().next().unwrap_or("");
    if first.is_empty() {
        return buf.to_string();
    }
    let start = buf.find(|c: char| !c.is_whitespace()).unwrap_or(0);
    let end = start + first.len();
    format!("{}{}{}{}{}", &buf[..start], color, first, RESET, &buf[end..])
}

fn input_hint(buf: &str, mode: Mode, jinx_names: &[String]) -> Option<String> {
    if buf.trim().is_empty() {
        return None;
    }
    match classify_input(buf, jinx_names) {
        TokenClass::BashCommand if mode == Mode::Agent => {
            let first = buf.split_whitespace().next().unwrap_or("");
            if first == "cd" || is_terminal_editor(buf) || is_interactive(buf) {
                return None;
            }
            Some("will execute as bash".to_string())
        }
        TokenClass::UnknownSlash => Some("unknown command".to_string()),
        _ => None,
    }
}

fn visible_len(s: &str) -> usize {
    let mut count = 0;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{001b}' {
            while let Some(next) = chars.next() {
                if next.is_alphabetic() {
                    break;
                }
            }
        } else {
            count += 1;
        }
    }
    count
}

enum ReadlineResult {
    Input(String),
    Cancel,
    Eof,
    Reattach,
}

fn readline_raw(
    prompt: &str,
    history: &mut Vec<String>,
    mode: Mode,
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
                    buf.insert_str(pos, &text);
                    pos += text.len();
                    redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                    io::stdout().flush()?;
                    continue;
                }
                Event::Key(key) => match key.code {
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
                                    redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                                }
                                'o' => {
                                    print!("\r\n");
                                    if let Some(p) = kernel.get_process(current_pid) {
                                        let mut tool_calls: Vec<&npcrs::r#gen::ToolCall> =
                                            Vec::new();
                                        for m in p.messages.iter().rev().take(10) {
                                            if let Some(ref tc) = m.tool_calls {
                                                for t in tc.iter().rev() {
                                                    tool_calls.push(t);
                                                }
                                            }
                                        }
                                        if tool_calls.is_empty() {
                                            println!(
                                                "{DIM}(no tool calls in recent messages){RESET}"
                                            );
                                        } else {
                                            let total = tool_calls.len().min(5);
                                            println!(
                                                "{BOLD}═══ Last {} tool call{} ═══{RESET}",
                                                total,
                                                if total > 1 { "s" } else { "" }
                                            );
                                            for (i, tc) in tool_calls.iter().take(5).enumerate() {
                                                println!(
                                                    "  [{}/{}] {CYAN}{}{RESET}",
                                                    i + 1,
                                                    total,
                                                    tc.function.name
                                                );
                                                let args = &tc.function.arguments;
                                                let preview = if args.len() > 200 {
                                                    let mut boundary = 200;
                                                    while boundary > 0
                                                        && !args.is_char_boundary(boundary)
                                                    {
                                                        boundary -= 1;
                                                    }
                                                    format!("{}…", &args[..boundary])
                                                } else {
                                                    args.to_string()
                                                };
                                                println!("    {}", preview);
                                            }
                                            println!("{BOLD}═{RESET}");
                                        }
                                    }
                                    redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                                }
                                _ => {}
                            }
                        } else {
                            tab_matches.clear();
                            if pos == buf.len() {
                                buf.push(c);
                            } else {
                                buf.insert(pos, c);
                            }
                            pos += c.len_utf8();
                            redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Backspace => {
                        tab_matches.clear();
                        if pos > 0 {
                            let prev = buf[..pos].chars().next_back().unwrap();
                            pos -= prev.len_utf8();
                            buf.remove(pos);
                            redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Delete => {
                        tab_matches.clear();
                        if pos < buf.len() {
                            buf.remove(pos);
                            redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Enter => {
                        print!("\r\n");
                        break Ok(ReadlineResult::Input(buf));
                    }
                    KeyCode::Left => {
                        if pos > 0 {
                            let prev = buf[..pos].chars().next_back().unwrap();
                            pos -= prev.len_utf8();
                            print!("\x1b[D");
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Right => {
                        if pos < buf.len() {
                            let next = buf[pos..].chars().next().unwrap();
                            pos += next.len_utf8();
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
                                redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                            }
                        } else if !history.is_empty() {
                            let new_idx = history.len() - 1;
                            *history_index = Some(new_idx);
                            buf = history[new_idx].clone();
                            pos = buf.len();
                            redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                        }
                    }
                    KeyCode::Down => {
                        if let Some(idx) = *history_index {
                            if idx + 1 < history.len() {
                                let new_idx = idx + 1;
                                *history_index = Some(new_idx);
                                buf = history[new_idx].clone();
                                pos = buf.len();
                                redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                            } else {
                                *history_index = None;
                                buf.clear();
                                pos = 0;
                                redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                            }
                        } else if buf.is_empty() {
                            break Ok(ReadlineResult::Reattach);
                        }
                    }
                    KeyCode::Tab => {
                        if tab_matches.is_empty() {
                            let (word_start, matches) = helper.complete(&buf, pos);
                            if matches.len() == 1 {
                                let replacement = &matches[0].replacement;
                                let new_buf =
                                    format!("{}{}{}", &buf[..word_start], replacement, &buf[pos..]);
                                pos = word_start + replacement.len();
                                buf = new_buf;
                                redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                                tab_matches.clear();
                            } else if !matches.is_empty() {
                                tab_matches = matches;
                                tab_index = 0;
                                print!("\r\n");
                                let cols = terminal::size().map(|(c, _)| c as usize).unwrap_or(80);
                                let max_len = tab_matches
                                    .iter()
                                    .map(|m| m.display.len())
                                    .max()
                                    .unwrap_or(0)
                                    + 2;
                                let col_width = max_len.max(16);
                                let ncols = (cols / col_width).max(1);
                                for (i, m) in tab_matches.iter().enumerate() {
                                    if i > 0 && i % ncols == 0 {
                                        print!("\r\n");
                                    }
                                    print!("{:<width$}", m.display, width = col_width);
                                }
                                print!("\r\n");
                                redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                            }
                        } else {
                            if !tab_matches.is_empty() {
                                tab_index = (tab_index + 1) % tab_matches.len();
                                let word_start = buf[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
                                let replacement = &tab_matches[tab_index].replacement;
                                let new_buf =
                                    format!("{}{}{}", &buf[..word_start], replacement, &buf[pos..]);
                                pos = word_start + replacement.len();
                                buf = new_buf;
                                redraw_prompt(prompt, &buf, pos, mode, &helper.jinx_names);
                            }
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    };

    let _ = terminal::disable_raw_mode();

    print!("\x1b[?2004l");
    let _ = io::stdout().flush();

    result
}

fn redraw_prompt(prompt: &str, buf: &str, pos: usize, mode: Mode, jinx_names: &[String]) {
    let colored = colorize_input(buf, jinx_names);
    let hint = input_hint(buf, mode, jinx_names);

    let (cols, _) = tui::term_size();
    let cols = cols.max(1);

    let prompt_visible = visible_len(prompt);
    let end_abs = prompt_visible + visible_len(buf);
    let end_line = end_abs / cols;

    let cursor_abs = prompt_visible + visible_len(&buf[..pos]);
    let cursor_line = cursor_abs / cols;
    let cursor_col = cursor_abs % cols;

    // Clear from the start of the prompt line downward.
    print!("\x1b[G");
    for _ in 0..cursor_line {
        print!("\x1b[A");
    }
    print!("\x1b[G\x1b[J");

    print!("{}", prompt);
    print!("{}", colored);

    let has_hint = hint.is_some();
    if let Some(h) = hint {
        print!("\r\n\x1b[2K\x1b[90m{}\x1b[0m", h);
    }

    // Place the cursor at the requested position within the input.
    let target_line = if has_hint { end_line + 1 } else { end_line };
    let lines_up = target_line.saturating_sub(cursor_line);
    if lines_up > 0 {
        print!("\x1b[{}A", lines_up);
    }
    print!("\x1b[G");
    if cursor_col > 0 {
        print!("\x1b[{}C", cursor_col);
    }

    let _ = io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_len_strips_ansi() {
        assert_eq!(visible_len("\x1b[31mhello\x1b[0m"), 5);
        assert_eq!(visible_len("\x1b[1;36m/\x1b[0mhelp"), 5);
    }

    #[test]
    fn classify_known_core_command() {
        let jinx_names: &[String] = &[];
        assert_eq!(classify_input("/help", jinx_names), TokenClass::CoreCommand);
        assert_eq!(classify_input("/exit now", jinx_names), TokenClass::CoreCommand);
    }

    #[test]
    fn classify_npc_ref() {
        let jinx_names: &[String] = &[];
        assert_eq!(classify_input("@alice", jinx_names), TokenClass::NpcRef);
        assert_eq!(classify_input("@bob hi", jinx_names), TokenClass::NpcRef);
    }

    #[test]
    fn classify_unknown_slash() {
        let jinx_names: &[String] = &[];
        assert_eq!(classify_input("/foobar", jinx_names), TokenClass::UnknownSlash);
    }

    #[test]
    fn colorize_core_command_wraps_first_token() {
        let jinx_names: &[String] = &[];
        let out = colorize_input("/help foo", jinx_names);
        assert!(out.starts_with(CYAN));
        assert!(out.contains("/help"));
        assert!(out.ends_with(" foo"));

        let with_space = colorize_input("  /help", jinx_names);
        assert!(with_space.starts_with("  "));
        assert!(with_space.contains("/help"));
    }

    #[test]
    fn colorize_bash_command_first_token() {
        let jinx_names: &[String] = &[];
        let out = colorize_input("ls -la", jinx_names);
        assert!(out.starts_with(YELLOW));
        assert!(out.contains("ls"));
        assert!(out.contains("\u{001b}[0m -la"));
    }

    #[test]
    fn colorize_text_untouched() {
        let jinx_names: &[String] = &[];
        assert_eq!(colorize_input("hello world", jinx_names), "hello world");
    }

    #[test]
    fn input_hint_for_bash_in_agent_mode() {
        let jinx_names: &[String] = &[];
        assert_eq!(
            input_hint("ls -la", Mode::Agent, jinx_names),
            Some("will execute as bash".to_string())
        );
    }

    #[test]
    fn input_hint_no_hint_for_cd_or_editors() {
        let jinx_names: &[String] = &[];
        assert_eq!(input_hint("cd /tmp", Mode::Agent, jinx_names), None);
        assert_eq!(input_hint("vim file.txt", Mode::Agent, jinx_names), None);
    }

    #[test]
    fn input_hint_no_hint_in_chat_mode() {
        let jinx_names: &[String] = &[];
        assert_eq!(input_hint("ls -la", Mode::Chat, jinx_names), None);
    }
}
