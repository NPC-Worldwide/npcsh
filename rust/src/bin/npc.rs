use npcrs::error::Result;
use npcsh::{
    agent_turn, exec_jinx_file, exec_npc_file, find_team_dir, init_team, resolve_team_layout,
};
use std::path::PathBuf;

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "5237";

fn server_url() -> String {
    std::env::var("NPCSH_SERVER_URL")
        .unwrap_or_else(|_| format!("http://{DEFAULT_HOST}:{DEFAULT_PORT}"))
}

fn resolve_agent_tool(name: &str) -> Option<&'static str> {
    match name.to_lowercase().as_str() {
        "claude" | "claude-code" => Some("claude"),
        "codex" => Some("codex"),
        "gemini" => Some("gemini"),
        "opencode" => Some("opencode"),
        "aider" => Some("aider"),
        "amp" => Some("amp"),
        _ => None,
    }
}

fn resolve_agent_jinx(tool: &str) -> Option<String> {
    let name = format!("{}.jinx", tool);
    let team_dir = find_team_dir();
    let global_team = shellexpand::tilde("~/.npcsh/npc_team").to_string();

    for base in [team_dir, global_team] {
        let jinxes_dir = PathBuf::from(&base).join("jinxes");
        if !jinxes_dir.is_dir() {
            continue;
        }
        for sub in ["", "lib", "lib/agents", "usr", "sys", "etc", "skills"] {
            let candidate = jinxes_dir.join(sub).join(&name);
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
    }
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    resolve_team_layout();

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-v") {
        println!("npc {}", env!("NPCSH_VERSION"));
        return Ok(());
    }
    let mut positional: Vec<&str> = Vec::new();
    let mut override_model: Option<String> = None;
    let mut override_provider: Option<String> = None;
    let mut override_npc: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-m" | "--model" => {
                if i + 1 < args.len() {
                    override_model = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "-pr" | "--provider" => {
                if i + 1 < args.len() {
                    override_provider = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "-n" | "--npc" => {
                if i + 1 < args.len() {
                    override_npc = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            _ => {}
        }
        positional.push(arg);
        i += 1;
    }

    if let Some(file) = positional.first() {
        if *file == "init" {
            let dir = positional.get(1).copied().unwrap_or(".");
            return init_team(dir);
        } else if file.ends_with(".jinx") {
            let jinx_args: Vec<&str> = positional[1..].to_vec();
            return exec_jinx_file(file, &jinx_args).await;
        } else if file.ends_with(".npc") {
            let client = reqwest::Client::new();
            return exec_npc_file(
                file,
                positional.get(1).copied(),
                &client,
                &server_url(),
                override_model.as_deref(),
                override_provider.as_deref(),
            )
            .await;
        } else if let Some(tool) = resolve_agent_tool(file) {
            let jinx_path = resolve_agent_jinx(tool).unwrap_or_else(|| {
                eprintln!(
                    "No {} launcher jinx found. Install it in your team jinxes directory or run `npc init`.",
                    tool
                );
                std::process::exit(1);
            });

            let mut jinx_args: Vec<String> = Vec::new();
            if let Some(npc) = &override_npc {
                jinx_args.push(format!("npc_name={}", npc));
            }

            let mut extra: Vec<&str> = positional[1..].to_vec();
            if extra.first() == Some(&"--") {
                extra.remove(0);
            }
            if !extra.is_empty() {
                jinx_args.push(format!("extra_args={}", extra.join(" ")));
            }

            let jinx_arg_refs: Vec<&str> = jinx_args.iter().map(|s| s.as_str()).collect();
            return exec_jinx_file(&jinx_path, &jinx_arg_refs).await;
        } else {
            let prompt = positional.join(" ");
            if prompt.is_empty() {
                eprintln!("Usage: npc <prompt> [-n NPC] [-m MODEL] [-pr PROVIDER]");
                eprintln!("       npc <file.npc|file.jinx|init> [args...]");
                eprintln!("       npc <agent> [-n NPC] [-- <tool args>]");
                eprintln!("       agents: claude, codex, gemini, opencode, aider, amp");
                std::process::exit(1);
            }
            // Use the same refactored one-shot agent loop that `npcsh -c` uses so
            // jinxes, tool calls, and the full kernel context behave identically.
            return agent_turn::run_command(
                &prompt,
                override_npc.as_deref(),
                override_model.as_deref(),
                override_provider.as_deref(),
            )
            .await;
        }
    }

    eprintln!("Usage: npc <prompt> [-n NPC] [-m MODEL] [-pr PROVIDER]");
    eprintln!("       npc <file.npc|file.jinx|init> [args...]");
    eprintln!("       npc <agent> [-n NPC] [-- <tool args>]");
    eprintln!("       agents: claude, codex, gemini, opencode, aider, amp");
    std::process::exit(1);
}
