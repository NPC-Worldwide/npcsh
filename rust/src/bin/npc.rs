use npcrs::error::Result;
use npcsh::{
    agent_turn, exec_jinx_file, exec_npc_file, find_team_dir, init_team, resolve_team_layout,
};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "5237";
const DEFAULT_NPC: &str = "sibiji";

fn server_url() -> String {
    std::env::var("NPCSH_SERVER_URL")
        .unwrap_or_else(|_| format!("http://{DEFAULT_HOST}:{DEFAULT_PORT}"))
}

fn resolve_npc_file(name: &str) -> Option<String> {
    if name.ends_with(".npc") {
        if std::path::Path::new(name).is_file() {
            return Some(name.to_string());
        }
        return None;
    }

    let team_dir = find_team_dir();
    let candidates = [
        format!("./{}.npc", name),
        format!("{}/{}.npc", team_dir, name),
        format!("{}/npc_team/{}.npc", team_dir, name),
    ];
    for candidate in &candidates {
        if std::path::Path::new(candidate).is_file() {
            return Some(candidate.clone());
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
    // npc mcp-server [team] [--npc name] — stdio MCP server for agent coders
    if args.len() > 1 && args[1] == "mcp-server" {
        let mut team_path: Option<String> = None;
        let mut npc_name: Option<String> = None;
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--npc" => {
                    npc_name = args.get(i + 1).cloned();
                    i += 2;
                }
                _ => {
                    if team_path.is_none() && !args[i].starts_with("--") {
                        team_path = Some(args[i].clone());
                    }
                    i += 1;
                }
            }
        }
        let tp = team_path.unwrap_or_else(find_team_dir);
        let mut server = npcsh::mcp_server::McpServer::load(&tp, npc_name.as_deref())?;
        return server.run_stdio().await;
    }

    // npc <agent-coder> [args...] — e.g. npc claude, npc codex --npc corca
    if let Some(first) = args.get(1)
        && npcsh::agent_coders::is_agent_coder(first)
    {
        let extra: Vec<String> = args[2..].to_vec();
        let (npc_name, extra): (Option<String>, Vec<String>) = {
            if let Some(pos) = extra.iter().position(|a| a == "--npc") {
                let npc = extra.get(pos + 1).cloned();
                let mut rest = extra.clone();
                let _ = rest.splice(pos..=(pos + 1).min(rest.len() - 1), []);
                (npc, rest)
            } else {
                (None, extra)
            }
        };
        return npcsh::agent_coders::run_agent_coder(first, None, npc_name.as_deref(), &extra);
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
        } else {
            let prompt = positional.join(" ");
            if prompt.is_empty() {
                eprintln!("Usage: npc <prompt> [-n NPC] [-m MODEL] [-pr PROVIDER]");
                eprintln!("       npc <file.npc|file.jinx|init> [args...]");
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
    std::process::exit(1);
}
