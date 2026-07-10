use npcrs::error::Result;
use npcsh::{exec_jinx_file, exec_npc_file, find_team_dir, init_team, resolve_team_layout};

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
            let npc_name = override_npc.as_deref().unwrap_or(DEFAULT_NPC);
            let npc_file = resolve_npc_file(npc_name).unwrap_or_else(|| {
                eprintln!("Error: could not find NPC file for '{}'", npc_name);
                std::process::exit(1);
            });
            let client = reqwest::Client::new();
            return exec_npc_file(
                &npc_file,
                Some(&prompt),
                &client,
                &server_url(),
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
