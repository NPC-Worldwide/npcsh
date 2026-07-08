//! `npc` — dedicated NPC CLI binary.
//!
//! Executes `.npc` files, `.jinx` files, and `npc init`. It contains none of
//! the `npcsh` shell/REPL/TUI code and depends on the `npcsh` library for the
//! shared implementation.

use npcrs::error::Result;
use npcsh::{exec_jinx_file, exec_npc_file, init_team};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "5237";

fn server_url() -> String {
    std::env::var("NPCSH_SERVER_URL")
        .unwrap_or_else(|_| format!("http://{DEFAULT_HOST}:{DEFAULT_PORT}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let args: Vec<String> = std::env::args().collect();
    if let Some(file) = args.get(1) {
        if file == "init" {
            let dir = args.get(2).map(|s| s.as_str()).unwrap_or(".");
            return init_team(dir);
        } else if file.ends_with(".jinx") {
            let jinx_args: Vec<&str> = args[2..].iter().map(|s| s.as_str()).collect();
            return exec_jinx_file(file, &jinx_args).await;
        } else if file.ends_with(".npc") {
            let client = reqwest::Client::new();
            return exec_npc_file(file, args.get(2).map(|s| s.as_str()), &client, &server_url()).await;
        }
    }

    eprintln!("Usage: npc <file.npc|file.jinx|init> [args...]");
    std::process::exit(1);
}
