//! Agent-coder interception layer for npcsh.
//!
//! When a known agent coder (claude, codex, gemini, opencode, ...) is run
//! inside npcsh or via `npc <coder>`, this layer:
//!   1. loads the resolved npcsh team,
//!   2. shows an NPC picker (or takes `--npc <name>`),
//!   3. registers the NPC MCP server with the coder (native `mcp add` CLI or
//!      config write), and
//!   4. launches the coder so it discovers team jinxes as MCP tools.
//!
//! Attach metadata is tracked in ~/.npcsh/agent_coders.json.

use npcrs::error::{NpcError, Result};
use npcrs::npc_compiler::Team;
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Stdio;

pub const KNOWN_CODERS: &[&str] = &["claude", "codex", "gemini", "opencode", "amp", "aider"];

pub fn is_agent_coder(cmd: &str) -> bool {
    KNOWN_CODERS.contains(&cmd)
}

fn home_dir() -> PathBuf {
    // Prefer the npcsh-resolved user home (guards against sudo/HOME quirks),
    // falling back to $HOME.
    crate::real_user_home()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
        })
}

fn registry_path() -> PathBuf {
    crate::npcsh_home().join("agent_coders.json")
}

pub fn track_attach(tool: &str, npc: &str, team_path: &str) {
    let path = registry_path();
    let mut root: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    root.as_object_mut().unwrap().insert(
        tool.to_string(),
        serde_json::json!({
            "npc": npc,
            "team_path": team_path,
            "attached_at": now,
            "mcp_server": "npcsh mcp-server",
        }),
    );
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(
        &path,
        serde_json::to_string_pretty(&root).unwrap_or_default(),
    );
}

fn pick_npc(team: &Team) -> Result<String> {
    let npcs: Vec<&String> = team.npcs.keys().collect();
    if npcs.is_empty() {
        return Err(NpcError::Shell("No NPCs in team".into()));
    }
    if npcs.len() == 1 {
        return Ok(npcs[0].clone());
    }
    eprintln!("\nTeam NPCs:");
    for (i, name) in npcs.iter().enumerate() {
        let first = team
            .npcs
            .get(*name)
            .and_then(|n| n.primary_directive.as_deref())
            .and_then(|d| d.lines().next().map(|s| s.to_string()))
            .unwrap_or_default();
        let marker = if team.forenpc.as_deref() == Some(name.as_str()) {
            " (forenpc)"
        } else {
            ""
        };
        eprintln!("  {}) {}{} — {}", i + 1, name, marker, first);
    }
    eprint!("Pick an NPC for this agent coder [1-{}]: ", npcs.len());
    io::stderr().flush().ok();
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|e| NpcError::Shell(format!("read: {}", e)))?;
    let choice = input.trim();
    let default_idx = team
        .forenpc
        .as_ref()
        .and_then(|f| npcs.iter().position(|n| *n == f))
        .unwrap_or(0);
    let idx = if choice.is_empty() {
        default_idx
    } else {
        choice
            .parse::<usize>()
            .unwrap_or(default_idx + 1)
            .saturating_sub(1)
            .min(npcs.len() - 1)
    };
    Ok(npcs[idx].clone())
}

fn mcp_server_cmd(team_path: &str, npc: &str) -> String {
    format!("npcsh mcp-server {} --npc {}", team_path, npc)
}

/// Register the MCP server with the coder via its own CLI where supported.
fn register_via_cli(tool: &str, team_path: &str, npc: &str) -> bool {
    let server_cmd = mcp_server_cmd(team_path, npc);
    let status = match tool {
        "claude" | "codex" => std::process::Command::new(tool)
            .args(["mcp", "add", "npc", "--"])
            .args(server_cmd.split_whitespace())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status(),
        "gemini" => std::process::Command::new(tool)
            .args([
                "mcp",
                "add",
                "npc",
                "npcsh",
                "mcp-server",
                team_path,
                "--npc",
                npc,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status(),
        _ => return false,
    };
    let ok = status.map(|s| s.success()).unwrap_or(false);
    if !ok {
        eprintln!(
            "[npc] note: could not auto-register with {} — add an MCP server manually:\n  {}",
            tool, server_cmd
        );
    }
    true
}

fn register_opencode(team_path: &str, npc: &str) {
    let cwd = std::env::current_dir().unwrap_or_default();
    let config = cwd.join("opencode.json");
    let mut root: serde_json::Value = config
        .exists()
        .then(|| std::fs::read_to_string(&config).ok())
        .flatten()
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let mcp = root
        .as_object_mut()
        .unwrap()
        .entry("mcp".to_string())
        .or_insert(serde_json::json!({}));
    mcp.as_object_mut().unwrap().insert(
        "npc".to_string(),
        serde_json::json!({
            "type": "local",
            "command": ["npcsh", "mcp-server", team_path, "--npc", npc],
            "enabled": true,
        }),
    );
    let _ = std::fs::write(
        &config,
        serde_json::to_string_pretty(&root).unwrap_or_default(),
    );
    eprintln!("[npc] wrote MCP entry to {}", config.display());
}

pub fn run_agent_coder(
    tool: &str,
    team_path: Option<&str>,
    npc_name: Option<&str>,
    extra_args: &[String],
) -> Result<()> {
    let team_path = team_path
        .map(|s| s.to_string())
        .unwrap_or_else(crate::find_team_dir);
    let team = npcrs::npc_compiler::load_team_from_directory(&team_path)?;
    let npc = match npc_name {
        Some(n) => n.to_string(),
        None => pick_npc(&team)?,
    };
    eprintln!("[npc] launching {} as @{} (team: {})", tool, npc, team.name);

    match tool {
        "claude" | "codex" | "gemini" => {
            register_via_cli(tool, &team_path, &npc);
        }
        "opencode" => {
            register_opencode(&team_path, &npc);
        }
        _ => {
            eprintln!(
                "[npc] {} has no MCP registration path yet; server command:\n  {}",
                tool,
                mcp_server_cmd(&team_path, &npc)
            );
        }
    }
    track_attach(tool, &npc, &team_path);

    let status = std::process::Command::new(tool)
        .args(extra_args)
        .status()
        .map_err(|e| NpcError::Shell(format!("launch {}: {}", tool, e)))?;
    if !status.success() {
        return Err(NpcError::Shell(format!("{} exited with error", tool)));
    }
    Ok(())
}

pub fn installed_registry() -> HashMap<String, serde_json::Value> {
    std::fs::read_to_string(registry_path())
        .ok()
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_default()
}
