//! Agent-coder interception layer for npcsh.
//!
//! When a known agent coder (claude, codex, gemini, opencode, ...) is run
//! inside npcsh or via `npc <coder>`, this layer:
//!   1. loads the resolved npcsh team,
//!   2. shows an NPC picker (or takes `--npc <name>`),
//!   3. registers the NPC MCP server (`npcsh mcp-server`) with the coder's
//!      own MCP config (via its CLI: `claude mcp add`, `codex mcp add`, ...),
//!   4. injects the persona via the coder's CLI arguments / env, and
//!   5. launches the coder.
//!
//! No files are written in the project directory. The persona is delivered as
//! CLI args (claude: --append-system-prompt/--agents; codex: -c instructions=;
//! gemini: GEMINI_SYSTEM_MD tempfile env) and jinxes arrive through MCP.
//! Attach metadata is tracked in ~/.npcsh/agent_coders.json.

use npcrs::error::{NpcError, Result};
use npcrs::npc_compiler::Team;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Stdio;

pub const KNOWN_CODERS: &[&str] = &["claude", "codex", "gemini", "opencode", "amp", "aider"];

pub fn is_agent_coder(cmd: &str) -> bool {
    KNOWN_CODERS.contains(&cmd)
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

/// The stdio command a coder should run to get the NPC MCP server.
fn mcp_server_cmd(team_path: &str, npc: &str) -> String {
    format!("npcsh mcp-server {} --npc {}", team_path, npc)
}

/// Register the MCP server through the coder's own CLI (writes into the
/// coder's config store, never into the project).
fn register_via_cli(tool: &str, team_path: &str, npc: &str) -> bool {
    let status = match tool {
        "claude" | "codex" => std::process::Command::new(tool)
            .args([
                "mcp",
                "add",
                "npc",
                "--",
                "npcsh",
                "mcp-server",
                team_path,
                "--npc",
                npc,
            ])
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
            "[npc] note: could not auto-register with {} — add this MCP server in its config:\n  {}",
            tool,
            mcp_server_cmd(team_path, npc)
        );
    }
    true
}

/// Persona as a plain prompt string: directive + tools + team roster + MCP note.
fn persona_prompt(npc_name: &str, team: &Team) -> String {
    let npc = team.npcs.get(npc_name);
    let directive = npc
        .and_then(|n| n.primary_directive.as_deref())
        .unwrap_or("")
        .trim();
    let tools: Vec<&str> = npc
        .map(|n| n.jinx_names.iter().map(|s| s.as_str()).collect())
        .unwrap_or_default();
    let mut out = format!("You are {}.\n\n{}", npc_name, directive);
    if !tools.is_empty() {
        out.push_str(&format!(
            "\n\nYour tools (via the `npc` MCP server): {}",
            tools.join(", ")
        ));
    }
    let others: Vec<&str> = team
        .npcs
        .keys()
        .filter(|k| k.as_str() != npc_name)
        .map(|k| k.as_str())
        .collect();
    if !others.is_empty() {
        out.push_str("\n\nOther NPCs on the team:");
        for o in others {
            let first = team
                .npcs
                .get(o)
                .and_then(|n| n.primary_directive.as_deref())
                .and_then(|d| d.lines().next())
                .unwrap_or("");
            out.push_str(&format!("\n  @{}: {}", o, first));
        }
    }
    out.push_str("\n");
    out
}

/// Claude `--agents` JSON: every team NPC as a subagent definition.
fn roster_agents_json(team: &Team) -> String {
    let mut map = serde_json::Map::new();
    for (name, npc) in &team.npcs {
        let first = npc
            .primary_directive
            .as_deref()
            .and_then(|d| d.lines().next())
            .unwrap_or("")
            .to_string();
        map.insert(
            name.clone(),
            serde_json::json!({"description": first, "prompt": npc.primary_directive.clone().unwrap_or_default()}),
        );
    }
    serde_json::Value::Object(map).to_string()
}

/// Escape for use inside a TOML basic string (codex -c instructions="...").
fn toml_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
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

    // MCP registration first, so the server is live in the coder's config
    // before the UI boots.
    if matches!(tool, "claude" | "codex" | "gemini") {
        register_via_cli(tool, &team_path, &npc);
    } else {
        eprintln!(
            "[npc] {} has no MCP CLI add path — register manually if it supports MCP:\n  {}",
            tool,
            mcp_server_cmd(&team_path, &npc)
        );
    }
    track_attach(tool, &npc, &team_path);

    let prompt = persona_prompt(&npc, &team);
    let mut cmd = std::process::Command::new(tool);
    let mut gemini_tmp: Option<PathBuf> = None;

    match tool {
        "claude" => {
            cmd.arg("--append-system-prompt")
                .arg(&prompt)
                .arg("--agents")
                .arg(roster_agents_json(&team));
        }
        "codex" => {
            cmd.arg("-c")
                .arg(format!("instructions=\"{}\"", toml_escape(&prompt)));
        }
        "gemini" => {
            // GEMINI_SYSTEM_MD expects a file path; use a session tempfile and
            // delete after the run (no project pollution).
            let tmp = std::env::temp_dir().join(format!(
                "npcsh_gemini_{}_{}.md",
                std::process::id(),
                npc
            ));
            std::fs::write(&tmp, &prompt)
                .map_err(|e| NpcError::Shell(format!("tempfile: {}", e)))?;
            cmd.env("GEMINI_SYSTEM_MD", &tmp);
            gemini_tmp = Some(tmp);
        }
        _ => {
            eprintln!(
                "[npc] {} has no known system-prompt CLI flag; launching without persona injection — the coder can still adopt the persona via the `npc` MCP server.",
                tool
            );
        }
    }

    cmd.args(extra_args);
    let status = cmd
        .status()
        .map_err(|e| NpcError::Shell(format!("launch {}: {}", tool, e)))?;

    if let Some(tmp) = gemini_tmp {
        let _ = std::fs::remove_file(&tmp);
    }

    if !status.success() {
        return Err(NpcError::Shell(format!("{} exited with error", tool)));
    }
    Ok(())
}

pub fn installed_registry() -> serde_json::Map<String, serde_json::Value> {
    std::fs::read_to_string(registry_path())
        .ok()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}
