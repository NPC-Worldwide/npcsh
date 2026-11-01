//! Minimal stdio MCP server exposing an npcsh team's jinxes as MCP tools.
//!
//! Agent coders spawn it via their MCP config as
//! `npcsh mcp-server [team] [--npc name]`. Speaks MCP over JSON-RPC on
//! stdin/stdout: initialize, tools/list, tools/call.

use npcrs::error::Result;
use npcrs::npc_compiler::{Team, execute_jinx_with_npc};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

pub struct McpServer {
    team: Team,
    npc_name: String,
}

impl McpServer {
    pub fn load(team_path: &str, npc: Option<&str>) -> Result<Self> {
        let team = npcrs::npc_compiler::load_team_from_directory(team_path)?;
        let npc_name = match npc {
            Some(n) if team.npcs.contains_key(n) => n.to_string(),
            Some(n) => {
                eprintln!("[npc-mcp] npc '{}' not found, using forenpc", n);
                team.forenpc
                    .clone()
                    .or_else(|| team.npcs.keys().next().cloned())
                    .unwrap_or_default()
            }
            None => team
                .forenpc
                .clone()
                .or_else(|| team.npcs.keys().next().cloned())
                .unwrap_or_default(),
        };
        eprintln!(
            "[npc-mcp] team={} npc={} jinxes={}",
            team.name,
            npc_name,
            team.jinxes.len()
        );
        Ok(Self { team, npc_name })
    }

    pub async fn run_stdio(&mut self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let req: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let id = req.get("id").cloned();
            match method {
                "initialize" => {
                    self.send(&stdout, serde_json::json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": format!("npc-{}", self.team.name), "version": "0.1.0"}
                        }
                    }));
                }
                "notifications/initialized" | "initialized" => {}
                "tools/list" => {
                    let tools: Vec<serde_json::Value> = self
                        .team
                        .jinxes
                        .values()
                        .filter_map(|j| j.to_tool_def())
                        .map(|t| {
                            serde_json::json!({
                                "name": t.function.name,
                                "description": t.function.description,
                                "inputSchema": t.function.parameters,
                            })
                        })
                        .collect();
                    self.send(
                        &stdout,
                        serde_json::json!({"jsonrpc": "2.0", "id": id, "result": {"tools": tools}}),
                    );
                }
                "tools/call" => {
                    let name = req
                        .pointer("/params/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let args: HashMap<String, String> = req
                        .pointer("/params/arguments")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default();
                    let npc = self.team.npcs.get(&self.npc_name);
                    let out = match self.team.jinxes.get(name) {
                        Some(jinx) => {
                            match execute_jinx_with_npc(jinx, &args, &self.team.jinxes, npc, None)
                                .await
                            {
                                Ok(res) => {
                                    if res.success {
                                        res.output
                                    } else {
                                        format!("jinx error: {}", res.error.unwrap_or_default())
                                    }
                                }
                                Err(e) => format!("jinx error: {}", e),
                            }
                        }
                        None => format!("unknown tool: {}", name),
                    };
                    self.send(
                        &stdout,
                        serde_json::json!({
                            "jsonrpc": "2.0", "id": id,
                            "result": {"content": [{"type": "text", "text": out}]}
                        }),
                    );
                }
                _ => {
                    if let Some(id) = id {
                        self.send(&stdout, serde_json::json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "method not found"}}));
                    }
                }
            }
        }
        Ok(())
    }

    fn send(&self, stdout: &io::Stdout, msg: serde_json::Value) {
        let mut lock = stdout.lock();
        let _ = writeln!(lock, "{}", serde_json::to_string(&msg).unwrap_or_default());
        let _ = lock.flush();
    }
}
