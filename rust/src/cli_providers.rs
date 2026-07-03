use npcrs::r#gen::Usage;
use std::collections::HashMap;
use tokio::io::AsyncBufReadExt;

pub const CLI_PROVIDERS: &[&str] = &[
    "opencode", "claude_code", "claude", "codex", "amp",
    "gemini_cli", "aider", "kimi", "kimi_code", "kilo",
];

const TUI_PROVIDERS: &[&str] = &["opencode", "kilo"];

pub struct CliResult {
    pub text: String,
    pub usage: Option<Usage>,
    pub cost_usd: f64,
    pub session_id: Option<String>,
}

fn wrap_with_system(prompt: &str, system_prompt: &str, session_id: Option<&str>) -> String {
    if session_id.is_some() || system_prompt.is_empty() {
        prompt.to_string()
    } else {
        format!("<system>\n{}\n</system>\n\n{}", system_prompt, prompt)
    }
}

fn build_cli_cmd(
    provider: &str,
    model: &str,
    prompt: &str,
    system_prompt: &str,
    session_id: Option<&str>,
) -> Option<Vec<String>> {
    match provider {
        "claude_code" | "claude" => {
            let sid = session_id.map(str::to_owned).unwrap_or_else(|| uuid_v4());
            let mut cmd = vec![
                "claude".to_string(), "-p".to_string(), prompt.to_string(),
                "--output-format".to_string(), "stream-json".to_string(),
                "--verbose".to_string(),
                "--session-id".to_string(), sid,
            ];
            if !model.is_empty() {
                cmd.extend(["--model".to_string(), model.to_string()]);
            }
            if !system_prompt.is_empty() {
                cmd.extend(["--system-prompt".to_string(), system_prompt.to_string()]);
            }
            Some(cmd)
        }
        "opencode" => {
            let full = wrap_with_system(prompt, system_prompt, session_id);
            let mut cmd = vec![
                "opencode".to_string(), "run".to_string(), full,
                "--format".to_string(), "json".to_string(),
            ];
            if !model.is_empty() { cmd.extend(["-m".to_string(), model.to_string()]); }
            if let Some(sid) = session_id { cmd.extend(["-s".to_string(), sid.to_string()]); }
            Some(cmd)
        }
        "codex" => {
            let full = wrap_with_system(prompt, system_prompt, session_id);
            let mut cmd = match session_id {
                Some("last") => vec![
                    "codex".to_string(), "exec".to_string(), "resume".to_string(),
                    "--last".to_string(), "--json".to_string(), full,
                ],
                Some(sid) => vec![
                    "codex".to_string(), "exec".to_string(), "resume".to_string(),
                    sid.to_string(), "--json".to_string(), full,
                ],
                None => vec![
                    "codex".to_string(), "exec".to_string(),
                    "--json".to_string(), full,
                ],
            };
            if !model.is_empty() { cmd.extend(["--model".to_string(), model.to_string()]); }
            Some(cmd)
        }
        "kimi" | "kimi_code" => {
            let full = wrap_with_system(prompt, system_prompt, session_id);
            let mut cmd = vec![
                "kimi".to_string(), "--print".to_string(),
                "--output-format".to_string(), "text".to_string(),
                "-p".to_string(), full,
            ];
            if !model.is_empty() { cmd.extend(["-m".to_string(), model.to_string()]); }
            if let Some(sid) = session_id { cmd.extend(["-S".to_string(), sid.to_string()]); }
            Some(cmd)
        }
        "kilo" => {
            let full = wrap_with_system(prompt, system_prompt, session_id);
            let mut cmd = vec![
                "kilo".to_string(), "run".to_string(), full,
                "--format".to_string(), "json".to_string(),
            ];
            if !model.is_empty() { cmd.extend(["-m".to_string(), model.to_string()]); }
            if let Some(sid) = session_id { cmd.extend(["-s".to_string(), sid.to_string()]); }
            Some(cmd)
        }
        "gemini_cli" => {
            let full = wrap_with_system(prompt, system_prompt, session_id);
            Some(vec!["gemini".to_string(), "-p".to_string(), full])
        }
        "amp" => {
            let full = wrap_with_system(prompt, system_prompt, session_id);
            Some(vec!["amp".to_string(), "run".to_string(), full])
        }
        "aider" => {
            let full = wrap_with_system(prompt, system_prompt, session_id);
            Some(vec![
                "aider".to_string(), "--message".to_string(),
                full, "--no-pretty".to_string(),
            ])
        }
        _ => None,
    }
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}", t, t >> 16, t & 0x0fff, 0x8000 | (t & 0x3fff), t as u64)
}

struct ParserResult {
    text: String,
    usage: Option<Usage>,
    cost: f64,
    parsed_session_id: Option<String>,
}

struct ClaudeStreamParser {
    last_text: String,
    final_text: String,
    input: u64,
    output: u64,
    cost: f64,
    saw_event: bool,
}

impl ClaudeStreamParser {
    fn new() -> Self {
        Self { last_text: String::new(), final_text: String::new(), input: 0, output: 0, cost: 0.0, saw_event: false }
    }

    fn feed(&mut self, line: &str) -> String {
        let stripped = line.trim();
        if stripped.is_empty() { return String::new(); }
        let ev: serde_json::Value = match serde_json::from_str(stripped) {
            Ok(v) => v,
            Err(_) => return String::new(),
        };
        self.saw_event = true;
        match ev["type"].as_str() {
            Some("assistant") => {
                let mut new_text = String::new();
                if let Some(content) = ev["message"]["content"].as_array() {
                    for c in content {
                        if c["type"] == "text" {
                            new_text.push_str(c["text"].as_str().unwrap_or(""));
                        }
                    }
                }
                if new_text.is_empty() { return String::new(); }
                let delta = if new_text.starts_with(self.last_text.as_str()) {
                    new_text[self.last_text.len()..].to_string()
                } else {
                    new_text.clone()
                };
                self.last_text = new_text.clone();
                self.final_text = new_text;
                delta
            }
            Some("result") => {
                let u = &ev["usage"];
                self.input = u["input_tokens"].as_u64().unwrap_or(0);
                self.output = u["output_tokens"].as_u64().unwrap_or(0);
                self.cost = ev["total_cost_usd"].as_f64().unwrap_or(0.0);
                String::new()
            }
            _ => String::new(),
        }
    }

    fn finalize(self) -> ParserResult {
        let total = self.input + self.output;
        let usage = if self.input > 0 || self.output > 0 {
            Some(Usage { prompt_tokens: self.input, completion_tokens: self.output, total_tokens: total })
        } else { None };
        ParserResult { text: self.final_text, usage, cost: self.cost, parsed_session_id: None }
    }
}

struct OpencodeStreamParser {
    parts: Vec<String>,
    input: u64,
    output: u64,
    cost: f64,
}

impl OpencodeStreamParser {
    fn new() -> Self { Self { parts: Vec::new(), input: 0, output: 0, cost: 0.0 } }

    fn feed(&mut self, line: &str) -> String {
        let stripped = line.trim();
        if stripped.is_empty() { return String::new(); }
        let ev: serde_json::Value = match serde_json::from_str(stripped) {
            Ok(v) => v,
            Err(_) => return String::new(),
        };
        match ev["type"].as_str() {
            Some("text") => {
                if let Some(t) = ev["part"]["text"].as_str() {
                    if !t.is_empty() { self.parts.push(t.to_string()); return t.to_string(); }
                }
                String::new()
            }
            Some("step_finish") => {
                let tokens = &ev["part"]["tokens"];
                self.input += tokens["input"].as_u64().unwrap_or(0);
                self.output += tokens["output"].as_u64().unwrap_or(0);
                self.cost += ev["part"]["cost"].as_f64().unwrap_or(0.0);
                String::new()
            }
            _ => String::new(),
        }
    }

    fn finalize(self) -> ParserResult {
        let total = self.input + self.output;
        let usage = if self.input > 0 || self.output > 0 {
            Some(Usage { prompt_tokens: self.input, completion_tokens: self.output, total_tokens: total })
        } else { None };
        ParserResult { text: self.parts.join(""), usage, cost: self.cost, parsed_session_id: None }
    }
}

struct CodexStreamParser {
    parts: Vec<String>,
    input: u64,
    output: u64,
    thread_id: Option<String>,
}

impl CodexStreamParser {
    fn new() -> Self { Self { parts: Vec::new(), input: 0, output: 0, thread_id: None } }

    fn feed(&mut self, line: &str) -> String {
        let stripped = line.trim();
        if stripped.is_empty() { return String::new(); }
        let ev: serde_json::Value = match serde_json::from_str(stripped) {
            Ok(v) => v,
            Err(_) => return String::new(),
        };
        match ev["type"].as_str() {
            Some("thread.started") => { self.thread_id = ev["thread_id"].as_str().map(str::to_owned); }
            Some("turn.completed") => {
                let u = &ev["usage"];
                self.input = u["input_tokens"].as_u64().unwrap_or(0);
                self.output = u["output_tokens"].as_u64().unwrap_or(0);
            }
            Some("item.completed") => {
                let item = &ev["item"];
                if item["type"] == "agent_message" {
                    if let Some(t) = item["text"].as_str() { if !t.is_empty() { self.parts.push(t.to_string()); } }
                }
            }
            _ => {}
        }
        String::new()
    }

    fn finalize(self) -> ParserResult {
        let total = self.input + self.output;
        let usage = if self.input > 0 || self.output > 0 {
            Some(Usage { prompt_tokens: self.input, completion_tokens: self.output, total_tokens: total })
        } else { None };
        ParserResult { text: self.parts.join(""), usage, cost: 0.0, parsed_session_id: self.thread_id }
    }
}

struct KimiStreamParser {
    parts: Vec<String>,
}

impl KimiStreamParser {
    fn new() -> Self { Self { parts: Vec::new() } }

    fn feed(&mut self, line: &str) -> String {
        self.parts.push(format!("{}\n", line));
        if line.trim().is_empty() { String::new() } else { format!("{}\n", line) }
    }

    fn finalize(self) -> ParserResult {
        ParserResult { text: self.parts.join("").trim().to_string(), usage: None, cost: 0.0, parsed_session_id: None }
    }
}

enum CliStreamParser {
    Claude(ClaudeStreamParser),
    Opencode(OpencodeStreamParser),
    Codex(CodexStreamParser),
    Kimi(KimiStreamParser),
}

impl CliStreamParser {
    fn for_provider(provider: &str) -> Option<Self> {
        match provider {
            "claude" | "claude_code" => Some(Self::Claude(ClaudeStreamParser::new())),
            "opencode" | "kilo" => Some(Self::Opencode(OpencodeStreamParser::new())),
            "codex" => Some(Self::Codex(CodexStreamParser::new())),
            "kimi" | "kimi_code" => Some(Self::Kimi(KimiStreamParser::new())),
            _ => None,
        }
    }

    fn feed(&mut self, line: &str) -> String {
        match self {
            Self::Claude(p) => p.feed(line),
            Self::Opencode(p) => p.feed(line),
            Self::Codex(p) => p.feed(line),
            Self::Kimi(p) => p.feed(line),
        }
    }

    fn finalize(self) -> ParserResult {
        match self {
            Self::Claude(p) => p.finalize(),
            Self::Opencode(p) => p.finalize(),
            Self::Codex(p) => p.finalize(),
            Self::Kimi(p) => p.finalize(),
        }
    }

    fn saw_any_claude_event(&self) -> bool {
        match self { Self::Claude(p) => p.saw_event, _ => false }
    }
}

fn resolve_session_id(provider: &str, parser_result: &ParserResult, pre_assigned_sid: Option<String>) -> Option<String> {
    match provider {
        "claude" | "claude_code" => pre_assigned_sid,
        "opencode" => fetch_opencode_session_id(),
        "kilo" => fetch_kilo_session_id(),
        "codex" => parser_result.parsed_session_id.clone().or_else(|| Some("last".to_string())),
        "kimi" | "kimi_code" => fetch_kimi_session_id(),
        _ => None,
    }
}

fn parse_claude_output(raw: &str) -> (String, Option<Usage>, f64) {
    let mut p = ClaudeStreamParser::new();
    for line in raw.lines() { p.feed(line); }
    let saw_any = p.saw_event;
    let r = p.finalize();
    if saw_any { return (r.text, r.usage, r.cost); }
    if let Ok(data) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
        let text = data["result"].as_str().unwrap_or("").to_string();
        let input = data["usage"]["input_tokens"].as_u64().unwrap_or(0);
        let output = data["usage"]["output_tokens"].as_u64().unwrap_or(0);
        let cost = data["total_cost_usd"].as_f64().unwrap_or(0.0);
        let usage = if input > 0 || output > 0 { Some(Usage { prompt_tokens: input, completion_tokens: output, total_tokens: input + output }) } else { None };
        return (text, usage, cost);
    }
    (r.text, r.usage, r.cost)
}

fn parse_opencode_output(raw: &str) -> (String, Option<Usage>, f64) {
    let mut p = OpencodeStreamParser::new();
    for line in raw.lines() { p.feed(line); }
    let r = p.finalize();
    (r.text, r.usage, r.cost)
}

fn parse_codex_output(raw: &str) -> (String, Option<Usage>, Option<String>) {
    let mut p = CodexStreamParser::new();
    for line in raw.lines() { p.feed(line); }
    let r = p.finalize();
    (r.text, r.usage, r.parsed_session_id)
}

fn fetch_opencode_session_id() -> Option<String> {
    let out = std::process::Command::new("opencode").args(["session", "list"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(first) = line.split_whitespace().next() {
            if first.starts_with("ses_") { return Some(first.to_string()); }
        }
    }
    None
}

fn fetch_kilo_session_id() -> Option<String> {
    let out = std::process::Command::new("kilo").args(["session", "list"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(first) = line.split_whitespace().next() {
            if first.starts_with("ses_") { return Some(first.to_string()); }
        }
    }
    None
}

fn fetch_kimi_session_id() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let cwd_str = cwd.to_string_lossy();
    let digest = md5_hex(cwd_str.as_bytes());
    let base = dirs_base(&digest);
    let base_short = dirs_base(&digest[..8]);
    let sessions_dir = if std::path::Path::new(&base).is_dir() { base } else { base_short };
    if !std::path::Path::new(&sessions_dir).is_dir() { return None; }
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for entry in std::fs::read_dir(&sessions_dir).ok()?.flatten() {
        if entry.path().is_dir() {
            if let Ok(meta) = entry.metadata() {
                if let Ok(mtime) = meta.modified() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) { best = Some((mtime, name)); }
                }
            }
        }
    }
    best.map(|(_, name)| name)
}

fn dirs_base(hash: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    format!("{}/.kimi/sessions/{}", home, hash)
}

fn md5_hex(data: &[u8]) -> String {
    let out = std::process::Command::new("python3")
        .args(["-c", &format!(
            "import hashlib,sys; sys.stdout.write(hashlib.md5({:?}.encode()).hexdigest())",
            std::str::from_utf8(data).unwrap_or("")
        )])
        .output();
    if let Ok(o) = out { return String::from_utf8_lossy(&o.stdout).to_string(); }
    String::new()
}

pub async fn run_cli_provider(
    provider: &str,
    model: &str,
    prompt: &str,
    system_prompt: &str,
    session_id: Option<&str>,
) -> Option<CliResult> {
    let cmd = build_cli_cmd(provider, model, prompt, system_prompt, session_id)?;

    let mut env: HashMap<String, String> = std::env::vars().collect();
    if TUI_PROVIDERS.contains(&provider) {
        env.insert("TERM".to_string(), "dumb".to_string());
    }

    let mut child = tokio::process::Command::new(&cmd[0])
        .args(&cmd[1..])
        .envs(&env)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;

    let stdout = child.stdout.take()?;
    let mut reader = tokio::io::BufReader::new(stdout).lines();

    let mut output_lines: Vec<String> = Vec::new();
    let mut parser = CliStreamParser::for_provider(provider);

    while let Ok(Some(line)) = reader.next_line().await {
        output_lines.push(line.clone() + "\n");
        if let Some(p) = parser.as_mut() { let _ = p.feed(&line); }
    }

    let _ = child.wait().await;

    let full_output = output_lines.join("");
    let pre_assigned_sid: Option<String> = cmd.iter()
        .position(|a| a == "--session-id")
        .and_then(|i| cmd.get(i + 1).cloned());

    let (response_text, usage, cost, new_session_id) = if let Some(p) = parser {
        let r = p.finalize();
        let sid = resolve_session_id(provider, &r, pre_assigned_sid);
        (r.text, r.usage, r.cost, sid)
    } else {
        (full_output.trim().to_string(), None, 0.0, None)
    };

    if !response_text.is_empty() { println!("{}", response_text); }

    Some(CliResult {
        text: response_text,
        usage,
        cost_usd: cost,
        session_id: new_session_id,
    })
}
