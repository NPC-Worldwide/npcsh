use crate::cron::{parse_duration, CronJobKind, CronRegistry};
use crate::tui::{clear_all, footer_line, header_line, hide_cursor, term_size, wline, RawModeGuard};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq)]
enum Step {
    Welcome,
    Model,
    Provider,
    ApiKeys,
    Heartbeats,
    Skills,
    Finish,
}

pub fn run_tutorial_tui(registry: &Arc<Mutex<CronRegistry>>, kernel: &mut npcrs::kernel::Kernel) -> crate::tui::Result<()> {
    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();
    hide_cursor(&mut out);

    let mut step = Step::Welcome;
    let mut model = std::env::var("NPCSH_CHAT_MODEL").unwrap_or_default();
    let mut provider = std::env::var("NPCSH_CHAT_PROVIDER").unwrap_or_default();
    let mut edit_buf = String::new();
    let mut cursor = 0usize;
    let mut editing = false;
    let mut edit_label = "";

    let mut heartbeats: Vec<(String, String, String)> = Vec::new();
    let mut hb_sel: usize = 0;
    let mut hb_field = 0;

    let mut skills: HashMap<String, bool> = [
        ("git-workflow".to_string(), false),
        ("npcpy-prompting".to_string(), false),
        ("technical-writing".to_string(), false),
        ("weak-writing-generator".to_string(), false),
    ]
    .into_iter()
    .collect();

    let mut state_path = PathBuf::from(shellexpand::tilde("~/.npcsh").to_string());
    state_path.push("tutorial_state.yaml");

    loop {
        let (cols, rows) = term_size();
        clear_all(&mut out);
        header_line(&mut out, cols, " NPCSH Tutorial ");
        hr(&mut out, cols, 2);

        let body_start = 4;
        let body_end = rows - 3;

        match step {
            Step::Welcome => render_welcome(&mut out, cols, body_start, body_end),
            Step::Model => render_model(&mut out, cols, body_start, body_end, &model, &provider, editing, &edit_buf, cursor),
            Step::Provider => render_provider(&mut out, cols, body_start, body_end, &provider, editing, &edit_buf, cursor),
            Step::ApiKeys => render_api_keys(&mut out, cols, body_start, body_end, editing, &edit_label, &edit_buf, cursor),
            Step::Heartbeats => render_heartbeats(&mut out, cols, body_start, body_end, &heartbeats, hb_sel, hb_field, editing, &edit_buf, cursor),
            Step::Skills => render_skills(&mut out, cols, body_start, body_end, &skills),
            Step::Finish => render_finish(&mut out, cols, body_start, body_end, &model, &provider, &heartbeats, &skills),
        }

        hr(&mut out, cols, rows - 2);
        let foot = match step {
            Step::Welcome => " [Enter] Start  [q] Quit ",
            Step::Finish => " [Enter] Save & Finish  [b] Back  [q] Quit ",
            _ => {
                if editing { " [Enter] Confirm  [Esc] Cancel " }
                else { " [j/k] Nav  [e] Edit  [Enter] Next  [b] Back  [q] Quit " }
            }
        };
        footer_line(&mut out, cols, rows, foot);
        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            let c = key.code;

            if editing {
                match c {
                    KeyCode::Esc | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => {
                        editing = false; edit_buf.clear(); cursor = 0;
                    }
                    KeyCode::Enter => {
                        match step {
                            Step::Model => model = edit_buf.trim().to_string(),
                            Step::Provider => provider = edit_buf.trim().to_string(),
                            Step::ApiKeys => {
                                if !edit_label.is_empty() {
                                    if edit_buf.trim().is_empty() {
                                        unsafe { std::env::remove_var(&edit_label); }
                                    } else {
                                        unsafe { std::env::set_var(&edit_label, edit_buf.trim()); }
                                    }
                                }
                            }
                            Step::Heartbeats => {
                                if let Some(hb) = heartbeats.get_mut(hb_sel) {
                                    match hb_field {
                                        0 => hb.0 = edit_buf.trim().to_string(),
                                        1 => hb.1 = edit_buf.trim().to_string(),
                                        2 => hb.2 = edit_buf.trim().to_string(),
                                        _ => {}
                                    }
                                }
                            }
                            _ => {}
                        }
                        editing = false; edit_buf.clear(); cursor = 0;
                    }
                    KeyCode::Backspace => { if cursor > 0 { cursor -= 1; edit_buf.remove(cursor); } }
                    KeyCode::Delete => { if cursor < edit_buf.len() { edit_buf.remove(cursor); } }
                    KeyCode::Left => { if cursor > 0 { cursor -= 1; } }
                    KeyCode::Right => { if cursor < edit_buf.len() { cursor += 1; } }
                    KeyCode::Home => { cursor = 0; }
                    KeyCode::End => { cursor = edit_buf.len(); }
                    KeyCode::Char(ch) => { if key.modifiers == KeyModifiers::CONTROL && ch == 'c' { break; } edit_buf.insert(cursor, ch); cursor += 1; }
                    _ => {}
                }
                continue;
            }

            match c {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Char('q') => break,
                KeyCode::Enter => {
                    match step {
                        Step::Welcome => step = Step::Model,
                        Step::Model => step = Step::Provider,
                        Step::Provider => step = Step::ApiKeys,
                        Step::ApiKeys => step = Step::Heartbeats,
                        Step::Heartbeats => step = Step::Skills,
                        Step::Skills => step = Step::Finish,
                        Step::Finish => {
                            finish_tutorial(&model, &provider, &heartbeats, &skills, &state_path, registry, kernel);
                            break;
                        }
                    }
                }
                KeyCode::Char('b') | KeyCode::Esc if step != Step::Welcome => {
                    step = match step {
                        Step::Model => Step::Welcome,
                        Step::Provider => Step::Model,
                        Step::ApiKeys => Step::Provider,
                        Step::Heartbeats => Step::ApiKeys,
                        Step::Skills => Step::Heartbeats,
                        Step::Finish => Step::Skills,
                        _ => step,
                    };
                }
                KeyCode::Char('e') | KeyCode::Enter => {
                    match step {
                        Step::Model => { editing = true; edit_buf = model.clone(); cursor = edit_buf.len(); }
                        Step::Provider => { editing = true; edit_buf = provider.clone(); cursor = edit_buf.len(); }
                        Step::ApiKeys => {
                            editing = true;
                            edit_label = pick_api_key(hb_sel);
                            edit_buf = std::env::var(&edit_label).unwrap_or_default();
                            cursor = edit_buf.len();
                        }
                        Step::Heartbeats => {
                            editing = true;
                            edit_buf = match hb_field {
                                0 => heartbeats.get(hb_sel).map(|h| h.0.clone()).unwrap_or_default(),
                                1 => heartbeats.get(hb_sel).map(|h| h.1.clone()).unwrap_or_default(),
                                2 => heartbeats.get(hb_sel).map(|h| h.2.clone()).unwrap_or_default(),
                                _ => String::new(),
                            };
                            cursor = edit_buf.len();
                        }
                        _ => {}
                    }
                }
                KeyCode::Char('j') | KeyCode::Down => {
                    match step {
                        Step::ApiKeys => { if hb_sel < 3 { hb_sel += 1; } }
                        Step::Heartbeats => {
                            hb_field += 1;
                            if hb_field > 2 { hb_field = 0; hb_sel = (hb_sel + 1).min(heartbeats.len().saturating_sub(1).max(0)); }
                        }
                        Step::Skills => {
                            let mut keys: Vec<String> = skills.keys().cloned().collect();
                            keys.sort();
                            if let Some(idx) = keys.iter().position(|k| *skills.get(k).unwrap_or(&false)) {
                            }
                        }
                        _ => {}
                    }
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    match step {
                        Step::ApiKeys => { if hb_sel > 0 { hb_sel -= 1; } }
                        Step::Heartbeats => {
                            if hb_field == 0 {
                                if hb_sel > 0 { hb_sel -= 1; hb_field = 2; }
                            } else {
                                hb_field -= 1;
                            }
                        }
                        _ => {}
                    }
                }
                KeyCode::Char('a') if step == Step::Heartbeats => {
                    heartbeats.push(("sibiji".to_string(), "30s".to_string(), "check for new emails and summarize".to_string()));
                    hb_sel = heartbeats.len() - 1; hb_field = 0;
                }
                KeyCode::Char('d') if step == Step::Heartbeats => {
                    if !heartbeats.is_empty() && hb_sel < heartbeats.len() {
                        heartbeats.remove(hb_sel);
                        if hb_sel > 0 && hb_sel >= heartbeats.len() { hb_sel -= 1; }
                    }
                }
                KeyCode::Char(' ') if step == Step::Skills => {
                    let mut keys: Vec<String> = skills.keys().cloned().collect();
                    keys.sort();
                    if let Some(k) = keys.get(hb_sel) {
                        let v = skills.get(k).copied().unwrap_or(false);
                        skills.insert(k.clone(), !v);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn render_welcome(out: &mut io::Stdout, cols: usize, start: usize, end: usize) {
    let lines = [
        "Welcome to npcsh!",
        "",
        "This tutorial will help you set up:",
        "  - Default chat model and provider",
        "  - API keys for cloud providers",
        "  - Recurring agent heartbeats / cron tasks",
        "  - Useful skills",
        "",
        "Everything is stored as plain-text YAML under ~/.npcsh",
        "so you can edit it later with any editor.",
        "",
        "Press Enter to begin.",
    ];
    for (i, line) in lines.iter().enumerate() {
        if start + i > end { break; }
        wline(out, start + i, &format!("  {}", line.chars().take(cols - 4).collect::<String>()));
    }
}

fn render_model(out: &mut io::Stdout, cols: usize, start: usize, _end: usize, model: &str, provider: &str, editing: bool, buf: &str, cursor: usize) {
    wline(out, start, "  Default chat model");
    wline(out, start + 1, &format!("  Active model: {} ({})", model, provider));
    wline(out, start + 3, "  Examples: qwen3.5:9b, gpt-4o, claude-3-5-sonnet-20241022");
    if editing {
        wline(out, start + 5, "  Edit model:");
        let before = &buf[..cursor.min(buf.len())];
        let cur = buf.chars().nth(cursor).map(|c| c.to_string()).unwrap_or_else(|| " ".to_string());
        let after = &buf[cursor.min(buf.len())..];
        wline(out, start + 6, &format!("  {}{}{}", before, cur, after));
    } else {
        wline(out, start + 5, "  [e] Edit model");
    }
}

fn render_provider(out: &mut io::Stdout, cols: usize, start: usize, _end: usize, provider: &str, editing: bool, buf: &str, cursor: usize) {
    wline(out, start, "  Provider");
    wline(out, start + 1, &format!("  Active provider: {}", provider));
    wline(out, start + 3, "  Examples: ollama, openai, anthropic, gemini, deepseek, lmstudio");
    if editing {
        wline(out, start + 5, "  Edit provider:");
        let before = &buf[..cursor.min(buf.len())];
        let cur = buf.chars().nth(cursor).map(|c| c.to_string()).unwrap_or_else(|| " ".to_string());
        let after = &buf[cursor.min(buf.len())..];
        wline(out, start + 6, &format!("  {}{}{}", before, cur, after));
    } else {
        wline(out, start + 5, "  [e] Edit provider");
    }
}

fn render_api_keys(out: &mut io::Stdout, cols: usize, start: usize, _end: usize, editing: bool, label: &str, buf: &str, cursor: usize) {
    wline(out, start, "  API Keys");
    let keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"];
    for (i, key) in keys.iter().enumerate() {
        let present = std::env::var(key).is_ok();
        let icon = if present { "\x1b[32m✓\x1b[0m" } else { "\x1b[31m✗\x1b[0m" };
        wline(out, start + 2 + i, &format!("  {} {}", icon, key));
    }
    if editing {
        wline(out, start + 7, &format!("  Edit {}:", label));
        let masked = "*".repeat(buf.len());
        let before = &masked[..cursor.min(masked.len())];
        let cur = "*";
        let after = &masked[cursor.min(masked.len())..];
        wline(out, start + 8, &format!("  {}{}{}", before, cur, after));
    } else {
        wline(out, start + 7, "  [j/k] Select  [e] Edit  [Enter] Next");
    }
}

fn render_heartbeats(out: &mut io::Stdout, cols: usize, start: usize, end: usize, heartbeats: &[(String, String, String)], sel: usize, field: usize, editing: bool, buf: &str, cursor: usize) {
    wline(out, start, "  Agent Heartbeats / Cron Tasks");
    wline(out, start + 1, "  Recurring tasks run in the background and report to the shell.");
    if heartbeats.is_empty() {
        wline(out, start + 3, "  \x1b[90mNo heartbeats configured. Press [a] to add one.\x1b[0m");
    } else {
        for (i, (npc, interval, task)) in heartbeats.iter().enumerate() {
            let marker = if i == sel { ">" } else { " " };
            let (f0, f1, f2) = if i == sel {
                match field { 0 => ("[npc]", " interval", " task"), 1 => (" npc", "[interval]", " task"), _ => (" npc", " interval", "[task]"), }
            } else { (" npc", " interval", " task") };
            wline(out, start + 3 + i, &format!("  {} {}: {} {}: {} {}: {}", marker, f0, npc, f1, interval, f2, task.chars().take(cols - 45).collect::<String>()));
        }
    }
    if editing {
        let label = match field { 0 => "NPC", 1 => "interval", _ => "task" };
        wline(out, end - 2, &format!("  Edit {}:", label));
        let before = &buf[..cursor.min(buf.len())];
        let cur = buf.chars().nth(cursor).map(|c| c.to_string()).unwrap_or_else(|| " ".to_string());
        let after = &buf[cursor.min(buf.len())..];
        wline(out, end - 1, &format!("  {}{}{}", before, cur, after));
    } else {
        wline(out, end - 1, "  [a] Add  [d] Delete  [e] Edit  [j/k] Navigate");
    }
}

fn render_skills(out: &mut io::Stdout, cols: usize, start: usize, _end: usize, skills: &HashMap<String, bool>) {
    wline(out, start, "  Skills");
    wline(out, start + 1, "  Toggle skills to install as jinxes in your team.");
    let mut keys: Vec<String> = skills.keys().cloned().collect();
    keys.sort();
    for (i, k) in keys.iter().enumerate() {
        let enabled = skills.get(k).copied().unwrap_or(false);
        let icon = if enabled { "\x1b[32m[✓]\x1b[0m" } else { "\x1b[90m[ ]\x1b[0m" };
        wline(out, start + 3 + i, &format!("  {} {}", icon, k));
    }
    wline(out, start + 3 + keys.len() + 1, "  [Space] Toggle  [Enter] Next");
}

fn render_finish(out: &mut io::Stdout, cols: usize, start: usize, _end: usize, model: &str, provider: &str, heartbeats: &[(String, String, String)], skills: &HashMap<String, bool>) {
    wline(out, start, "  Summary");
    wline(out, start + 2, &format!("  Model: {} / {}", model, provider));
    wline(out, start + 3, &format!("  Heartbeats: {}", heartbeats.len()));
    for (i, (npc, interval, task)) in heartbeats.iter().enumerate() {
        wline(out, start + 4 + i, &format!("    - @{npc} every {interval}: {task}", task = task.chars().take(cols - 35).collect::<String>()));
    }
    let enabled_skills: Vec<String> = skills.iter().filter(|(_, v)| **v).map(|(k, _)| k.clone()).collect();
    wline(out, start + 5 + heartbeats.len(), &format!("  Skills: {}", if enabled_skills.is_empty() { "none".to_string() } else { enabled_skills.join(", ") }));
    wline(out, start + 7 + heartbeats.len(), "  [Enter] Save everything to ~/.npcsh and finish");
}

fn pick_api_key(idx: usize) -> &'static str {
    ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"][idx.min(3)]
}

fn hr(out: &mut io::Stdout, cols: usize, row: usize) {
    let _ = write!(out, "\x1b[{};1H\x1b[K\x1b[90m{}\x1b[0m", row, "─".repeat(cols));
}

fn finish_tutorial(
    model: &str,
    provider: &str,
    heartbeats: &[(String, String, String)],
    skills: &HashMap<String, bool>,
    state_path: &PathBuf,
    registry: &Arc<Mutex<CronRegistry>>,
    kernel: &mut npcrs::kernel::Kernel,
) {
    let rc_path = shellexpand::tilde("~/.npcshrc").to_string();
    let mut lines: Vec<String> = Vec::new();
    let mut found_model = false;
    let mut found_provider = false;
    if let Ok(content) = std::fs::read_to_string(&rc_path) {
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some((k, _)) = trimmed.strip_prefix("export ").and_then(|s| s.split_once('=')) {
                if k.trim() == "NPCSH_CHAT_MODEL" { lines.push(format!("export NPCSH_CHAT_MODEL={}", model)); found_model = true; continue; }
                if k.trim() == "NPCSH_CHAT_PROVIDER" { lines.push(format!("export NPCSH_CHAT_PROVIDER={}", provider)); found_provider = true; continue; }
            }
            lines.push(line.to_string());
        }
    }
    if !found_model { lines.push(format!("export NPCSH_CHAT_MODEL={}", model)); }
    if !found_provider { lines.push(format!("export NPCSH_CHAT_PROVIDER={}", provider)); }
    let _ = std::fs::write(rc_path, lines.join("\n") + "\n");
    unsafe {
        std::env::set_var("NPCSH_CHAT_MODEL", model);
        std::env::set_var("NPCSH_CHAT_PROVIDER", provider);
    }

    let user_jinxes = PathBuf::from(shellexpand::tilde("~/.npcsh/npc_team/jinxes/usr").to_string());
    let _ = std::fs::create_dir_all(&user_jinxes);
    if let Ok(entries) = std::fs::read_dir(&user_jinxes) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("jinx") {
                if let Ok(content) = std::fs::read_to_string(&p) {
                    if content.contains("# tutorial heartbeat") {
                        let _ = std::fs::remove_file(&p);
                    }
                }
            }
        }
    }

    let mut reg = registry.lock().unwrap();
    for (i, (npc, interval, task)) in heartbeats.iter().enumerate() {
        let name = format!("tutorial_heartbeat_{}", i + 1);
        let file = user_jinxes.join(format!("{}.jinx", name));
        let yaml = format!(
            "#!/usr/bin/env npc\n# tutorial heartbeat\njinx_name: {name}\ndescription: Auto-generated tutorial heartbeat for @{npc}\ninputs:\n- task: \"{task}\"\ncron:\n- npc: {npc}\n  interval: {interval}\n  task: \"{task}\"\n  kind: chat\nsteps:\n- name: run\n  engine: python\n  code: |\n    task = {{{{ task | tojson }}}}\n    # This jinx is executed by the rust scheduler; the cron block above registers it.\n    context['output'] = task\n"
        );
        let _ = std::fs::write(&file, yaml);
        let secs = parse_duration(interval);
        reg.add(npc.clone(), secs, task.clone(), CronJobKind::Chat);
    }

    let skills_dir = PathBuf::from(shellexpand::tilde("~/.npcsh/npc_team/jinxes/skills").to_string());
    for (skill, enabled) in skills {
        if !enabled { continue; }
        let src = skills_dir.join(format!("{}.jinx", skill));
        let dst = user_jinxes.join(format!("{}.jinx", skill));
        if src.exists() && !dst.exists() {
            let _ = std::fs::copy(src, dst);
        }
    }

    let state = serde_yaml::to_string(&TutorialState {
        completed_at: chrono::Utc::now().to_rfc3339(),
        model: model.to_string(),
        provider: provider.to_string(),
        heartbeats: heartbeats.to_vec(),
        skills: skills.clone(),
    })
    .unwrap_or_default();
    let _ = std::fs::write(state_path, state);

    println!("\x1b[32mTutorial complete! Configuration saved to ~/.npcsh\x1b[0m");
    if !heartbeats.is_empty() {
        println!("\x1b[36m{} heartbeat(s) registered.\x1b[0m", heartbeats.len());
    }
}

#[derive(serde::Serialize)]
struct TutorialState {
    completed_at: String,
    model: String,
    provider: String,
    heartbeats: Vec<(String, String, String)>,
    skills: HashMap<String, bool>,
}
