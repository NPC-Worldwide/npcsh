use npcrs::kernel::Kernel;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal;

const ONE_WEEK: std::time::Duration = std::time::Duration::from_secs(7 * 24 * 60 * 60);

// ────────────────────────────────
// Common helpers
// ────────────────────────────────

pub type Result<T> = std::result::Result<T, npcrs::NpcError>;

pub struct RawModeGuard;
impl RawModeGuard {
    pub fn new() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        // Save cursor position, switch to alternate screen buffer, and hide cursor.
        let _ = io::stdout().write_all(b"\x1b[?1049h\x1b[?25l");
        let _ = io::stdout().flush();
        Ok(Self)
    }
}
impl Drop for RawModeGuard {
    fn drop(&mut self) {
        // Clear alternate screen, restore main screen, re-enable cursor, and disable raw mode.
        let _ = io::stdout().write_all(b"\x1b[2J\x1b[H\x1b[?1049l\x1b[?25h");
        let _ = io::stdout().flush();
        let _ = terminal::disable_raw_mode();
    }
}

pub fn term_size() -> (usize, usize) {
    let (c, r) = terminal::size().unwrap_or((80, 24));
    (c as usize, r as usize)
}

pub fn clear_all(out: &mut io::Stdout) {
    let _ = out.write_all(b"\x1b[2J\x1b[H");
}

pub fn hide_cursor(out: &mut io::Stdout) {
    let _ = out.write_all(b"\x1b[?25l");
}

pub fn wline(out: &mut io::Stdout, row: usize, text: &str) {
    let _ = write!(out, "\x1b[{};1H\x1b[K{}", row, text);
}

pub fn header_line(out: &mut io::Stdout, cols: usize, text: &str) {
    let pad = "═".repeat(cols);
    let mid = if text.is_empty() {
        String::new()
    } else {
        let t = format!(" {} ", text);
        let start = (cols.saturating_sub(t.len())) / 2;
        format!("\x1b[{};{}H{}", 1, start + 1, t)
    };
    let _ = write!(out, "\x1b[1;1H\x1b[7;1m{}\x1b[0m{}", pad, mid);
}

pub fn footer_line(out: &mut io::Stdout, cols: usize, rows: usize, text: &str) {
    let text = text.chars().take(cols).collect::<String>();
    let _ = write!(out, "\x1b[{};1H\x1b[K\x1b[7m{}\x1b[0m", rows, text.pad(cols));
}

pub fn hr(out: &mut io::Stdout, cols: usize, row: usize) {
    let _ = write!(out, "\x1b[{};1H\x1b[K\x1b[90m{}\x1b[0m", row, "─".repeat(cols));
}

trait StringPad {
    fn pad(&self, n: usize) -> String;
}
impl StringPad for str {
    fn pad(&self, n: usize) -> String {
        if self.len() >= n {
            self.to_string()
        } else {
            format!("{}{}", self, " ".repeat(n - self.len()))
        }
    }
}

fn run_git<P: AsRef<Path>>(repo: P, args: &[&str]) -> std::result::Result<std::process::Output, std::io::Error> {
    std::process::Command::new("git")
        .args(args)
        .current_dir(repo.as_ref())
        .output()
}

fn git_ok<P: AsRef<Path>>(repo: P, args: &[&str]) -> bool {
    run_git(repo, args).map(|o| o.status.success()).unwrap_or(false)
}

fn git_str(output: std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn git_err(output: std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

// ────────────────────────────────
// /config TUI
// ────────────────────────────────

pub fn run_config_tui() -> Result<()> {
    let rc_path = shellexpand::tilde("~/.npcshrc").to_string();
    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();

    #[derive(Clone)]
    enum ItemType { Text, Toggle, Choice }
    struct Item {
        key: &'static str,
        label: &'static str,
        ty: ItemType,
        choices: &'static [&'static str],
    }

    let items: Vec<Item> = vec![
        Item { key: "NPCSH_CHAT_MODEL", label: "Chat Model", ty: ItemType::Text, choices: &[] },
        Item { key: "NPCSH_CHAT_PROVIDER", label: "Chat Provider", ty: ItemType::Text, choices: &[] },
        Item { key: "NPCSH_VISION_MODEL", label: "Vision Model", ty: ItemType::Text, choices: &[] },
        Item { key: "NPCSH_VISION_PROVIDER", label: "Vision Provider", ty: ItemType::Text, choices: &[] },
        Item { key: "NPCSH_EMBEDDING_MODEL", label: "Embedding Model", ty: ItemType::Text, choices: &[] },
        Item { key: "NPCSH_EMBEDDING_PROVIDER", label: "Embedding Provider", ty: ItemType::Text, choices: &[] },
        Item { key: "NPCSH_REASONING_MODEL", label: "Reasoning Model", ty: ItemType::Text, choices: &[] },
        Item { key: "NPCSH_REASONING_PROVIDER", label: "Reasoning Provider", ty: ItemType::Text, choices: &[] },
        Item { key: "NPCSH_DEFAULT_MODE", label: "Default Mode", ty: ItemType::Choice, choices: &["agent", "chat", "cmd"] },
        Item { key: "NPCSH_STREAM_OUTPUT", label: "Stream Output", ty: ItemType::Toggle, choices: &[] },
        Item { key: "NPCSH_BUILD_KG", label: "Build Knowledge Graph", ty: ItemType::Toggle, choices: &[] },
        Item { key: "NPCSH_SEARCH_PROVIDER", label: "Search Provider", ty: ItemType::Choice, choices: &["duckduckgo", "google", "bing", "perplexity"] },
    ];

    let mut values = std::collections::HashMap::<String, String>::new();
    if let Ok(content) = std::fs::read_to_string(&rc_path) {
        for line in content.lines() {
            let line = line.trim();
            if let Some((k, v)) = line.strip_prefix("export ").and_then(|l| l.split_once('=')) {
                values.insert(k.to_string(), v.trim_matches('"').trim_matches('\'').to_string());
            }
        }
    }
    for item in &items {
        values.entry(item.key.to_string()).or_insert_with(|| String::new());
    }

    let mut sel: usize = 0;
    let mut editing = false;
    let mut edit_buf = String::new();
    let mut edit_cursor: usize = 0;

    loop {
        let (cols, rows) = term_size();
        clear_all(&mut out);
        header_line(&mut out, cols, " npcsh config ");
        hr(&mut out, cols, 2);

        let start = 4;
        let visible = rows.saturating_sub(start + 3).max(1);
        if sel >= items.len() { sel = items.len().saturating_sub(1); }
        let scroll = if sel < visible { 0 } else { sel - visible + 1 };

        for (i, item) in items.iter().enumerate() {
            let row = start + i.saturating_sub(scroll);
            if i < scroll || row >= rows - 2 { continue; }
            let val = values.get(item.key).cloned().unwrap_or_default();
            let display_val = match item.ty {
                ItemType::Toggle => if val == "1" || val.eq_ignore_ascii_case("true") || val.eq_ignore_ascii_case("yes") { "on" } else { "off" },
                _ => val.as_str(),
            };
            let marker = if i == sel { "> " } else { "  " };
            let line = format!("{}{:<26} {}", marker, item.label, display_val);
            if i == sel {
                wline(&mut out, row, &format!("\x1b[7m{}\x1b[0m", line.pad(cols)));
            } else {
                wline(&mut out, row, &line);
            }
        }

        if editing {
            let row = rows - 3;
            let prompt = format!("{}: ", items[sel].label);
            let mut full = format!("{}{}", prompt, edit_buf);
            let max_len = cols.saturating_sub(4);
            let display = if full.len() > max_len { full.chars().skip(full.len() - max_len).collect::<String>() } else { full };
            wline(&mut out, row, &format!("\x1b[90m{}\x1b[0m{}", prompt, display));
            if let Some(pos) = display.len().checked_sub(edit_buf.len() - edit_cursor + 1) {
                let _ = write!(out, "\x1b[{};{}H", row, pos + prompt.len() + 1);
            }
        }

        hr(&mut out, cols, rows - 2);
        let foot = if editing { " [Enter] Save  [Esc] Cancel " } else { " [j/k] Nav  [Enter] Edit/Toggle  [s] Save  [q] Quit " };
        footer_line(&mut out, cols, rows, foot);
        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Esc => {
                    if editing {
                        editing = false;
                        edit_buf.clear();
                        edit_cursor = 0;
                    } else {
                        break;
                    }
                }
                KeyCode::Char('s') if !editing => {
                    let mut lines: Vec<String> = items.iter().map(|item| {
                        let val = values.get(item.key).cloned().unwrap_or_default();
                        format!("export {}={}", item.key, val)
                    }).collect();
                    let extra = if let Ok(content) = std::fs::read_to_string(&rc_path) {
                        content.lines().filter(|l| {
                            let trimmed = l.trim();
                            !trimmed.is_empty() && !items.iter().any(|item| trimmed.starts_with(&format!("export {}", item.key)))
                        }).map(|l| l.to_string()).collect::<Vec<_>>()
                    } else { Vec::new() };
                    lines.extend(extra);
                    let _ = std::fs::write(&rc_path, lines.join("\n") + "\n");
                }
                KeyCode::Char('j') | KeyCode::Down if !editing => { if sel + 1 < items.len() { sel += 1; } }
                KeyCode::Char('k') | KeyCode::Up if !editing => { if sel > 0 { sel -= 1; } }
                KeyCode::Enter => {
                    let item = &items[sel];
                    if matches!(item.ty, ItemType::Toggle) {
                        let current = values.get(item.key).cloned().unwrap_or_default();
                        let next = if current == "1" || current.eq_ignore_ascii_case("true") { "0" } else { "1" };
                        values.insert(item.key.to_string(), next.to_string());
                    } else if matches!(item.ty, ItemType::Choice) {
                        let current = values.get(item.key).cloned().unwrap_or_default();
                        let idx = item.choices.iter().position(|c| *c == current).unwrap_or(0);
                        let next = item.choices[(idx + 1) % item.choices.len()];
                        values.insert(item.key.to_string(), next.to_string());
                    } else if !editing {
                        edit_buf = values.get(item.key).cloned().unwrap_or_default();
                        edit_cursor = edit_buf.len();
                        editing = true;
                    } else {
                        values.insert(item.key.to_string(), edit_buf.clone());
                        editing = false;
                        edit_buf.clear();
                        edit_cursor = 0;
                    }
                }
                KeyCode::Char(c) if editing => {
                    edit_buf.insert(edit_cursor, c);
                    edit_cursor += 1;
                }
                KeyCode::Backspace if editing && edit_cursor > 0 => {
                    edit_cursor -= 1;
                    edit_buf.remove(edit_cursor);
                }
                KeyCode::Left if editing && edit_cursor > 0 => { edit_cursor -= 1; }
                KeyCode::Right if editing && edit_cursor < edit_buf.len() => { edit_cursor += 1; }
                _ => {}
            }
        }
    }
    Ok(())
}

// ────────────────────────────────
// /gitt TUI
// ────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum GitTab { Status, Log, Branches, Stash, Diff }

#[derive(Clone)]
struct GitFile { path: String, staged: bool, modified: bool, untracked: bool }

pub fn run_gitt_tui(path: Option<&str>) -> Result<()> {
    let repo = path.map(PathBuf::from).unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let repo_str = repo.to_string_lossy().to_string();
    if !git_ok(&repo, &["rev-parse", "--is-inside-work-tree"]) {
        return Err(npcrs::NpcError::Other(format!("{} is not a git repository", repo_str)));
    }

    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();
    let mut tab = GitTab::Status;
    let mut sel: usize = 0;
    let mut scroll: usize = 0;
    let mut status = String::new();

    let mut files: Vec<GitFile> = Vec::new();
    let mut branches: Vec<(String, bool)> = Vec::new();
    let mut stash: Vec<String> = Vec::new();
    let mut log: Vec<String> = Vec::new();
    let mut diff_text: Vec<String> = Vec::new();

    let mut detail = false;
    let mut detail_scroll: usize = 0;

    let refresh = |t: &GitTab, f: &mut Vec<GitFile>, b: &mut Vec<(String, bool)>, s: &mut Vec<String>, l: &mut Vec<String>, d: &mut Vec<String>, st: &mut String, repo: &PathBuf| {
        match t {
            GitTab::Status => {
                f.clear();
                let out = run_git(repo, &["status", "--porcelain=v1", "-u"]).ok().map(git_str).unwrap_or_default();
                for line in out.lines() {
                    if line.len() < 3 { continue; }
                    let idx = &line[..2];
                    let path = line[3..].to_string();
                    f.push(GitFile { path, staged: idx.starts_with('A') || idx.starts_with('M') || idx.starts_with('D') || idx.starts_with('R'), modified: idx.chars().nth(1).map(|c| c == 'M' || c == 'D').unwrap_or(false), untracked: idx == "??" });
                }
                *st = format!("{} files", f.len());
            }
            GitTab::Log => {
                l.clear();
                let out = run_git(repo, &["log", "--oneline", "--decorate", "-n", "100"]).ok().map(git_str).unwrap_or_default();
                l.extend(out.lines().map(|s| s.to_string()));
                *st = format!("{} commits", l.len());
            }
            GitTab::Branches => {
                b.clear();
                let out = run_git(repo, &["branch", "-vv"]).ok().map(git_str).unwrap_or_default();
                for line in out.lines() {
                    let current = line.starts_with('*');
                    let name = line.trim_start_matches('*').trim_start().split_whitespace().next().unwrap_or("?").to_string();
                    b.push((name, current));
                }
                *st = format!("{} branches", b.len());
            }
            GitTab::Stash => {
                s.clear();
                let out = run_git(repo, &["stash", "list"]).ok().map(git_str).unwrap_or_default();
                s.extend(out.lines().map(|s| s.to_string()));
                *st = format!("{} stashes", s.len());
            }
            GitTab::Diff => {
                d.clear();
                let out = run_git(repo, &["diff", "--cached"]).ok().map(git_str).unwrap_or_default();
                let out2 = run_git(repo, &["diff"]).ok().map(git_str).unwrap_or_default();
                let full = if out.is_empty() { out2 } else { out };
                d.extend(full.lines().map(|s| s.to_string()));
                *st = format!("{} lines", d.len());
            }
        }
    };

    refresh(&tab, &mut files, &mut branches, &mut stash, &mut log, &mut diff_text, &mut status, &repo);

    loop {
        let (cols, rows) = term_size();
        let body_h = rows.saturating_sub(5).max(1);

        if detail {
            let lines = match tab {
                GitTab::Status => {
                    let f = files.get(sel);
                    if let Some(f) = f {
                        let diff = if f.staged { run_git(&repo, &["diff", "--cached", "--", &f.path]) } else { run_git(&repo, &["diff", "--", &f.path]) };
                        diff.ok().map(git_str).unwrap_or_default().lines().map(|s| s.to_string()).collect()
                    } else { Vec::new() }
                }
                _ => diff_text.clone(),
            };
            clear_all(&mut out);
            header_line(&mut out, cols, &format!(" gitt / {:?} ", tab));
            hr(&mut out, cols, 2);
            wline(&mut out, 3, &format!("  {} | {}", repo_str, status));
            hr(&mut out, cols, 4);
            for r in 0..body_h {
                let idx = detail_scroll + r;
                let row = 5 + r;
                if idx >= lines.len() { wline(&mut out, row, ""); }
                else {
                    let line = &lines[idx];
                    let rendered = if line.starts_with('+') { format!("\x1b[32m{}\x1b[0m", line) }
                        else if line.starts_with('-') { format!("\x1b[31m{}\x1b[0m", line) }
                        else if line.starts_with("@@") { format!("\x1b[36m{}\x1b[0m", line) }
                        else { line.to_string() };
                    let truncated = rendered.chars().take(cols).collect::<String>();
                    wline(&mut out, row, &format!("  {}", truncated));
                }
            }
            hr(&mut out, cols, rows - 2);
            footer_line(&mut out, cols, rows, " [j/k] Scroll  [q/Esc] Back ");
        } else {
            let items: usize = match tab {
                GitTab::Status => files.len(),
                GitTab::Log => log.len(),
                GitTab::Branches => branches.len(),
                GitTab::Stash => stash.len(),
                GitTab::Diff => diff_text.len(),
            };
            if sel >= items && items > 0 { sel = items - 1; }
            if sel < scroll { scroll = sel; }
            else if sel >= scroll + body_h { scroll = sel.saturating_sub(body_h).saturating_add(1); }

            clear_all(&mut out);
            header_line(&mut out, cols, " gitt ");
            hr(&mut out, cols, 2);
            let tabs = "[1:status] [2:log] [3:branches] [4:stash] [5:diff]";
            wline(&mut out, 3, &format!("  {} | {} | {}", repo_str, status, tabs));
            hr(&mut out, cols, 4);

            for r in 0..body_h {
                let idx = scroll + r;
                let row = 5 + r;
                if idx >= items { wline(&mut out, row, ""); continue; }
                let text = match tab {
                    GitTab::Status => {
                        let f = &files[idx];
                        let marker = if f.staged { "A" } else if f.untracked { "?" } else { "M" };
                        format!("[{}] {}", marker, f.path)
                    }
                    GitTab::Log => log[idx].clone(),
                    GitTab::Branches => {
                        let (name, current) = &branches[idx];
                        if *current { format!("* {}", name) } else { format!("  {}", name) }
                    }
                    GitTab::Stash => stash[idx].clone(),
                    GitTab::Diff => diff_text[idx].clone(),
                };
                let truncated = text.chars().take(cols.saturating_sub(4)).collect::<String>();
                if idx == sel { wline(&mut out, row, &format!("\x1b[7m  > {}\x1b[0m", truncated).pad(cols)); }
                else { wline(&mut out, row, &format!("    {}", truncated)); }
            }
            hr(&mut out, cols, rows - 2);
            footer_line(&mut out, cols, rows, " [Tab] Switch  [j/k] Nav  [Enter] Diff/Stage  [s] Stage  [u] Unstage  [c] Commit  [q] Quit ");
        }

        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Esc => { if detail { detail = false; detail_scroll = 0; } else { break; } }
                KeyCode::Char('q') => { if detail { detail = false; detail_scroll = 0; } else { break; } }
                KeyCode::Tab => {
                    let tabs = [GitTab::Status, GitTab::Log, GitTab::Branches, GitTab::Stash, GitTab::Diff];
                    let idx = tabs.iter().position(|t| t == &tab).unwrap_or(0);
                    tab = tabs[(idx + 1) % tabs.len()];
                    sel = 0; scroll = 0; detail = matches!(tab, GitTab::Diff);
                    refresh(&tab, &mut files, &mut branches, &mut stash, &mut log, &mut diff_text, &mut status, &repo);
                }
                KeyCode::BackTab => {
                    let tabs = [GitTab::Status, GitTab::Log, GitTab::Branches, GitTab::Stash, GitTab::Diff];
                    let idx = tabs.iter().position(|t| t == &tab).unwrap_or(0);
                    tab = tabs[(idx + tabs.len() - 1) % tabs.len()];
                    sel = 0; scroll = 0; detail = matches!(tab, GitTab::Diff);
                    refresh(&tab, &mut files, &mut branches, &mut stash, &mut log, &mut diff_text, &mut status, &repo);
                }
                KeyCode::Char('j') | KeyCode::Down => {
                    if detail {
                        let lines = match tab { GitTab::Diff => diff_text.len(), _ => files.get(sel).map(|_| 1000).unwrap_or(0) };
                        if detail_scroll + 1 < lines { detail_scroll += 1; }
                    } else {
                        let items = match tab { GitTab::Status => files.len(), GitTab::Log => log.len(), GitTab::Branches => branches.len(), GitTab::Stash => stash.len(), GitTab::Diff => diff_text.len() };
                        if sel + 1 < items { sel += 1; }
                    }
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    if detail { detail_scroll = detail_scroll.saturating_sub(1); }
                    else if sel > 0 { sel -= 1; }
                }
                KeyCode::Enter => {
                    if tab == GitTab::Status {
                        detail = !detail;
                        detail_scroll = 0;
                    } else if tab == GitTab::Log {
                        let commit = log.get(sel).and_then(|l| l.split_whitespace().next()).map(|s| s.to_string());
                        if let Some(commit) = commit {
                            diff_text = run_git(&repo, &["show", "--stat", &commit]).ok().map(git_str).unwrap_or_default().lines().map(|s| s.to_string()).collect();
                            detail = true;
                            detail_scroll = 0;
                        }
                    }
                }
                KeyCode::Char('s') => {
                    if tab == GitTab::Status {
                        if let Some(f) = files.get(sel) {
                            let _ = run_git(&repo, &["add", "--", &f.path]);
                            refresh(&tab, &mut files, &mut branches, &mut stash, &mut log, &mut diff_text, &mut status, &repo);
                        }
                    }
                }
                KeyCode::Char('u') => {
                    if tab == GitTab::Status {
                        if let Some(f) = files.get(sel) {
                            let _ = run_git(&repo, &["reset", "HEAD", "--", &f.path]);
                            refresh(&tab, &mut files, &mut branches, &mut stash, &mut log, &mut diff_text, &mut status, &repo);
                        }
                    }
                }
                KeyCode::Char('c') => {
                    if tab == GitTab::Status {
                        let _ = terminal::disable_raw_mode();
                        let _ = io::stdout().write_all(b"\x1b[?1049l\x1b[?25h");
                        let _ = io::stdout().flush();
                        let mut msg = String::new();
                        print!("Commit message: ");
                        let _ = io::stdout().flush();
                        let _ = std::io::stdin().read_line(&mut msg);
                        let msg = msg.trim();
                        if !msg.is_empty() {
                            let _ = run_git(&repo, &["commit", "-m", msg]);
                        }
                        let _ = terminal::enable_raw_mode();
                        let _ = io::stdout().write_all(b"\x1b[?1049h\x1b[?25l");
                        let _ = io::stdout().flush();
                        refresh(&tab, &mut files, &mut branches, &mut stash, &mut log, &mut diff_text, &mut status, &repo);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

// ────────────────────────────────
// /model TUI
// ────────────────────────────────

#[derive(Clone)]
struct ModelEntry { provider: String, id: String, name: String }

fn api_key_hash() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let keys = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
        "GROQ_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY", "PERPLEXITY_API_KEY",
        "TOGETHER_API_KEY", "FIREWORKS_API_KEY", "CEREBRAS_API_KEY", "AI21_API_KEY",
        "AZURE_API_KEY", "COHERE_API_KEY", "OPENROUTER_API_KEY", "NOVITA_API_KEY",
        "HYPERBOLIC_API_KEY", "SAMBANOVA_API_KEY", "NEBIUS_API_KEY", "MOONSHOT_API_KEY",
        "OLLAMA_HOST", "GGUF_DIR",
    ];
    let mut hasher = DefaultHasher::new();
    for key in keys {
        std::env::var(key).unwrap_or_default().hash(&mut hasher);
        key.hash(&mut hasher);
    }
    format!("{:x}", hasher.finish())
}

fn detect_models(providers: &mut Vec<String>, models: &mut std::collections::HashMap<String, Vec<String>>) {
    providers.clear(); models.clear();

    let cache_path = shellexpand::tilde("~/.npcsh/available_models.yaml").to_string();
    let key_hash = api_key_hash();

    let cache = (|| -> Option<(Vec<ModelEntry>, String, u64)> {
        let content = std::fs::read_to_string(&cache_path).ok()?;
        let parsed: serde_yaml::Value = serde_yaml::from_str(&content).ok()?;
        let cached_hash = parsed.get("key_hash")?.as_str()?.to_string();
        let ts = parsed.get("timestamp")?.as_u64()?;
        let arr = parsed.get("models")?.as_sequence()?;
        let entries = arr.iter().filter_map(|v| {
            Some(ModelEntry {
                provider: v.get("provider")?.as_str()?.to_string(),
                id: v.get("id")?.as_str()?.to_string(),
                name: v.get("name")?.as_str()?.to_string(),
            })
        }).collect();
        Some((entries, cached_hash, ts))
    })();

    if let Some((entries, cached_hash, ts)) = cache {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if cached_hash == key_hash && now.saturating_sub(ts) < ONE_WEEK.as_secs() {
            for e in entries {
                models.entry(e.provider.clone()).or_default().push(e.id);
                if !providers.contains(&e.provider) { providers.push(e.provider); }
            }
            sort_providers_and_models(providers, models);
            return;
        }
    }

    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let cwd_json = serde_json::to_string(&cwd).unwrap_or_else(|_| "\".\"".to_string());
    let python_one_liner = format!(
        r#"
import json
from npcpy.npc_sysenv import get_locally_available_models
from litellm import utils as lu, get_valid_models
out = {{}}
try:
    data = get_locally_available_models({})
    if isinstance(data, dict):
        for mid, prov in data.items():
            if isinstance(mid, str) and isinstance(prov, str):
                out.setdefault(prov, []).append(mid)
except Exception:
    pass
try:
    for pv in lu._infer_valid_provider_from_env_vars():
        prov = pv.value
        try:
            out[prov] = [str(x) for x in get_valid_models(custom_llm_provider=prov)[:50]]
        except Exception:
            pass
except Exception:
    pass
print(json.dumps(out))
"#,
        cwd_json
    );
    if let Ok(output) = std::process::Command::new("python3")
        .args(["-c", &python_one_liner])
        .output()
    {
        if output.status.success() {
            if let Ok(parsed) = serde_json::from_slice::<std::collections::HashMap<String, Vec<String>>>(&output.stdout) {
                for (p, ms) in parsed {
                    models.entry(p.clone()).or_default().extend(ms);
                    if !providers.contains(&p) { providers.push(p); }
                }
            }
        }
    }

    sort_providers_and_models(providers, models);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut cache_yaml = serde_yaml::Mapping::new();
    cache_yaml.insert("key_hash".into(), key_hash.into());
    cache_yaml.insert("timestamp".into(), now.into());
    let models_yaml: serde_yaml::Value = providers
        .iter()
        .flat_map(|p| {
            let ms = models.get(p).cloned().unwrap_or_default();
            ms.into_iter().map(move |m| {
                let mut map = serde_yaml::Mapping::new();
                map.insert("provider".into(), p.clone().into());
                map.insert("id".into(), m.clone().into());
                map.insert("name".into(), m.into());
                serde_yaml::Value::Mapping(map)
            })
        })
        .collect();
    cache_yaml.insert("models".into(), models_yaml);
    let _ = std::fs::create_dir_all(Path::new(&cache_path).parent().unwrap_or(Path::new(".")));
    let _ = std::fs::write(&cache_path, serde_yaml::to_string(&cache_yaml).unwrap_or_default());
}

fn sort_providers_and_models(providers: &mut Vec<String>, models: &mut std::collections::HashMap<String, Vec<String>>) {
    providers.sort_by_key(|p| {
        if is_local_provider(p) { (0, p.clone()) }
        else if is_cli_provider(p) { (1, p.clone()) }
        else { (2, p.clone()) }
    });
    for ms in models.values_mut() { ms.sort(); ms.dedup(); }
}

fn detect_models_list() -> Vec<ModelEntry> {
    let mut providers: Vec<String> = Vec::new();
    let mut models: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    detect_models(&mut providers, &mut models);
    providers
        .into_iter()
        .flat_map(|p| {
            let ms = models.remove(&p).unwrap_or_default();
            ms.into_iter().map(move |m| {
                let name = m.split('/').last().unwrap_or(&m).to_string();
                ModelEntry { provider: p.clone(), id: m, name }
            })
        })
        .collect()
}

pub fn run_model_tui() -> Result<()> {
    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();

    let mut providers: Vec<String> = Vec::new();
    let mut provider_models: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut chat_model = std::env::var("NPCSH_CHAT_MODEL").unwrap_or_default();
    let mut chat_provider = std::env::var("NPCSH_CHAT_PROVIDER").unwrap_or_default();
    let mut level = 0; // 0 providers, 1 models
    let mut active_provider = String::new();
    let mut sel: usize = 0;
    let mut scroll: usize = 0;
    let mut status = String::from("Detecting models...");

    let (detect_tx, detect_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut providers: Vec<String> = Vec::new();
        let mut models: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        detect_models(&mut providers, &mut models);
        let _ = detect_tx.send((providers, models));
    });

    let mut detected = false;
    let mut frame: usize = 0;

    hide_cursor(&mut out);
    loop {
        if !detected {
            if let Ok((p, m)) = detect_rx.try_recv() {
                providers = p;
                provider_models = m;
                detected = true;
                if providers.is_empty() { status = "No providers detected.".to_string(); }
                else { status = format!("Found {} providers, {} models", providers.len(), provider_models.values().map(|v| v.len()).sum::<usize>()); }
            }
        }
        let (cols, rows) = term_size();
        let body_h = rows.saturating_sub(6).max(1);
        if sel < scroll { scroll = sel; }
        else {
            let items_len = if level == 0 { providers.len() } else { provider_models.get(&active_provider).map(|v| v.len()).unwrap_or(0) };
            if items_len > 0 && sel >= items_len { sel = items_len - 1; }
            if sel >= scroll + body_h { scroll = sel - body_h + 1; }
        }

        clear_all(&mut out);
        header_line(&mut out, cols, "Set Chat Model");
        hr(&mut out, cols, 2);

        if !detected {
            let spinner = ["|", "/", "-", "\\"];
            let msg = format!("Detecting models {} ", spinner[frame % spinner.len()]);
            let start = (cols.saturating_sub(msg.len())) / 2;
            wline(&mut out, rows / 2, &format!("\x1b[{};{}H\x1b[90m{}\x1b[0m", rows / 2, start + 1, msg));
            let _ = out.flush();
            frame += 1;

            if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
                let _ = event::read();
            }
            std::thread::sleep(std::time::Duration::from_millis(80));
            continue;
        }

        let breadcrumb = if level == 0 {
            format!("  Providers  |  {}", status)
        } else {
            format!("  {} → Models  |  {}", active_provider, status)
        };
        wline(&mut out, 3, &format!("\x1b[1m{}\x1b[0m", breadcrumb.chars().take(cols).collect::<String>()));

        if level == 0 {
            for r in 0..body_h {
                let idx = scroll + r; let row = 4 + r;
                if idx >= providers.len() { wline(&mut out, row, ""); continue; }
                let p = &providers[idx];
                let count = provider_models.get(p).map(|v| v.len()).unwrap_or(0);
                let icon = provider_icon(p);
                let color = provider_color(p);
                let line = format!("  ▶ {} {}  ({} models)", icon, p, count);
                if idx == sel { wline(&mut out, row, &format!("\x1b[7m> {}\x1b[0m", line.chars().take(cols - 2).collect::<String>().pad(cols - 2))); }
                else { wline(&mut out, row, &format!("\x1b[{}m{}\x1b[0m", color, line.chars().take(cols).collect::<String>())); }
            }
            if providers.is_empty() { wline(&mut out, 4, "  \x1b[90mNo providers found. Press [d] to refresh.\x1b[0m"); }
        } else {
            let models = provider_models.get(&active_provider).cloned().unwrap_or_default();
            for r in 0..body_h {
                let idx = scroll + r; let row = 4 + r;
                if idx >= models.len() { wline(&mut out, row, ""); continue; }
                let m = &models[idx];
                let active = m == &chat_model && active_provider == chat_provider;
                if idx == sel { wline(&mut out, row, &format!("\x1b[7m {} {}\x1b[0m", if active { ">" } else { " " }, m.chars().take(cols - 4).collect::<String>().pad(cols - 4))); }
                else { wline(&mut out, row, &format!("{} {}", if active { "  *" } else { "   " }, m.chars().take(cols - 4).collect::<String>())); }
            }
        }

        let active_line = if chat_model.is_empty() { "  \x1b[90mNo active model set.\x1b[0m".to_string() } else { format!("  Active: {} / {}", chat_model, chat_provider) };
        hr(&mut out, cols, rows - 2);
        wline(&mut out, rows - 1, &format!("\x1b[33m{}\x1b[0m", active_line.chars().take(cols - 2).collect::<String>()));
        let foot = if level == 0 { " [j/k] Nav  [Enter] Expand  [q] Quit  [d] Refresh " } else { " [j/k] Nav  [Enter] Set  [h/Esc] Back  [q] Quit " };
        footer_line(&mut out, cols, rows, foot);
        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            let c = key.code;
            match c {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Char('q') => break,
                KeyCode::Char('d') if level == 0 => {
                    let cache_path = shellexpand::tilde("~/.npcsh/available_models.yaml").to_string();
                    let _ = std::fs::remove_file(&cache_path);
                    detect_models(&mut providers, &mut provider_models);
                    status = if providers.is_empty() { "No providers detected.".to_string() } else { format!("Found {} providers, {} models", providers.len(), provider_models.values().map(|v| v.len()).sum::<usize>()) };
                }
                KeyCode::Char('h') | KeyCode::Esc if level == 1 => { level = 0; sel = providers.iter().position(|p| p == &active_provider).unwrap_or(0); scroll = 0; }
                KeyCode::Char('j') | KeyCode::Down => {
                    let mx = if level == 0 { providers.len() } else { provider_models.get(&active_provider).map(|v| v.len()).unwrap_or(0) };
                    if sel + 1 < mx { sel += 1; }
                }
                KeyCode::Char('k') | KeyCode::Up => { if sel > 0 { sel -= 1; } }
                KeyCode::Enter => {
                    if level == 0 {
                        if let Some(p) = providers.get(sel) {
                            if provider_models.get(p).map(|v| !v.is_empty()).unwrap_or(false) {
                                active_provider = p.clone(); level = 1; sel = 0; scroll = 0;
                            } else { status = format!("{} has no models", p); }
                        }
                    } else {
                        if let Some(m) = provider_models.get(&active_provider).and_then(|v| v.get(sel)) {
                            set_npcsh_config_value("NPCSH_CHAT_MODEL", m);
                            set_npcsh_config_value("NPCSH_CHAT_PROVIDER", &active_provider);
                            chat_model = m.clone(); chat_provider = active_provider.clone();
                            status = format!("Set to {} / {}", m, active_provider);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn provider_color(p: &str) -> u8 {
    if is_cli_provider(p) { 33 } else if is_local_provider(p) { 36 } else { 35 }
}
fn provider_icon(p: &str) -> &'static str {
    if is_cli_provider(p) { "[CLI]" } else if is_local_provider(p) { "[LOC]" } else { "[CLD]" }
}
fn is_local_provider(p: &str) -> bool { matches!(p, "ollama" | "llamacpp" | "lmstudio" | "mlx" | "lora") }
fn is_cli_provider(p: &str) -> bool { matches!(p, "claude_code" | "opencode" | "codex" | "kimi_code" | "kilo") }

fn set_npcsh_config_value(key: &str, value: &str) {
    let rc_path = shellexpand::tilde("~/.npcshrc").to_string();
    let mut lines: Vec<String> = Vec::new();
    let mut found = false;
    if let Ok(content) = std::fs::read_to_string(&rc_path) {
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some((k, _)) = trimmed.strip_prefix("export ").and_then(|s| s.split_once('=')) {
                if k.trim() == key { lines.push(format!("export {}={}", key, value)); found = true; continue; }
            }
            lines.push(line.to_string());
        }
    }
    if !found { lines.push(format!("export {}={}", key, value)); }
    let _ = std::fs::write(rc_path, lines.join("\n") + "\n");
    unsafe { std::env::set_var(key, value); }
}

// ────────────────────────────────
// /setup TUI
// ────────────────────────────────

pub fn run_setup_tui() -> Result<()> {
    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();

    let steps = vec![
        ("Welcome", "npcsh setup wizard. Detecting models and API keys."),
        ("Chat Model", "Choose your default chat model."),
        ("API Keys", "Verify required API keys are set."),
        ("Done", "Setup complete. Press q to exit."),
    ];
    let mut step: usize = 0;
    let mut models: Vec<ModelEntry> = Vec::new();
    let mut sel: usize = 0;

    let (detect_tx, detect_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || { detect_tx.send(detect_models_list()).ok(); });

    loop {
        if let Ok(found) = detect_rx.try_recv() { models = found; }

        let (cols, rows) = term_size();
        clear_all(&mut out);
        header_line(&mut out, cols, " Setup ");
        hr(&mut out, cols, 2);

        let (title, desc) = steps[step];
        wline(&mut out, 3, &format!("  Step {} of {}: {} - {}", step + 1, steps.len(), title, desc));
        hr(&mut out, cols, 4);

        let body_h = rows.saturating_sub(6).max(1);
        match step {
            0 => {
                wline(&mut out, 6, "  npcsh needs a running Python server and a chosen model.");
                wline(&mut out, 8, "  This wizard detects available models from API keys.");
            }
            1 => {
                for r in 0..body_h {
                    let idx = r;
                    let row = 5 + r;
                    if idx >= models.len() { wline(&mut out, row, ""); continue; }
                    let m = &models[idx];
                    let text = format!("{} / {} ({})", m.provider, m.id, m.name);
                    if idx == sel { wline(&mut out, row, &format!("\x1b[7m  > {}\x1b[0m", text).pad(cols)); }
                    else { wline(&mut out, row, &format!("    {}", text)); }
                }
            }
            2 => {
                let keys = [("OPENAI_API_KEY", "OpenAI"), ("ANTHROPIC_API_KEY", "Anthropic"), ("GEMINI_API_KEY", "Gemini"), ("PERPLEXITY_API_KEY", "Perplexity")];
                for (i, (key, label)) in keys.iter().enumerate() {
                    let set = std::env::var(key).is_ok();
                    let status = if set { "\x1b[32mset\x1b[0m" } else { "\x1b[31mnot set\x1b[0m" };
                    wline(&mut out, 6 + i, &format!("  {}: {}", label, status));
                }
            }
            3 => {
                wline(&mut out, 6, "  Setup complete. ~/.npcshrc has been updated.");
                wline(&mut out, 8, "  Start chatting by typing at the prompt.");
            }
            _ => {}
        }

        hr(&mut out, cols, rows - 2);
        footer_line(&mut out, cols, rows, " [n] Next  [p] Prev  [Enter] Select  [q] Quit ");
        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Esc | KeyCode::Char('q') => break,
                KeyCode::Char('n') => { if step + 1 < steps.len() { step += 1; } }
                KeyCode::Char('p') => { if step > 0 { step -= 1; } }
                KeyCode::Char('j') | KeyCode::Down => { if step == 1 && sel + 1 < models.len() { sel += 1; } }
                KeyCode::Char('k') | KeyCode::Up => { if step == 1 && sel > 0 { sel -= 1; } }
                KeyCode::Enter => {
                    if step == 1 {
                        if let Some(m) = models.get(sel) {
                            set_npcsh_config_value("NPCSH_CHAT_MODEL", &m.id);
                            set_npcsh_config_value("NPCSH_CHAT_PROVIDER", &m.provider);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

// ────────────────────────────────
// /team TUI
// ────────────────────────────────

pub fn run_team_tui(kernel: &mut Kernel) -> Result<()> {
    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();

    let npcs: Vec<String> = kernel.ps().iter().map(|p| p.npc.name.clone()).collect();
    let jinx_names: Vec<String> = kernel.jinx_names().into_iter().map(String::from).collect();
    let team_dir = kernel.team.source_dir.clone().unwrap_or_default();

    #[derive(Clone, Copy, PartialEq)]
    enum Tab { NPCs, Jinxes, Context }
    let mut tab = Tab::NPCs;
    let mut sel: usize = 0;
    let mut scroll: usize = 0;

    loop {
        let (cols, rows) = term_size();
        let body_h = rows.saturating_sub(6).max(1);
        clear_all(&mut out);
        header_line(&mut out, cols, " Team ");
        hr(&mut out, cols, 2);
        wline(&mut out, 3, &format!("  {} | [{}] NPCs  [{}] Jinxes  [{}] Context", team_dir, if matches!(tab, Tab::NPCs) { "1" } else { "_" }, if matches!(tab, Tab::Jinxes) { "2" } else { "_" }, if matches!(tab, Tab::Context) { "3" } else { "_" }));
        hr(&mut out, cols, 4);

        let items = match tab {
            Tab::NPCs => npcs.len(),
            Tab::Jinxes => jinx_names.len(),
            Tab::Context => 1,
        };
        if sel >= items && items > 0 { sel = items - 1; }
        if sel < scroll { scroll = sel; }
        else if sel >= scroll + body_h { scroll = sel.saturating_sub(body_h) + 1; }

        for r in 0..body_h {
            let idx = scroll + r;
            let row = 5 + r;
            if idx >= items { wline(&mut out, row, ""); continue; }
            let text = match tab {
                Tab::NPCs => format!("@{}", npcs[idx]),
                Tab::Jinxes => format!("/{}", jinx_names[idx]),
                Tab::Context => kernel.team.context.clone().unwrap_or_else(|| "(none)".to_string()),
            };
            let truncated = text.chars().take(cols.saturating_sub(6)).collect::<String>();
            if idx == sel { wline(&mut out, row, &format!("\x1b[7m  > {}\x1b[0m", truncated).pad(cols)); }
            else { wline(&mut out, row, &format!("    {}", truncated)); }
        }

        hr(&mut out, cols, rows - 2);
        footer_line(&mut out, cols, rows, " [Tab] Switch  [j/k] Nav  [q] Quit ");
        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Esc | KeyCode::Char('q') => break,
                KeyCode::Tab => {
                    let tabs = [Tab::NPCs, Tab::Jinxes, Tab::Context];
                    let idx = tabs.iter().position(|t| t == &tab).unwrap_or(0);
                    tab = tabs[(idx + 1) % tabs.len()];
                    sel = 0; scroll = 0;
                }
                KeyCode::BackTab => {
                    let tabs = [Tab::NPCs, Tab::Jinxes, Tab::Context];
                    let idx = tabs.iter().position(|t| t == &tab).unwrap_or(0);
                    tab = tabs[(idx + tabs.len() - 1) % tabs.len()];
                    sel = 0; scroll = 0;
                }
                KeyCode::Char('j') | KeyCode::Down => { if sel + 1 < items { sel += 1; } }
                KeyCode::Char('k') | KeyCode::Up => { if sel > 0 { sel -= 1; } }
                _ => {}
            }
        }
    }
    Ok(())
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for para in text.split('\n') {
        let mut cur = String::new();
        for word in para.split_whitespace() {
            if cur.is_empty() { cur.push_str(word); }
            else if cur.len() + 1 + word.len() <= width { cur.push(' '); cur.push_str(word); }
            else { lines.push(cur); cur = word.to_string(); }
        }
        if !cur.is_empty() { lines.push(cur); }
    }
    lines
}

// ────────────────────────────────
// /commit TUI
// ────────────────────────────────

pub fn run_commit_tui() -> Result<()> {
    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();

    let repo = std::env::current_dir().unwrap_or_default();
    let status_out = run_git(&repo, &["status", "--short"]).ok().map(git_str).unwrap_or_default();
    let mut files: Vec<String> = status_out.lines().map(|l| l[3..].to_string()).collect();

    let mut sel: usize = 0;
    let mut msg = String::new();
    let mut stage: usize = 0; // 0 = pick files, 1 = message

    loop {
        let (cols, rows) = term_size();
        clear_all(&mut out);
        header_line(&mut out, cols, " commit ");
        hr(&mut out, cols, 2);

        if stage == 0 {
            wline(&mut out, 3, "  Select files to stage (Space toggles, Enter commits all shown):");
            hr(&mut out, cols, 4);
            let body_h = rows.saturating_sub(6).max(1);
            for r in 0..body_h {
                let idx = r;
                let row = 5 + r;
                if idx >= files.len() { wline(&mut out, row, ""); continue; }
                let marker = "[x]";
                let text = format!("{} {}", marker, files[idx]);
                if idx == sel { wline(&mut out, row, &format!("\x1b[7m  > {}\x1b[0m", text).pad(cols)); }
                else { wline(&mut out, row, &format!("    {}", text)); }
            }
            hr(&mut out, cols, rows - 2);
            footer_line(&mut out, cols, rows, " [j/k] Nav  [Enter] Message  [q] Quit ");
        } else {
            wline(&mut out, 3, "  Commit message:");
            wline(&mut out, 5, &format!("  > {}", msg));
            let _ = write!(out, "\x1b[{};{}H", 5, 6 + msg.len());
            hr(&mut out, cols, rows - 2);
            footer_line(&mut out, cols, rows, " [Enter] Commit  [Esc] Back ");
        }

        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Esc => { if stage == 1 { stage = 0; } else { break; } }
                KeyCode::Char('j') | KeyCode::Down if stage == 0 => { if sel + 1 < files.len() { sel += 1; } }
                KeyCode::Char('k') | KeyCode::Up if stage == 0 => { if sel > 0 { sel -= 1; } }
                KeyCode::Enter => {
                    if stage == 0 { stage = 1; }
                    else {
                        let _ = run_git(&repo, &["add", "-A"]);
                        if !msg.is_empty() {
                            let _ = run_git(&repo, &["commit", "-m", &msg]);
                        }
                        break;
                    }
                }
                KeyCode::Char(c) if stage == 1 => msg.push(c),
                KeyCode::Backspace if stage == 1 && !msg.is_empty() => { msg.pop(); }
                _ => {}
            }
        }
    }
    Ok(())
}

// ────────────────────────────────
// /jinxes TUI
// ────────────────────────────────

pub fn run_jinxes_tui(kernel: &mut Kernel) -> Result<()> {
    let mut all: Vec<(String, String, String, String)> = Vec::new(); // source, folder, name, description
    let team_dir = kernel.team.source_dir.clone();
    let global_dir = shellexpand::tilde("~/.npcsh/npc_team").to_string();
    scan_jinxes(&team_dir, "team", &mut all);
    if global_dir != team_dir.clone().unwrap_or_default() { scan_jinxes(&Some(global_dir), "global", &mut all); }
    all.sort_by(|a, b| a.2.cmp(&b.2));

    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();
    let mut sel: usize = 0;
    let mut scroll: usize = 0;
    let mut detail = false;
    let mut detail_scroll = 0usize;
    hide_cursor(&mut out);

    loop {
        let (cols, rows) = term_size();
        let body_h = rows.saturating_sub(6).max(1);
        if sel < scroll { scroll = sel; }
        else if !all.is_empty() && sel >= all.len() { sel = all.len() - 1; }
        else if sel >= scroll + body_h { scroll = sel - body_h + 1; }

        clear_all(&mut out);
        header_line(&mut out, cols, " Jinxes ");
        hr(&mut out, cols, 2);
        wline(&mut out, 3, &format!("  {} jinxes loaded", all.len()));
        hr(&mut out, cols, 4);

        if detail {
            if let Some(item) = all.get(sel) {
                let desc = item.3.clone();
                let lines: Vec<String> = desc.split('\n').flat_map(|para| {
                    let mut out = Vec::new();
                    let mut cur = String::new();
                    for word in para.split_whitespace() {
                        if cur.is_empty() { cur.push_str(word); }
                        else if cur.len() + 1 + word.len() <= cols.saturating_sub(4) { cur.push(' '); cur.push_str(word); }
                        else { out.push(cur); cur = word.to_string(); }
                    }
                    if !cur.is_empty() { out.push(cur); }
                    out
                }).collect();
                for r in 0..body_h {
                    let idx = detail_scroll + r; let row = 5 + r;
                    if idx >= lines.len() { wline(&mut out, row, ""); }
                    else { wline(&mut out, row, &format!("  {}", lines[idx])); }
                }
            }
        } else {
            for r in 0..body_h {
                let idx = scroll + r; let row = 5 + r;
                if idx >= all.len() { wline(&mut out, row, ""); continue; }
                let (src, folder, name, desc) = &all[idx];
                let label = format!("{}/{}", folder, name);
                let d = desc.chars().take(cols - 28).collect::<String>();
                if idx == sel { wline(&mut out, row, &format!("\x1b[7m  > {:<22} [{}] {}\x1b[0m", label, src, d).pad(cols)); }
                else { wline(&mut out, row, &format!("    {:<22} \x1b[90m[{}]\x1b[0m {}", label, src, d)); }
            }
        }

        hr(&mut out, cols, rows - 2);
        let foot = if detail { " [j/k] Scroll  [q/Esc] Back " } else { " [j/k] Nav  [Enter] Detail  [q] Quit " };
        footer_line(&mut out, cols, rows, foot);
        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Char('q') | KeyCode::Esc => { if detail { detail = false; detail_scroll = 0; } else { break; } }
                KeyCode::Char('j') | KeyCode::Down => { if detail { detail_scroll += 1; } else if sel + 1 < all.len() { sel += 1; } }
                KeyCode::Char('k') | KeyCode::Up => { if detail { detail_scroll = detail_scroll.saturating_sub(1); } else if sel > 0 { sel -= 1; } }
                KeyCode::Enter => { if !detail { detail = true; detail_scroll = 0; } }
                _ => {}
            }
        }
    }
    Ok(())
}

fn scan_jinxes(dir: &Option<String>, source: &str, out: &mut Vec<(String, String, String, String)>) {
    let Some(d) = dir else { return };
    let jdir = PathBuf::from(d).join("jinxes");
    if let Ok(entries) = std::fs::read_dir(&jdir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                let folder = p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
                if let Ok(files) = std::fs::read_dir(&p) {
                    for f in files.flatten() {
                        let fp = f.path();
                        if fp.extension().and_then(|s| s.to_str()) == Some("jinx") {
                            let name = fp.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
                            let desc = if let Ok(content) = std::fs::read_to_string(&fp) {
                                content.lines().find(|l| l.trim_start().starts_with("description:"))
                                    .map(|l| l.split_once(':').map(|(_, v)| v.trim().to_string()).unwrap_or_default())
                                    .unwrap_or_default()
                            } else { String::new() };
                            out.push((source.to_string(), folder.clone(), name, desc));
                        }
                    }
                }
            }
        }
    }
}

// ────────────────────────────────
// /agent dashboard TUI
// ────────────────────────────────

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
struct AgentLog {
    timestamp: String,
    role: String,
    content: String,
    source: String,
    cost: Option<f64>,
}

fn load_agent_logs(kernel: &Kernel, npc_name: &str, limit: usize) -> Vec<AgentLog> {
    let mut logs: Vec<AgentLog> = Vec::new();

    // Conversation history messages for this NPC.
    if let Ok(msgs) = kernel.history.get_messages_by_npc(npc_name, limit) {
        for m in msgs {
            logs.push(AgentLog {
                timestamp: String::new(),
                role: m.role.clone(),
                content: m.content.unwrap_or_default(),
                source: format!("{} / {}", m.model.unwrap_or_default(), m.provider.unwrap_or_default()),
                cost: m.cost.and_then(|c| c.parse::<f64>().ok()),
            });
        }
    }

    // Jinx executions for this NPC via direct DB query.
    if let Ok(conn) = rusqlite::Connection::open(&kernel.history.db_path) {
        let mut stmt = conn.prepare(
            "SELECT jinx_name, input, output, status, timestamp, error_message FROM jinx_executions WHERE npc = ?1 ORDER BY timestamp DESC LIMIT ?2"
        ).ok();
        if let Some(mut stmt) = stmt {
            let rows = stmt.query_map(rusqlite::params![npc_name, limit as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5).unwrap_or_default(),
                ))
            });
            if let Ok(rows) = rows {
                for r in rows.flatten() {
                    let (jinx_name, input, output, status, timestamp, error) = r;
                    let content = if status == "error" {
                        format!("jinx /{} error: {} | {}", jinx_name, error, output)
                    } else {
                        format!("jinx /{} | input: {} | {}", jinx_name, input, output)
                    };
                    logs.push(AgentLog {
                        timestamp,
                        role: "jinx".to_string(),
                        content,
                        source: format!("jinx:{}", jinx_name),
                        cost: None,
                    });
                }
            }
        }
    }

    // NPC executions (direct LLM/tool runs) for this NPC.
    if let Ok(rows) = kernel.history.get_npc_executions(npc_name, limit) {
        for row in rows {
            let input = row.get("input").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let timestamp = row.get("timestamp").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let model = row.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let provider = row.get("provider").and_then(|v| v.as_str()).unwrap_or("").to_string();
            logs.push(AgentLog {
                timestamp,
                role: "execution".to_string(),
                content: input,
                source: format!("{} / {}", model, provider),
                cost: None,
            });
        }
    }

    logs.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    logs.truncate(limit);
    logs.reverse();
    logs
}

fn task_slug(task: &str, is_jinx: bool) -> String {
    let base = if is_jinx { format!("jinx_{}", task) } else { task.to_string() };
    base.to_lowercase()
        .replace(|c: char| !c.is_alphanumeric(), "_")
        .replace("__", "_")
        .trim_matches('_')
        .chars()
        .take(60)
        .collect::<String>()
}

fn load_task_runs(npc: &str, task: &str, is_jinx: bool) -> Vec<(String, String)> {
    let slug = task_slug(task, is_jinx);
    let base = PathBuf::from(shellexpand::tilde("~/.npcsh/loops").to_string())
        .join(npc)
        .join(slug)
        .join("runs");
    if !base.is_dir() { return Vec::new(); }
    let mut runs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&base) {
        let mut files: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().and_then(|s| s.to_str()) == Some("txt")).collect();
        files.sort();
        for f in files.iter().rev().take(50) {
            let ts = f.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
            let content = std::fs::read_to_string(f).unwrap_or_default();
            runs.push((ts, content));
        }
    }
    runs.reverse();
    runs
}

fn show_run_view(content: &str) -> Result<()> {
    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();
    let mut scroll: usize = 0;
    let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    loop {
        let (cols, rows) = term_size();
        let body_h = rows.saturating_sub(4).max(1);
        clear_all(&mut out);
        header_line(&mut out, cols, " Run output ");
        hr(&mut out, cols, 2);
        for r in 0..body_h {
            let idx = scroll + r;
            let row = 3 + r;
            if idx >= lines.len() { wline(&mut out, row, ""); }
            else {
                let line = lines[idx].chars().take(cols.saturating_sub(2)).collect::<String>();
                wline(&mut out, row, &format!("  {}", line));
            }
        }
        hr(&mut out, cols, rows - 2);
        footer_line(&mut out, cols, rows, " [j/k] Scroll  [q/Esc] Back ");
        let _ = out.flush();
        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Char('j') | KeyCode::Down => { if scroll + 1 < lines.len() { scroll += 1; } }
                KeyCode::Char('k') | KeyCode::Up => { if scroll > 0 { scroll -= 1; } }
                _ => {}
            }
        }
    }
    Ok(())
}

pub fn run_agent_dashboard_tui(
    kernel: &mut Kernel,
    npc_name: &str,
    registry: &Arc<Mutex<crate::cron::CronRegistry>>,
) -> Result<()> {
    let _guard = RawModeGuard::new().map_err(|e| npcrs::NpcError::Other(e.to_string()))?;
    let mut out = io::stdout();

    let Some(proc) = kernel.find_by_name(npc_name).map(|p| p.pid) else {
        return Err(npcrs::NpcError::Other(format!("NPC '{}' not found", npc_name)));
    };
    let pid = proc;

    #[derive(Clone, PartialEq)]
    enum AgentTab { Tasks, TaskRuns, Logs, Details }
    impl AgentTab {
        fn next(&self) -> AgentTab {
            match self {
                AgentTab::Tasks => AgentTab::TaskRuns,
                AgentTab::TaskRuns => AgentTab::Logs,
                AgentTab::Logs => AgentTab::Details,
                AgentTab::Details => AgentTab::Tasks,
            }
        }
        fn prev(&self) -> AgentTab {
            match self {
                AgentTab::Tasks => AgentTab::Details,
                AgentTab::TaskRuns => AgentTab::Tasks,
                AgentTab::Logs => AgentTab::TaskRuns,
                AgentTab::Details => AgentTab::Logs,
            }
        }
    }
    let mut tab = AgentTab::Tasks;
    let mut sel: usize = 0;
    let mut scroll: usize = 0;
    let mut selected_job: Option<crate::cron::CronJob> = None;
    let mut runs: Vec<(String, String)> = Vec::new(); // (timestamp_label, content)

    loop {
        let (cols, rows) = term_size();
        let body_h = rows.saturating_sub(6).max(1);
        clear_all(&mut out);
        header_line(&mut out, cols, &format!(" Agent @{} ", npc_name));
        hr(&mut out, cols, 2);
        let tabs = format!(
            "  {} Tasks    {} Runs    {} Logs    {} Details",
            if tab == AgentTab::Tasks { "▸" } else { " " },
            if tab == AgentTab::TaskRuns { "▸" } else { " " },
            if tab == AgentTab::Logs { "▸" } else { " " },
            if tab == AgentTab::Details { "▸" } else { " " }
        );
        wline(&mut out, 3, &tabs);
        hr(&mut out, cols, 4);

        let reg = registry.lock().unwrap();
        let jobs: Vec<crate::cron::CronJob> = reg.list().iter().filter(|j| j.npc == npc_name).cloned().collect();
        let logs = load_agent_logs(kernel, npc_name, 200);
        drop(reg);

        let items = match tab {
            AgentTab::Tasks => jobs.len(),
            AgentTab::TaskRuns => {
                if runs.is_empty() && selected_job.is_some() {
                    let j = selected_job.as_ref().unwrap();
                    runs = load_task_runs(&j.npc, &j.task, j.kind == crate::cron::CronJobKind::Jinx);
                }
                runs.len()
            }
            AgentTab::Logs => logs.len(),
            AgentTab::Details => 1,
        };
        if sel >= items && items > 0 { sel = items - 1; }
        if sel < scroll { scroll = sel; }
        else if sel >= scroll + body_h { scroll = sel.saturating_sub(body_h) + 1; }

        for r in 0..body_h {
            let idx = scroll + r;
            let row = 5 + r;
            if idx >= items { wline(&mut out, row, ""); continue; }
            let text = match tab {
                AgentTab::Tasks => {
                    let j = &jobs[idx];
                    let kind = if j.kind == crate::cron::CronJobKind::Jinx { "jinx" } else { "chat" };
                    let status = if j.enabled { "\x1b[32menabled\x1b[0m" } else { "\x1b[31mdisabled\x1b[0m" };
                    format!("[{}] every {} [{}] {} ({}) {}", j.id, crate::cron::format_duration(j.interval_secs), kind, j.task, if j.last_run.is_some() { "ran" } else { "never" }, status)
                }
                AgentTab::TaskRuns => {
                    let (ts, _) = &runs[idx];
                    format!("run {}", ts)
                }
                AgentTab::Logs => {
                    let log = &logs[idx];
                    let preview = log.content.chars().take(cols.saturating_sub(28)).collect::<String>();
                    let cost_str = log.cost.map(|c| format!(" ${:.4}", c)).unwrap_or_default();
                    let ts = if log.timestamp.is_empty() { String::new() } else { format!("{} ", log.timestamp) };
                    format!("{}{}{}{}{}", ts, log.role, if log.source.is_empty() { "" } else { ":" }, log.source, cost_str)
                }
                AgentTab::Details => {
                    if let Some(p) = kernel.get_process(pid) {
                        format!("model={} provider={} tokens={}/{} cost=${:.4} turns={}",
                            p.npc.resolved_model(),
                            p.npc.resolved_provider(),
                            p.usage.total_input_tokens,
                            p.usage.total_output_tokens,
                            p.usage.total_cost_usd,
                            p.usage.total_turns)
                    } else { "(process not found)".to_string() }
                }
            };
            let truncated = text.chars().take(cols.saturating_sub(4)).collect::<String>();
            if idx == sel { wline(&mut out, row, &format!("\x1b[7m  > {}\x1b[0m", truncated).pad(cols)); }
            else { wline(&mut out, row, &format!("    {}", truncated)); }
        }

        hr(&mut out, cols, rows - 2);
        let foot = if tab == AgentTab::TaskRuns {
            " [Tab] Cycle  [j/k] Nav  [Enter] View run  [q] Quit "
        } else if tab == AgentTab::Tasks {
            " [Tab] Cycle  [j/k] Nav  [Enter] View runs  [Space] Toggle  [d] Delete  [q] Quit "
        } else {
            " [Tab] Cycle  [j/k] Nav  [q] Quit "
        };
        footer_line(&mut out, cols, rows, foot);
        let _ = out.flush();

        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Release { continue; }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => break,
                KeyCode::Esc | KeyCode::Char('q') => break,
                KeyCode::Tab => {
                    if key.modifiers.contains(KeyModifiers::SHIFT) {
                        tab = tab.prev(); sel = 0; scroll = 0;
                    } else {
                        tab = tab.next(); sel = 0; scroll = 0;
                    }
                }
                KeyCode::BackTab => { tab = tab.prev(); sel = 0; scroll = 0; }
                KeyCode::Char('j') | KeyCode::Down => { if sel + 1 < items { sel += 1; } }
                KeyCode::Char('k') | KeyCode::Up => { if sel > 0 { sel -= 1; } }
                KeyCode::Enter => {
                    if tab == AgentTab::Tasks && !jobs.is_empty() {
                        selected_job = Some(jobs[sel].clone());
                        runs = load_task_runs(&jobs[sel].npc, &jobs[sel].task, jobs[sel].kind == crate::cron::CronJobKind::Jinx);
                        tab = AgentTab::TaskRuns;
                        sel = 0; scroll = 0;
                    } else if tab == AgentTab::TaskRuns && !runs.is_empty() {
                        show_run_view(&runs[sel].1)?;
                    }
                }
                KeyCode::Char(' ') if tab == AgentTab::Tasks && !jobs.is_empty() => {
                    let id = jobs[sel].id;
                    let new_state = !jobs[sel].enabled;
                    let _ = registry.lock().unwrap().enable(id, new_state);
                }
                KeyCode::Char('d') | KeyCode::Char('D') if tab == AgentTab::Tasks && !jobs.is_empty() => {
                    let id = jobs[sel].id;
                    registry.lock().unwrap().remove(id);
                    if sel >= jobs.len().saturating_sub(1) { sel = jobs.len().saturating_sub(2); }
                }
                _ => {}
            }
        }
    }
    Ok(())
}
