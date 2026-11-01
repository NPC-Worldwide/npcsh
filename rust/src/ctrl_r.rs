//! Ctrl+R reverse history search for npcsh.
//!
//! Searches ~/.npcsh_history (deduped, most-recent-first). Typing narrows
//! matches live, Ctrl+R / arrows cycle candidates, Enter accepts the current
//! match into the input line, Esc/Ctrl+G/Ctrl+C abort.

use crossterm::{
    event::{Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal,
};
use std::io::{self, Write};
use std::path::PathBuf;

fn history_file() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".npcsh_history")
}

/// Load history, most-recent-first, deduped.
pub fn load_history() -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    if let Ok(content) = std::fs::read_to_string(history_file()) {
        for line in content.lines().rev() {
            let line = line.trim();
            if !line.is_empty() && seen.insert(line.to_string()) {
                out.push(line.to_string());
            }
        }
    }
    out
}

struct State<'a> {
    orig_line: &'a str,
    query: String,
    idx: usize,
    entries: Vec<String>,
}

impl<'a> State<'a> {
    fn matches(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| self.query.is_empty() || e.contains(&self.query))
            .map(|e| e.as_str())
            .collect()
    }

    fn current(&self) -> Option<&str> {
        self.matches().get(self.idx).copied()
    }
}

fn render(st: &State, cols: u16) {
    let cand = st.current().unwrap_or("(no match)");
    let w = cols as usize;
    // clamp display
    let disp: String = cand.chars().take(w.saturating_sub(30)).collect();
    let q_disp: String = st.query.chars().take(40).collect();
    print!(
        "\r\x1b[K\x1b[90m(reverse-i-search)`{}'\x1b[0m {}",
        q_disp, disp
    );
    let _ = io::stdout().flush();
}

fn clear_line() {
    print!("\r\x1b[K");
    let _ = io::stdout().flush();
}

/// Interactive Ctrl+R loop. Returns the accepted command text (or empty
/// string on abort — the caller should restore the original input line).
pub fn reverse_search(orig_line: &str) -> String {
    let entries = load_history();
    if entries.is_empty() {
        eprintln!("\r\x1b[90m(no npcsh history)\x1b[0m");
        let _ = io::stderr().flush();
        return orig_line.to_string();
    }

    let mut st = State {
        orig_line,
        query: String::new(),
        idx: 0,
        entries,
    };

    let (cols, _) = terminal::size().unwrap_or((120, 24));
    let _ = render_initial(&st, cols);

    let result = loop {
        if let Ok(Event::Key(key)) = crossterm::event::read() {
            if key.kind == KeyEventKind::Release {
                continue;
            }
            match (key.code, key.modifiers) {
                (KeyCode::Char('r'), m) if m.contains(KeyModifiers::CONTROL) => {
                    st.idx += 1;
                    if st.idx >= st.matches().len() {
                        st.idx = 0;
                    }
                }
                (KeyCode::Char('g'), m) if m.contains(KeyModifiers::CONTROL) => {
                    break None;
                }
                (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => {
                    break None;
                }
                (KeyCode::Esc, _) => break None,
                (KeyCode::Enter, _) => {
                    break st.current().map(|s| s.to_string());
                }
                (KeyCode::Backspace, _) => {
                    st.query.pop();
                    st.idx = 0;
                }
                (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                    st.query.push(c);
                    st.idx = 0;
                }
                (KeyCode::Down, _) => {
                    let n = st.matches().len();
                    if n > 0 {
                        st.idx = (st.idx + 1) % n;
                    }
                }
                (KeyCode::Up, _) => {
                    st.idx = st.idx.saturating_sub(1);
                }
                _ => {}
            }
            render(&st, cols);
        }
    };

    clear_line();
    match result {
        Some(s) => s,
        None => orig_line.to_string(),
    }
}

fn render_initial(st: &State, cols: u16) -> std::io::Result<()> {
    clear_line();
    print!("\x1b[90m(reverse-i-search) query: \x1b[0m");
    let _ = io::stdout().flush();
    let _ = st;
    let _ = cols;
    Ok(())
}
