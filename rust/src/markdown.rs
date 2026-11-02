use std::io::IsTerminal;
use std::time::{Duration, Instant};

const BOLD: &str = "\x1b[1m";
const ITALIC: &str = "\x1b[3m";
const DIM: &str = "\x1b[90m";
const CYAN: &str = "\x1b[36m";
const UNDERLINE: &str = "\x1b[4m";
const RESET: &str = "\x1b[0m";

fn format_inline_markdown(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut code = String::new();
            let mut closed = false;
            while let Some(c) = chars.next() {
                if c == '`' {
                    closed = true;
                    break;
                }
                code.push(c);
            }
            out.push_str(CYAN);
            out.push_str(&code);
            if closed {
                out.push_str(RESET);
            }
        } else if ch == '*' {
            if chars.peek() == Some(&'*') {
                chars.next();
                let mut inner = String::new();
                let mut closed = false;
                while let Some(c) = chars.next() {
                    if c == '*' && chars.peek() == Some(&'*') {
                        chars.next();
                        closed = true;
                        break;
                    }
                    inner.push(c);
                }
                out.push_str(BOLD);
                out.push_str(&format_inline_markdown(&inner));
                if closed {
                    out.push_str(RESET);
                }
            } else {
                let mut inner = String::new();
                let mut closed = false;
                while let Some(c) = chars.next() {
                    if c == '*' {
                        closed = true;
                        break;
                    }
                    inner.push(c);
                }
                out.push_str(ITALIC);
                out.push_str(&format_inline_markdown(&inner));
                if closed {
                    out.push_str(RESET);
                }
            }
        } else if ch == '_' {
            if chars.peek() == Some(&'_') {
                chars.next();
                let mut inner = String::new();
                let mut closed = false;
                while let Some(c) = chars.next() {
                    if c == '_' && chars.peek() == Some(&'_') {
                        chars.next();
                        closed = true;
                        break;
                    }
                    inner.push(c);
                }
                out.push_str(BOLD);
                out.push_str(&format_inline_markdown(&inner));
                if closed {
                    out.push_str(RESET);
                }
            } else {
                let mut inner = String::new();
                let mut closed = false;
                while let Some(c) = chars.next() {
                    if c == '_' {
                        closed = true;
                        break;
                    }
                    inner.push(c);
                }
                out.push_str(ITALIC);
                out.push_str(&format_inline_markdown(&inner));
                if closed {
                    out.push_str(RESET);
                }
            }
        } else if ch == '~' && chars.peek() == Some(&'~') {
            chars.next();
            let mut inner = String::new();
            let mut closed = false;
            while let Some(c) = chars.next() {
                if c == '~' && chars.peek() == Some(&'~') {
                    chars.next();
                    closed = true;
                    break;
                }
                inner.push(c);
            }
            out.push_str(DIM);
            out.push_str(&inner);
            if closed {
                out.push_str(RESET);
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn format_line_markdown(line: &str) -> String {
    if line.starts_with("```") {
        return format!("{}{}{}", DIM, line, RESET);
    }
    if line.starts_with("# ") {
        return format!(
            "{}{}{}{}{}",
            BOLD,
            UNDERLINE,
            format_inline_markdown(&line[2..]),
            RESET,
            RESET
        );
    }
    if line.starts_with("## ") || line.starts_with("### ") || line.starts_with("#### ") {
        let start = line.find(' ').unwrap_or(0) + 1;
        return format!(
            "{}{}{}{}",
            BOLD,
            format_inline_markdown(&line[start..]),
            RESET,
            RESET
        );
    }
    if line.trim() == "---" || line.trim() == "***" || line.trim() == "___" {
        let width = 60;
        return format!("{}{}{}", DIM, "─".repeat(width), RESET);
    }
    format_inline_markdown(line)
}

pub fn render_block(md: &str) -> String {
    md.lines()
        .map(format_line_markdown)
        .collect::<Vec<_>>()
        .join("\n")
}

pub struct StreamRenderer {
    buffer: String,
    at_line_start: bool,
    last_render: Instant,
    min_interval: Duration,
    disabled: bool,
    started: bool,
}

impl StreamRenderer {
    pub fn new() -> Self {
        let disabled = false;
        Self {
            buffer: String::new(),
            at_line_start: true,
            last_render: Instant::now(),
            min_interval: Duration::from_millis(50),
            disabled,
            started: false,
        }
    }

    fn emit_line(&mut self, line: &str) {
        let line = line.trim_start();
        eprint!("{}\r\n", format_line_markdown(line));
        let _ = std::io::Write::flush(&mut std::io::stderr());
        self.last_render = Instant::now();
    }

    pub fn push(&mut self, text: &str) {
        if self.disabled {
            eprint!("{}", text);
            let _ = std::io::Write::flush(&mut std::io::stderr());
            return;
        }
        if !self.started {
            self.started = true;
            eprint!("\r\n");
        }
        for ch in text.chars() {
            if ch == '\n' {
                self.flush_line();
                self.at_line_start = true;
            } else if self.at_line_start && (ch == ' ' || ch == '\t') {
                continue;
            } else {
                self.buffer.push(ch);
                self.at_line_start = false;
            }
        }
    }

    fn flush_line(&mut self) {
        let line = self.buffer.trim_end_matches(['\n', '\r']).trim_start();
        if !line.is_empty() {
            eprint!("{}\r\n", format_line_markdown(line));
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
        self.buffer.clear();
    }

    pub fn flush(&mut self) {
        if self.disabled {
            return;
        }
        self.flush_line();
        self.at_line_start = true;
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.at_line_start = true;
    }
}

impl Default for StreamRenderer {
    fn default() -> Self {
        Self::new()
    }
}
