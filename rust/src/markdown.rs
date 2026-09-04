use std::io::IsTerminal;
use std::time::{Duration, Instant};

/// Render a markdown string to an ANSI-styled string using the terminal width.
pub fn render_block(md: &str) -> String {
    if md.trim().is_empty() {
        return md.to_string();
    }
    let skin = termimad::MadSkin::default();
    format!("{}", skin.term_text(md))
}

/// Renderer that streams markdown deltas to stderr.
///
/// During streaming we only emit *complete* lines as raw text.  Markdown
/// wrapping is applied only to the final trailing partial line on flush().
/// This avoids termimad wrapping partial output and prevents progressive
/// indentation drift.
pub struct StreamRenderer {
    /// Raw accumulated source text since the last newline.
    buffer: String,
    skin: termimad::MadSkin,
    last_render: Instant,
    min_interval: Duration,
    disabled: bool,
}

impl StreamRenderer {
    pub fn new() -> Self {
        let disabled = !std::io::stderr().is_terminal();
        Self {
            buffer: String::new(),
            skin: termimad::MadSkin::default(),
            last_render: Instant::now(),
            min_interval: Duration::from_millis(50),
            disabled,
        }
    }

    /// Append a raw markdown delta.
    ///
    /// Complete lines are emitted as raw text immediately.  The final
    /// trailing partial line is left buffered and markdown-rendered on flush.
    pub fn push(&mut self, text: &str) {
        if self.disabled {
            eprint!("{}", text);
            let _ = std::io::Write::flush(&mut std::io::stderr());
            return;
        }

        self.buffer.push_str(text);

        // Emit any complete lines that just ended.
        while let Some(pos) = self.buffer.find('\n') {
            let line = &self.buffer[..=pos];
            // Strip trailing \r for CRLF streams.
            let line = line.strip_suffix('\n').unwrap_or(line);
            let line = line.strip_suffix('\r').unwrap_or(line);
            eprintln!("{}", line);
            self.buffer.replace_range(..=pos, "");
            self.last_render = Instant::now();
        }
    }

    /// Force a final flush of any remaining unemitted text.
    pub fn flush(&mut self) {
        if self.disabled {
            return;
        }
        if self.buffer.trim_end_matches(['\n', '\r']).is_empty() {
            self.buffer.clear();
            return;
        }
        let rendered = format!("{}", self.skin.term_text(&self.buffer));
        let rendered = rendered.trim_end_matches(['\n', '\r']).to_string();
        if !rendered.is_empty() {
            eprint!("{}", rendered);
        }
        self.buffer.clear();
        self.last_render = Instant::now();
    }

    /// Clear the accumulated buffer and forget emitted progress.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

impl Default for StreamRenderer {
    fn default() -> Self {
        Self::new()
    }
}
