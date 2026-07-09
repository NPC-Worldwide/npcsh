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

/// Renderer that streams markdown while avoiding duplication and offsets.
///
/// The server sometimes sends the full accumulated content in every delta, so
/// we only append the genuinely new suffix.  To make output visible as it
/// streams we overwrite the current block in place using absolute cursor moves
/// based on the *previously emitted* height.  We commit (move to a new line)
/// when the content ends with a newline so the next block starts fresh.
pub struct StreamRenderer {
    buffer: String,
    skin: termimad::MadSkin,
    last_height: usize,
    last_render: Instant,
    min_interval: Duration,
    disabled: bool,
    dirty: bool,
}

impl StreamRenderer {
    pub fn new() -> Self {
        let disabled = !std::io::stderr().is_terminal();
        Self {
            buffer: String::new(),
            skin: termimad::MadSkin::default(),
            last_height: 0,
            last_render: Instant::now(),
            min_interval: Duration::from_millis(50),
            disabled,
            dirty: false,
        }
    }

    /// Append a raw markdown delta and re-render if enough time has passed or
    /// the delta ends with a newline.
    pub fn push(&mut self, text: &str) {
        if self.disabled {
            eprint!("{}", text);
            self.buffer.push_str(text);
            let _ = std::io::Write::flush(&mut std::io::stderr());
            return;
        }
        let was_empty = self.buffer.is_empty();
        self.buffer.push_str(text);
        self.dirty = true;

        let ends_with_newline = self.buffer.ends_with('\n') || self.buffer.ends_with("\r\n");
        let now = Instant::now();
        if was_empty
            || now.duration_since(self.last_render) >= self.min_interval
            || ends_with_newline
        {
            self.render();
            if ends_with_newline {
                eprintln!();
                self.last_height = 0;
                self.buffer.clear();
                self.dirty = false;
            }
        }
    }

    /// Force a final render and reset the saved height.
    pub fn flush(&mut self) {
        if self.disabled {
            return;
        }
        self.render();
        self.last_height = 0;
    }

    /// Clear the accumulated buffer and forget the previous height.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_height = 0;
        self.dirty = false;
    }

    fn render(&mut self) {
        if self.buffer.is_empty() || !self.dirty {
            return;
        }

        let raw = self.buffer.trim_end_matches(['\n', '\r']);
        if raw.is_empty() {
            return;
        }

        let rendered = format!("{}", self.skin.term_text(raw));
        let rendered = rendered.trim_end_matches(['\n', '\r']).to_string();
        if rendered.is_empty() {
            return;
        }

        if self.last_height > 0 {
            let up = self.last_height.saturating_sub(1);
            if up > 0 {
                eprint!("\x1b[{}A", up);
            }
            eprint!("\x1b[G\x1b[J");
        }

        eprint!("{}", rendered);

        self.last_height = rendered.chars().filter(|c| *c == '\n').count() + 1;
        self.last_render = Instant::now();
        self.dirty = false;
    }
}

impl Default for StreamRenderer {
    fn default() -> Self {
        Self::new()
    }
}
