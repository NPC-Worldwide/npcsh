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

/// Renderer that updates a growing markdown buffer in place on the terminal.
/// It uses termimad to render the accumulated markdown to ANSI and overwrites
/// the previous render using cursor control sequences.
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

    /// Append a raw markdown delta and, if enough time has passed, re-render.
    pub fn push(&mut self, text: &str) {
        if self.disabled {
            eprint!("{}", text);
            return;
        }
        let was_empty = self.buffer.is_empty();
        self.buffer.push_str(text);
        self.dirty = true;
        let now = Instant::now();
        if was_empty || now.duration_since(self.last_render) >= self.min_interval {
            self.render();
        }
    }

    /// Force an immediate re-render of the current buffer.
    pub fn flush(&mut self) {
        if self.disabled {
            return;
        }
        self.render();
    }

    /// Clear the internal buffer and forget any previous on-screen height.
    /// Call this after flushing and printing something else inline (e.g. a
    /// tool result) so the next content block starts fresh below it.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_height = 0;
        self.dirty = false;
    }

    fn render(&mut self) {
        if self.buffer.is_empty() || !self.dirty {
            return;
        }

        let rendered = format!("{}", self.skin.term_text(&self.buffer));
        let rendered = rendered.trim_end_matches(['\n', '\r']).to_string();
        if rendered.is_empty() {
            return;
        }

        let height = rendered.chars().filter(|&c| c == '\n').count() + 1;

        if self.last_height > 0 {
            let up = self.last_height.saturating_sub(1);
            if up > 0 {
                eprint!("\x1b[{}A", up);
            }
            eprint!("\x1b[G\x1b[J");
        }

        eprint!("{}", rendered);

        self.last_height = height;
        self.last_render = Instant::now();
        self.dirty = false;
    }
}

impl Default for StreamRenderer {
    fn default() -> Self {
        Self::new()
    }
}
