//! Nesting-safe terminal raw mode.
//!
//! `crossterm::terminal::enable_raw_mode()` is not reference counted: it
//! returns early when the terminal is already raw and remembers the *original*
//! termios once. Its counterpart `disable_raw_mode()` unconditionally restores
//! that original termios. So two overlapping owners (for example the streaming
//! turn and a permission prompt raised from inside it) break each other: the
//! inner prompt's teardown drops the terminal out of raw mode while the outer
//! turn is still running.
//!
//! That matters for Ctrl+C. In raw mode the terminal does not generate SIGINT
//! and Ctrl+C arrives as a key event, which the shell can turn into a graceful
//! cancel. Outside raw mode the tty generates a real SIGINT, and without a
//! signal handler the default disposition kills the process.
//!
//! `RawModeGuard` remembers whether the terminal was already raw when it was
//! created and only restores the previous state if it was the owner that
//! enabled raw mode. Nested guards therefore leave raw mode alone.

use std::io;

pub struct RawModeGuard {
    we_enabled: bool,
}

impl RawModeGuard {
    /// Enter raw mode, or record that it was already enabled.
    pub fn acquire() -> io::Result<Self> {
        let already_raw = crossterm::terminal::is_raw_mode_enabled().unwrap_or(false);
        if !already_raw {
            crossterm::terminal::enable_raw_mode()?;
        }
        Ok(Self {
            we_enabled: !already_raw,
        })
    }

    /// Best-effort variant: a failure to enter raw mode is not fatal, the guard
    /// simply does nothing on drop.
    pub fn acquire_lossy() -> Self {
        Self::acquire().unwrap_or(Self { we_enabled: false })
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        if self.we_enabled {
            let _ = crossterm::terminal::disable_raw_mode();
        }
    }
}
