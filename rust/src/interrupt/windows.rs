use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Windows interrupt listener.
///
/// On Windows, tokio::signal::ctrl_c() maps Ctrl+C and Ctrl+Break events to
/// the same signal. Combining that with crossterm key-event polling gives the
/// most reliable interrupt handling while keeping the Enter/queue behavior.
pub fn spawn_listener(
    interrupt_tx: tokio::sync::mpsc::UnboundedSender<()>,
    queue_tx: tokio::sync::mpsc::UnboundedSender<String>,
    running: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    let signal_interrupt_tx = interrupt_tx.clone();
    let signal_running = running.clone();
    tokio::spawn(async move {
        // Re-arm after each delivery so one Ctrl+C does not disable interrupt
        // handling for the rest of a long-lived shell session.
        while tokio::signal::ctrl_c().await.is_ok() {
            let _ = signal_interrupt_tx.send(());
            if !signal_running.load(Ordering::Relaxed) {
                break;
            }
        }
    });

    tokio::task::spawn_blocking(move || {
        let mut buf = String::new();
        while running.load(Ordering::Relaxed) {
            if event::poll(Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == crossterm::event::KeyEventKind::Release {
                        continue;
                    }
                    match key.code {
                        KeyCode::Esc => {
                            let _ = interrupt_tx.send(());
                        }
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            let _ = interrupt_tx.send(());
                        }
                        KeyCode::Enter => {
                            let line = std::mem::take(&mut buf);
                            if !line.is_empty() {
                                let _ = queue_tx.send(line);
                            }
                        }
                        KeyCode::Char(c) => {
                            buf.push(c);
                        }
                        KeyCode::Backspace => {
                            buf.pop();
                        }
                        _ => {}
                    }
                }
            }
        }
    })
}
