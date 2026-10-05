use crate::npcsh_home;
use nql::compiler::{Compiler, Target};
use rusqlite::{Connection, params};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const MIGRATIONS_TABLE_SQL: &str = "
CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
";

const BUNDLED_MIGRATIONS: &[(&str, &str)] = &[(
    "001_fix_duplicate_cost_totals_1005_2026.sql",
    include_str!("../../npcsh/db_migrations/001_fix_duplicate_cost_totals_1005_2026.sql"),
)];

fn migrations_dir() -> PathBuf {
    if let Ok(test_dir) = std::env::var("NPCSH_MIGRATIONS_TEST_DIR") {
        return Path::new(&test_dir).to_path_buf();
    }
    npcsh_home().join("db_migrations")
}

fn ensure_migrations_dir() -> Result<PathBuf, String> {
    let dir = migrations_dir();
    if !dir.is_dir() {
        fs::create_dir_all(&dir)
            .map_err(|e| format!("failed to create migrations dir {}: {e}", dir.display()))?;
    }
    for (name, contents) in BUNDLED_MIGRATIONS {
        let path = dir.join(name);
        if !path.is_file() {
            fs::write(&path, contents).map_err(|e| {
                format!("failed to write bundled migration {}: {e}", path.display())
            })?;
        }
    }
    Ok(dir)
}

fn start_spinner(message: &str) -> Arc<AtomicBool> {
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    let msg = message.to_string();
    thread::spawn(move || {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let mut i = 0;
        while running_clone.load(Ordering::Relaxed) {
            eprint!("\r{} {}...", frames[i % frames.len()], msg);
            let _ = std::io::Write::flush(&mut std::io::stderr());
            i += 1;
            thread::sleep(Duration::from_millis(80));
        }
        eprint!("\r{} done.\n", msg);
        let _ = std::io::Write::flush(&mut std::io::stderr());
    });
    running
}

fn stop_spinner(running: Arc<AtomicBool>) {
    running.store(false, Ordering::Relaxed);
    thread::sleep(Duration::from_millis(120));
}

fn pending_migrations(
    conn: &Connection,
    migrations_dir: &Path,
) -> Result<Vec<(String, PathBuf)>, String> {
    let mut entries: Vec<PathBuf> = fs::read_dir(migrations_dir)
        .map_err(|e| {
            format!(
                "failed to read migrations dir {}: {}",
                migrations_dir.display(),
                e
            )
        })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sql"))
        .collect();
    entries.sort();

    let mut pending = Vec::new();
    for path in entries {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            continue;
        }
        let applied: bool = conn
            .query_row(
                "SELECT 1 FROM schema_migrations WHERE name = ?1",
                params![&name],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !applied {
            pending.push((name, path));
        }
    }
    Ok(pending)
}

fn prompt_proceed(pending: &[(String, PathBuf)]) -> bool {
    if std::env::var("NPCSH_MIGRATIONS_AUTO_CONFIRM").is_ok() {
        return true;
    }
    if pending.len() == 1 {
        eprintln!("MIGRATION DETECTED FOR DB: {}", pending[0].0);
    } else {
        eprintln!("MIGRATIONS DETECTED FOR DB: {}", pending.len());
        for (name, _) in pending {
            eprintln!("  - {}", name);
        }
    }
    eprint!("PROCEED? [y/N] ");
    let _ = std::io::Write::flush(&mut std::io::stderr());

    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
}

pub fn run_migrations(db_path: &str) -> Result<(), String> {
    let spinner = start_spinner("checking database migrations");

    let migrations_dir = ensure_migrations_dir()?;

    let conn = Connection::open(db_path)
        .map_err(|e| format!("failed to open history db for migrations: {e}"))?;

    conn.execute_batch(MIGRATIONS_TABLE_SQL)
        .map_err(|e| format!("failed to create schema_migrations table: {e}"))?;

    let pending = pending_migrations(&conn, &migrations_dir)?;

    stop_spinner(spinner);

    if pending.is_empty() {
        return Ok(());
    }

    if !prompt_proceed(&pending) {
        return Err("database migration declined by user".to_string());
    }

    let spinner = start_spinner("applying database migrations");

    let compiler = Compiler::new(Target::Sqlite);
    for (name, path) in pending {
        let sql = fs::read_to_string(&path)
            .map_err(|e| format!("failed to read migration {}: {}", path.display(), e))?;
        let compiled = compiler.compile(&sql, Target::Sqlite);

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("failed to begin migration transaction: {e}"))?;
        tx.execute_batch(&compiled)
            .map_err(|e| format!("migration {name} failed: {e}"))?;
        tx.execute(
            "INSERT INTO schema_migrations (name) VALUES (?1)",
            params![&name],
        )
        .map_err(|e| format!("failed to record migration {name}: {e}"))?;
        tx.commit()
            .map_err(|e| format!("failed to commit migration {name}: {e}"))?;

        eprintln!("[npcsh] applied db migration: {}", name);
    }

    stop_spinner(spinner);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;
    use std::env;

    #[test]
    fn test_migration_cleans_duplicate_cost_rows() {
        let tmp_dir =
            std::env::temp_dir().join(format!("npcsh_migrations_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).unwrap();
        let migration_file = tmp_dir.join("001_fix_duplicate_cost_totals_1005_2026.sql");
        fs::write(&migration_file, BUNDLED_MIGRATIONS[0].1).unwrap();

        let db_path = tmp_dir.join("history.db").to_string_lossy().to_string();
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE NOT NULL,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                conversation_id TEXT,
                directory_path TEXT,
                model TEXT,
                provider TEXT,
                npc TEXT,
                team TEXT,
                reasoning_content TEXT,
                tool_calls TEXT,
                tool_results TEXT,
                parent_message_id TEXT,
                device_id TEXT,
                device_name TEXT,
                params TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost TEXT
            );",
        )
        .unwrap();

        let conv_id = "conv-1";
        let rows = [
            ("user", "hello", None, None::<u64>, None::<u64>, None::<f64>),
            (
                "assistant",
                "hi there",
                Some("test-model"),
                Some(100_u64),
                Some(50_u64),
                Some(0.0015_f64),
            ),
            (
                "user",
                "hello",
                Some("test-model"),
                Some(10000_u64),
                None,
                None,
            ),
            (
                "assistant",
                "hi there",
                Some("test-model"),
                Some(10000_u64),
                Some(5000_u64),
                Some(0.15_f64),
            ),
        ];
        for (role, content, model, inp, out, cost) in rows {
            let cost_str = cost.map(|c| format!("{:.6}", c));
            conn.execute(
                "INSERT INTO conversation_history
                 (message_id, timestamp, role, content, conversation_id, model,
                  input_tokens, output_tokens, cost)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    uuid::Uuid::new_v4().to_string(),
                    "2026-01-01 00:00:00",
                    role,
                    content,
                    conv_id,
                    model,
                    inp,
                    out,
                    cost_str,
                ],
            )
            .unwrap();
        }

        unsafe { env::set_var("NPCSH_MIGRATIONS_TEST_DIR", &tmp_dir) };
        unsafe { env::set_var("NPCSH_MIGRATIONS_AUTO_CONFIRM", "1") };
        let original_home = env::var("HOME").ok();
        unsafe { env::set_var("HOME", &tmp_dir) };
        run_migrations(&db_path).unwrap();
        if let Some(h) = original_home {
            unsafe { env::set_var("HOME", h) };
        } else {
            unsafe { env::remove_var("HOME") };
        }
        unsafe { env::remove_var("NPCSH_MIGRATIONS_TEST_DIR") };
        unsafe { env::remove_var("NPCSH_MIGRATIONS_AUTO_CONFIRM") };

        let user_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM conversation_history WHERE role = 'user'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let assistant_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM conversation_history WHERE role = 'assistant'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let total_cost: f64 = conn
            .query_row(
                "SELECT COALESCE(SUM(CAST(cost AS REAL)), 0) FROM conversation_history",
                [],
                |r| r.get(0),
            )
            .unwrap();

        assert_eq!(user_count, 1, "duplicate user re-save should be deleted");
        assert_eq!(
            assistant_count, 1,
            "duplicate assistant re-save should be deleted"
        );
        assert!(
            (total_cost - 0.0015).abs() < 0.0001,
            "cost total should be ~0.0015, got {}",
            total_cost
        );

        let applied: i64 = conn
            .query_row("SELECT COUNT(*) FROM schema_migrations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(applied, 1, "migration should be recorded");

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
