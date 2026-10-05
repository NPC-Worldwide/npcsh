use crate::{expand_tilde, npcsh_home};
use nql::compiler::{Compiler, Target};
use rusqlite::{Connection, params};
use std::fs;
use std::path::PathBuf;

const MIGRATIONS_TABLE_SQL: &str = "
CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
";

/// Resolve the directory that holds .sql migration files.
///
/// Order of precedence:
/// 1. `NPCSH_DB_MIGRATIONS_DIR` environment variable.
/// 2. `~/.npcsh/db_migrations` (canonical installed location).
/// 3. Next to the running executable:
///    - `<exe_dir>/db_migrations`  (installed layout: `~/.npcsh/bin/npcsh` + `~/.npcsh/bin/db_migrations`)
///    - `<exe_dir>/../../npcsh/db_migrations` (dev layout: `npcsh/rust/target/debug/npcsh`)
fn migrations_dir() -> Option<PathBuf> {
    if let Ok(env_dir) = std::env::var("NPCSH_DB_MIGRATIONS_DIR") {
        let p = expand_tilde(&env_dir);
        if p.is_dir() {
            return Some(p);
        }
    }

    let canonical = npcsh_home().join("db_migrations");
    if canonical.is_dir() {
        return Some(canonical);
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let sidecar = exe_dir.join("db_migrations");
            if sidecar.is_dir() {
                return Some(sidecar);
            }
            let installed = exe_dir.join("..").join("npcsh").join("db_migrations");
            let installed = installed.canonicalize().unwrap_or(installed);
            if installed.is_dir() {
                return Some(installed);
            }
            let dev_fallback = exe_dir
                .join("..")
                .join("..")
                .join("..")
                .join("npcsh")
                .join("db_migrations");
            let dev_fallback = dev_fallback.canonicalize().unwrap_or(dev_fallback);
            if dev_fallback.is_dir() {
                return Some(dev_fallback);
            }
        }
    }

    None
}

/// Read every `.sql` file in the migrations directory, sort by filename, and
/// execute any that are not recorded in `schema_migrations`.  SQL is compiled
/// through nql (Target::Sqlite) so migrations can use nql.* helpers or
/// `{{ ref('...') }}` if needed.
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
        let real_migration = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("npcsh")
            .join("db_migrations")
            .join("001_fix_duplicate_cost_totals_1005_2026.sql");
        fs::write(
            &migration_file,
            fs::read_to_string(&real_migration).unwrap(),
        )
        .unwrap();

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
            // original user message from kernel
            ("user", "hello", None, None::<u64>, None::<u64>, None::<f64>),
            // assistant response from kernel (per-call)
            (
                "assistant",
                "hi there",
                Some("gpt-4o"),
                Some(100_u64),
                Some(50_u64),
                Some(0.0015_f64),
            ),
            // duplicate re-saved user row with cumulative input_tokens
            ("user", "hello", Some("gpt-4o"), Some(150_u64), None, None),
            // duplicate re-saved assistant row with identical usage/cost
            (
                "assistant",
                "hi there",
                Some("gpt-4o"),
                Some(100_u64),
                Some(50_u64),
                Some(0.0015_f64),
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

        unsafe { env::set_var("NPCSH_DB_MIGRATIONS_DIR", &tmp_dir) };
        run_migrations(&db_path).unwrap();

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

        unsafe { env::remove_var("NPCSH_DB_MIGRATIONS_DIR") };
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}

pub fn run_migrations(db_path: &str) -> Result<(), String> {
    let migrations_dir = migrations_dir().ok_or_else(|| {
        "NPCSH DB migrations directory not found. \
         Set NPCSH_DB_MIGRATIONS_DIR or ensure ~/.npcsh/db_migrations exists."
            .to_string()
    })?;

    let conn = Connection::open(db_path)
        .map_err(|e| format!("failed to open history db for migrations: {e}"))?;
    conn.execute_batch(MIGRATIONS_TABLE_SQL)
        .map_err(|e| format!("failed to create schema_migrations table: {e}"))?;

    let mut entries: Vec<PathBuf> = fs::read_dir(&migrations_dir)
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

    let compiler = Compiler::new(Target::Sqlite);

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
        if applied {
            continue;
        }

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

    Ok(())
}
