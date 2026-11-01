//! Ensure the Ctrl+R history loader dedupes and returns most-recent-first,
//! as it backs the reverse-search surface exposed in the shell.

use serial_test::serial;
use std::env;
use std::fs;

fn with_temp_home<F: FnOnce()>(content: &str, f: F) {
    let dir = env::temp_dir().join(format!("npcsh_ctrlr_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join(".npcsh_history"), content).unwrap();
    let prev = env::var("HOME").ok();
    env::set_var("HOME", &dir);
    f();
    match prev {
        Some(v) => env::set_var("HOME", v),
        None => env::remove_var("HOME"),
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[serial]
fn history_dedupes_and_reverses() {
    let content = "ls -la\npwd\ngit status\npwd\n";
    with_temp_home(content, || {
        let h = npcsh::ctrl_r::load_history();
        assert_eq!(h, vec!["pwd", "git status", "ls -la"]);
    });
}

#[test]
#[serial]
fn empty_history_returns_empty_vec() {
    with_temp_home("", || {
        let h = npcsh::ctrl_r::load_history();
        assert!(h.is_empty());
    });
}
