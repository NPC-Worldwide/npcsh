use std::fs;
use std::path::PathBuf;

fn tmp_root() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("npcsh_team_layout_test_{}", uuid::Uuid::new_v4()));
    p
}

fn write(path: impl AsRef<std::path::Path>, content: &str) {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

#[test]
fn npc_team_layout_is_default_when_only_npc_team() {
    let root = tmp_root();
    let team_dir = root.join("npc_team");
    fs::create_dir_all(&team_dir).unwrap();
    write(
        team_dir.join("team.ctx"),
        "model: qwen3.5:2b\nprovider: ollama\n",
    );
    write(
        team_dir.join("sibiji.npc"),
        "name: sibiji\nprimary_directive: Orchestrator.\n",
    );

    let resolved = npcsh::resolve_team_layout_at(&root).unwrap();
    assert!(resolved.ends_with("npc_team"), "resolved={}", resolved);

    let kernel = npcrs::Kernel::boot(&resolved, root.join("db.sqlite").to_str().unwrap()).unwrap();
    assert!(kernel.team.get_npc("sibiji").is_some());
}

#[test]
fn agents_layout_uses_synthetic_team_dir_when_only_agents() {
    let root = tmp_root();
    // In agents mode the context file and jinxes live at the project root.
    write(
        root.join("team.ctx"),
        "model: qwen3.5:2b\nprovider: ollama\n",
    );
    write(root.join("agents.md"), "## summarizer\nYou summarize.\n");
    write(
        root.join("jinxes").join("hello.jinx"),
        "jinx_name: hello\nsteps:\n  - engine: llm\n    prompt: hi\n",
    );

    let resolved = npcsh::resolve_team_layout_at(&root).unwrap();
    assert!(resolved.ends_with(".npcsh_team"), "resolved={}", resolved);

    let kernel = npcrs::Kernel::boot(&resolved, root.join("db.sqlite").to_str().unwrap()).unwrap();
    assert!(kernel.team.get_npc("summarizer").is_some());
    assert!(kernel.team.jinx_names().contains(&"hello"));
}

#[test]
fn both_layouts_respect_pref_file_for_agents() {
    let root = tmp_root();
    let team_dir = root.join("npc_team");
    fs::create_dir_all(&team_dir).unwrap();
    write(
        team_dir.join("team.ctx"),
        "model: qwen3.5:2b\nprovider: ollama\n",
    );
    write(
        team_dir.join("sibiji.npc"),
        "name: sibiji\nprimary_directive: NPC team agent.\n",
    );
    write(root.join("agents.md"), "## summarizer\nYou summarize.\n");
    write(root.join(".NPCSH_PREFERRED_TEAM_NAME"), "agents\n");

    let resolved = npcsh::resolve_team_layout_at(&root).unwrap();
    assert!(resolved.ends_with(".npcsh_team"), "resolved={}", resolved);

    let kernel = npcrs::Kernel::boot(&resolved, root.join("db.sqlite").to_str().unwrap()).unwrap();
    assert!(kernel.team.get_npc("summarizer").is_some());
    assert!(kernel.team.get_npc("sibiji").is_none());
}

#[test]
fn both_layouts_respect_pref_file_for_npc_team() {
    let root = tmp_root();
    let team_dir = root.join("npc_team");
    fs::create_dir_all(&team_dir).unwrap();
    write(
        team_dir.join("team.ctx"),
        "model: qwen3.5:2b\nprovider: ollama\n",
    );
    write(
        team_dir.join("sibiji.npc"),
        "name: sibiji\nprimary_directive: NPC team agent.\n",
    );
    write(root.join("agents.md"), "## summarizer\nYou summarize.\n");
    write(root.join(".NPCSH_PREFERRED_TEAM_NAME"), "npc_team\n");

    let resolved = npcsh::resolve_team_layout_at(&root).unwrap();
    assert!(resolved.ends_with("npc_team"), "resolved={}", resolved);

    let kernel = npcrs::Kernel::boot(&resolved, root.join("db.sqlite").to_str().unwrap()).unwrap();
    assert!(kernel.team.get_npc("sibiji").is_some());
}
