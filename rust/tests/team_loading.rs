use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

fn tmp_root() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("npcsh_team_loading_test_{}", uuid::Uuid::new_v4()));
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
fn loads_npc_ctx_and_jinx() {
    let root = tmp_root();
    let team_dir = root.join("npc_team");
    fs::create_dir_all(&team_dir).unwrap();
    let db = root.join("history.db");

    write(
        team_dir.join("team.ctx"),
        "forenpc: sibiji\nmodel: qwen3.5:2b\nprovider: ollama\n",
    );
    write(
        team_dir.join("sibiji.npc"),
        "name: sibiji\nprimary_directive: Orchestrator.\n",
    );
    write(
        team_dir.join("corca.npc"),
        "name: corca\nprimary_directive: Coder.\n",
    );
    write(
        team_dir.join("jinxes").join("hello.jinx"),
        "jinx_name: hello\nsteps:\n  - engine: llm\n    prompt: say hi\n",
    );

    let kernel = npcrs::Kernel::boot(team_dir.to_str().unwrap(), db.to_str().unwrap()).unwrap();
    let names: HashSet<_> = kernel.team.npc_names().into_iter().collect();
    assert!(names.contains("sibiji"));
    assert!(names.contains("corca"));
    assert!(kernel.team.jinx_names().contains(&"hello"));
    assert_eq!(kernel.team.forenpc.as_deref(), Some("sibiji"));
}

#[test]
fn loads_agents_md() {
    let root = tmp_root();
    let team_dir = root.join("npc_team");
    fs::create_dir_all(&team_dir).unwrap();
    let db = root.join("history.db");

    write(
        team_dir.join("team.ctx"),
        "model: qwen3.5:2b\nprovider: ollama\n",
    );
    write(
        root.join("agents.md"),
        "## summarizer\nYou summarize.\n\n## fact_checker\nYou check facts.\n",
    );

    let kernel = npcrs::Kernel::boot(team_dir.to_str().unwrap(), db.to_str().unwrap()).unwrap();
    let names: HashSet<_> = kernel.team.npc_names().into_iter().collect();
    assert!(names.contains("summarizer"));
    assert!(names.contains("fact_checker"));
}

#[test]
fn loads_agents_dir() {
    let root = tmp_root();
    let team_dir = root.join("npc_team");
    fs::create_dir_all(&team_dir).unwrap();
    let agents = root.join("agents");
    fs::create_dir_all(&agents).unwrap();
    let db = root.join("history.db");

    write(
        team_dir.join("team.ctx"),
        "model: qwen3.5:2b\nprovider: ollama\n",
    );
    write(
        agents.join("translator.md"),
        "---\nmodel: gemini-2.5-flash\nprovider: gemini\n---\nYou translate.\n",
    );
    write(
        agents.join("custom.md"),
        "---\nname: custom\nmodel: qwen3.5:4b\nprovider: ollama\n---\nCustom agent.\n",
    );

    let kernel = npcrs::Kernel::boot(team_dir.to_str().unwrap(), db.to_str().unwrap()).unwrap();
    let names: HashSet<_> = kernel.team.npc_names().into_iter().collect();
    assert!(names.contains("translator"));
    assert!(names.contains("custom"));

    let translator = kernel.team.get_npc("translator").unwrap();
    assert_eq!(translator.model.as_deref(), Some("gemini-2.5-flash"));
    assert_eq!(translator.provider.as_deref(), Some("gemini"));

    let custom = kernel.team.get_npc("custom").unwrap();
    assert_eq!(custom.model.as_deref(), Some("qwen3.5:4b"));
}

#[test]
fn loads_real_repo_team() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let team_dir = std::path::Path::new(&manifest)
        .join("..")
        .join("npcsh")
        .join("npc_team");
    let db = std::env::temp_dir().join(format!("npcsh_real_team_test_{}.db", uuid::Uuid::new_v4()));

    let kernel = npcrs::Kernel::boot(team_dir.to_str().unwrap(), db.to_str().unwrap()).unwrap();

    let names: HashSet<_> = kernel.team.npc_names().into_iter().collect();
    assert!(
        names.contains("sibiji") || names.contains("corca"),
        "expected at least sibiji or corca in the bundled team"
    );

    let jinxes: HashSet<_> = kernel.team.jinx_names().into_iter().collect();
    assert!(
        !jinxes.is_empty(),
        "expected at least one jinx to load from the bundled team"
    );
}
