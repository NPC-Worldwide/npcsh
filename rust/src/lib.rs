pub mod markdown;
pub mod stream_client;

use npcrs::calculate_cost;
use npcrs::error::Result;
use npcrs::kernel::Kernel;
use std::io::IsTerminal;
use std::sync::OnceLock;

const PREF_FILE: &str = ".NPCSH_PREFERRED_TEAM_NAME";
const AGENTS_TEAM_DIR: &str = ".npcsh_team";

static RESOLVED_TEAM_DIR: OnceLock<String> = OnceLock::new();

pub fn find_team_dir() -> String {
    if let Some(resolved) = RESOLVED_TEAM_DIR.get() {
        return resolved.clone();
    }

    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--team") {
        if let Some(dir) = args.get(pos + 1) {
            return dir.clone();
        }
    }

    if std::path::Path::new("./npc_team").exists() {
        return "./npc_team".to_string();
    }

    let global = shellexpand::tilde("~/.npcsh/npc_team").to_string();
    if std::path::Path::new(&global).exists() {
        return global;
    }

    ".".to_string()
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let dest = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &dest)?;
        } else {
            std::fs::copy(&path, &dest)?;
        }
    }
    Ok(())
}

fn build_agents_team_dir(cwd: &std::path::Path) -> String {
    let team = cwd.join(AGENTS_TEAM_DIR);
    let _ = std::fs::remove_dir_all(&team);
    let _ = std::fs::create_dir_all(team.join("jinxes"));

    // Copy any .ctx file from the project root into the synthetic team dir.
    if let Ok(entries) = std::fs::read_dir(cwd) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("ctx") {
                let dest = team.join(path.file_name().unwrap_or_default());
                let _ = std::fs::copy(&path, &dest);
            }
        }
    }

    // Copy the jinxes directory from the project root if present.
    let jinxes_src = cwd.join("jinxes");
    if jinxes_src.is_dir() {
        let _ = copy_dir(&jinxes_src, &team.join("jinxes"));
    }

    team.to_string_lossy().to_string()
}

fn prompt_for_layout(cwd: &std::path::Path) -> Option<String> {
    use std::io;

    eprintln!(
        "Found both npc_team/ and agents.md/agents/ in {}.",
        cwd.display()
    );
    eprintln!("Which layout should npcsh use?");
    eprintln!("  1) npc_team");
    eprintln!("  2) agents");

    loop {
        eprint!("Enter 1 or 2: ");
        let _ = io::Write::flush(&mut io::stderr());
        let mut buf = String::new();
        if io::stdin().read_line(&mut buf).is_err() {
            return None;
        }
        match buf.trim() {
            "1" | "npc_team" => return Some("npc_team".to_string()),
            "2" | "agents" => return Some("agents".to_string()),
            _ => eprintln!("Invalid choice. Enter 1 or 2."),
        }
    }
}

#[doc(hidden)]
pub fn resolve_team_layout_at(cwd: &std::path::Path) -> Option<String> {
    let has_npc_team = cwd.join("npc_team").is_dir();
    let has_agents = cwd.join("agents.md").is_file() || cwd.join("agents").is_dir();

    let pref_path = cwd.join(PREF_FILE);
    let pref = if pref_path.exists() {
        std::fs::read_to_string(&pref_path)
            .ok()
            .map(|s| s.trim().to_lowercase())
    } else {
        None
    };

    let mode = match (has_npc_team, has_agents) {
        (true, false) => Some("npc_team".to_string()),
        (false, true) => Some("agents".to_string()),
        (true, true) => pref.or_else(|| {
            if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
                prompt_for_layout(cwd)
            } else {
                eprintln!(
                    "Warning: both npc_team/ and agents layout found; defaulting to npc_team. \
                     Set {} to choose.",
                    PREF_FILE
                );
                Some("npc_team".to_string())
            }
        }),
        (false, false) => None,
    };

    let mode = mode?;

    // Persist the choice so the user is not asked again.
    if !pref_path.exists() {
        let _ = std::fs::write(&pref_path, &mode);
    }

    let team_dir = if mode == "agents" {
        build_agents_team_dir(cwd)
    } else {
        cwd.join("npc_team").to_string_lossy().to_string()
    };

    RESOLVED_TEAM_DIR.set(team_dir.clone()).ok();
    Some(team_dir)
}

/// Detect the project layout before booting the kernel.
///
/// If both `npc_team/` and `agents.md`/`agents/` exist, the user is prompted
/// (in interactive terminals) and the choice is saved in `.NPCSH_PREFERRED_TEAM_NAME`.
/// The resolved team directory is cached for `find_team_dir()` to use.
pub fn resolve_team_layout() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    resolve_team_layout_at(&cwd)
}

pub async fn exec_jinx_file(jinx_file: &str, args: &[&str]) -> Result<()> {
    use npcrs::npc_compiler::{
        execute_jinx_with_npc, load_jinx_from_file, load_team_from_directory, Jinx,
    };

    let jinx = load_jinx_from_file(jinx_file)?;

    let mut input_values = std::collections::HashMap::new();
    let mut positional_idx = 0;

    for arg in args {
        if let Some((k, v)) = arg.split_once('=') {
            input_values.insert(k.to_string(), v.to_string());
        } else if let Some(input) = jinx.inputs.get(positional_idx) {
            input_values.insert(input.name.clone(), arg.to_string());
            positional_idx += 1;
        }
    }

    // Boot the full team like npcpy does, so every sub-jinx and the lead NPC are available.
    let team_dir = resolve_team_layout().unwrap_or_else(|| ".".to_string());
    let team = load_team_from_directory(&team_dir)?;
    let mut available_jinxes = team.jinxes.clone();

    let npc = if let Some(mut lead) = team.lead_npc().cloned() {
        lead.team = Some(Box::new(team.clone()));
        Some(lead)
    } else {
        None
    };

    let result = execute_jinx_with_npc(&jinx, &input_values, &available_jinxes, npc.as_ref()).await?;

    if !result.output.is_empty() {
        println!("{}", result.output);
    }

    if !result.success {
        if let Some(err) = result.error {
            eprintln!("Error: {}", err);
        }
        std::process::exit(1);
    }

    Ok(())
}

pub async fn exec_npc_file(
    npc_file: &str,
    command: Option<&str>,
    client: &reqwest::Client,
    server_url: &str,
    override_model: Option<&str>,
    override_provider: Option<&str>,
) -> Result<()> {
    use npcrs::npc_compiler::NPC;

    let mut npc = NPC::from_file(npc_file)?;
    if let Some(m) = override_model {
        npc.model = Some(m.to_string());
    }
    if let Some(p) = override_provider {
        npc.provider = Some(p.to_string());
    }
    let npc_name = npc.name.clone();

    let model = npc.resolved_model();
    let provider = npc.resolved_provider();

    if let Some(cmd) = command {
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        let conv_id = uuid::Uuid::new_v4().to_string();
        let request = stream_client::StreamRequest {
            model: model.clone(),
            provider: provider.clone(),
            messages: vec![
                npcrs::Message::system(npc.system_prompt(None)),
                npcrs::Message::user(cmd.to_string()),
            ],
            commandstr: cmd.to_string(),
            npc: Some(npc_name.clone()),
            registered_teams: None,
            conversation_id: Some(conv_id.clone()),
            current_path: Some(cwd.clone()),
            execution_mode: "chat".to_string(),
        };
        let response = stream_client::call_stream(client, server_url, &request, None)
            .await
            .map_err(|e| npcrs::NpcError::Other(e))?;
        let output = response.message.content.clone().unwrap_or_default();
        if !output.is_empty() {
            println!("{}", output);
        }
        let in_tok = response
            .usage
            .as_ref()
            .map(|u| u.prompt_tokens)
            .unwrap_or(0);
        let out_tok = response
            .usage
            .as_ref()
            .map(|u| u.completion_tokens)
            .unwrap_or(0);
        let cost = response
            .usage
            .as_ref()
            .map(|u| calculate_cost(&request.model, u.prompt_tokens, u.completion_tokens))
            .unwrap_or(0.0);
        let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
        let team_dir = find_team_dir();
        if let Ok(kernel) = npcrs::Kernel::boot(&team_dir, &db_path) {
            let team_name_str = kernel
                .team
                .source_dir
                .as_deref()
                .and_then(|d| std::path::Path::new(d).file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("npc")
                .to_string();
            let _ = kernel.history.save_conversation_message(
                &conv_id,
                "user",
                cmd,
                &cwd,
                Some(&model),
                Some(&provider),
                Some(&npc_name),
                Some(&team_name_str),
                None,
                None,
                None,
                Some(in_tok),
                None,
                None,
            );
            let _ = kernel.history.save_conversation_message(
                &conv_id,
                "assistant",
                &output,
                &cwd,
                Some(&model),
                Some(&provider),
                Some(&npc_name),
                Some(&team_name_str),
                None,
                None,
                None,
                None,
                Some(out_tok),
                Some(cost),
            );
        }
    } else {
        let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();
        let team_dir = find_team_dir();
        let mut kernel = Kernel::boot(&team_dir, &db_path)?;

        if let Some(p) = kernel.get_process_mut(0) {
            p.npc = npc;
        }

        eprintln!("\x1b[1;94m{}\x1b[0m", npc_file);
        eprintln!(
            "NPC: {} | model: {} | provider: {}",
            kernel
                .get_process(0)
                .map(|p| p.npc.name.as_str())
                .unwrap_or("?"),
            model,
            provider
        );
        eprintln!();

        let mut rl = rustyline::DefaultEditor::new().unwrap();
        loop {
            let input = match rl.readline("\x1b[35m> \x1b[0m") {
                Ok(line) => line.trim().to_string(),
                Err(_) => break,
            };
            if input.is_empty() {
                continue;
            }
            if input == "exit" || input == "quit" {
                break;
            }
            rl.add_history_entry(&input).ok();

            let system = kernel
                .get_process(0)
                .map(|p| p.npc.system_prompt(kernel.team.context.as_deref()))
                .unwrap_or_default();
            let request = stream_client::StreamRequest {
                model: model.clone(),
                provider: provider.clone(),
                messages: vec![
                    npcrs::Message::system(system),
                    npcrs::Message::user(input.clone()),
                ],
                commandstr: input.clone(),
                npc: Some(npc_name.clone()),
                registered_teams: None,
                conversation_id: Some(uuid::Uuid::new_v4().to_string()),
                current_path: Some(
                    std::env::current_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| ".".to_string()),
                ),
                execution_mode: "chat".to_string(),
            };
            match stream_client::call_stream(client, server_url, &request, None).await {
                Ok(response) => {
                    let output = response.message.content.clone().unwrap_or_default();
                    if !output.is_empty() {
                        println!("\n{}", output);
                    }
                }
                Err(e) => eprintln!("\x1b[31mError: {}\x1b[0m", e),
            }
        }
    }

    Ok(())
}

pub fn init_team(dir: &str) -> Result<()> {
    let dir = std::path::Path::new(dir)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(dir));
    let team_dir = dir.join("npc_team");

    if team_dir.exists()
        && std::fs::read_dir(&team_dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
    {
        eprintln!("npc_team/ already exists at {}", team_dir.display());
        return Ok(());
    }

    std::fs::create_dir_all(team_dir.join("jinxes")).unwrap();

    let global_jinxes = shellexpand::tilde("~/.npcsh/npc_team/jinxes/lib").to_string();
    let dest_lib = team_dir.join("jinxes").join("lib");
    if std::path::Path::new(&global_jinxes).is_dir() && !dest_lib.exists() {
        fn copy_dir(src: &std::path::Path, dst: &std::path::Path) {
            std::fs::create_dir_all(dst).ok();
            if let Ok(entries) = std::fs::read_dir(src) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let dest = dst.join(entry.file_name());
                    if path.is_dir() {
                        copy_dir(&path, &dest);
                    } else {
                        std::fs::copy(&path, &dest).ok();
                    }
                }
            }
        }
        copy_dir(std::path::Path::new(&global_jinxes), &dest_lib);
    }

    let forenpc = "#!/usr/bin/env npc\n\
name: forenpc\n\
primary_directive: You are the team coordinator. Delegate tasks to specialists and synthesize results.\n\
model: qwen3.5:2b\n\
provider: ollama\n\
jinxes:\n\
  - sh\n\
  - python\n\
  - edit_file\n\
  - load_file\n\
  - web_search\n\
  - file_search\n\
  - delegate\n";
    let fp = team_dir.join("forenpc.npc");
    std::fs::write(&fp, forenpc).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fp, std::fs::Permissions::from_mode(0o755)).ok();
    }

    let coder = "#!/usr/bin/env npc\n\
name: coder\n\
primary_directive: You are a coding specialist. Write, debug, and refactor code. Run tests. Edit files.\n\
model: qwen3.5:2b\n\
provider: ollama\n\
jinxes:\n\
  - sh\n\
  - python\n\
  - edit_file\n\
  - load_file\n\
  - file_search\n";
    let cp = team_dir.join("coder.npc");
    std::fs::write(&cp, coder).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&cp, std::fs::Permissions::from_mode(0o755)).ok();
    }

    let ctx = "context: A development team.\nforenpc: forenpc\n";
    std::fs::write(team_dir.join("team.ctx"), ctx).unwrap();

    println!("Created npc_team/ at {}", team_dir.display());
    println!("  forenpc.npc — coordinator");
    println!("  coder.npc   — coding specialist");
    println!("  team.ctx    — team context");
    println!();
    println!("Run npcsh to start, or:");
    println!("  npc {} 'what can you do?'", fp.display());
    println!(
        "  npc {} 'list all TODO comments in this project'",
        cp.display()
    );

    Ok(())
}
