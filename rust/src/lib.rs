pub mod markdown;
pub mod stream_client;

use npcrs::error::Result;
use npcrs::kernel::Kernel;
use npcrs::{calculate_cost, Message};

pub fn find_team_dir() -> String {
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

pub async fn exec_jinx_file(jinx_file: &str, args: &[&str]) -> Result<()> {
    use npcrs::npc_compiler::{execute_jinx, load_jinx_from_file};

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

    let empty_jinxes = std::collections::HashMap::new();
    let result = execute_jinx(&jinx, &input_values, &empty_jinxes).await?;

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
        let in_tok = response.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
        let out_tok = response.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0);
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
