use std::fs;
use std::path::{Path, PathBuf};

const RESERVED_CTX_KEYS: &[&str] = &[
    "context",
    "databases",
    "forenpc",
    "mcp_servers",
    "providers",
    "available_executables",
    "name",
];

pub fn ensure_user_subteam(home: &Path) -> std::io::Result<PathBuf> {
    let team = home.join(".npcsh").join("npc_team");
    let usr = team.join("usr");
    let jinxes = usr.join("jinxes");
    if !jinxes.exists() {
        fs::create_dir_all(&jinxes)?;
    }
    let user_npc = usr.join("user.npc");
    if !user_npc.exists() {
        fs::write(
            &user_npc,
            "# User NPC\nname: user\ndescription: The human user.\n",
        )?;
    }
    let usr_ctx = usr.join("usr.ctx");
    if !usr_ctx.exists() {
        fs::write(
            &usr_ctx,
            "# User context\n# Put personal context, preferences, and reminders here.\n",
        )?;
    }
    Ok(usr)
}

pub fn sync_team(source: &Path, target: &Path) -> std::io::Result<()> {
    if !source.exists() || !source.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(target)?;
    ensure_user_subteam(&target.ancestors().nth(2).unwrap_or(target))?;

    let mut preserved_ctx_values: std::collections::HashMap<String, serde_yaml::Value> =
        std::collections::HashMap::new();
    let target_ctx_path = target.join("npcsh.ctx");
    if target_ctx_path.exists() {
        if let Ok(content) = fs::read_to_string(&target_ctx_path) {
            if let Ok(parsed) = serde_yaml::from_str::<serde_yaml::Value>(&content) {
                if let Some(mapping) = parsed.as_mapping() {
                    for (key, value) in mapping {
                        let key_str = key.as_str().unwrap_or("").to_string();
                        if !RESERVED_CTX_KEYS.contains(&key_str.as_str()) {
                            preserved_ctx_values.insert(key_str, value.clone());
                        }
                    }
                }
            }
        }
    }

    for entry in fs::read_dir(target)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "usr" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            fs::remove_dir_all(&path)?;
        } else {
            fs::remove_file(&path)?;
        }
    }

    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "usr" {
            continue;
        }
        let src = entry.path();
        let dst = target.join(&name);
        if src.is_dir() {
            copy_dir_all(&src, &dst)?;
        } else {
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }
            if dst.file_name().and_then(|s| s.to_str()) == Some("npcsh.ctx")
                && !preserved_ctx_values.is_empty()
            {
                copy_ctx_with_preserved_values(&src, &dst, &preserved_ctx_values)?;
            } else {
                fs::copy(&src, &dst)?;
            }
        }
    }

    Ok(())
}

fn copy_ctx_with_preserved_values(
    src: &Path,
    dst: &Path,
    preserved: &std::collections::HashMap<String, serde_yaml::Value>,
) -> std::io::Result<()> {
    let content = fs::read_to_string(src)?;
    let mut parsed: serde_yaml::Value = serde_yaml::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(mapping) = parsed.as_mapping_mut() {
        for (key, value) in preserved {
            mapping.insert(serde_yaml::Value::String(key.clone()), value.clone());
        }
    }
    let out = serde_yaml::to_string(&parsed)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    fs::write(dst, out)?;
    Ok(())
}

fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> std::io::Result<()> {
    fs::create_dir_all(&dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(name))?;
        } else {
            fs::copy(entry.path(), dst.as_ref().join(name))?;
        }
    }
    Ok(())
}
