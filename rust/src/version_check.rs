use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_TTL_SECONDS: u64 = 86400;
const GITHUB_RELEASES_URL: &str = "https://api.github.com/repos/NPC-Worldwide/npcsh/releases/latest";
const CRATES_URL: &str = "https://crates.io/api/v1/crates/npcsh";
const BREW_FORMULA_URL: &str = "https://raw.githubusercontent.com/NPC-Worldwide/homebrew-npcsh/main/Formula/npcsh.rb";

#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub source: &'static str,
    pub current: String,
    pub latest: String,
    pub command: &'static str,
}

fn cache_path() -> PathBuf {
    std::path::PathBuf::from(shellexpand::tilde("~/.npcsh").as_ref()).join(".version_check.json")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn detect_source() -> &'static str {
    let exe = std::env::current_exe().ok();
    let exe_str = exe.as_ref().map(|p| p.display().to_string()).unwrap_or_default();

    let brew_roots = ["/opt/homebrew", "/usr/local", "/home/linuxbrew"];
    for root in &brew_roots {
        if exe_str.starts_with(root) {
            return "brew";
        }
    }

    if exe_str.contains(".cargo") || exe_str.contains("cargo-install") {
        return "cargo";
    }

    "binary"
}

fn read_cache() -> Option<(String, u64)> {
    let path = cache_path();
    let text = std::fs::read_to_string(&path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let checked_at = value.get("checked_at")?.as_u64()?;
    if now_secs() - checked_at >= CACHE_TTL_SECONDS {
        return None;
    }
    let latest = value.get("latest")?.as_str()?.to_string();
    Some((latest, checked_at))
}

fn write_cache(latest: &str) {
    let path = cache_path();
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(&path));
    let value = serde_json::json!({
        "latest": latest,
        "checked_at": now_secs(),
    });
    let _ = std::fs::write(path, value.to_string());
}

async fn fetch_github_latest(client: &reqwest::Client) -> Option<String> {
    let resp = client
        .get(GITHUB_RELEASES_URL)
        .header("User-Agent", "npcsh-version-check")
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    json.get("tag_name")?.as_str().map(|t| t.trim_start_matches('v').to_string())
}

async fn fetch_crates_latest(client: &reqwest::Client) -> Option<String> {
    let resp = client
        .get(CRATES_URL)
        .header("User-Agent", "npcsh-version-check")
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    json.get("crate")?
        .get("max_stable_version")?
        .as_str()
        .map(String::from)
}

async fn fetch_brew_latest(client: &reqwest::Client) -> Option<String> {
    let resp = client
        .get(BREW_FORMULA_URL)
        .header("User-Agent", "npcsh-version-check")
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .ok()?;
    let text = resp.text().await.ok()?;
    for line in text.lines() {
        if line.trim().starts_with("version") {
            if let Some(start) = line.find('"') {
                if let Some(end) = line[start + 1..].find('"') {
                    return Some(line[start + 1..start + 1 + end].to_string());
                }
            }
        }
    }
    None
}

fn parse_version(v: &str) -> Option<semver::Version> {
    semver::Version::parse(v.trim_start_matches('v')).ok()
}

fn update_command(source: &str) -> &'static str {
    match source {
        "cargo" => "cargo install npcsh --force",
        "brew" => "brew upgrade npcsh",
        _ => "curl -fsSL https://enpisi.com/install-npcsh.sh | sh",
    }
}

pub async fn check_version(client: &reqwest::Client, current: &str) -> Option<UpdateInfo> {
    let source = detect_source();

    let latest = if let Some((cached, _)) = read_cache() {
        cached
    } else {
        let latest = match source {
            "cargo" => fetch_crates_latest(client).await,
            "brew" => fetch_brew_latest(client).await,
            _ => fetch_github_latest(client).await,
        }
        .unwrap_or_default();
        if !latest.is_empty() {
            write_cache(&latest);
        }
        latest
    };

    if latest.is_empty() {
        return None;
    }

    let current_ver = parse_version(current)?;
    let latest_ver = parse_version(&latest)?;

    if latest_ver > current_ver {
        Some(UpdateInfo {
            source,
            current: current.to_string(),
            latest,
            command: update_command(source),
        })
    } else {
        None
    }
}

pub fn format_update_notice(info: &UpdateInfo) -> String {
    format!(
        "  \x1b[33mUpdate available:\x1b[0m {} v{} → v{}\n  Run: \x1b[36m{}\x1b[0m",
        info.source, info.current, info.latest, info.command
    )
}
