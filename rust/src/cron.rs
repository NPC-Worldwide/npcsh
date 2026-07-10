use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CronJobKind {
    Chat,
    Jinx,
}

#[derive(Debug, Clone)]
pub struct CronJob {
    pub id: u32,
    pub npc: String,
    pub interval_secs: u64,
    pub task: String,
    pub kind: CronJobKind,
    pub enabled: bool,
    pub last_run: Option<Instant>,
    pub next_run: Instant,
}

#[derive(Debug, Clone, Default)]
pub struct CronRegistry {
    next_id: u32,
    jobs: Vec<CronJob>,
    cron_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CronFile {
    cron: Vec<CronFileJob>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CronFileJob {
    npc: String,
    interval: String,
    #[serde(default = "default_enabled")]
    enabled: bool,
    #[serde(flatten)]
    task: CronFileTask,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum CronFileTask {
    Chat {
        task: String,
    },
    Jinx {
        jinx: String,
        #[serde(default)]
        args: String,
    },
}

fn default_enabled() -> bool {
    true
}

impl CronRegistry {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            jobs: Vec::new(),
            cron_file: None,
        }
    }

    pub fn with_file(path: impl Into<String>) -> Self {
        let mut reg = Self::new();
        reg.cron_file = Some(path.into());
        reg.load();
        reg
    }

    fn cron_path(&self) -> Option<String> {
        self.cron_file
            .as_ref()
            .map(|p| shellexpand::tilde(p).to_string())
    }

    fn load(&mut self) {
        let path = match self.cron_path() {
            Some(p) => p,
            None => return,
        };
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let raw = if content.starts_with("#!") {
            content.splitn(2, '\n').nth(1).unwrap_or("").to_string()
        } else {
            content
        };
        let file: CronFile = match serde_yaml::from_str(&raw) {
            Ok(f) => f,
            Err(_) => return,
        };
        let now = Instant::now();
        for j in file.cron {
            let (kind, task) = match j.task {
                CronFileTask::Chat { task } => (CronJobKind::Chat, task),
                CronFileTask::Jinx { jinx, .. } => (CronJobKind::Jinx, jinx),
            };
            let secs = parse_duration(&j.interval);
            let id = self.next_id;
            self.next_id += 1;
            self.jobs.push(CronJob {
                id,
                npc: j.npc,
                interval_secs: secs,
                task,
                kind,
                enabled: j.enabled,
                last_run: None,
                next_run: now + Duration::from_secs(secs),
            });
        }
    }

    fn save(&self) {
        let path = match self.cron_path() {
            Some(p) => p,
            None => return,
        };
        let mut jobs: Vec<CronFileJob> = self
            .jobs
            .iter()
            .map(|j| {
                let task = match j.kind {
                    CronJobKind::Chat => CronFileTask::Chat {
                        task: j.task.clone(),
                    },
                    CronJobKind::Jinx => CronFileTask::Jinx {
                        jinx: j.task.clone(),
                        args: String::new(),
                    },
                };
                CronFileJob {
                    npc: j.npc.clone(),
                    interval: format_duration(j.interval_secs),
                    enabled: j.enabled,
                    task,
                }
            })
            .collect();
        jobs.sort_by(|a, b| a.npc.cmp(&b.npc).then(a.interval.cmp(&b.interval)));
        let file = CronFile { cron: jobs };
        let body = format!(
            "#!/usr/bin/env npc\njinx_name: cron\ndescription: Scheduled heartbeat tasks\ncron:\n{}\n",
            serde_yaml::to_string(&file)
                .unwrap_or_default()
                .strip_prefix("cron:\n")
                .unwrap_or("")
        );
        let _ = std::fs::create_dir_all(Path::new(&path).parent().unwrap_or(Path::new(".")));
        let _ = std::fs::write(&path, body);
    }

    pub fn add(&mut self, npc: String, interval_secs: u64, task: String, kind: CronJobKind) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.jobs.push(CronJob {
            id,
            npc,
            interval_secs,
            task,
            kind,
            enabled: true,
            last_run: None,
            next_run: Instant::now() + Duration::from_secs(interval_secs),
        });
        self.save();
        id
    }

    pub fn remove(&mut self, id: u32) -> bool {
        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != id);
        let removed = self.jobs.len() < before;
        if removed {
            self.save();
        }
        removed
    }

    pub fn enable(&mut self, id: u32, enabled: bool) -> bool {
        if let Some(j) = self.jobs.iter_mut().find(|j| j.id == id) {
            j.enabled = enabled;
            self.save();
            true
        } else {
            false
        }
    }

    pub fn list(&self) -> &[CronJob] {
        &self.jobs
    }

    pub fn check_due(&mut self) -> Vec<CronJob> {
        let now = Instant::now();
        let mut due = Vec::new();
        for j in &mut self.jobs {
            if !j.enabled {
                continue;
            }
            if now >= j.next_run {
                j.last_run = Some(now);
                j.next_run = now + Duration::from_secs(j.interval_secs);
                due.push(j.clone());
            }
        }
        due
    }

    pub fn load_from_jinxes(&mut self, team_dir: &str) {
        let jdir = Path::new(team_dir).join("jinxes");
        if let Ok(entries) = std::fs::read_dir(&jdir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if let Ok(files) = std::fs::read_dir(&p) {
                        for f in files.flatten() {
                            let fp = f.path();
                            if fp.extension().and_then(|s| s.to_str()) == Some("jinx") {
                                self.parse_jinx_cron(&fp);
                            }
                        }
                    }
                }
            }
        }
        let global = std::path::PathBuf::from(shellexpand::tilde("~/.npcsh/npc_team").to_string())
            .join("jinxes");
        if global != jdir {
            if let Ok(entries) = std::fs::read_dir(&global) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        if let Ok(files) = std::fs::read_dir(&p) {
                            for f in files.flatten() {
                                let fp = f.path();
                                if fp.extension().and_then(|s| s.to_str()) == Some("jinx") {
                                    self.parse_jinx_cron(&fp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn parse_jinx_cron(&mut self, path: &Path) {
        let Ok(content) = std::fs::read_to_string(path) else {
            return;
        };
        let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(&content) else {
            return;
        };
        let Some(cron_list) = value.get("cron") else {
            return;
        };
        let Some(items) = cron_list.as_sequence() else {
            return;
        };
        for item in items {
            let npc = item
                .get("npc")
                .and_then(|v| v.as_str())
                .unwrap_or("sibiji")
                .to_string();
            let task = item
                .get("task")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let interval = item
                .get("interval")
                .and_then(|v| v.as_str())
                .unwrap_or("60s");
            let kind = match item.get("kind").and_then(|v| v.as_str()).unwrap_or("chat") {
                "jinx" | "tool" => CronJobKind::Jinx,
                _ => CronJobKind::Chat,
            };
            if task.is_empty() {
                continue;
            }
            let secs = parse_duration(interval);
            self.add(npc, secs, task, kind);
        }
    }
}

pub fn parse_duration(s: &str) -> u64 {
    let s = s.trim();
    if let Ok(n) = s.parse::<u64>() {
        return n;
    }
    let (num, unit) = s.split_at(s.len().saturating_sub(1));
    let num: u64 = num.parse().unwrap_or(60);
    match unit {
        "s" => num,
        "m" => num * 60,
        "h" => num * 60 * 60,
        "d" => num * 24 * 60 * 60,
        _ => s.parse().unwrap_or(60),
    }
}

pub fn spawn_cron_ticker(
    registry: Arc<Mutex<CronRegistry>>,
    tx: tokio::sync::mpsc::UnboundedSender<CronJob>,
) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let due = registry.lock().unwrap().check_due();
            for job in due {
                let _ = tx.send(job);
            }
        }
    });
}

pub fn format_duration(secs: u64) -> String {
    if secs >= 86400 {
        format!("{}d", secs / 86400)
    } else if secs >= 3600 {
        format!("{}h", secs / 3600)
    } else if secs >= 60 {
        format!("{}m", secs / 60)
    } else {
        format!("{}s", secs)
    }
}
