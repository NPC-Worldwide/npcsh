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

/// Schedule phase that must outlive the process.
///
/// `CronJob::next_run` is an `Instant`, which is meaningless across restarts.
/// Without this, `load()` re-phases every job to `now + interval` on boot, so a
/// 12h loop on a machine restarted twice a day would never fire. We persist the
/// wall-clock due time per job key and restore it on load.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PhaseFile {
    #[serde(default)]
    next_run_at: std::collections::HashMap<String, String>,
}

fn phase_path(cron_file: &str) -> std::path::PathBuf {
    std::path::Path::new(&cron_file)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("loops.phase.json")
}

fn job_phase_key(npc: &str, interval_secs: u64, task: &str) -> String {
    format!("{npc}|{interval_secs}|{task}")
}

impl CronRegistry {
    fn phase_file(&self) -> Option<PhaseFile> {
        let path = phase_path(&self.cron_path()?);
        let raw = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&raw).ok()
    }

    /// Persist the wall-clock due time of every job.
    fn save_phase(&self) {
        let Some(path_str) = self.cron_path() else {
            return;
        };
        let path = phase_path(&path_str);
        let mut pf = PhaseFile::default();
        let now = Instant::now();
        for j in &self.jobs {
            let remaining = j.next_run.saturating_duration_since(now);
            let due =
                chrono::Utc::now() + chrono::Duration::from_std(remaining).unwrap_or_default();
            pf.next_run_at.insert(
                job_phase_key(&j.npc, j.interval_secs, &j.task),
                due.to_rfc3339(),
            );
        }
        if let Ok(body) = serde_json::to_string_pretty(&pf) {
            let _ = std::fs::write(path, body);
        }
    }
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

    /// The loops file this registry watches, with `~` expanded.
    pub fn cron_file_path(&self) -> Option<String> {
        self.cron_path()
    }

    /// Re-read loops.yaml and jinx-declared crons in place.
    ///
    /// Preserves `next_run` for jobs that already exist, so an unrelated edit
    /// does not reset every interval's phase (which would starve long loops).
    pub fn reload_from_disk(&mut self, team_dir: &str) {
        let preserved: Vec<(String, Instant)> = self
            .jobs
            .iter()
            .map(|j| (job_phase_key(&j.npc, j.interval_secs, &j.task), j.next_run))
            .collect();

        self.jobs.clear();
        self.next_id = 1;
        self.load();

        // Reapply phases for surviving jobs.
        for j in self.jobs.iter_mut() {
            let key = job_phase_key(&j.npc, j.interval_secs, &j.task);
            if let Some((_, next)) = preserved.iter().find(|(k, _)| *k == key) {
                j.next_run = *next;
            }
        }

        self.load_from_jinxes(team_dir);
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
        let saved_phase = self.phase_file().unwrap_or_default();
        for j in file.cron {
            let (kind, task) = match j.task {
                CronFileTask::Chat { task } => (CronJobKind::Chat, task),
                CronFileTask::Jinx { jinx, args } => (
                    CronJobKind::Jinx,
                    if args.trim().is_empty() {
                        jinx
                    } else {
                        // Preserve the jinx's single positional arg: task
                        // strings are later split on the first space.
                        format!("{} {}", jinx, args.trim())
                    },
                ),
            };
            let secs = parse_duration(&j.interval);
            let id = self.next_id;
            self.next_id += 1;

            // Restore the persisted due time so a restart does not reset every
            // loop's phase. If the saved time is already past, run on next tick
            // (clamp to now) rather than firing a burst of catch-up runs.
            let next_run = saved_phase
                .next_run_at
                .get(&job_phase_key(&j.npc, secs, &task))
                .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                .map(|due| {
                    let delta = due.with_timezone(&chrono::Utc) - chrono::Utc::now();
                    let secs_left = delta.num_seconds().max(0) as u64;
                    now + Duration::from_secs(secs_left)
                })
                .unwrap_or_else(|| now + Duration::from_secs(secs));

            self.jobs.push(CronJob {
                id,
                npc: j.npc,
                interval_secs: secs,
                task,
                kind,
                enabled: j.enabled,
                last_run: None,
                next_run,
            });
        }
        // Persist immediately so a fresh schedule survives a restart that
        // happens before the first tick. Otherwise phase is only written when
        // a job fires, and a restart loop can starve a long-interval job.
        self.save_phase();
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
        // Deliberately do NOT sort here. Job ids are positional: `load()` assigns
        // ids in file order. Sorting the file while leaving the in-memory Vec in
        // append order means the shell and the daemon disagree about which id
        // belongs to which job, so `/loop` reports an id that `/loop <id>` and
        // `/loops` resolve to a different job. Keeping append order preserves
        // id identity across save/load.
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
        let mut advanced = false;
        for j in &mut self.jobs {
            if !j.enabled {
                continue;
            }
            if now >= j.next_run {
                j.last_run = Some(now);
                j.next_run = now + Duration::from_secs(j.interval_secs);
                due.push(j.clone());
                advanced = true;
            }
        }
        if advanced {
            self.save_phase();
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

/// Start the cron ticker.
///
/// The ticker *executes* due jobs itself, on the async runtime it is handed.
/// It does not merely enqueue them for a REPL to pick up: a loop must keep
/// running when nobody is sitting at a prompt, when a long agent turn is in
/// progress, and when a TUI is open. `notify` is best-effort only — a UI that
/// wants to redraw on a tick can listen, and a dropped notification must never
/// mean a dropped run.
static TICK_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn spawn_cron_ticker(
    registry: Arc<Mutex<CronRegistry>>,
    handle: tokio::runtime::Handle,
    cfg: crate::loop_runner::DetachedLoopConfig,
    notify: Option<tokio::sync::mpsc::UnboundedSender<CronJob>>,
) {
    // The registry is a snapshot from load time, so edits to `loops.yaml` by
    // hand would otherwise be invisible until a restart. Track the file's mtime
    // and reload when it changes: the file is the user's editing surface and
    // must behave like one.
    let watched = registry.lock().ok().and_then(|r| r.cron_file_path());
    let mut last_mtime = watched.as_ref().and_then(|p| mtime_of(p));

    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(1));

            if let Some(path) = watched.as_ref() {
                let now_mtime = mtime_of(path);
                if now_mtime != last_mtime {
                    last_mtime = now_mtime;
                    if let Ok(mut r) = registry.lock() {
                        r.reload_from_disk(&cfg.team_dir);
                    }
                }
            }

            let due = match registry.lock() {
                Ok(mut r) => r.check_due(),
                Err(_) => continue,
            };
            for job in due {
                if let Some(tx) = notify.as_ref() {
                    // Never let notification backpressure drop a run.
                    let _ = tx.send(job.clone());
                }

                // Back off a loop that keeps failing to even start (missing NPC,
                // malformed task). Without this it retries at the configured
                // interval forever and floods its log. Successful runs reset the
                // streak, so a transient outage costs nothing.
                let key = crate::loop_runner::loop_key(&job);
                let fails = crate::loop_runner::failure_count(&key);
                if fails > 0 {
                    let skip = 1u32 << fails.min(5); // 2,4,8,16,32
                    if skip > 1 {
                        let counter =
                            TICK_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if counter % skip as u64 != 0 {
                            continue;
                        }
                    }
                }

                crate::loop_runner::spawn_detached_loop(&handle, job, cfg.clone());
            }
        }
    });
}

fn mtime_of(path: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
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
