//! `npcshd` — the npcsh loop daemon.
//!
//! Loops must not die with the shell that started them. This process owns the
//! cron registry, the ticker, and the kernel that runs loop turns. `npcsh`
//! becomes a client: it can create loops, list them, run one now, stop one, and
//! watch their output — but it does not own execution.
//!
//! Discovery: the daemon binds an ephemeral loopback port and records it in the
//! `daemon_state` table keyed by its team directory. Clients look up their own
//! team, so a daemon serving team A can never be mistaken for team B's.
//!
//! Single-instance *per team*: an exclusive `flock` on a team-derived lock file
//! is held for the daemon's lifetime, so two shells racing to start a daemon for
//! the same team cannot both win, while different teams run side by side.

use npcsh::cron::CronRegistry;
use npcsh::daemon::{self, DaemonStatus, JobInfo, Request, Response};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

const HEARTBEAT_SECS: u64 = 5;
const IDLE_CHECK_SECS: u64 = 60;
const IDLE_EXIT_SECS: u64 = 600;

fn arg_value(args: &[String], names: &[&str]) -> Option<String> {
    for n in names {
        if let Some(pos) = args.iter().position(|a| a == n) {
            if let Some(v) = args.get(pos + 1) {
                return Some(v.clone());
            }
        }
    }
    None
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn log_path() -> PathBuf {
    let dir = shellexpand::tilde("~/.npcsh/logs").to_string();
    let _ = std::fs::create_dir_all(&dir);
    PathBuf::from(dir).join("npcshd.log")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if has_flag(&args, "--version") || has_flag(&args, "-V") {
        println!("npcshd {}", env!("NPCSH_VERSION"));
        return;
    }

    if !has_flag(&args, "--foreground") {
        // Re-exec detached, with output going to the log file rather than a
        // terminal that will soon disappear.
        let exe = match std::env::current_exe() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("npcshd: cannot resolve own path: {e}");
                std::process::exit(1);
            }
        };
        let mut cmd = std::process::Command::new(exe);
        for a in args.iter().skip(1) {
            if a != "--foreground" {
                cmd.arg(a);
            }
        }
        cmd.arg("--foreground");
        if let Ok(f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path())
        {
            if let Ok(f2) = f.try_clone() {
                cmd.stdout(std::process::Stdio::from(f));
                cmd.stderr(std::process::Stdio::from(f2));
            }
        } else {
            cmd.stdout(std::process::Stdio::null());
            cmd.stderr(std::process::Stdio::null());
        }
        cmd.stdin(std::process::Stdio::null());
        #[cfg(unix)]
        unsafe {
            use std::os::unix::process::CommandExt;
            // New session: survive the parent shell exiting and lose its tty.
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        match cmd.spawn() {
            Ok(_) => std::process::exit(0),
            Err(e) => {
                eprintln!("npcshd: failed to detach: {e}");
                std::process::exit(1);
            }
        }
    }

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("npcshd: runtime: {e}");
            std::process::exit(1);
        }
    };
    let team_hint = arg_value(&args, &["--team"]).unwrap_or_else(npcsh::find_team_dir);
    if let Err(e) = rt.block_on(run(args)) {
        eprintln!("npcshd: {e}");
        let db = daemon::default_db_path();
        daemon::clear_status(&db, &team_hint);
        std::process::exit(1);
    }
}

async fn run(args: Vec<String>) -> Result<(), String> {
    npcsh::load_env();

    let db_path = arg_value(&args, &["--db"]).unwrap_or_else(daemon::default_db_path);
    let server_url = arg_value(&args, &["--server-url"])
        .or_else(|| std::env::var("NPCSH_SERVER_URL").ok())
        .unwrap_or_else(|| "http://127.0.0.1:5237".to_string());
    let team_dir = arg_value(&args, &["--team"]).unwrap_or_else(npcsh::find_team_dir);
    npcsh::daemon::set_current_team(&team_dir);

    // ── single instance, per team ────────────────────────────────────────────
    let _lock = match acquire_lock(&team_dir) {
        Some(l) => l,
        None => {
            // A daemon for this team already holds the lock: success, not error.
            let _ = writeln!(
                std::io::stderr(),
                "npcshd: a daemon for team {} is already running",
                team_dir
            );
            return Ok(());
        }
    };

    if let Err(e) = npcsh::db_migrations::run_migrations(&db_path) {
        return Err(format!("database migration failed: {e}"));
    }

    // ── registry ─────────────────────────────────────────────────────────────
    let cron_file = shellexpand::tilde("~/.npcsh/loops.yaml").to_string();
    let registry = Arc::new(Mutex::new(CronRegistry::with_file(cron_file)));
    if let Ok(mut reg) = registry.lock() {
        reg.load_from_jinxes(&team_dir);
    }

    // ── listener ─────────────────────────────────────────────────────────────
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| format!("bind failed: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("local_addr failed: {e}"))?
        .port();

    let status = DaemonStatus {
        pid: std::process::id(),
        port,
        team_dir: daemon::team_key(&team_dir),
        started_at: chrono::Utc::now().to_rfc3339(),
        last_heartbeat: chrono::Utc::now().to_rfc3339(),
        status: "running".to_string(),
    };
    daemon::write_status(&db_path, &status)
        .ok_or_else(|| "could not record daemon_state".to_string())?;

    let cfg = npcsh::loop_runner::DetachedLoopConfig::new(
        team_dir.clone(),
        db_path.clone(),
        server_url.clone(),
    );

    // ── ticker: the daemon owns execution ────────────────────────────────────
    npcsh::cron::spawn_cron_ticker(
        registry.clone(),
        tokio::runtime::Handle::current(),
        cfg.clone(),
        None,
    );

    let _ = writeln!(
        std::io::stderr(),
        "npcshd: pid {} listening on 127.0.0.1:{} | team {} | {} loop(s)",
        std::process::id(),
        port,
        team_dir,
        registry.lock().map(|r| r.list().len()).unwrap_or(0)
    );

    let shutdown = Arc::new(tokio::sync::Notify::new());

    // ── heartbeat + idle exit ────────────────────────────────────────────────
    //
    // A daemon with no configured loops has nothing to keep alive, so it exits
    // after a grace period rather than lingering forever on machines where
    // npcsh was opened once. It never exits while a job exists: that is the
    // only case where exiting would silently stop someone's work.
    {
        let db = db_path.clone();
        let team = cfg.team_dir.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(HEARTBEAT_SECS));
            loop {
                tick.tick().await;
                daemon::touch_heartbeat(&db, &team, "running");
            }
        });
    }

    // Separate task owns the exit decision so the heartbeat stays simple.
    {
        let sd = shutdown.clone();
        let reg = registry.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(IDLE_CHECK_SECS)).await;
                let idle = reg.lock().map(|r| r.list().is_empty()).unwrap_or(false)
                    && npcsh::loop_runner::running_keys().is_empty();
                if idle {
                    let _ = writeln!(
                        std::io::stderr(),
                        "npcshd: no loops configured for {}s; exiting",
                        IDLE_EXIT_SECS
                    );
                    sd.notify_waiters();
                    break;
                }
            }
        });
    }

    // ── signal handling ──────────────────────────────────────────────────────
    {
        let sd = shutdown.clone();
        tokio::spawn(async move {
            #[cfg(unix)]
            {
                let mut term =
                    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    {
                        Ok(s) => s,
                        Err(_) => {
                            let _ = tokio::signal::ctrl_c().await;
                            sd.notify_waiters();
                            return;
                        }
                    };
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
            sd.notify_waiters();
        });
    }

    // ── accept loop ──────────────────────────────────────────────────────────
    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                let _ = writeln!(std::io::stderr(), "npcshd: shutting down");
                break;
            }
            accepted = listener.accept() => {
                let (stream, _peer) = match accepted {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = writeln!(std::io::stderr(), "npcshd: accept: {e}");
                        continue;
                    }
                };
                let reg = registry.clone();
                let cfg = cfg.clone();
                let db = db_path.clone();
                let sd = shutdown.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_conn(stream, reg, cfg, db, sd).await {
                        let _ = writeln!(std::io::stderr(), "npcshd: conn: {e}");
                    }
                });
            }
        }
    }

    daemon::clear_status(&db_path, &team_dir);
    // Give in-flight loops a moment to persist their partial transcripts.
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    Ok(())
}

fn lock_path_for(team_dir: &str) -> String {
    let key = daemon::team_key(team_dir);
    // Hash the team path into a short stable name so the lock is per-team.
    let digest = md5::compute(key.as_bytes());
    shellexpand::tilde(&format!("~/.npcsh/npcshd-{:x}.lock", digest)).to_string()
}

fn acquire_lock(team_dir: &str) -> Option<std::fs::File> {
    let path = lock_path_for(team_dir);
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let f = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&path)
        .ok()?;

    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            return None;
        }
    }
    // Record our pid for humans reading the file.
    let mut fw = &f;
    let _ = fw.set_len(0);
    let _ = writeln!(fw, "{}", std::process::id());
    Some(f)
}

async fn handle_conn(
    stream: TcpStream,
    registry: Arc<Mutex<CronRegistry>>,
    cfg: npcsh::loop_runner::DetachedLoopConfig,
    db_path: String,
    shutdown: Arc<tokio::sync::Notify>,
) -> Result<(), String> {
    let (rd, mut wr) = stream.into_split();
    let mut reader = BufReader::new(rd);
    let mut line = String::new();
    let n = reader
        .read_line(&mut line)
        .await
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Ok(());
    }

    let req: Request = match serde_json::from_str(line.trim()) {
        Ok(r) => r,
        Err(e) => {
            let body = serde_json::to_string(&Response::err(format!("bad request: {e}")))
                .map_err(|e| e.to_string())?;
            let _ = wr.write_all(format!("{body}\n").as_bytes()).await;
            return Ok(());
        }
    };

    let resp = match req {
        Request::Ping => {
            let mut r = Response::ok();
            r.pid = Some(std::process::id());
            r.uptime_secs = Some(0);
            r
        }
        Request::List => {
            let jobs: Vec<JobInfo> = registry
                .lock()
                .map(|reg| {
                    reg.list()
                        .iter()
                        .map(|j| {
                            let key = format!("loop-{}-{}", j.npc, j.id);
                            JobInfo {
                                id: j.id,
                                npc: j.npc.clone(),
                                interval_secs: j.interval_secs,
                                task: j.task.clone(),
                                kind: if j.kind == npcsh::cron::CronJobKind::Jinx {
                                    "jinx".to_string()
                                } else {
                                    "chat".to_string()
                                },
                                enabled: j.enabled,
                                running: npcsh::loop_runner::is_in_flight(&key),
                                next_in_secs: j
                                    .next_run
                                    .saturating_duration_since(std::time::Instant::now())
                                    .as_secs(),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mut r = Response::ok();
            r.jobs = jobs;
            r.running = npcsh::loop_runner::running_keys();
            r
        }
        Request::Run { id } => {
            let job = registry
                .lock()
                .ok()
                .and_then(|reg| reg.list().iter().find(|j| j.id == id).cloned());
            match job {
                Some(job) => {
                    npcsh::loop_runner::spawn_detached_loop(
                        &tokio::runtime::Handle::current(),
                        job,
                        cfg.clone(),
                    );
                    Response::ok()
                }
                None => Response::err(format!("no loop with id {id}")),
            }
        }
        Request::Stop { id } => {
            let key = registry.lock().ok().and_then(|reg| {
                reg.list()
                    .iter()
                    .find(|j| j.id == id)
                    .map(|j| format!("loop-{}-{}", j.npc, j.id))
            });
            match key {
                Some(k) if npcsh::loop_runner::interrupt_loop(&k) => Response::ok(),
                Some(_) => Response::err(format!("loop {id} is not running")),
                None => Response::err(format!("no loop with id {id}")),
            }
        }
        Request::Enable { id, enabled } => {
            let ok = registry
                .lock()
                .map(|mut reg| reg.enable(id, enabled))
                .unwrap_or(false);
            if ok {
                Response::ok()
            } else {
                Response::err(format!("no loop with id {id}"))
            }
        }
        Request::Attach { id } => {
            // Find the key for this job, then relay its broadcast stream as
            // JSON lines. The client renders these; it does not run anything.
            let key = registry.lock().ok().and_then(|reg| {
                reg.list()
                    .iter()
                    .find(|j| j.id == id)
                    .map(|j| format!("loop-{}-{}", j.npc, j.id))
            });
            let Some(key) = key else {
                let body = serde_json::to_string(&Response::err(format!("no loop with id {id}")))
                    .map_err(|e| e.to_string())?;
                let _ = wr.write_all(format!("{body}\n").as_bytes()).await;
                return Ok(());
            };
            let body = serde_json::to_string(&Response::ok()).map_err(|e| e.to_string())?;
            wr.write_all(format!("{body}\n").as_bytes())
                .await
                .map_err(|e| e.to_string())?;

            // Wait for the run to appear, then relay until it ends.
            let mut rx = None;
            for _ in 0..120 {
                if let Some(r) = npcsh::loop_runner::subscribe_loop(&key) {
                    rx = Some(r);
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
            let Some(mut rx) = rx else {
                let note = serde_json::json!({"event": "idle", "key": key});
                let _ = wr.write_all(format!("{note}\n").as_bytes()).await;
                return Ok(());
            };
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        let payload = npcsh::loop_runner::event_to_json(&ev);
                        if wr
                            .write_all(format!("{payload}\n").as_bytes())
                            .await
                            .is_err()
                        {
                            break;
                        }
                        if matches!(ev, npcsh::loop_runner::LoopEvent::Finished) {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
            wr.flush().await.ok();
            return Ok(());
        }
        Request::Reload => {
            // Same code path as the file watcher, so an explicit reload and an
            // on-disk edit can never diverge.
            let ok = registry
                .lock()
                .map(|mut reg| {
                    reg.reload_from_disk(&cfg.team_dir);
                    true
                })
                .unwrap_or(false);
            if ok {
                Response::ok()
            } else {
                Response::err("could not reload loops.yaml")
            }
        }
        Request::Shutdown => {
            shutdown.notify_waiters();
            Response::ok()
        }
    };

    let body = serde_json::to_string(&resp).map_err(|e| e.to_string())?;
    wr.write_all(format!("{body}\n").as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    wr.flush().await.ok();
    let _ = &db_path;
    Ok(())
}
