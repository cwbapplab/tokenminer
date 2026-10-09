//! Pearl (PRL) Stratum session over the TokenMiner proxy.
//!
//! Speaks the Pearl named-params dialect the proxy and pool implement:
//! `mining.authorize {wallet, worker, pass, agent}` (no subscribe), a
//! `mining.notify {header, height, job_id, target, cert_version}` that may arrive
//! **before** the authorize ack, and `mining.submit {job_id, plain_proof}`.
//!
//! Framing/parsing comes from `sp-protocol` — the proxy's own crate — so the wire format
//! cannot drift. Proofs are produced by [`crate::miners::pearl_gpu`] and encoded by
//! [`crate::miners::pearl_mining`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::BufReader;
use tokio::net::TcpStream;
use tokio::sync::mpsc;

use sp_protocol::{read_line, write_line, Message, METHOD_AUTHORIZE, METHOD_NOTIFY, METHOD_SUBMIT};

use tauri::AppHandle;

use crate::miners::{emit_log, emit_status, now, MinerKind, MinerState, MinerStatus};

const AGENT: &str = concat!("tokenminer-desktop/", env!("CARGO_PKG_VERSION"));

/// Seconds between hashrate reports to the UI.
///
/// Five, because that is the search's own slice (`MAX_SECONDS_PER_SEARCH`): the loop comes back to
/// here at least that often, so this is as fast as the number can honestly change. It used to be 25,
/// which is faster than nothing and slower than the dashboard can show — the chart's window is 60
/// seconds, so a 25-second cadence put two or three distinct values in the whole graph and the Pearl
/// line read as a staircase rather than a rate.
const PROGRESS_EVERY: u64 = 5;

/// The Pearl hashrate window.
///
/// A rate is tiles divided by time, and the only honest question is *which* time. Between two
/// searches the miner assembles and verifies a share, encodes it, sends it, and waits for the pool
/// to hand over the next job -- and sleeps for a quarter of a second to a second whenever no usable
/// job is in hand. None of that folds a tile, so none of it belongs in the denominator of a rate
/// whose numerator counts only tiles. A share costs ~490ms to assemble on the way out, so at any
/// ordinary share rate that is not a rounding error.
///
/// Separate from the cadence: the window is reported every [`PROGRESS_EVERY`] seconds of wall clock
/// because that is how fast the UI can show a change, and it is measured over search time because
/// that is what the number describes.
#[derive(Debug)]
struct RateWindow {
    tiles: u64,
    /// The engine's cumulative tile count as of the last window close, so a caller's running total
    /// can be differenced against it without the caller tracking that itself.
    seen_tiles: u64,
    searched: f64,
    opened_at: std::time::Instant,
}

impl RateWindow {
    fn new() -> Self {
        Self {
            tiles: 0,
            seen_tiles: 0,
            searched: 0.0,
            opened_at: std::time::Instant::now(),
        }
    }

    /// Record one completed search: `total_tiles` is the engine's cumulative count and `seconds` is
    /// how long that search took.
    fn add(&mut self, total_tiles: u64, seconds: f64) {
        self.tiles += total_tiles.saturating_sub(self.seen_tiles);
        self.seen_tiles = total_tiles;
        self.searched += seconds;
    }

    /// The rate to publish if the reporting cadence has elapsed, closing the window either way.
    ///
    /// `Some(rate)` means publish it. A zero-length search is possible -- a job superseded before
    /// the first batch launches -- so the divisor is guarded rather than assumed.
    fn take_if_due(&mut self, now: std::time::Instant) -> Option<f64> {
        if now.duration_since(self.opened_at).as_secs_f64() < PROGRESS_EVERY as f64 {
            return None;
        }
        let rate = if self.searched > 0.0 {
            self.tiles as f64 / self.searched
        } else {
            0.0
        };
        self.tiles = 0;
        self.searched = 0.0;
        self.opened_at = now;
        Some(rate)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrlConfig {
    /// `host:port` of the stratum proxy (from the parsed miner command).
    pub host: String,
    pub wallet: String,
    #[serde(default)]
    pub worker: String,
    /// Mining configuration; defaults to the reference miner's settings.
    #[serde(default)]
    pub mining: crate::miners::pearl_mining::PearlMining,
    /// Which certificate/seed derivation the network is on. Decides how the noise seeds are
    /// derived from the committed roots, so it is part of every proof this session produces.
    #[serde(default = "default_cert_version")]
    pub cert_version: u32,
}

fn default_cert_version() -> u32 {
    2
}

#[derive(Debug, Clone)]
struct Job {
    job_id: String,
    header_hex: String,
    cert_version: u32,
    target: String,
}

fn job_from_notify(params: &Value) -> Option<Job> {
    let object = params.as_object()?;

    Some(Job {
        job_id: object.get("job_id").and_then(Value::as_str)?.to_string(),
        header_hex: object.get("header").and_then(Value::as_str)?.to_string(),
        cert_version: object
            .get("cert_version")
            .and_then(Value::as_u64)
            .unwrap_or(2) as u32,
        target: object
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    })
}

/// State shared between the reader loop, the writer task and the miner thread.
struct Shared {
    app: AppHandle,
    status: Arc<Mutex<MinerStatus>>,
    /// The job currently being worked.
    job: Mutex<Option<Job>>,
    /// Submit id -> job id, so a pool reply can be attributed.
    pending: Mutex<HashMap<u64, String>>,
    next_id: AtomicU64,
    /// Set by `stop_miner` (via the handle `spawn` hands back) to tell the
    /// miner thread to unwind. An `Arc` rather than a bare `AtomicBool`
    /// because the caller has to hold the same flag the thread polls —
    /// aborting the reader task alone would leave the thread looping with
    /// the device and its memory still held.
    stop: Arc<AtomicBool>,
    /// Whether submits are gzipped. Decided by the authorize ack, not by configuration: pools
    /// negotiate it, and Kryptex's v2 session is accepted in an object rather than a `true`.
    gzip: AtomicBool,
    /// Outbound submits: (id, job_id, plain_proof base64).
    tx: mpsc::UnboundedSender<(u64, String, String)>,
}

/// How the pool wants proofs framed on submit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofEncoding {
    Plain,
    Gzip,
}

/// Reads the authorize ack and decides the submit framing.
///
/// Not [`sp_protocol::Response::is_accepted`]: that only accepts a literal `true`, so a v2 session's
/// object answer (`{"type":"v2"}`) would look like a rejection. Per the captured pool dialects, a
/// `type` of `v2` means every submit carries `base64(gzip(bincode))`.
pub fn read_authorize_result(response: &sp_protocol::Response) -> Result<ProofEncoding, String> {
    if let Some(error) = &response.error {
        return Err(error.to_string());
    }

    match &response.result {
        Some(Value::Bool(true)) => Ok(ProofEncoding::Plain),
        Some(Value::Bool(false)) | None => {
            Err("the proxy did not accept the authorize".to_string())
        }
        Some(Value::Object(object)) => {
            let v2 = object.get("type").and_then(Value::as_str) == Some("v2");

            Ok(if v2 {
                ProofEncoding::Gzip
            } else {
                ProofEncoding::Plain
            })
        }
        Some(other) => Err(format!("unexpected authorize result: {other}")),
    }
}

impl Shared {
    fn log(&self, level: &str, message: impl Into<String>) {
        emit_log(&self.app, MinerKind::Pearl, level, message);
    }

    fn set_message(&self, message: impl Into<String>) {
        if let Ok(mut current) = self.status.lock() {
            current.message = Some(message.into());
            current.updated_at = now();
            let snapshot = current.clone();
            drop(current);
            emit_status(&self.app, &snapshot);
        }
    }

    fn current_job_id(&self) -> Option<String> {
        self.job
            .lock()
            .ok()
            .and_then(|current| current.clone())
            .map(|job| job.job_id)
    }
}

fn update_status(
    app: &AppHandle,
    status: &Arc<Mutex<MinerStatus>>,
    apply: impl FnOnce(&mut MinerStatus),
) {
    let Ok(mut current) = status.lock() else {
        return;
    };
    apply(&mut current);
    current.updated_at = now();
    let snapshot = current.clone();
    drop(current);
    emit_status(app, &snapshot);
}

/// Runs the PRL session on a background task.
///
/// Returns the task handle alongside the session's stop signal. The caller
/// stores the signal and sets it to stop: the miner thread is a plain OS
/// thread that aborting the task cannot reach, so it has to observe the
/// flag and exit on its own.
pub fn spawn(
    app: AppHandle,
    config: PrlConfig,
    status: Arc<Mutex<MinerStatus>>,
) -> (tauri::async_runtime::JoinHandle<()>, Arc<AtomicBool>) {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_run = stop.clone();

    let handle = tauri::async_runtime::spawn(async move {
        match run(&app, &config, &status, stop_for_run).await {
            Ok(()) => {
                update_status(&app, &status, |s| {
                    s.state = MinerState::Stopped;
                    s.active_jobs = 0;
                    s.message = Some("proxy closed the connection".into());
                });
                emit_log(&app, MinerKind::Pearl, "info", "Stratum session ended.");
            }
            Err(error) => {
                update_status(&app, &status, |s| {
                    s.state = MinerState::Error;
                    s.active_jobs = 0;
                    s.message = Some(error.clone());
                });
                emit_log(&app, MinerKind::Pearl, "error", error);
            }
        }
    });

    (handle, stop)
}

async fn run(
    app: &AppHandle,
    config: &PrlConfig,
    status: &Arc<Mutex<MinerStatus>>,
    stop: Arc<AtomicBool>,
) -> Result<(), String> {
    emit_log(
        app,
        MinerKind::Pearl,
        "info",
        format!("Connecting to the stratum proxy at {}…", config.host),
    );

    let stream = TcpStream::connect(&config.host).await.map_err(|e| {
        format!(
            "Failed to connect to the stratum target '{}': {e}. It must be host:port — set the \
             Stratum endpoint in Settings (default localhost:3333).",
            config.host
        )
    })?;

    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    let (tx, mut rx) = mpsc::unbounded_channel::<(u64, String, String)>();
    let shared = Arc::new(Shared {
        app: app.clone(),
        status: status.clone(),
        job: Mutex::new(None),
        pending: Mutex::new(HashMap::new()),
        next_id: AtomicU64::new(1),
        stop,
        gzip: AtomicBool::new(config.mining.gzip),
        tx,
    });

    // Authorize first, while we still own the write half — the first notify can arrive before
    // the ack, so the reader must be running by then; the writer task takes over afterwards.
    let authorize_id = shared.next_id.fetch_add(1, Ordering::SeqCst);
    let authorize = json!({
        "id": authorize_id,
        "method": METHOD_AUTHORIZE,
        "params": {
            "wallet": config.wallet,
            "worker": config.worker,
            "pass": "x",
            "agent": AGENT,
        }
    });

    write_line(&mut write_half, &authorize.to_string())
        .await
        .map_err(|e| format!("Failed to send authorize: {e}"))?;

    emit_log(
        app,
        MinerKind::Pearl,
        "info",
        format!(
            "→ authorize wallet={} worker={}",
            config.wallet, config.worker
        ),
    );

    // Writer: every proof the miner produces goes out on this connection.
    tauri::async_runtime::spawn(async move {
        while let Some((id, job_id, proof)) = rx.recv().await {
            let line = json!({
                "id": id,
                "method": METHOD_SUBMIT,
                "params": { "job_id": job_id, "plain_proof": proof }
            })
            .to_string();

            if write_line(&mut write_half, &line).await.is_err() {
                break;
            }
        }
    });

    spawn_miner(shared.clone(), config.clone());

    while let Some(line) = read_line(&mut reader).await.map_err(|e| e.to_string())? {
        if line.trim().is_empty() {
            continue;
        }

        let message = match Message::parse(&line) {
            Ok(message) => message,
            Err(_) => {
                emit_log(
                    app,
                    MinerKind::Pearl,
                    "debug",
                    format!("← (unparsed) {line}"),
                );
                continue;
            }
        };

        match message {
            Message::Notification(notification) if notification.method == METHOD_NOTIFY => {
                let Some(job) = job_from_notify(&notification.params) else {
                    emit_log(
                        app,
                        MinerKind::Pearl,
                        "debug",
                        "← mining.notify (unrecognised params)",
                    );
                    continue;
                };

                emit_log(
                    app,
                    MinerKind::Pearl,
                    "info",
                    format!(
                        "← notify job={} cert_version={} target={}",
                        job.job_id, job.cert_version, job.target
                    ),
                );

                update_status(app, status, |s| {
                    s.active_jobs = 1;
                });
                shared.set_message(format!("job {}", job.job_id));

                if let Ok(mut current) = shared.job.lock() {
                    *current = Some(job);
                }
            }
            Message::Notification(notification) => {
                emit_log(
                    app,
                    MinerKind::Pearl,
                    "debug",
                    format!("← {} {}", notification.method, notification.params),
                );
            }
            Message::Response(response) => {
                if response.id.as_u64() == Some(authorize_id) {
                    let encoding = read_authorize_result(&response)
                        .map_err(|error| format!("Proxy rejected authorize: {error}"))?;

                    if encoding == ProofEncoding::Gzip {
                        shared.gzip.store(true, Ordering::SeqCst);
                    }

                    emit_log(
                        app,
                        MinerKind::Pearl,
                        "info",
                        format!(
                            "← authorized as {}.{}{}",
                            config.wallet,
                            config.worker,
                            match encoding {
                                ProofEncoding::Gzip => " (v2 session: proofs are gzipped)",
                                ProofEncoding::Plain => "",
                            }
                        ),
                    );
                    update_status(app, status, |s| {
                        s.state = MinerState::Running;
                        s.workers = 1;
                    });
                    shared
                        .set_message(format!("authorized as {}.{}", config.wallet, config.worker));
                    continue;
                }

                if let Some(id) = response.id.as_u64() {
                    let job_id = shared
                        .pending
                        .lock()
                        .ok()
                        .and_then(|mut pending| pending.remove(&id));

                    if let Some(job_id) = job_id {
                        if response.is_accepted() {
                            emit_log(
                                app,
                                MinerKind::Pearl,
                                "info",
                                format!("✓ share accepted ({job_id})"),
                            );
                            shared.set_message(format!("share accepted ({job_id})"));
                        } else {
                            emit_log(
                                app,
                                MinerKind::Pearl,
                                "warn",
                                format!(
                                    "✗ share rejected ({job_id}): {}",
                                    response
                                        .error
                                        .as_ref()
                                        .map(Value::to_string)
                                        .unwrap_or_else(|| "unknown".into())
                                ),
                            );
                            shared.set_message(format!("share rejected ({job_id})"));
                        }
                    }
                }
            }
            Message::Request(request) => {
                emit_log(
                    app,
                    MinerKind::Pearl,
                    "debug",
                    format!("← {} {}", request.method, request.params),
                );
            }
        }
    }

    shared.stop.store(true, Ordering::SeqCst);

    Ok(())
}

/// The search thread: takes the current job, mines it against the job's *share* target on the GPU,
/// and hands any solution to the writer.
fn spawn_miner(shared: Arc<Shared>, config: PrlConfig) {
    use crate::miners::pearl_gpu::GpuMiner;
    use crate::miners::pearl_mining;
    use crate::miners::pearl_pow;

    std::thread::spawn(move || {
        let mining = config.mining.clone();

        shared.log(
            "info",
            format!(
                "Mining on GPU: m={} n={} k={} rank={} (proof framing is negotiated on authorize)",
                mining.m, mining.n, mining.k, mining.rank
            ),
        );

        // Open the device once. Without a working, self-tested backend there is nothing to do, so
        // say so and stop rather than spinning on every job.
        let mut miner = match GpuMiner::open(mining) {
            Ok(miner) => miner,
            Err(error) => {
                shared.log("error", format!("Pearl cannot mine: {error}"));
                update_status(&shared.app, &shared.status, |s| {
                    s.state = MinerState::Error;
                    s.message = Some(format!("GPU unavailable: {error}"));
                });
                return;
            }
        };

        let mut rate = RateWindow::new();
        // A job whose target we cannot use is retried only after a new job arrives, so one bad
        // notify does not spam the console.
        let mut unusable_job: Option<String> = None;

        loop {
            if shared.stop.load(Ordering::SeqCst) {
                return;
            }

            let job = shared.job.lock().ok().and_then(|current| current.clone());
            let Some(job) = job else {
                std::thread::sleep(Duration::from_millis(250));
                continue;
            };

            let header = match pearl_mining::parse_header(&job.header_hex) {
                Ok(header) => header,
                Err(error) => {
                    shared.log("error", error);
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };

            // Mine against the pool's share target rather than the block difficulty committed in
            // the header: a share is by definition easier than a block, so the block bound would
            // reject every candidate the pool accepts.
            let (share_target, bound) = match pearl_pow::share_search(&job.target, miner.mining()) {
                Ok(pair) => pair,
                Err(error) => {
                    if unusable_job.as_deref() != Some(job.job_id.as_str()) {
                        shared.log(
                            "warn",
                            format!(
                                "Job {} cannot be mined: {error} Waiting for the next job.",
                                job.job_id
                            ),
                        );
                        unusable_job = Some(job.job_id.clone());
                    }
                    std::thread::sleep(Duration::from_millis(500));
                    continue;
                }
            };

            shared.log(
                "info",
                format!(
                    "Searching job {} (nbits={:#010x}, share bound={:#x})…",
                    job.job_id, header.nbits, bound
                ),
            );

            // The certificate version comes off the job, not the config: it selects the seed
            // derivation, and a proof built under the wrong one will not verify.
            let cert_version = if job.cert_version == 0 {
                config.cert_version
            } else {
                job.cert_version
            };

            let job_id = job.job_id.clone();
            let stopped = || {
                shared.stop.load(Ordering::SeqCst)
                    || shared.current_job_id().as_deref() != Some(job_id.as_str())
            };

            let search_started = std::time::Instant::now();
            let searched_for = miner.search(&header, cert_version, bound, &stopped);
            // Recorded before the result is unwrapped, so a search that ends in an error has still
            // been paid for and still counts against the interval it was spent in.
            rate.add(miner.tiles(), search_started.elapsed().as_secs_f64());

            let found = match searched_for {
                Ok(found) => found,
                Err(error) => {
                    shared.log("error", format!("GPU mining failed: {error}"));
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };

            if let Some(tiles_per_second) = rate.take_if_due(std::time::Instant::now()) {
                let th_per_second = config.mining.th_per_second(tiles_per_second);
                update_status(&shared.app, &shared.status, |s| {
                    // TH/s, the unit every other Pearl miner prints and the one the reference
                    // derives from its tile count the same way — see
                    // `PearlMining::th_per_second`.
                    //
                    // `hashrate` is the field the dashboard's Pearl card reads; `gpu_hashrate` is
                    // the same number, filed separately because that is the whole engine. Setting
                    // only the second left the card reading zero forever, and a Pearl engine
                    // running at full tilt looked identical to one that had never started.
                    s.hashrate = th_per_second;
                    s.gpu_hashrate = th_per_second;
                });
            }

            let Some(proof) = found else {
                continue;
            };

            // The pool re-verifies everything we send. So do we, first — a proof that fails here
            // is a bug in the GPU path, and submitting it would only burn pool goodwill.
            if let Err(error) =
                pearl_pow::verify_share_locally(&header, cert_version, &proof, share_target)
            {
                shared.log(
                    "error",
                    format!("Mined proof failed local verification: {error}"),
                );
                continue;
            }

            let gzip = shared.gzip.load(Ordering::SeqCst);
            let encoded = match pearl_mining::encode_plain_proof(&proof, gzip) {
                Ok(encoded) => encoded,
                Err(error) => {
                    shared.log("error", error);
                    continue;
                }
            };

            let id = shared.next_id.fetch_add(1, Ordering::SeqCst);
            if let Ok(mut pending) = shared.pending.lock() {
                pending.insert(id, job.job_id.clone());
            }

            shared.log(
                "info",
                format!(
                    "⛏ solution found for job {} ({} chars, {}) — submitting",
                    job.job_id,
                    encoded.len(),
                    if gzip { "gzip+bincode" } else { "bincode" }
                ),
            );

            if shared.tx.send((id, job.job_id.clone(), encoded)).is_err() {
                return;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the window exists for: time that was not spent folding does not reach the
    /// divisor. A hashrate is tiles per second of *mining*, and the loop spends real seconds
    /// assembling shares, verifying them and waiting for jobs.
    #[test]
    fn the_rate_divides_by_search_time_and_not_by_wall_clock() {
        let start = std::time::Instant::now();
        let mut window = RateWindow::new();
        window.opened_at = start;

        // Two searches, five seconds of searching between them, and ten seconds of wall clock --
        // the extra five being a share assembled and verified, which is what the loop actually does.
        window.add(1_000_000, 2.5);
        let midway = start + Duration::from_secs(3);
        window.add(3_000_000, 2.5);

        // Not due yet: the cadence is five seconds of wall clock, and only three have passed.
        assert_eq!(window.take_if_due(midway), None);

        // Due now. The rate is 3,000,000 tiles over 5.0 seconds of search -- 600,000/s -- and *not*
        // over the 10 seconds of wall clock, which would read 300,000/s and blame the pool for half
        // the hashrate the card was producing.
        let rate = window
            .take_if_due(start + Duration::from_secs(10))
            .expect("ten seconds is past the cadence");
        assert!(
            (rate - 600_000.0).abs() < 1.0,
            "expected 600000 tiles/s from search time, got {rate}"
        );

        // The window closed, so the next report starts from zero rather than double-counting.
        assert_eq!(window.tiles, 0);
        assert_eq!(window.searched, 0.0);
    }

    /// A search can return without folding anything -- a job superseded before the first batch
    /// launches -- so an interval can close with no search time in it. Dividing by that is not an
    /// error to propagate into the UI; the number is simply zero.
    #[test]
    fn a_window_with_no_search_time_reports_zero_rather_than_dividing_by_zero() {
        let start = std::time::Instant::now();
        let mut window = RateWindow::new();
        window.opened_at = start;
        window.add(0, 0.0);
        assert_eq!(window.take_if_due(start + Duration::from_secs(6)), Some(0.0));
    }

    /// The engine's tile count is cumulative and never reset, so the window differences it. If it
    /// counted the running total instead, the second search in an interval would report its own
    /// total plus the previous one and the rate would climb without bound.
    #[test]
    fn a_cumulative_tile_count_is_differenced_not_summed() {
        let start = std::time::Instant::now();
        let mut window = RateWindow::new();
        window.opened_at = start;

        window.add(500, 1.0); // engine total is now 500
        window.add(900, 1.0); // and now 900: this search folded 400, not 900

        let rate = window
            .take_if_due(start + Duration::from_secs(6))
            .expect("six seconds is past the cadence");
        assert!(
            (rate - 450.0).abs() < 0.5,
            "expected 450 tiles/s from 900 tiles folded, got {rate}"
        );
    }

    #[test]
    fn reads_a_pearl_job_off_a_notify() {
        let params = json!({
            "header": "00004020",
            "job_id": "18e4eadf_1",
            "target": "000000044b",
            "cert_version": 3
        });

        let job = job_from_notify(&params).expect("a job");
        assert_eq!(job.job_id, "18e4eadf_1");
        assert_eq!(job.cert_version, 3);
    }

    #[test]
    fn a_notify_without_a_header_is_ignored() {
        assert!(job_from_notify(&json!({ "job_id": "x" })).is_none());
    }

    #[test]
    fn a_v2_authorize_ack_is_an_acceptance_and_selects_gzip() {
        // Kryptex's v2 session answers with an object. `Response::is_accepted` only accepts a
        // literal `true`, so it would read this as a rejection and drop the session.
        let ack = sp_protocol::Response {
            id: json!(1),
            result: Some(json!({ "type": "v2" })),
            error: None,
        };

        assert_eq!(read_authorize_result(&ack), Ok(ProofEncoding::Gzip));
    }

    #[test]
    fn a_plain_authorize_ack_selects_plain_framing() {
        let ack = sp_protocol::Response {
            id: json!(1),
            result: Some(json!(true)),
            error: None,
        };

        assert_eq!(read_authorize_result(&ack), Ok(ProofEncoding::Plain));
    }

    #[test]
    fn a_refused_authorize_is_reported_rather_than_accepted() {
        // The shape the proxy sends when the wallet is not in its policy.
        let errored = sp_protocol::Response {
            id: json!(1),
            result: None,
            error: Some(json!([24, "unauthorised wallet", null])),
        };
        assert!(read_authorize_result(&errored).is_err());

        let refused = sp_protocol::Response {
            id: json!(1),
            result: Some(json!(false)),
            error: None,
        };
        assert!(read_authorize_result(&refused).is_err());

        // No result and no error is not an acceptance either.
        let empty = sp_protocol::Response {
            id: json!(1),
            result: None,
            error: None,
        };
        assert!(read_authorize_result(&empty).is_err());
    }
}
