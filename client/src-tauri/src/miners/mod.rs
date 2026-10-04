//! Native, in-process miner management.
//!
//! Both engines are compiled into this binary and driven from here — no child
//! processes. Each engine is gated behind a Cargo feature so it can be built
//! independently (`quantus`, `pearl`).

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::state::AppState;

#[cfg(feature = "quantus")]
pub mod quantus;
#[cfg(not(feature = "quantus"))]
pub mod quantus {
    //! Fallback when the `quantus` feature is not enabled.

    use std::sync::Arc;

    pub fn start(
        _app: &tauri::AppHandle,
        _config: Option<serde_json::Value>,
        _status: Arc<std::sync::Mutex<super::MinerStatus>>,
    ) -> Result<super::Started, String> {
        Err("Quantus support is not compiled in. Rebuild with --features quantus.".into())
    }

    pub fn stop(_app: &tauri::AppHandle) {}
}

#[cfg(feature = "pearl")]
pub mod pearl;
#[cfg(feature = "pearl")]
pub mod pearl_gpu;
#[cfg(feature = "pearl")]
pub mod pearl_mining;
#[cfg(feature = "pearl")]
pub mod pearl_pow;
#[cfg(not(feature = "pearl"))]
pub mod pearl {
    //! Fallback when the `pearl` feature is not enabled.

    use super::{MinerStatus, Started};
    use std::sync::{Arc, Mutex};

    pub fn start(
        _app: &tauri::AppHandle,
        _config: Option<serde_json::Value>,
        _status: Arc<Mutex<MinerStatus>>,
    ) -> Result<Started, String> {
        Err("Pearl support is not compiled in. Rebuild with --features pearl.".into())
    }

    pub fn stop() {}
}

pub mod command;
pub mod gpu;

/// Which mining backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MinerKind {
    Quantus,
    Pearl,
}

impl MinerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MinerKind::Quantus => "quantus",
            MinerKind::Pearl => "pearl",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MinerState {
    Stopped,
    Starting,
    Running,
    Error,
}

/// Snapshot of an engine, mirrored by `MinerStatus` in the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MinerStatus {
    pub kind: MinerKind,
    pub state: MinerState,
    pub hashrate: f64,
    pub cpu_hashrate: f64,
    pub gpu_hashrate: f64,
    pub active_jobs: i64,
    pub workers: i64,
    pub message: Option<String>,
    pub updated_at: String,
}

impl MinerStatus {
    pub fn new(kind: MinerKind) -> Self {
        Self {
            kind,
            state: MinerState::Stopped,
            hashrate: 0.0,
            cpu_hashrate: 0.0,
            gpu_hashrate: 0.0,
            active_jobs: 0,
            workers: 0,
            message: None,
            updated_at: now(),
        }
    }
}

/// Handles plus shared status returned by an engine's `start`.
pub struct Started {
    pub status: Arc<Mutex<MinerStatus>>,
    pub miner: Option<tauri::async_runtime::JoinHandle<()>>,
    pub poller: Option<tauri::async_runtime::JoinHandle<()>>,
}

/// One engine's live state and background tasks.
pub struct EngineSlot {
    pub status: Arc<Mutex<MinerStatus>>,
    pub miner: Option<tauri::async_runtime::JoinHandle<()>>,
    pub poller: Option<tauri::async_runtime::JoinHandle<()>>,
}

impl EngineSlot {
    fn new(kind: MinerKind) -> Self {
        Self {
            status: Arc::new(Mutex::new(MinerStatus::new(kind))),
            miner: None,
            poller: None,
        }
    }

    fn snapshot(&self) -> MinerStatus {
        self.status.lock().map(|s| s.clone()).unwrap_or_else(|_| MinerStatus::new(MinerKind::Pearl))
    }

    fn is_active(&self) -> bool {
        matches!(
            self.snapshot().state,
            MinerState::Starting | MinerState::Running
        )
    }
}

pub struct MinerManager {
    pub quantus: EngineSlot,
    pub pearl: EngineSlot,
}

impl MinerManager {
    pub fn new() -> Self {
        Self {
            quantus: EngineSlot::new(MinerKind::Quantus),
            pearl: EngineSlot::new(MinerKind::Pearl),
        }
    }

    fn slot(&self, kind: MinerKind) -> &EngineSlot {
        match kind {
            MinerKind::Quantus => &self.quantus,
            MinerKind::Pearl => &self.pearl,
        }
    }

    fn slot_mut(&mut self, kind: MinerKind) -> &mut EngineSlot {
        match kind {
            MinerKind::Quantus => &mut self.quantus,
            MinerKind::Pearl => &mut self.pearl,
        }
    }

    pub fn statuses(&self) -> Vec<MinerStatus> {
        vec![self.quantus.snapshot(), self.pearl.snapshot()]
    }
}

impl Default for MinerManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// True when `value` is a bare `host:port` usable by `TcpStream::connect`.
fn is_host_port(value: &str) -> bool {
    match value.rsplit_once(':') {
        Some((host, port)) => !host.is_empty() && port.parse::<u16>().is_ok(),
        None => false,
    }
}

/// Emits a `miner-log` event so the UI console can show engine activity.
pub fn emit_log(app: &AppHandle, kind: MinerKind, level: &str, message: impl Into<String>) {
    let payload = serde_json::json!({
        "kind": kind.as_str(),
        "level": level,
        "message": message.into(),
        "timestamp": now(),
    });
    let _ = app.emit("miner-log", payload);
}

/// Emits the current status of an engine.
pub fn emit_status(app: &AppHandle, status: &MinerStatus) {
    let _ = app.emit("miner-status", status.clone());
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Starts an engine in-process. Shared by `start_miner` and `start_session_miner`.
fn start_engine(
    app: &AppHandle,
    state: &AppState,
    kind: MinerKind,
    config: Option<serde_json::Value>,
) -> Result<MinerStatus, String> {
    let mut manager = state.miners.lock().map_err(|_| "miner state is poisoned")?;

    if manager.slot(kind).is_active() {
        return Ok(manager.slot(kind).snapshot());
    }

    let status = manager.slot(kind).status.clone();
    if let Ok(mut s) = status.lock() {
        s.state = MinerState::Starting;
        s.message = None;
        s.updated_at = now();
    }

    emit_log(app, kind, "info", format!("Starting {} engine…", kind.as_str()));

    let started = match kind {
        MinerKind::Quantus => quantus::start(app, config, status.clone()),
        MinerKind::Pearl => pearl::start(app, config, status.clone()),
    };

    match started {
        Ok(started) => {
            let mut s = started
                .status
                .lock()
                .map_err(|_| "miner state is poisoned".to_string())?;
            s.state = MinerState::Running;
            // Keep whatever the engine set (e.g. "idle — …" for a target-less Pearl).
            if s.message.is_none() {
                s.message = Some(format!("{} engine running", kind.as_str()));
            }
            s.updated_at = now();
            let snapshot = s.clone();
            drop(s);

            let slot = manager.slot_mut(kind);
            slot.status = started.status;
            slot.miner = started.miner;
            slot.poller = started.poller;

            emit_log(app, kind, "info", format!("{} engine started.", kind.as_str()));
            emit_status(app, &snapshot);
            Ok(snapshot)
        }
        Err(error) => {
            if let Ok(mut s) = status.lock() {
                s.state = MinerState::Error;
                s.message = Some(error.clone());
                s.updated_at = now();
            }
            let snapshot = status.lock().map(|s| s.clone()).unwrap_or_else(|_| MinerStatus::new(kind));
            emit_log(app, kind, "error", error.clone());
            emit_status(app, &snapshot);
            Err(error)
        }
    }
}

#[tauri::command]
pub fn start_miner(
    app: AppHandle,
    state: State<'_, AppState>,
    kind: MinerKind,
    config: Option<serde_json::Value>,
) -> Result<MinerStatus, String> {
    start_engine(&app, &state, kind, config)
}

/// A mining session started from the API's rendered command.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionConfig {
    /// The launch command the API rendered (empty when the algorithm uses a structured config).
    pub command: String,
    /// The algorithm configuration the API rendered — the structured form we prefer.
    #[serde(default)]
    pub miner_config: Option<serde_json::Value>,
    #[serde(default)]
    pub coin_code: Option<String>,
    #[serde(default)]
    pub algorithm_code: Option<String>,
    /// The client's configured Stratum endpoint (the proxy), used as a fallback only.
    #[serde(default)]
    pub stratum_endpoint: Option<String>,
    /// Engine configuration from the client's settings, per engine.
    #[serde(default)]
    pub quantus: Option<serde_json::Value>,
    #[serde(default)]
    pub pearl: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStarted {
    pub kind: MinerKind,
    pub params: command::MinerParams,
    pub status: MinerStatus,
}

/// Starts a session from the API's config and drives the matching embedded engine in-process.
/// No child processes are created.
#[tauri::command]
pub fn start_session_miner(
    app: AppHandle,
    state: State<'_, AppState>,
    config: SessionConfig,
) -> Result<SessionStarted, String> {
    // The structured config is the preferred source; the command string is a legacy fallback.
    let from_config = |name: &str| {
        config
            .miner_config
            .as_ref()
            .and_then(|value| value.get(name))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };

    if let Some(miner_config) = &config.miner_config {
        log::info!("miner config from API: {miner_config}");
    }

    let params = command::parse(&config.command)?;
    let kind = command::resolve_kind(
        config.coin_code.as_deref(),
        config.algorithm_code.as_deref(),
        &params.program,
    );

    // The API's endpoint is authoritative (structured config, then the session field). Never fall
    // back to a host mentioned in the command string — that may be a pool URL.
    let host = config
        .stratum_endpoint
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| from_config("endpoint"));

    let wallet = from_config("wallet").or_else(|| params.wallet.clone());
    let worker = from_config("workerId").or_else(|| params.worker.clone());

    match &host {
        Some(host) if !is_host_port(host) => {
            return Err(format!(
                "Stratum endpoint '{host}' is not host:port — the API must return the proxy address."
            ));
        }
        None if kind == MinerKind::Pearl => {
            return Err(
                "The API did not provide a Stratum endpoint for this session \
                 (Mining:PublicStratumEndpoint)."
                    .into(),
            );
        }
        _ => {}
    }

    let mut engine_config = match kind {
        MinerKind::Quantus => config.quantus,
        MinerKind::Pearl => config.pearl,
    };
    if let Some(object) = engine_config.as_mut().and_then(|value| value.as_object_mut()) {
        if let Some(cpu) = params.cpu_workers {
            object.insert("cpuWorkers".into(), serde_json::json!(cpu));
        }
        if let Some(gpu) = params.gpu_devices {
            object.insert("gpuDevices".into(), serde_json::json!(gpu));
        }
        // The stratum target is what lets the engine actually connect.
        if let Some(host) = &host {
            object.insert("stratumHost".into(), serde_json::json!(host));
        }
        if let Some(wallet) = &wallet {
            object.insert("wallet".into(), serde_json::json!(wallet));
        }
        if let Some(worker) = &worker {
            object.insert("worker".into(), serde_json::json!(worker));
        }
    }

    emit_log(&app, kind, "info", format!("Session command: {}", params.describe()));
    // Also to the log file / Backend log so the raw command is easy to inspect.
    log::info!("session command from API: {}", config.command);
    if let Some(host) = &host {
        emit_log(&app, kind, "info", format!("Stratum target: {host}"));
    }
    if let Some(worker) = &params.worker {
        emit_log(&app, kind, "info", format!("Worker: {worker}"));
    }

    let status = start_engine(&app, &state, kind, engine_config)?;

    Ok(SessionStarted {
        kind,
        params,
        status,
    })
}

#[tauri::command]
pub fn stop_miner(app: AppHandle, state: State<'_, AppState>, kind: MinerKind) -> Result<MinerStatus, String> {
    let mut manager = state.miners.lock().map_err(|_| "miner state is poisoned")?;

    match kind {
        MinerKind::Quantus => quantus::stop(&app),
        MinerKind::Pearl => pearl::stop(),
    }

    let slot = manager.slot_mut(kind);
    if let Some(handle) = slot.miner.take() {
        handle.abort();
    }
    if let Some(handle) = slot.poller.take() {
        handle.abort();
    }

    let snapshot = {
        let mut s = slot.status.lock().map_err(|_| "miner state is poisoned".to_string())?;
        s.state = MinerState::Stopped;
        s.hashrate = 0.0;
        s.cpu_hashrate = 0.0;
        s.gpu_hashrate = 0.0;
        s.active_jobs = 0;
        s.workers = 0;
        s.message = None;
        s.updated_at = now();
        s.clone()
    };

    emit_log(&app, kind, "info", format!("{} engine stopped.", kind.as_str()));
    emit_status(&app, &snapshot);
    Ok(snapshot)
}

#[tauri::command]
pub fn get_miner_status(state: State<'_, AppState>) -> Result<Vec<MinerStatus>, String> {
    let manager = state.miners.lock().map_err(|_| "miner state is poisoned")?;
    Ok(manager.statuses())
}
