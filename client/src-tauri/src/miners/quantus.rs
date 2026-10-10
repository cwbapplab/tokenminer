//! Quantus engine — the `miner-service` workspace crate driven in-process.
//!
//! We call `miner_service::run(ServiceConfig)` on a background task and read live
//! status from the miner's Prometheus exporter (the crate's registry is private,
//! so the HTTP endpoint is the only supported read path). Stopping is done by
//! aborting the task — the upstream service has no stop handle.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use tauri::AppHandle;

use super::{emit_log, emit_status, now, MinerKind, MinerState, MinerStatus, Started};

static EXPORTER_STARTED: AtomicBool = AtomicBool::new(false);

const DEFAULT_CPU_BATCH: u64 = 10_000;
const DEFAULT_GPU_BATCH: u32 = 1_000_000;
const CUDA_GPU_BATCH: u32 = 32_000_000;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantusConfig {
    pub node_addr: String,
    pub auth_token_file: String,
    pub tls_cert_sha256_file: String,
    #[serde(default)]
    pub cpu_workers: u32,
    #[serde(default)]
    pub gpu_devices: u32,
    #[serde(default)]
    pub cuda_gpu: bool,
    #[serde(default)]
    pub opencl_gpu: bool,
    #[serde(default)]
    pub allow_integrated: bool,
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
}

fn default_metrics_port() -> u16 {
    9900
}

fn read_trimmed(path: &str) -> Result<String, String> {
    if path.trim().is_empty() {
        return Err("No auth token file configured.".into());
    }
    std::fs::read_to_string(path)
        .map(|contents| contents.trim().to_string())
        .map_err(|error| format!("Failed to read {path}: {error}"))
}

pub fn start(
    app: &AppHandle,
    config: Option<serde_json::Value>,
    status: Arc<Mutex<MinerStatus>>,
) -> Result<Started, String> {
    let config: QuantusConfig = match config {
        Some(value) => {
            serde_json::from_value(value).map_err(|e| format!("Invalid Quantus config: {e}"))?
        }
        None => {
            return Err("Quantus requires a node address, auth token file and TLS pin file.".into())
        }
    };

    let auth_token = read_trimmed(&config.auth_token_file)?;
    let tls_cert_sha256 = read_trimmed(&config.tls_cert_sha256_file)?;

    // Fail closed on a malformed pin/token before spawning any workers.
    quic_transport::validate_auth_config(&auth_token, &tls_cert_sha256)
        .map_err(|e| e.to_string())?;

    let node_addr: SocketAddr = config
        .node_addr
        .parse()
        .map_err(|e| format!("Invalid node address '{}': {e}", config.node_addr))?;

    let port = config.metrics_port;

    if !EXPORTER_STARTED.swap(true, Ordering::SeqCst) {
        tauri::async_runtime::spawn(async move {
            if let Err(error) = miner_metrics::start_http_exporter(port).await {
                log::error!("failed to start miner metrics exporter: {error}");
                // The latch is set before the bind, so a failed bind has to
                // clear it — otherwise a restart after a port conflict finds
                // the one-shot already spent and leaves telemetry dead for the
                // rest of the process.
                EXPORTER_STARTED.store(false, Ordering::SeqCst);
            }
        });
    }

    let gpu_batch_size = if config.cuda_gpu {
        CUDA_GPU_BATCH
    } else {
        DEFAULT_GPU_BATCH
    };
    let service = miner_service::ServiceConfig {
        node_addr,
        auth_token,
        tls_cert_sha256,
        cpu_workers: (config.cpu_workers > 0).then_some(config.cpu_workers as usize),
        gpu_devices: (config.gpu_devices > 0).then_some(config.gpu_devices as usize),
        gpu_batch_size,
        cpu_batch_size: DEFAULT_CPU_BATCH,
        gpu_throttle_ms: 0,
        allow_integrated: config.allow_integrated,
        cuda_gpu: config.cuda_gpu,
        opencl_gpu: config.opencl_gpu,
    };

    let miner_status = status.clone();
    let miner_app = app.clone();
    let miner = tauri::async_runtime::spawn(async move {
        if let Err(error) = miner_service::run(service).await {
            log::error!("quantus miner stopped: {error}");
            emit_log(
                &miner_app,
                MinerKind::Quantus,
                "error",
                format!("Miner stopped: {error}"),
            );
            if let Ok(mut s) = miner_status.lock() {
                s.state = MinerState::Error;
                s.message = Some(error.to_string());
                s.updated_at = now();
            }
            if let Ok(snapshot) = miner_status.lock().map(|s| s.clone()) {
                emit_status(&miner_app, &snapshot);
            }
        }
    });

    let poll_app = app.clone();
    let poll_status = status.clone();
    let poller = tauri::async_runtime::spawn(async move {
        poll_metrics(poll_app, port, poll_status).await;
    });

    Ok(Started {
        status,
        miner: Some(miner),
        poller: Some(poller),
        // Quantus has no flag to set: its workers cancel when their pool is
        // dropped (see `WorkerPool`'s `Drop` in the service crate), which
        // aborting the task below does.
        stop: None,
    })
}

pub fn stop(_app: &AppHandle) {
    // The engine task is aborted by the manager; nothing else to do here.
}

/// Polls `GET /metrics` and mirrors the gauges into the shared status.
async fn poll_metrics(app: AppHandle, port: u16, status: Arc<Mutex<MinerStatus>>) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            log::error!("failed to build metrics client: {error}");
            return;
        }
    };
    let url = format!("http://127.0.0.1:{port}/metrics");

    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;

        let Ok(response) = client.get(&url).send().await else {
            continue;
        };
        let Ok(body) = response.text().await else {
            continue;
        };

        let metrics = parse_prometheus(&body);
        let snapshot = {
            let Ok(mut current) = status.lock() else {
                return;
            };
            current.hashrate = metric(&metrics, "miner_hash_rate");
            current.cpu_hashrate = metric(&metrics, "miner_cpu_hash_rate");
            current.gpu_hashrate = metric(&metrics, "miner_gpu_hash_rate");
            current.active_jobs = metric(&metrics, "miner_active_jobs") as i64;
            current.workers = metric(&metrics, "miner_workers") as i64;
            current.updated_at = now();
            current.clone()
        };

        emit_status(&app, &snapshot);
    }
}

fn metric(metrics: &HashMap<String, f64>, name: &str) -> f64 {
    metrics.get(name).copied().unwrap_or(0.0)
}

/// Parses the Prometheus text exposition format into `name -> value` (gauges/counters).
fn parse_prometheus(body: &str) -> HashMap<String, f64> {
    let mut out = HashMap::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();
        let Some(raw_name) = parts.next() else {
            continue;
        };
        let Some(raw_value) = parts.next() else {
            continue;
        };

        // Drop any label set, e.g. `miner_hash_rate{engine="cpu"}`.
        let name = raw_name.split('{').next().unwrap_or(raw_name);
        if let Ok(value) = raw_value.parse::<f64>() {
            out.insert(name.to_string(), value);
        }
    }

    out
}
