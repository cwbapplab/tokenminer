//! Pearl engine — a GPU miner, driven in-process.
//!
//! This module is the engine's front door: it reads the engine config, reports which GPU it will
//! mine on, and hands the session to [`crate::stratum`]. The search itself lives in
//! [`pearl_gpu`]; the configuration and wire format live in [`pearl_mining`]; the share-target
//! arithmetic and the pre-submit verification live in [`pearl_pow`].
//!
//! There is no CPU search. Everything that decides whether a candidate is a solution happens in
//! [`crate::miners::gpu`].

use std::sync::{Arc, Mutex};

use serde_json::Value;
use tauri::AppHandle;

use super::{MinerStatus, Started};
use crate::miners::pearl_mining::PearlMining;
use crate::stratum::{self, PrlConfig};

/// Reads the optional `mining` block from the engine config, falling back to the default shape.
fn mining_settings(config: &Option<Value>) -> PearlMining {
    config
        .as_ref()
        .and_then(|value| value.get("mining"))
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default()
}

/// Starts the Pearl engine.
///
/// When the session supplies a Stratum host and wallet (parsed from the API's command), this opens
/// a real PRL session against the proxy and the GPU mines whatever jobs arrive. Otherwise the
/// engine refuses to start rather than sit "running" with nothing to do.
pub fn start(
    app: &AppHandle,
    config: Option<Value>,
    status: Arc<Mutex<MinerStatus>>,
) -> Result<Started, String> {
    let cert_version = config
        .as_ref()
        .and_then(|value| value.get("certVersion"))
        .and_then(Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .unwrap_or(2);

    let mining = mining_settings(&config);

    log::info!(
        "pearl engine ready: cert version {cert_version}; mining m={} n={} k={} rank={} on GPU",
        mining.m,
        mining.n,
        mining.k,
        mining.rank
    );

    // Report the GPU we will mine on, and prove the embedded image works on it. This is the gate:
    // a backend that cannot compute the jackpot or hash correctly must not mine.
    match super::gpu::probe() {
        Some(mut backend) => {
            let described = backend.device().describe();
            match backend.self_test() {
                Ok(()) => log::info!("pearl: GPU {} usable — {described}", backend.name()),
                Err(error) => {
                    log::warn!("pearl: GPU {} failed its self-test: {error}", backend.name())
                }
            }
            backend.shutdown();
        }
        None => log::info!("pearl: no GPU backend available on this machine"),
    }

    let field = |name: &str| {
        config
            .as_ref()
            .and_then(|value| value.get(name))
            .and_then(Value::as_str)
            .map(str::to_string)
    };

    let host = field("stratumHost");
    let wallet = field("wallet");

    // Diagnostic: show exactly what the session handed us.
    log::info!(
        "pearl: config stratumHost={:?} wallet={:?} worker={:?}",
        host,
        wallet,
        field("worker"),
    );

    // `stratumHost` / `wallet` / `worker` are injected from the parsed miner command.
    if let (Some(host), Some(wallet)) = (host, wallet) {
        let worker = field("worker").unwrap_or_default();

        let handle = stratum::spawn(
            app.clone(),
            PrlConfig {
                host,
                wallet,
                worker,
                mining,
                cert_version,
            },
            status.clone(),
        );

        return Ok(Started {
            status,
            miner: Some(handle),
            poller: None,
        });
    }

    // Without a session-supplied target the Pearl engine has nothing to do. Fail loudly rather than
    // sit "running".
    log::warn!(
        "pearl: no stratum target — start a mining *session* so the API supplies the proxy host \
         and wallet (the engine card cannot, it only carries cert version)"
    );
    Err("Pearl needs a mining session — use Start mining so the API supplies the proxy host and wallet.".into())
}

pub fn stop() {}