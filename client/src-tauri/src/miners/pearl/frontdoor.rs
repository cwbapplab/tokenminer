//! Pearl engine — the front door.
//!
//! Reads the engine config, reports which GPU it will mine on, and hands the
//! session to [`crate::stratum`]. The search itself lives in [`super::engine`]
//! (llmjob's CUDA core); the configuration is [`crate::miners::pearl_mining`];
//! the share-target arithmetic and the pre-submit verification live in
//! [`super::target`] and [`super::proof`].
//!
//! There is no CPU search. Everything that decides whether a candidate is a
//! solution happens on the GPU.

use std::sync::{Arc, Mutex};

use serde_json::Value;
use tauri::AppHandle;

use super::super::{MinerStatus, Started};
use crate::miners::pearl_mining::PearlMining;
use crate::stratum::{self, PrlConfig};

/// Reads the optional `mining` block from the engine config, falling back to the default shape.
///
/// `k` and `rank` are **not** settings. The vendored core is compiled for exactly one pair
/// (`PEARL_FOLD_K`/`PEARL_FOLD_RANK`) and `pearl_host_submit` refuses any other with "the fold
/// kernel is built for k=… rank=…", so honouring a stored `k`/`rank` only trades a config
/// problem for a refusal that reads like a hardware fault. A client settings blob written by an
/// older build still carries the retired `4096/256` pair, which is exactly how a working box
/// stops mining. They are re-pinned to the shipped profile here, in the single place the config
/// is read; `m`/`n` remain the client's own workload dimensions and are left alone.
fn mining_settings(config: &Option<Value>) -> PearlMining {
    let mut mining: PearlMining = config
        .as_ref()
        .and_then(|value| value.get("mining"))
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default();

    let core = super::config::PROFILE;
    if mining.k != core.k as usize || mining.rank != core.rank {
        log::info!(
            "pearl: ignoring a configured k={} rank={} — the CUDA core is built for k={} rank={}",
            mining.k,
            mining.rank,
            core.k,
            core.rank
        );
        mining.k = core.k as usize;
        mining.rank = core.rank;
    }

    mining
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

    // The GPU is not probed here. `GpuMiner::open` self-tests the device on the
    // miner thread as its gate — a backend that cannot compute the jackpot or hash
    // correctly must not mine — so a second self-test here would run the kernel
    // twice per start, once on this thread while the caller holds the global miner
    // mutex. The failure still surfaces, through the miner thread's own report.

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

        let (handle, stop) = stratum::spawn(
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
            stop: Some(stop),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `k` and `rank` are the protocol pair the CUDA core is compiled for, so a stored config that
    /// names a different pair must be re-pinned rather than handed to a core that will refuse it.
    /// The stale `4096/256` blob a previous build wrote is the case that took a working box down;
    /// `m`/`n` are the client's own dimensions and must survive untouched.
    #[test]
    fn a_stale_configured_k_and_rank_are_re_pinned_to_the_core() {
        let mining = mining_settings(&Some(json!({
            "mining": { "m": 131072, "n": 131072, "k": 4096, "rank": 256 }
        })));

        let core = super::super::config::PROFILE;
        assert_eq!(mining.k, core.k as usize, "k must be the core's own");
        assert_eq!(mining.rank, core.rank, "rank must be the core's own");
        assert_eq!(mining.m, 131_072, "m is the client's and must be kept");
        assert_eq!(mining.n, 131_072, "n is the client's and must be kept");
    }

    /// An engine config with no `mining` block (the API's current shape) yields the shipped
    /// default, and a config that already names the core pair is left exactly as written.
    #[test]
    fn a_config_without_a_mining_block_uses_the_shipped_default() {
        let default = mining_settings(&None);
        assert_eq!(default.k, PearlMining::default().k);
        assert_eq!(default.rank, PearlMining::default().rank);

        let matching = mining_settings(&Some(json!({
            "mining": { "m": 131072, "n": 262144, "k": 2048, "rank": 128 }
        })));
        assert_eq!(matching.k, 2048);
        assert_eq!(matching.rank, 128);
        assert_eq!(matching.n, 262_144);
    }
}
