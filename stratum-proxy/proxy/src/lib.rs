//! TokenMiner stratum proxy.
//!
//! Miners reach this service through nginx. For each connection the proxy identifies the worker,
//! opens a connection to that worker's pool and relays stratum traffic verbatim in both
//! directions. The only thing it *acts* on is an accepted share: at that point it reports the
//! share to the API with a signed request. Rejected shares are relayed and forgotten.

pub mod cache;
pub mod config;
pub mod relay;
pub mod session;
pub mod share;

use std::sync::Arc;

use anyhow::Context;
use sp_api_client::{ApiClient, StratumConfig, StratumWorker};
use tokio::net::TcpListener;
use tracing::{info, warn};

use crate::cache::TtlCache;
use crate::config::Config;

/// Single-slot key for the cached routing table.
const ROUTING_TABLE_KEY: &str = "current";

/// Shared state handed to every session.
pub struct AppContext {
    pub api: ApiClient,
    pub config: Config,

    routing_table: TtlCache<String, Arc<StratumConfig>>,
    workers: TtlCache<String, Option<Arc<StratumWorker>>>,
}

impl AppContext {
    pub fn new(config: Config) -> Self {
        let api = ApiClient::new(
            &config.api_base_url,
            &config.service_id,
            &config.service_secret,
        );

        Self {
            routing_table: TtlCache::new(config.config_cache_ttl),
            workers: TtlCache::new(config.worker_cache_ttl),
            config,
            api,
        }
    }

    /// Cached routing table, refetched once its TTL lapses. A change to any pool configuration
    /// changes the returned `version`, so a stale copy cannot outlive its TTL.
    pub async fn routing_table(&self) -> anyhow::Result<Arc<StratumConfig>> {
        if let Some(cached) = self.routing_table.get(&ROUTING_TABLE_KEY.to_string()).await {
            return Ok(cached);
        }

        let config = Arc::new(
            self.api
                .stratum_config()
                .await
                .context("fetching the stratum config")?,
        );

        self.routing_table
            .insert(ROUTING_TABLE_KEY.to_string(), Arc::clone(&config))
            .await;

        Ok(config)
    }

    /// Cached worker resolution. Negative results are cached too, so an unknown worker cannot
    /// use the proxy to hammer the API.
    pub async fn resolve_worker(
        &self,
        worker_id: &str,
    ) -> anyhow::Result<Option<Arc<StratumWorker>>> {
        if let Some(cached) = self.workers.get(&worker_id.to_string()).await {
            return Ok(cached);
        }

        let resolved = self.api.resolve_worker(worker_id).await?.map(Arc::new);

        self.workers
            .insert(worker_id.to_string(), resolved.clone())
            .await;

        Ok(resolved)
    }
}

/// Accepts connections until the listener fails.
pub async fn serve(listener: TcpListener, context: Arc<AppContext>) -> anyhow::Result<()> {
    let local = listener.local_addr().context("reading the listener address")?;
    info!(%local, "stratum proxy listening");

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(error) => {
                warn!(%error, "accepting a connection failed");
                continue;
            }
        };

        let context = Arc::clone(&context);

        tokio::spawn(async move {
            if let Err(error) = session::handle(stream, context).await {
                warn!(%peer, error = %error, "session ended");
            }
        });
    }
}
