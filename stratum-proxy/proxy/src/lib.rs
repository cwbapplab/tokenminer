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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Context;
use sp_api_client::{ApiClient, StratumConfig};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cache::TtlCache;
use crate::config::Config;

/// Single-slot key for the cached routing table.
const ROUTING_TABLE_KEY: &str = "current";

/// The wallets a miner is allowed to use, derived from the active pools' configuration.
///
/// Loaded at startup and refreshed with the routing table, it lets the authorize gate be answered
/// from memory instead of a round trip: a wallet no active pool pays out to is refused outright.
#[derive(Debug, Default)]
pub struct WalletPolicy {
    known: HashSet<String>,
    by_coin: HashMap<String, String>,
}

impl WalletPolicy {
    /// Collects every active pool's payout wallet and maps its coins to that wallet.
    pub fn from_config(config: &StratumConfig) -> Self {
        let mut known = HashSet::new();
        let mut by_coin = HashMap::new();

        for pool in &config.pools {
            let Some(wallet) = pool
                .payout_address
                .as_deref()
                .filter(|value| !value.is_empty())
            else {
                continue;
            };

            known.insert(wallet.to_string());

            for coin in &pool.coins {
                by_coin.insert(coin.code.to_ascii_lowercase(), wallet.to_string());
            }
        }

        Self { known, by_coin }
    }

    pub fn is_empty(&self) -> bool {
        self.known.is_empty()
    }

    pub fn wallet_count(&self) -> usize {
        self.known.len()
    }

    /// Whether the address is a wallet some active pool pays out to.
    pub fn knows(&self, wallet: &str) -> bool {
        self.known.contains(wallet)
    }

    /// The wallet the active pool for this coin pays out to.
    pub fn wallet_for_coin(&self, coin_code: &str) -> Option<&str> {
        self.by_coin
            .get(&coin_code.to_ascii_lowercase())
            .map(String::as_str)
    }
}

/// Shared state handed to every session.
pub struct AppContext {
    pub api: ApiClient,
    pub config: Config,

    routing_table: TtlCache<String, Arc<StratumConfig>>,
    wallets: RwLock<Arc<WalletPolicy>>,
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
            wallets: RwLock::new(Arc::new(WalletPolicy::default())),
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

        *self.wallets.write().await = Arc::new(WalletPolicy::from_config(&config));

        self.routing_table
            .insert(ROUTING_TABLE_KEY.to_string(), Arc::clone(&config))
            .await;

        Ok(config)
    }

    /// The in-memory wallet policy, refreshed alongside the routing table. The first call loads it.
    pub async fn wallet_policy(&self) -> anyhow::Result<Arc<WalletPolicy>> {
        let _ = self.routing_table().await?;

        let wallets = self.wallets.read().await;

        Ok(Arc::clone(&wallets))
    }
}

/// Accepts connections until the listener fails.
pub async fn serve(listener: TcpListener, context: Arc<AppContext>) -> anyhow::Result<()> {
    let local = listener.local_addr().context("reading the listener address")?;

    // Load the wallet policy up front so the authorize gate is answered from memory.
    match context.wallet_policy().await {
        Ok(policy) => info!(wallets = policy.wallet_count(), "loaded the wallet policy"),
        Err(error) => {
            warn!(%error, "could not load the wallet policy at startup; it will be retried per authorize")
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use sp_api_client::{StratumCoin, StratumPool};

    fn pool(system_pool_id: &str, wallet: Option<&str>, coins: &[&str]) -> StratumPool {
        StratumPool {
            pool_id: format!("{system_pool_id}-id"),
            system_pool_id: system_pool_id.to_string(),
            name: "pool".to_string(),
            stratum_endpoint: "pool.example.com:3333".to_string(),
            payout_address: wallet.map(str::to_string),
            coins: coins
                .iter()
                .map(|code| StratumCoin {
                    coin_id: format!("{code}-id"),
                    code: code.to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn policy_collects_wallets_and_maps_them_to_coins() {
        let config = StratumConfig {
            version: "1".to_string(),
            pools: vec![
                pool("kryptex-prl", Some("wallet-a"), &["PRL"]),
                pool("kryptex-qtc", Some("wallet-b"), &["QTC"]),
                pool("no-wallet", None, &["USDT"]),
            ],
        };

        let policy = WalletPolicy::from_config(&config);

        assert_eq!(policy.wallet_count(), 2);
        assert!(policy.knows("wallet-a"));
        assert!(policy.knows("wallet-b"));
        assert!(!policy.knows("wallet-c"));
        assert_eq!(policy.wallet_for_coin("prl"), Some("wallet-a"));
        assert_eq!(policy.wallet_for_coin("QTC"), Some("wallet-b"));
        assert_eq!(policy.wallet_for_coin("usdt"), None);
    }

    #[test]
    fn an_empty_config_yields_an_empty_policy() {
        let policy = WalletPolicy::from_config(&StratumConfig {
            version: "0".to_string(),
            pools: vec![],
        });

        assert!(policy.is_empty());
    }
}
