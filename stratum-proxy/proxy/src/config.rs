use std::time::Duration;

/// Runtime configuration, entirely from the environment.
#[derive(Debug, Clone)]
pub struct Config {
    /// Address miners (via nginx) connect to.
    pub listen_addr: String,

    /// Base URL of the TokenMiner API.
    pub api_base_url: String,

    /// Service identity presented to the API and used for replay protection.
    pub service_id: String,

    /// Shared secret used to sign internal API requests.
    pub service_secret: String,

    /// How long the routing table is reused before it is refetched.
    pub config_cache_ttl: Duration,

    /// How long a worker resolution is reused, including negative results.
    pub worker_cache_ttl: Duration,

    /// How long a client may stay connected without authorizing.
    pub authorize_timeout: Duration,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let service_secret = required("TOKENMINER_SERVICE_SECRET")?;

        Ok(Self {
            listen_addr: optional("STRATUM_LISTEN_ADDR").unwrap_or_else(|| "0.0.0.0:3333".to_string()),
            api_base_url: required("TOKENMINER_API_BASE_URL")?,
            service_id: optional("TOKENMINER_SERVICE_ID").unwrap_or_else(|| "stratum-proxy".to_string()),
            service_secret,
            config_cache_ttl: Duration::from_secs(int_env("CONFIG_CACHE_TTL_SECONDS", 60)),
            worker_cache_ttl: Duration::from_secs(int_env("WORKER_CACHE_TTL_SECONDS", 30)),
            authorize_timeout: Duration::from_secs(int_env("AUTHORIZE_TIMEOUT_SECONDS", 30)),
        })
    }
}

fn optional(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.trim().is_empty())
}

fn required(key: &str) -> anyhow::Result<String> {
    optional(key).ok_or_else(|| anyhow::anyhow!("{key} must be set"))
}

fn int_env(key: &str, default: u64) -> u64 {
    optional(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
