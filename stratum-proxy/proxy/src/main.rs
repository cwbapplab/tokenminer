use std::sync::Arc;

use anyhow::Context;
use sp_proxy::{config::Config, serve, AppContext};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env()?;

    let listen_addr = config.listen_addr.clone();
    let context = Arc::new(AppContext::new(config));

    let listener = TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("binding {listen_addr}"))?;

    serve(listener, context).await
}
