use std::sync::Arc;

use anyhow::{anyhow, Context};
use serde_json::Value;
use sp_api_client::StratumWorker;
use sp_protocol::{self as protocol, Message, METHOD_AUTHORIZE};
use tokio::io::{BufReader, BufWriter};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::relay::{self, PendingSubmits};
use crate::AppContext;

/// Serves one miner connection: identify the worker, connect to its pool, then relay.
pub async fn handle(stream: TcpStream, context: Arc<AppContext>) -> anyhow::Result<()> {
    let _ = stream.set_nodelay(true);

    let (client_read, client_write) = stream.into_split();
    let mut client_reader = BufReader::new(client_read);
    let mut client_writer = BufWriter::new(client_write);

    let mut buffered: Vec<String> = Vec::new();
    let mut worker: Option<Arc<StratumWorker>> = None;

    let deadline = tokio::time::sleep(context.config.authorize_timeout);
    tokio::pin!(deadline);

    while worker.is_none() {
        let line = tokio::select! {
            _ = &mut deadline => return Err(anyhow!("timed out waiting for mining.authorize")),
            line = protocol::read_line(&mut client_reader) => line.context("reading from the miner")?,
        };

        // A clean close before authorizing is a normal, uninteresting disconnect.
        let Some(line) = line else {
            return Ok(());
        };

        if let Some(worker_id) = authorized_worker_id(&line) {
            match context.resolve_worker(&worker_id).await {
                Ok(Some(resolved)) => worker = Some(resolved),
                Ok(None) => {
                    // No active session for this worker: refuse rather than proxy blindly.
                    warn!(%worker_id, "refusing an unauthorised worker");
                    let reply = protocol::error_response(&request_id(&line), 24, "unauthorised worker");
                    let _ = protocol::write_line(&mut client_writer, &reply).await;
                    return Ok(());
                }
                Err(error) => return Err(error.context("resolving the worker")),
            }
        }

        buffered.push(line);
    }

    let worker = worker.expect("the loop only exits once the worker is resolved");

    // The routing table is authoritative for the live endpoint; the resolution is authoritative
    // for identity.
    let endpoint = context
        .routing_table()
        .await
        .ok()
        .and_then(|table| {
            table
                .pools
                .iter()
                .find(|pool| pool.pool_id == worker.pool_id)
                .map(|pool| pool.stratum_endpoint.clone())
        })
        .unwrap_or_else(|| worker.stratum_endpoint.clone());

    let pool = TcpStream::connect(&endpoint)
        .await
        .with_context(|| format!("connecting to the pool at {endpoint}"))?;
    let _ = pool.set_nodelay(true);

    info!(worker = %worker.worker_id, pool = %endpoint, "worker authorised");

    let (pool_read, pool_write) = pool.into_split();
    let pool_reader = BufReader::new(pool_read);
    let mut pool_writer = BufWriter::new(pool_write);

    // Replay whatever the miner sent while we were resolving.
    for line in &buffered {
        protocol::write_line(&mut pool_writer, line).await?;
    }

    let pending: PendingSubmits = PendingSubmits::default();
    let difficulty = Arc::new(Mutex::new(None::<f64>));

    tokio::try_join!(
        relay::pump_up(client_reader, pool_writer, Arc::clone(&pending), Arc::clone(&worker)),
        relay::pump_down(pool_reader, client_writer, pending, difficulty, Arc::clone(&context)),
    )?;

    Ok(())
}

/// The worker identity from a `mining.authorize` line, if this line is one.
fn authorized_worker_id(line: &str) -> Option<String> {
    let message = Message::parse(line).ok()?;
    let request = message.as_request()?;

    if request.method != METHOD_AUTHORIZE {
        return None;
    }

    let username = request.params.get(0)?.as_str()?;

    Some(protocol::worker_id_from_username(username).to_string())
}

fn request_id(line: &str) -> Value {
    Message::parse(line)
        .ok()
        .and_then(|message| message.as_request().map(|request| request.id.clone()))
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_the_worker_from_an_authorize_request() {
        let line = r#"{"id":2,"method":"mining.authorize","params":["krxYR9NJVQ.user-1","x"]}"#;

        assert_eq!(authorized_worker_id(line).as_deref(), Some("user-1"));
    }

    #[test]
    fn ignores_other_methods() {
        let subscribe = r#"{"id":1,"method":"mining.subscribe","params":["cgminer/4.0"]}"#;
        let submit = r#"{"id":3,"method":"mining.submit","params":["w","job","aa","bb","cc"]}"#;

        assert!(authorized_worker_id(subscribe).is_none());
        assert!(authorized_worker_id(submit).is_none());
    }

    #[test]
    fn ignores_malformed_lines() {
        assert!(authorized_worker_id("not json").is_none());
        assert!(authorized_worker_id(r#"{"method":"mining.authorize"}"#).is_none());
    }

    #[test]
    fn reads_the_request_id_for_error_replies() {
        let line = r#"{"id":42,"method":"mining.authorize","params":["w.h"]}"#;

        assert_eq!(request_id(line), json!(42));
    }

    #[test]
    fn falls_back_to_a_null_id() {
        assert_eq!(request_id("not json"), json!(null));
    }
}
