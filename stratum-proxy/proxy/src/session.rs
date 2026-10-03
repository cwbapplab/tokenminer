use std::sync::Arc;

use anyhow::{anyhow, Context};
use serde_json::Value;
use sp_protocol::{self as protocol, Message, METHOD_AUTHORIZE};
use tokio::io::{AsyncWrite, BufReader, BufWriter};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::relay::{self, PendingSubmits};
use crate::share::ShareContext;
use crate::AppContext;

/// Serves one miner connection: read the identity from the username, pick the pool and coin from
/// the in-memory routing table, then relay. No API call is made before the pool connection — the
/// worker is attributed later, when an accepted share is reported.
pub async fn handle(stream: TcpStream, context: Arc<AppContext>) -> anyhow::Result<()> {
    let _ = stream.set_nodelay(true);

    let peer = stream
        .peer_addr()
        .map(|address| address.to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let (client_read, client_write) = stream.into_split();
    let mut client_reader = BufReader::new(client_read);
    let mut client_writer = BufWriter::new(client_write);

    let mut buffered: Vec<String> = Vec::new();
    let mut route: Option<Route> = None;

    let deadline = tokio::time::sleep(context.config.authorize_timeout);
    tokio::pin!(deadline);

    while route.is_none() {
        let line = tokio::select! {
            _ = &mut deadline => return Err(anyhow!("timed out waiting for mining.authorize")),
            line = protocol::read_line(&mut client_reader) => line.context("reading from the miner")?,
        };

        // A clean close before authorizing is a normal, uninteresting disconnect.
        let Some(line) = line else {
            return Ok(());
        };

        debug!(%peer, "miner -> {line}");

        if let Some(identity) = authorized_identity(&line) {
            match resolve_route(&context, &identity, &line, &mut client_writer).await? {
                Some(resolved) => route = Some(resolved),
                None => return Ok(()),
            }
        }

        buffered.push(line);
    }

    let route = route.expect("the loop only exits once a route is resolved");

    let pool = TcpStream::connect(&route.endpoint)
        .await
        .with_context(|| format!("connecting to the pool at {}", route.endpoint))?;
    let _ = pool.set_nodelay(true);

    info!(worker = %route.share.worker_identifier, pool = %route.endpoint, "worker authorised");

    let (pool_read, pool_write) = pool.into_split();
    let pool_reader = BufReader::new(pool_read);
    let mut pool_writer = BufWriter::new(pool_write);

    // Replay whatever the miner sent while we were deciding where to route it.
    for line in &buffered {
        protocol::write_line(&mut pool_writer, line).await?;
    }

    let pending: PendingSubmits = PendingSubmits::default();
    let difficulty = Arc::new(Mutex::new(None::<f64>));
    let share = Arc::new(route.share);

    tokio::try_join!(
        relay::pump_up(client_reader, pool_writer, Arc::clone(&pending), Arc::clone(&share)),
        relay::pump_down(pool_reader, client_writer, pending, difficulty, Arc::clone(&context)),
    )?;

    Ok(())
}

/// Where a connection goes and how its shares are attributed, resolved from the username alone.
struct Route {
    endpoint: String,
    share: ShareContext,
}

/// Validates the username against the in-memory configuration and returns the route. `Ok(None)`
/// means the connection was refused and answered on the wire.
async fn resolve_route<W>(
    context: &AppContext,
    identity: &protocol::AuthorizeIdentity,
    line: &str,
    writer: &mut BufWriter<W>,
) -> anyhow::Result<Option<Route>>
where
    W: AsyncWrite + Unpin,
{
    // Gate 1 (in memory): the wallet must belong to an active pool.
    let policy = context
        .wallet_policy()
        .await
        .context("loading the wallet policy")?;

    if !policy.is_empty() && !policy.knows(&identity.wallet) {
        warn!(wallet = %identity.wallet, "refusing an unknown wallet");
        refuse(writer, line, "unauthorised wallet").await;
        return Ok(None);
    }

    // The pool is named in the username, or implied by a wallet only one pool pays to.
    let routing = context
        .routing_table()
        .await
        .context("loading the routing table")?;

    let pool = match identity.pool.as_deref() {
        // A named pool must exist; a mistyped one is refused rather than silently re-routed.
        Some(system_id) => routing
            .pools
            .iter()
            .find(|pool| pool.system_pool_id == system_id),
        // No pool segment: fall back to the wallet, which is unique to a pool.
        None => routing
            .pools
            .iter()
            .find(|pool| pool.payout_address.as_deref() == Some(identity.wallet.as_str())),
    };

    let Some(pool) = pool else {
        warn!(
            pool = identity.pool.as_deref().unwrap_or("<none>"),
            wallet = %identity.wallet,
            "refusing a username that names no known pool"
        );
        refuse(writer, line, "unknown pool").await;
        return Ok(None);
    };

    let Some(coin) = pool.coins.first() else {
        warn!(pool = %pool.system_pool_id, "refusing a pool with no coin");
        refuse(writer, line, "pool has no coin").await;
        return Ok(None);
    };

    // The worker id is the device's hardware id in "N" form, opaque to the proxy; the API resolves
    // it to the device when the share is reported.
    Ok(Some(Route {
        endpoint: pool.stratum_endpoint.clone(),
        share: ShareContext {
            worker_identifier: identity.worker.clone(),
            pool_id: pool.pool_id.clone(),
            coin_id: coin.coin_id.clone(),
        },
    }))
}

/// Answers an authorize we are refusing, mirroring the request id.
async fn refuse<W>(writer: &mut BufWriter<W>, line: &str, reason: &str)
where
    W: AsyncWrite + Unpin,
{
    let reply = protocol::error_response(&request_id(line), 24, reason);
    let _ = protocol::write_line(writer, &reply).await;
}

/// The identity from a `mining.authorize` line, if this line is one. Handles the classic
/// positional form (`["wallet.worker", "x"]`) and Pearl's named form (`{"wallet": ..., "worker": ...}`).
fn authorized_identity(line: &str) -> Option<protocol::AuthorizeIdentity> {
    let message = Message::parse(line).ok()?;
    let request = message.as_request()?;

    if request.method != METHOD_AUTHORIZE {
        return None;
    }

    protocol::parse_authorize(&request.params)
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

    const WORKER: &str =
        "11111111-1111-1111-1111-111111111111-22222222-2222-2222-2222-222222222222";

    #[test]
    fn reads_the_identity_from_a_classic_authorize() {
        let line = format!(
            r#"{{"id":2,"method":"mining.authorize","params":["krxYR9NJVQ.kryptex-prl.{WORKER}","x"]}}"#
        );

        let identity = authorized_identity(&line).expect("an identity");

        assert_eq!(identity.wallet, "krxYR9NJVQ");
        assert_eq!(identity.pool.as_deref(), Some("kryptex-prl"));
        assert_eq!(identity.worker, WORKER);
    }

    #[test]
    fn reads_the_identity_from_a_pearl_named_authorize() {
        let line = format!(
            r#"{{"id":1,"method":"mining.authorize","params":{{"wallet":"krxYR9NJVQ.kryptex-prl.{WORKER}","worker":""}}}}"#
        );

        let identity = authorized_identity(&line).expect("an identity");

        assert_eq!(identity.wallet, "krxYR9NJVQ");
        assert_eq!(identity.pool.as_deref(), Some("kryptex-prl"));
        assert_eq!(identity.worker, WORKER);
    }

    #[test]
    fn ignores_other_methods() {
        let subscribe = r#"{"id":1,"method":"mining.subscribe","params":["cgminer/4.0"]}"#;
        let submit = r#"{"id":3,"method":"mining.submit","params":["w","job","aa","bb","cc"]}"#;

        assert!(authorized_identity(subscribe).is_none());
        assert!(authorized_identity(submit).is_none());
    }

    #[test]
    fn ignores_malformed_lines() {
        assert!(authorized_identity("not json").is_none());
        assert!(authorized_identity(r#"{"method":"mining.authorize"}"#).is_none());
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
