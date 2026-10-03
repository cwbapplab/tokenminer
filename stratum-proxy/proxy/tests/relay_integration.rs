//! End-to-end relay behaviour: a fake miner talks to the proxy, which fronts a mock pool and a
//! mock TokenMiner API.
//!
//! The proxy routes purely from the username and the in-memory routing table (no API call before
//! dialing the pool). The one thing it *acts* on is an accepted share, which it reports with a
//! correctly signed request; a rejected share is never reported.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use sp_api_client::{
    body_hash, signature, NONCE_HEADER, SERVICE_ID_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER,
};
use sp_proxy::{config::Config, serve, AppContext};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout, Instant};

const SECRET: &str = "integration-secret";
const WALLET: &str = "krxYR9NJVQ";
const SYSTEM_POOL_ID: &str = "mock-pool";

// The pool-facing worker id: the device's hardware id in "N" form (32 chars).
const WORKER: &str = "7ed82f7514bd4b1397183061e5e25a68";

fn username() -> String {
    format!("{WALLET}.{SYSTEM_POOL_ID}.{WORKER}")
}

#[derive(Clone, Default)]
struct ApiState {
    /// Every share payload the proxy reported.
    shares: Arc<Mutex<Vec<Value>>>,
    /// Signatures that did not verify, so a signing regression fails loudly.
    rejected_signatures: Arc<Mutex<Vec<String>>>,
    /// The pool endpoint the routing table advertises.
    pool_endpoint: Arc<Mutex<String>>,
}

impl ApiState {
    fn recorded_shares(&self) -> Vec<Value> {
        self.shares.lock().unwrap().clone()
    }
}

async fn stratum_config(State(state): State<ApiState>) -> Json<Value> {
    let endpoint = state.pool_endpoint.lock().unwrap().clone();

    Json(json!({
        "version": "1",
        "pools": [{
            "poolId": "pool-1",
            "systemPoolId": SYSTEM_POOL_ID,
            "name": "Mock pool",
            "stratumEndpoint": endpoint,
            "payoutAddress": WALLET,
            "coins": [{ "coinId": "coin-1", "code": "qtc" }]
        }]
    }))
}

async fn record_share(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: String,
) -> Json<Value> {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string()
    };

    let expected = signature(
        SECRET,
        "POST",
        "/internal/mining/shares",
        &header(TIMESTAMP_HEADER),
        &header(NONCE_HEADER),
        &body_hash(body.as_bytes()),
    );

    if expected != header(SIGNATURE_HEADER) || header(SERVICE_ID_HEADER) != "stratum-proxy" {
        state.rejected_signatures.lock().unwrap().push(body.clone());
    }

    let parsed: Value = serde_json::from_str(&body).expect("the proxy posts valid JSON");
    state.shares.lock().unwrap().push(parsed);

    Json(json!({
        "id": "share-1",
        "shareIdentifier": "share-1",
        "rewardStatus": "pending",
        "coinValue": null,
        "approxUsdValueAtTime": null,
        "alreadyRecorded": false,
        "createdAt": "2026-10-02T19:00:00Z"
    }))
}

/// A pool that answers every submission the same way.
async fn run_mock_pool(listener: TcpListener, accept_submissions: bool) {
    let (stream, _) = listener.accept().await.expect("a pool connection");
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();

    loop {
        line.clear();

        if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
            break;
        }

        let Ok(request) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };

        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let method = request.get("method").and_then(Value::as_str).unwrap_or_default();

        let response = match method {
            "mining.submit" if !accept_submissions => {
                json!({ "id": id, "result": null, "error": [23, "low difficulty share", null] })
            }
            _ => json!({ "id": id, "result": true, "error": null }),
        };

        let mut encoded = response.to_string();
        encoded.push('\n');

        if write_half.write_all(encoded.as_bytes()).await.is_err() {
            break;
        }

        let _ = write_half.flush().await;
    }
}

struct Stack {
    proxy_addr: std::net::SocketAddr,
    state: ApiState,
}

async fn spawn_stack(accept_submissions: bool) -> Stack {
    // Mock pool.
    let pool_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let pool_addr = pool_listener.local_addr().unwrap();

    // Mock API.
    let state = ApiState::default();
    *state.pool_endpoint.lock().unwrap() = pool_addr.to_string();

    let router = Router::new()
        .route("/internal/stratum/config", get(stratum_config))
        .route("/internal/mining/shares", post(record_share))
        .with_state(state.clone());

    let api_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_addr = api_listener.local_addr().unwrap();

    tokio::spawn(async move {
        let _ = axum::serve(api_listener, router).await;
    });

    // Proxy under test.
    let config = Config {
        listen_addr: "127.0.0.1:0".to_string(),
        api_base_url: format!("http://{api_addr}"),
        service_id: "stratum-proxy".to_string(),
        service_secret: SECRET.to_string(),
        config_cache_ttl: Duration::from_secs(60),
        worker_cache_ttl: Duration::from_secs(60),
        authorize_timeout: Duration::from_secs(5),
    };

    let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = proxy_listener.local_addr().unwrap();

    tokio::spawn(async move {
        let _ = serve(proxy_listener, Arc::new(AppContext::new(config))).await;
    });

    tokio::spawn(run_mock_pool(pool_listener, accept_submissions));

    Stack { proxy_addr, state }
}

/// Encodes one newline-delimited stratum message.
fn encode(message: Value) -> Vec<u8> {
    let mut encoded = message.to_string();
    encoded.push('\n');
    encoded.into_bytes()
}

/// Runs one miner session: subscribe, authorize, then submit one share.
async fn run_miner(proxy_addr: std::net::SocketAddr, user: &str) -> Vec<Value> {
    let stream = TcpStream::connect(proxy_addr).await.expect("connecting to the proxy");
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    write_half
        .write_all(&encode(json!({"id":1,"method":"mining.subscribe","params":["miner/1.0"]})))
        .await
        .unwrap();
    write_half
        .write_all(&encode(json!({"id":2,"method":"mining.authorize","params":[user,"x"]})))
        .await
        .unwrap();
    write_half.flush().await.unwrap();

    let mut responses = Vec::new();

    for _ in 0..2 {
        let mut line = String::new();
        timeout(Duration::from_secs(5), reader.read_line(&mut line))
            .await
            .expect("the proxy answers the handshake")
            .unwrap();
        responses.push(serde_json::from_str::<Value>(line.trim()).unwrap());
    }

    write_half
        .write_all(&encode(json!({
            "id": 3,
            "method": "mining.submit",
            "params": [user, "job-1", "deadbeef", "65a1b2c3", "0a1b2c3d"]
        })))
        .await
        .unwrap();
    write_half.flush().await.unwrap();

    let mut line = String::new();
    timeout(Duration::from_secs(5), reader.read_line(&mut line))
        .await
        .expect("the proxy relays the verdict")
        .unwrap();
    responses.push(serde_json::from_str::<Value>(line.trim()).unwrap());

    responses
}

/// Waits for the asynchronous share report to land.
async fn wait_for_shares(state: &ApiState, expected: usize) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(5);

    while Instant::now() < deadline {
        let shares = state.recorded_shares();
        if shares.len() >= expected {
            return shares;
        }
        sleep(Duration::from_millis(25)).await;
    }

    state.recorded_shares()
}

#[tokio::test]
async fn an_accepted_share_is_reported_with_a_valid_signature() {
    let stack = spawn_stack(true).await;

    let responses = run_miner(stack.proxy_addr, &username()).await;

    // The pool's own verdict is handed straight back to the miner.
    assert_eq!(responses[2]["result"], json!(true));

    let shares = wait_for_shares(&stack.state, 1).await;

    assert_eq!(shares.len(), 1, "exactly one share should be reported");
    assert!(
        stack.state.rejected_signatures.lock().unwrap().is_empty(),
        "the reported share must carry a valid signature"
    );

    let share = &shares[0];

    // The worker id comes from the username; the pool and coin from the routing table entry.
    assert_eq!(share["workerIdentifier"], json!(WORKER));
    assert_eq!(share["poolId"], json!("pool-1"));
    assert_eq!(share["coinId"], json!("coin-1"));
    assert_eq!(share["jobId"], json!("job-1"));
    assert_eq!(share["nonce"], json!("0a1b2c3d"));
    assert_eq!(share["extranonce"], json!("deadbeef"));

    // The identifier is the deterministic proof-of-work tuple, computed the same way here.
    let expected_identifier = {
        let canonical = format!("{WORKER}|job-1|deadbeef|65a1b2c3|0a1b2c3d");
        let digest = {
            use sha2::{Digest, Sha256};
            hex::encode_upper(Sha256::digest(canonical.as_bytes()))
        };
        digest[..32].to_string()
    };

    assert_eq!(share["shareIdentifier"], json!(expected_identifier));

    // The pool's response is kept verbatim for auditing.
    assert_eq!(share["poolResponse"]["result"], json!(true));
}

#[tokio::test]
async fn a_rejected_share_is_never_reported() {
    let stack = spawn_stack(false).await;

    let responses = run_miner(stack.proxy_addr, &username()).await;

    // The rejection reaches the miner unchanged...
    assert_eq!(responses[2]["error"][0], json!(23));

    // ...and, after giving the proxy time to (not) report it, the API has heard nothing.
    sleep(Duration::from_millis(300)).await;

    assert!(
        stack.state.recorded_shares().is_empty(),
        "a rejected share must never be reported"
    );
}

/// Sends one authorize and reads the proxy's single reply.
async fn authorize(proxy_addr: std::net::SocketAddr, username: &str) -> Value {
    let stream = TcpStream::connect(proxy_addr).await.unwrap();
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    write_half
        .write_all(&encode(json!({"id":7,"method":"mining.authorize","params":[username,"x"]})))
        .await
        .unwrap();
    write_half.flush().await.unwrap();

    let mut line = String::new();
    timeout(Duration::from_secs(5), reader.read_line(&mut line))
        .await
        .expect("the proxy replies to a refused authorize")
        .unwrap();

    serde_json::from_str::<Value>(line.trim()).unwrap()
}

#[tokio::test]
async fn a_wallet_only_username_routes_by_the_wallet() {
    let stack = spawn_stack(true).await;

    // No pool segment: the wallet is unique to a pool, so it picks the pool.
    let responses = run_miner(stack.proxy_addr, &format!("{WALLET}.{WORKER}")).await;

    assert_eq!(responses[2]["result"], json!(true));

    let shares = wait_for_shares(&stack.state, 1).await;

    assert_eq!(shares.len(), 1, "a wallet-only username must still be reported");
    assert_eq!(shares[0]["poolId"], json!("pool-1"));
    assert_eq!(shares[0]["workerIdentifier"], json!(WORKER));
}

#[tokio::test]
async fn an_unknown_wallet_is_refused() {
    let stack = spawn_stack(true).await;

    let response = authorize(stack.proxy_addr, &format!("some-other-wallet.{SYSTEM_POOL_ID}.{WORKER}")).await;

    assert_eq!(response["id"], json!(7));
    assert_eq!(response["error"][0], json!(24));
    assert_eq!(response["error"][1], json!("unauthorised wallet"));
    assert_eq!(response["result"], json!(null));
}

#[tokio::test]
async fn an_unknown_pool_is_refused() {
    let stack = spawn_stack(true).await;

    let response = authorize(stack.proxy_addr, &format!("{WALLET}.no-such-pool.{WORKER}")).await;

    assert_eq!(response["error"][0], json!(24));
    assert_eq!(response["error"][1], json!("unknown pool"));
}
