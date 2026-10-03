//! Stratum JSON-RPC line framing and message parsing.
//!
//! The proxy relays miner traffic verbatim; parsing exists so it can *observe* which worker is
//! talking and which submissions were accepted. Anything that fails to parse is simply relayed
//! unchanged, so an unusual pool dialect can never break the relay.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};

pub const METHOD_SUBSCRIBE: &str = "mining.subscribe";
pub const METHOD_AUTHORIZE: &str = "mining.authorize";
pub const METHOD_SUBMIT: &str = "mining.submit";
pub const METHOD_SET_DIFFICULTY: &str = "mining.set_difficulty";
pub const METHOD_NOTIFY: &str = "mining.notify";

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("message is not valid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("message is neither a request, a response nor a notification")]
    Unrecognised,

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Notification {
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub id: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

impl Response {
    /// The pool accepted the work.
    pub fn is_accepted(&self) -> bool {
        self.error.is_none() && matches!(self.result, Some(Value::Bool(true)))
    }

    /// The pool refused the work, either with an error or a `false` result.
    pub fn is_rejected(&self) -> bool {
        self.error.is_some() || matches!(self.result, Some(Value::Bool(false)))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    Request(Request),
    Response(Response),
    Notification(Notification),
}

impl Message {
    /// Parses a single stratum line. Batch (array) messages are not modelled; callers that only
    /// observe traffic treat a parse failure as "relay unchanged".
    pub fn parse(line: &str) -> Result<Self, ProtocolError> {
        let value: Value = serde_json::from_str(line)?;
        let object = value.as_object().ok_or(ProtocolError::Unrecognised)?;

        if let Some(method) = object.get("method").and_then(Value::as_str) {
            let params = object.get("params").cloned().unwrap_or(Value::Null);

            return match object.get("id") {
                Some(id) if !id.is_null() => Ok(Message::Request(Request {
                    id: id.clone(),
                    method: method.to_string(),
                    params,
                })),
                _ => Ok(Message::Notification(Notification {
                    method: method.to_string(),
                    params,
                })),
            };
        }

        match object.get("id") {
            Some(id) if !id.is_null() => Ok(Message::Response(Response {
                id: id.clone(),
                result: object.get("result").cloned().filter(|value| !value.is_null()),
                error: object.get("error").cloned().filter(|value| !value.is_null()),
            })),
            _ => Err(ProtocolError::Unrecognised),
        }
    }

    pub fn as_request(&self) -> Option<&Request> {
        match self {
            Message::Request(request) => Some(request),
            _ => None,
        }
    }

    pub fn as_response(&self) -> Option<&Response> {
        match self {
            Message::Response(response) => Some(response),
            _ => None,
        }
    }

    pub fn as_notification(&self) -> Option<&Notification> {
        match self {
            Message::Notification(notification) => Some(notification),
            _ => None,
        }
    }
}

/// A `mining.submit` observed on the way up to the pool.
///
/// The classic dialect is positional (`[worker, jobId, extranonce2, ntime, nonce, ...]`) and the
/// nonce is the proof of work. Pearl's named dialect sends `{job_id, plain_proof}` with no worker
/// or nonce, so the proof carries the work instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submit {
    pub worker: String,
    pub job_id: String,
    pub extranonce: Option<String>,
    pub ntime: Option<String>,
    pub nonce: String,
    pub proof: Option<String>,
}

/// Extracts the arguments of a `mining.submit`, in either dialect.
pub fn parse_submit(params: &Value) -> Option<Submit> {
    if let Some(values) = params.as_array() {
        let text = |index: usize| {
            values
                .get(index)
                .and_then(Value::as_str)
                .map(str::to_string)
        };

        return Some(Submit {
            worker: text(0)?,
            job_id: text(1)?,
            extranonce: text(2),
            ntime: text(3),
            nonce: text(4)?,
            proof: None,
        });
    }

    // Pearl: `{ "job_id": "...", "plain_proof": "<base64>" }`.
    let object = params.as_object()?;

    Some(Submit {
        worker: String::new(),
        job_id: object.get("job_id").and_then(Value::as_str)?.to_string(),
        extranonce: None,
        ntime: None,
        nonce: String::new(),
        proof: Some(object.get("plain_proof").and_then(Value::as_str)?.to_string()),
    })
}

/// The username from a `mining.authorize`, in either dialect.
///
/// Classic is positional (`["wallet.worker", "x"]`). Pearl is named (`{"wallet": "...", "worker":
/// "..."}`); a non-empty `worker` is joined to the wallet with a dot so the usual last-dot split
/// yields the worker, and miners that glue `wallet.worker` into `wallet` work unchanged.
pub fn authorize_username(params: &Value) -> Option<String> {
    if let Some(values) = params.as_array() {
        return values.first().and_then(Value::as_str).map(str::to_string);
    }

    let object = params.as_object()?;
    let wallet = object.get("wallet").and_then(Value::as_str)?;

    match object.get("worker").and_then(Value::as_str) {
        Some(worker) if !worker.is_empty() => Some(format!("{wallet}.{worker}")),
        _ => Some(wallet.to_string()),
    }
}

/// The wallet address from a `mining.authorize`, in either dialect.
///
/// Pools treat everything before the first dot as the wallet and the rest as the worker, so the
/// `wallet.worker` form and a separate `worker` field both reduce to the same address.
pub fn authorize_wallet(params: &Value) -> Option<String> {
    let username = authorize_username(params)?;

    Some(match username.split_once('.') {
        Some((wallet, _)) => wallet.to_string(),
        None => username,
    })
}

/// A `mining.authorize` username split into `wallet.systemPoolId.worker`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizeIdentity {
    pub wallet: String,
    /// The pool named in the username, absent for a bare `wallet.worker`.
    pub pool: Option<String>,
    /// The worker identity, `<userId>-<hardwareId>`.
    pub worker: String,
}

/// Splits a username into wallet, pool and worker. The pool is the segments between the first and
/// last dot, so `wallet.pool.worker` and `wallet.worker` both parse. The worker is opaque.
pub fn parse_authorize(params: &Value) -> Option<AuthorizeIdentity> {
    let username = authorize_username(params)?;
    let mut segments = username.split('.');

    let wallet = segments.next()?.to_string();
    let rest: Vec<&str> = segments.collect();

    let (pool, worker) = match rest.len() {
        0 => (None, wallet.clone()),
        1 => (None, rest[0].to_string()),
        _ => (
            Some(rest[..rest.len() - 1].join(".")),
            rest[rest.len() - 1].to_string(),
        ),
    };

    Some(AuthorizeIdentity { wallet, pool, worker })
}

/// Extracts the worker identity from a stratum username of the form `wallet.worker`.
///
/// Worker identities contain dashes but never dots, so splitting on the last dot is safe even
/// when no wallet prefix is present.
pub fn worker_id_from_username(username: &str) -> &str {
    match username.rsplit_once('.') {
        Some((_, worker)) if !worker.is_empty() => worker,
        _ => username,
    }
}

/// Reads one newline-delimited message. `Ok(None)` means the peer closed cleanly.
pub async fn read_line<R>(reader: &mut R) -> Result<Option<String>, ProtocolError>
where
    R: AsyncBufRead + Unpin,
{
    let mut buffer = String::new();

    if reader.read_line(&mut buffer).await? == 0 {
        return Ok(None);
    }

    Ok(Some(
        buffer
            .trim_end_matches(['\r', '\n'])
            .to_string(),
    ))
}

/// Writes one newline-delimited message.
pub async fn write_line<W>(writer: &mut W, line: &str) -> Result<(), ProtocolError>
where
    W: AsyncWrite + Unpin,
{
    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;

    Ok(())
}

/// Builds a stratum error response, which is always `[code, message, traceback]`.
pub fn error_response(id: &Value, code: i32, message: &str) -> String {
    serde_json::json!({
        "id": id,
        "result": null,
        "error": [code, message, null],
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_a_request_with_an_id() {
        let message = Message::parse(r#"{"id":1,"method":"mining.submit","params":[]}"#).unwrap();
        let request = message.as_request().expect("a request");

        assert_eq!(request.method, METHOD_SUBMIT);
        assert_eq!(request.id, json!(1));
    }

    #[test]
    fn parses_a_notification_without_an_id() {
        let message = Message::parse(r#"{"method":"mining.notify","params":[1,2]}"#).unwrap();

        assert_eq!(message.as_notification().unwrap().method, METHOD_NOTIFY);
    }

    #[test]
    fn a_null_id_is_a_notification_not_a_request() {
        let message = Message::parse(r#"{"id":null,"method":"mining.set_difficulty","params":[1]}"#).unwrap();

        assert!(message.as_notification().is_some());
    }

    #[test]
    fn parses_an_accepted_response() {
        let message = Message::parse(r#"{"id":1,"result":true,"error":null}"#).unwrap();
        let response = message.as_response().expect("a response");

        assert!(response.is_accepted());
        assert!(!response.is_rejected());
    }

    #[test]
    fn parses_a_rejected_response_with_an_error() {
        let message =
            Message::parse(r#"{"id":1,"result":null,"error":[23,"low difficulty",null]}"#).unwrap();
        let response = message.as_response().expect("a response");

        assert!(response.is_rejected());
        assert!(!response.is_accepted());
    }

    #[test]
    fn a_false_result_counts_as_rejected() {
        let message = Message::parse(r#"{"id":7,"result":false}"#).unwrap();
        let response = message.as_response().expect("a response");

        assert!(response.is_rejected());
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(Message::parse("not json").is_err());
        assert!(Message::parse("[1,2,3]").is_err());
        assert!(Message::parse(r#"{"nothing":"useful"}"#).is_err());
    }

    #[test]
    fn parses_submit_arguments() {
        let params = json!(["-acct.rig-1", "job-42", "deadbeef", "65a1b2c3", "0a1b2c3d", "00000002"]);

        let submit = parse_submit(&params).expect("a submit");

        assert_eq!(submit.worker, "-acct.rig-1");
        assert_eq!(submit.job_id, "job-42");
        assert_eq!(submit.extranonce.as_deref(), Some("deadbeef"));
        assert_eq!(submit.ntime.as_deref(), Some("65a1b2c3"));
        assert_eq!(submit.nonce, "0a1b2c3d");
    }

    #[test]
    fn rejects_incomplete_submit_arguments() {
        assert!(parse_submit(&json!(["-acct.rig-1", "job-42"])).is_none());
        assert!(parse_submit(&json!("nope")).is_none());
    }

    #[test]
    fn parses_a_pearl_named_submit() {
        let params = json!({ "job_id": "18e4eadf_1", "plain_proof": "AAACAAAA" });

        let submit = parse_submit(&params).expect("a submit");

        assert_eq!(submit.job_id, "18e4eadf_1");
        assert_eq!(submit.proof.as_deref(), Some("AAACAAAA"));
        assert!(submit.nonce.is_empty());
        assert!(submit.worker.is_empty());
    }

    #[test]
    fn rejects_a_named_submit_without_a_proof() {
        assert!(parse_submit(&json!({ "job_id": "18e4eadf_1" })).is_none());
    }

    #[test]
    fn parses_a_classic_positional_authorize() {
        assert_eq!(
            authorize_username(&json!(["krxYR9NJVQ.user-1", "x"])).as_deref(),
            Some("krxYR9NJVQ.user-1")
        );
    }

    #[test]
    fn parses_a_pearl_named_authorize() {
        assert_eq!(
            authorize_username(&json!({ "wallet": "krxYR9NJVQ.pc-home", "worker": "" })).as_deref(),
            Some("krxYR9NJVQ.pc-home")
        );
        assert_eq!(
            authorize_username(&json!({ "wallet": "prl1paddr", "worker": "rig1" })).as_deref(),
            Some("prl1paddr.rig1")
        );
    }

    #[test]
    fn a_named_authorize_without_a_wallet_is_none() {
        assert!(authorize_username(&json!({ "worker": "rig1" })).is_none());
    }

    #[test]
    fn reads_the_wallet_before_the_first_dot() {
        assert_eq!(
            authorize_wallet(&json!(["krxYR9NJVQ.user-1", "x"])).as_deref(),
            Some("krxYR9NJVQ")
        );
        assert_eq!(
            authorize_wallet(&json!({ "wallet": "krxYR9NJVQ.pc-home", "worker": "" })).as_deref(),
            Some("krxYR9NJVQ")
        );
        assert_eq!(
            authorize_wallet(&json!({ "wallet": "prl1paddr", "worker": "rig1" })).as_deref(),
            Some("prl1paddr")
        );
        assert_eq!(
            authorize_wallet(&json!(["barewallet", "x"])).as_deref(),
            Some("barewallet")
        );
    }

    #[test]
    fn splits_a_three_part_username() {
        let identity = parse_authorize(&json!([
            "krxYR9NJVQ.kryptex-prl.11111111-1111-1111-1111-111111111111-22222222-2222-2222-2222-222222222222",
            "x"
        ]))
        .expect("an identity");

        assert_eq!(identity.wallet, "krxYR9NJVQ");
        assert_eq!(identity.pool.as_deref(), Some("kryptex-prl"));
        assert_eq!(
            identity.worker,
            "11111111-1111-1111-1111-111111111111-22222222-2222-2222-2222-222222222222"
        );
    }

    #[test]
    fn a_two_part_username_has_no_pool() {
        let identity = parse_authorize(&json!(["krxYR9NJVQ.rig-1", "x"])).expect("an identity");

        assert_eq!(identity.wallet, "krxYR9NJVQ");
        assert!(identity.pool.is_none());
        assert_eq!(identity.worker, "rig-1");
    }

    #[test]
    fn extracts_the_worker_from_a_wallet_prefixed_username() {
        assert_eq!(worker_id_from_username("krxYR9NJVQ.11111111-2222"), "11111111-2222");
    }

    #[test]
    fn a_bare_username_is_its_own_worker() {
        assert_eq!(worker_id_from_username("11111111-2222"), "11111111-2222");
    }

    #[test]
    fn a_trailing_dot_falls_back_to_the_whole_username() {
        assert_eq!(worker_id_from_username("wallet."), "wallet.");
    }

    #[tokio::test]
    async fn framing_round_trips_and_strips_line_endings() {
        let mut buffer: Vec<u8> = Vec::new();
        write_line(&mut buffer, r#"{"id":1}"#).await.unwrap();
        write_line(&mut buffer, r#"{"id":2}"#).await.unwrap();

        assert_eq!(buffer, b"{\"id\":1}\n{\"id\":2}\n");

        let mut reader = tokio::io::BufReader::new(&buffer[..]);
        assert_eq!(read_line(&mut reader).await.unwrap().as_deref(), Some(r#"{"id":1}"#));
        assert_eq!(read_line(&mut reader).await.unwrap().as_deref(), Some(r#"{"id":2}"#));
        assert!(read_line(&mut reader).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn read_line_removes_carriage_returns() {
        let data = b"{\"id\":1}\r\n";
        let mut reader = tokio::io::BufReader::new(&data[..]);

        assert_eq!(read_line(&mut reader).await.unwrap().as_deref(), Some(r#"{"id":1}"#));
    }

    #[test]
    fn builds_stratum_error_responses() {
        let line = error_response(&json!(9), 24, "unauthorised");
        let value: Value = serde_json::from_str(&line).unwrap();

        assert_eq!(value["id"], json!(9));
        assert_eq!(value["error"][0], json!(24));
        assert_eq!(value["error"][1], json!("unauthorised"));
    }
}
