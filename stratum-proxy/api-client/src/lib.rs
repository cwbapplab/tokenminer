//! Signed client for the TokenMiner internal API.
//!
//! The signing scheme must match the API's `ServiceRequestSignature` exactly: HMAC-SHA256 over
//! `METHOD\nPATH\nTIMESTAMP\nNONCE\nSHA256(body)`, hex-encoded in upper case.

use chrono::{SecondsFormat, Utc};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

pub const SERVICE_ID_HEADER: &str = "X-Service-Id";
pub const TIMESTAMP_HEADER: &str = "X-Timestamp";
pub const NONCE_HEADER: &str = "X-Nonce";
pub const SIGNATURE_HEADER: &str = "X-Signature";

pub const STRATUM_CONFIG_PATH: &str = "/internal/stratum/config";
pub const SHARES_PATH: &str = "/internal/mining/shares";

/// Upper-case hex SHA-256 of the request body.
pub fn body_hash(body: &[u8]) -> String {
    hex::encode_upper(Sha256::digest(body))
}

/// Canonical request signature.
pub fn signature(
    secret: &str,
    method: &str,
    path: &str,
    timestamp: &str,
    nonce: &str,
    body_hash: &str,
) -> String {
    let canonical = format!(
        "{}\n{}\n{}\n{}\n{}",
        method.to_uppercase(),
        path,
        timestamp,
        nonce,
        body_hash
    );

    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC accepts a key of any length");

    mac.update(canonical.as_bytes());

    hex::encode_upper(mac.finalize().into_bytes())
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("request to {path} failed: {source}")]
    Transport {
        path: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("request to {path} returned HTTP {status}")]
    Status { path: String, status: u16 },

    #[error("could not decode the response from {path}: {source}")]
    Decode {
        path: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("could not encode the request body: {0}")]
    Encode(#[from] serde_json::Error),
}

// --- Wire types ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StratumConfig {
    pub version: String,
    #[serde(default)]
    pub pools: Vec<StratumPool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StratumPool {
    pub pool_id: String,
    pub system_pool_id: String,
    pub name: String,
    pub stratum_endpoint: String,
    pub payout_address: Option<String>,
    #[serde(default)]
    pub coins: Vec<StratumCoin>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StratumCoin {
    pub coin_id: String,
    pub code: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StratumWorker {
    pub worker_id: String,
    pub user_id: String,
    pub user_hardware_id: String,
    pub user_hardware_miner_id: String,
    pub pool_id: String,
    pub stratum_endpoint: String,
    pub payout_address: Option<String>,
    pub coin_id: String,
    pub coin_code: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShareReport {
    /// The pool-facing worker id (the device's hardware id in "N" form); the API resolves it.
    pub worker_identifier: String,
    pub pool_id: String,
    pub coin_id: String,
    pub share_identifier: String,
    pub job_id: Option<String>,
    pub nonce: Option<String>,
    pub extranonce: Option<String>,
    pub difficulty: Option<f64>,
    pub target: Option<String>,
    pub hash: Option<String>,
    pub timestamp: String,
    pub coin_value: Option<f64>,
    pub pool_response: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordedShare {
    pub id: String,
    pub share_identifier: String,
    pub reward_status: String,
    pub already_recorded: bool,
}

// --- Client -------------------------------------------------------------------------------

#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
    service_id: String,
    secret: String,
    http: reqwest::Client,
}

impl ApiClient {
    pub fn new(
        base_url: impl Into<String>,
        service_id: impl Into<String>,
        secret: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            service_id: service_id.into(),
            secret: secret.into(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .expect("reqwest client builds"),
        }
    }

    /// Current routing table. `version` changes whenever pool configuration changes.
    pub async fn stratum_config(&self) -> Result<StratumConfig, ApiError> {
        let response = self
            .send(reqwest::Method::GET, STRATUM_CONFIG_PATH, None)
            .await?;

        self.decode(STRATUM_CONFIG_PATH, response).await
    }

    /// Resolves a worker identity to its active session. `Ok(None)` means the worker is not
    /// currently allowed to mine.
    pub async fn resolve_worker(&self, worker_id: &str) -> Result<Option<StratumWorker>, ApiError> {
        let path = format!("/internal/stratum/workers/{worker_id}");

        let response = self.send(reqwest::Method::GET, &path, None).await?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(ApiError::Status {
                path,
                status: response.status().as_u16(),
            });
        }

        response
            .json::<StratumWorker>()
            .await
            .map(Some)
            .map_err(|source| ApiError::Decode { path, source })
    }

    /// Reports a pool-accepted share. Safe to retry: the API de-duplicates on the share id.
    pub async fn record_share(&self, report: &ShareReport) -> Result<RecordedShare, ApiError> {
        let body = serde_json::to_vec(report)?;

        let response = self
            .send(reqwest::Method::POST, SHARES_PATH, Some(body))
            .await?;

        self.decode(SHARES_PATH, response).await
    }

    async fn decode<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        response: reqwest::Response,
    ) -> Result<T, ApiError> {
        if !response.status().is_success() {
            return Err(ApiError::Status {
                path: path.to_string(),
                status: response.status().as_u16(),
            });
        }

        response.json::<T>().await.map_err(|source| ApiError::Decode {
            path: path.to_string(),
            source,
        })
    }

    async fn send(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> Result<reqwest::Response, ApiError> {
        let timestamp = Utc::now().timestamp().to_string();
        let nonce = Uuid::new_v4().simple().to_string();
        let hash = body_hash(body.as_deref().unwrap_or_default());

        let signature = signature(&self.secret, method.as_str(), path, &timestamp, &nonce, &hash);

        let mut request = self
            .http
            .request(method, format!("{}{path}", self.base_url))
            .header(SERVICE_ID_HEADER, &self.service_id)
            .header(TIMESTAMP_HEADER, &timestamp)
            .header(NONCE_HEADER, &nonce)
            .header(SIGNATURE_HEADER, &signature);

        if let Some(body) = body {
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body);
        }

        request
            .send()
            .await
            .map_err(|source| ApiError::Transport {
                path: path.to_string(),
                source,
            })
    }
}

/// RFC 3339 timestamp with milliseconds, which the API parses as a `DateTimeOffset`.
pub fn now_timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_hash_is_upper_case_hex_sha256() {
        assert_eq!(
            body_hash(b""),
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
        );
        assert_eq!(body_hash(b"abc"), body_hash(b"abc"));
        assert_ne!(body_hash(b"abc"), body_hash(b"abd"));
    }

    #[test]
    fn signature_is_deterministic_and_upper_case_hex() {
        let first = signature("secret", "POST", "/internal/mining/shares", "100", "nonce", "hash");
        let second = signature("secret", "POST", "/internal/mining/shares", "100", "nonce", "hash");

        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        assert!(first.chars().all(|character| character.is_ascii_hexdigit()));
        assert_eq!(first, first.to_uppercase());
    }

    #[test]
    fn signature_ignores_the_method_case() {
        assert_eq!(
            signature("secret", "post", "/p", "1", "n", "h"),
            signature("secret", "POST", "/p", "1", "n", "h")
        );
    }

    #[test]
    fn signature_changes_when_any_input_changes() {
        let base = signature("secret", "POST", "/p", "1", "n", "h");

        for other in [
            signature("other", "POST", "/p", "1", "n", "h"),
            signature("secret", "GET", "/p", "1", "n", "h"),
            signature("secret", "POST", "/q", "1", "n", "h"),
            signature("secret", "POST", "/p", "2", "n", "h"),
            signature("secret", "POST", "/p", "1", "m", "h"),
            signature("secret", "POST", "/p", "1", "n", "i"),
        ] {
            assert_ne!(base, other);
        }
    }

    #[test]
    fn timestamp_is_rfc3339() {
        let timestamp = now_timestamp();

        assert!(timestamp.ends_with('Z'));
        assert!(timestamp.contains('T'));
        assert_eq!(timestamp.len(), "2026-10-02T19:00:00.000Z".len());
    }

    /// The same vector is asserted by `ServiceSignatureVectorTests` on the .NET side. The two
    /// implementations must agree byte for byte or service authentication fails at runtime.
    #[test]
    fn matches_the_cross_language_signature_vector() {
        let hash = body_hash(br#"{"a":1}"#);

        assert_eq!(
            hash,
            "015ABD7F5CC57A2DD94B7590F04AD8084273905EE33EC5CEBEAE62276A97F862"
        );

        let signature = signature(
            "cross-language-secret",
            "POST",
            "/internal/mining/shares",
            "1767225600",
            "0123456789abcdef",
            &hash,
        );

        assert_eq!(
            signature,
            "0705551D50F70830D9F337F41476086C07942D3C4B156F128F2067C6952AC8B5"
        );
    }

    #[test]
    fn share_reports_serialise_with_camel_case_names() {
        let report = ShareReport {
            worker_identifier: "w".into(),
            pool_id: "p".into(),
            coin_id: "c".into(),
            share_identifier: "s".into(),
            job_id: None,
            nonce: None,
            extranonce: None,
            difficulty: None,
            target: None,
            hash: None,
            timestamp: now_timestamp(),
            coin_value: None,
            pool_response: None,
        };

        let value = serde_json::to_value(&report).unwrap();

        assert!(value.get("shareIdentifier").is_some());
        assert!(value.get("workerIdentifier").is_some());
        assert!(value.get("poolId").is_some());
    }
}
