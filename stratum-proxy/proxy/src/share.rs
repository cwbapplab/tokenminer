use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};
use sp_api_client::{now_timestamp, ShareReport, StratumWorker};
use sp_protocol::Submit;

/// A submission seen on its way to the pool, awaiting the pool's verdict.
#[derive(Debug, Clone)]
pub struct PendingSubmit {
    pub worker: Arc<StratumWorker>,
    pub submit: Submit,
    pub share_identifier: String,
}

/// Deterministic idempotency key for a submission.
///
/// The pool does not hand out a share id, so it is derived from the proof-of-work tuple. Two
/// submissions of the same nonce for the same job therefore collapse into one record, which is
/// exactly the de-duplication the API expects.
pub fn share_identifier(worker_id: &str, submit: &Submit) -> String {
    let canonical = format!(
        "{}|{}|{}|{}|{}",
        worker_id,
        submit.job_id,
        submit.extranonce.as_deref().unwrap_or_default(),
        submit.ntime.as_deref().unwrap_or_default(),
        submit.nonce
    );

    let digest = hex::encode_upper(Sha256::digest(canonical.as_bytes()));

    digest[..32].to_string()
}

/// Builds the payload reported to the API for an accepted share.
pub fn build_report(
    pending: &PendingSubmit,
    difficulty: Option<f64>,
    pool_response: Option<Value>,
) -> ShareReport {
    ShareReport {
        user_id: pending.worker.user_id.clone(),
        user_hardware_id: pending.worker.user_hardware_id.clone(),
        pool_id: pending.worker.pool_id.clone(),
        coin_id: pending.worker.coin_id.clone(),
        share_identifier: pending.share_identifier.clone(),
        job_id: Some(pending.submit.job_id.clone()),
        nonce: Some(pending.submit.nonce.clone()),
        extranonce: pending.submit.extranonce.clone(),
        difficulty,
        // The target and resulting hash are not part of a stratum submit; the pool's response
        // is stored verbatim instead so the share can still be audited.
        target: None,
        hash: None,
        timestamp: now_timestamp(),
        coin_value: None,
        pool_response,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sp_api_client::StratumWorker;

    fn worker() -> Arc<StratumWorker> {
        Arc::new(StratumWorker {
            worker_id: "user-hardware".into(),
            user_id: "user".into(),
            user_hardware_id: "hardware".into(),
            user_hardware_miner_id: "session".into(),
            pool_id: "pool".into(),
            stratum_endpoint: "pool.example.com:3333".into(),
            payout_address: Some("wallet".into()),
            coin_id: "coin".into(),
            coin_code: "qtc".into(),
        })
    }

    fn submit(nonce: &str) -> Submit {
        Submit {
            worker: "wallet.user-hardware".into(),
            job_id: "job-1".into(),
            extranonce: Some("deadbeef".into()),
            ntime: Some("65a1b2c3".into()),
            nonce: nonce.into(),
        }
    }

    #[test]
    fn share_identifier_is_stable_for_the_same_work() {
        let first = share_identifier("user-hardware", &submit("0a1b"));
        let second = share_identifier("user-hardware", &submit("0a1b"));

        assert_eq!(first, second);
        assert_eq!(first.len(), 32);
        assert!(first.chars().all(|character| character.is_ascii_hexdigit()));
    }

    #[test]
    fn share_identifier_changes_with_the_nonce_or_worker() {
        let base = share_identifier("user-hardware", &submit("0a1b"));

        assert_ne!(base, share_identifier("user-hardware", &submit("0a1c")));
        assert_ne!(base, share_identifier("other-worker", &submit("0a1b")));
    }

    #[test]
    fn report_maps_the_worker_identity_and_proof_of_work() {
        let pending = PendingSubmit {
            worker: worker(),
            submit: submit("0a1b"),
            share_identifier: share_identifier("user-hardware", &submit("0a1b")),
        };

        let report = build_report(&pending, Some(2.0), Some(serde_json::json!({"result": true})));

        assert_eq!(report.user_id, "user");
        assert_eq!(report.user_hardware_id, "hardware");
        assert_eq!(report.pool_id, "pool");
        assert_eq!(report.coin_id, "coin");
        assert_eq!(report.job_id.as_deref(), Some("job-1"));
        assert_eq!(report.nonce.as_deref(), Some("0a1b"));
        assert_eq!(report.extranonce.as_deref(), Some("deadbeef"));
        assert_eq!(report.difficulty, Some(2.0));
        assert!(report.target.is_none());
        assert!(report.hash.is_none());
        assert!(!report.timestamp.is_empty());
    }
}
