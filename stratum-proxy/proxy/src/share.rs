use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};
use sp_api_client::{now_timestamp, ShareReport};
use sp_protocol::Submit;

/// Everything about a connection that a reported share needs, resolved once at authorize.
#[derive(Debug, Clone)]
pub struct ShareContext {
    /// The pool-facing worker id — the device's hardware id in "N" form. The API resolves it.
    pub worker_identifier: String,
    pub pool_id: String,
    pub coin_id: String,
}

/// A submission seen on its way to the pool, awaiting the pool's verdict.
///
/// Only the fields the API report needs are kept. A Pearl `plain_proof` can be hundreds of KB, so
/// the raw [`Submit`] is not held here — it is hashed into the share identifier and dropped.
#[derive(Debug, Clone)]
pub struct PendingSubmit {
    pub context: Arc<ShareContext>,
    pub job_id: String,
    pub nonce: Option<String>,
    pub extranonce: Option<String>,
    pub share_identifier: String,
}

impl PendingSubmit {
    /// Reduces a parsed submit to the report fields, deriving the idempotency key from the work.
    pub fn new(context: Arc<ShareContext>, submit: &Submit) -> Self {
        let share_identifier = share_identifier(&context.worker_identifier, submit);

        Self {
            context,
            job_id: submit.job_id.clone(),
            nonce: (!submit.nonce.is_empty()).then(|| submit.nonce.clone()),
            extranonce: submit.extranonce.clone(),
            share_identifier,
        }
    }
}

/// Deterministic idempotency key for a submission.
///
/// The pool does not hand out a share id, so it is derived from the proof of work: classic work is
/// the nonce for a job, Pearl work is the `plain_proof`. Two submissions of the same work collapse
/// into one record, which is the de-duplication the API expects.
pub fn share_identifier(worker_id: &str, submit: &Submit) -> String {
    let mut canonical = format!(
        "{}|{}|{}|{}|{}",
        worker_id,
        submit.job_id,
        submit.extranonce.as_deref().unwrap_or_default(),
        submit.ntime.as_deref().unwrap_or_default(),
        submit.nonce
    );

    if let Some(proof) = submit.proof.as_deref() {
        canonical.push('|');
        canonical.push_str(proof);
    }

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
        worker_identifier: pending.context.worker_identifier.clone(),
        pool_id: pending.context.pool_id.clone(),
        coin_id: pending.context.coin_id.clone(),
        share_identifier: pending.share_identifier.clone(),
        job_id: Some(pending.job_id.clone()),
        nonce: pending.nonce.clone(),
        extranonce: pending.extranonce.clone(),
        difficulty,
        // The target and resulting hash are not part of a stratum submit; the pool's response is
        // stored verbatim instead so the share can still be audited.
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

    fn context() -> Arc<ShareContext> {
        Arc::new(ShareContext {
            worker_identifier: "7ed82f7514bd4b1397183061e5e25a68".into(),
            pool_id: "pool".into(),
            coin_id: "coin".into(),
        })
    }

    fn classic(nonce: &str) -> Submit {
        Submit {
            worker: "wallet.worker".into(),
            job_id: "job-1".into(),
            extranonce: Some("deadbeef".into()),
            ntime: Some("65a1b2c3".into()),
            nonce: nonce.into(),
            proof: None,
        }
    }

    fn pearl(proof: &str) -> Submit {
        Submit {
            worker: String::new(),
            job_id: "18e4eadf_1".into(),
            extranonce: None,
            ntime: None,
            nonce: String::new(),
            proof: Some(proof.into()),
        }
    }

    #[test]
    fn share_identifier_is_stable_for_the_same_work() {
        let first = share_identifier("worker", &classic("0a1b"));
        let second = share_identifier("worker", &classic("0a1b"));

        assert_eq!(first, second);
        assert_eq!(first.len(), 32);
        assert!(first.chars().all(|character| character.is_ascii_hexdigit()));
    }

    #[test]
    fn share_identifier_changes_with_the_nonce_or_worker() {
        let base = share_identifier("worker", &classic("0a1b"));

        assert_ne!(base, share_identifier("worker", &classic("0a1c")));
        assert_ne!(base, share_identifier("other-worker", &classic("0a1b")));
    }

    #[test]
    fn pearl_identifier_tracks_the_proof() {
        let base = share_identifier("worker", &pearl("AAACAAAA"));

        assert_eq!(base, share_identifier("worker", &pearl("AAACAAAA")));
        assert_ne!(base, share_identifier("worker", &pearl("AAACAAAB")));
        assert_ne!(base, share_identifier("worker", &classic("0a1b")));
    }

    #[test]
    fn report_maps_the_worker_identity_and_proof_of_work() {
        let pending = PendingSubmit::new(context(), &classic("0a1b"));

        let report = build_report(&pending, Some(2.0), Some(serde_json::json!({"result": true})));

        assert_eq!(report.worker_identifier, "7ed82f7514bd4b1397183061e5e25a68");
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

    #[test]
    fn pearl_report_has_a_job_but_no_nonce() {
        let pending = PendingSubmit::new(context(), &pearl("AAACAAAA"));

        let report = build_report(&pending, None, None);

        assert_eq!(report.job_id.as_deref(), Some("18e4eadf_1"));
        assert!(report.nonce.is_none());
        assert!(report.extranonce.is_none());
    }
}
