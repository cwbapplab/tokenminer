use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use sp_api_client::StratumWorker;
use sp_protocol::{Message, METHOD_SET_DIFFICULTY, METHOD_SUBMIT};
use tokio::io::{AsyncBufRead, AsyncWrite};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::share::{self, PendingSubmit};
use crate::AppContext;

/// Submissions awaiting the pool's verdict, keyed by the JSON-RPC request id.
pub type PendingSubmits = Arc<Mutex<HashMap<String, PendingSubmit>>>;

/// A submission the pool never answers must not grow the map without bound.
const MAX_PENDING_SUBMITS: usize = 512;

/// Miner → pool. Relays verbatim and remembers any submission so its verdict can be interpreted.
pub async fn pump_up<R, W>(
    mut reader: R,
    mut writer: W,
    pending: PendingSubmits,
    worker: Arc<StratumWorker>,
) -> anyhow::Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    while let Some(line) = sp_protocol::read_line(&mut reader).await? {
        observe_submission(&line, &pending, &worker).await;
        sp_protocol::write_line(&mut writer, &line).await?;
    }

    Ok(())
}

/// Pool → miner. Relays verbatim and, for a submission the pool accepted, reports the share.
pub async fn pump_down<R, W>(
    mut reader: R,
    mut writer: W,
    pending: PendingSubmits,
    difficulty: Arc<Mutex<Option<f64>>>,
    context: Arc<AppContext>,
) -> anyhow::Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    while let Some(line) = sp_protocol::read_line(&mut reader).await? {
        observe_verdict(&line, &pending, &difficulty, &context).await;
        sp_protocol::write_line(&mut writer, &line).await?;
    }

    Ok(())
}

async fn observe_submission(line: &str, pending: &PendingSubmits, worker: &Arc<StratumWorker>) {
    let Ok(Message::Request(request)) = Message::parse(line) else {
        return;
    };

    if request.method != METHOD_SUBMIT {
        return;
    }

    let Some(submit) = sp_protocol::parse_submit(&request.params) else {
        warn!("ignoring a submission whose arguments could not be read");
        return;
    };

    let identifier = share::share_identifier(&worker.worker_id, &submit);
    let key = request.id.to_string();

    let mut pending = pending.lock().await;

    if pending.len() >= MAX_PENDING_SUBMITS {
        // Drop an arbitrary entry rather than grow without bound; a lost verdict only means a
        // share goes unreported, which the pool's own accounting still covers.
        if let Some(stale) = pending.keys().next().cloned() {
            pending.remove(&stale);
        }
    }

    pending.insert(
        key,
        PendingSubmit {
            worker: Arc::clone(worker),
            submit,
            share_identifier: identifier,
        },
    );
}

async fn observe_verdict(
    line: &str,
    pending: &PendingSubmits,
    difficulty: &Arc<Mutex<Option<f64>>>,
    context: &Arc<AppContext>,
) {
    let Ok(message) = Message::parse(line) else {
        return;
    };

    match message {
        Message::Notification(notification) if notification.method == METHOD_SET_DIFFICULTY => {
            if let Some(value) = notification.params.get(0).and_then(Value::as_f64) {
                *difficulty.lock().await = Some(value);
            }
        }

        Message::Response(response) => {
            let key = response.id.to_string();

            // Only submissions are tracked, so an unrelated response is a no-op.
            let Some(observed) = pending.lock().await.remove(&key) else {
                return;
            };

            if !response.is_accepted() {
                // Rejected work is relayed to the miner and otherwise ignored.
                debug!(share = %observed.share_identifier, "pool rejected the submission");
                return;
            }

            let report = share::build_report(
                &observed,
                *difficulty.lock().await,
                serde_json::to_value(&response).ok(),
            );

            let context = Arc::clone(context);

            // Reporting must never stall the relay, so it happens off to the side.
            tokio::spawn(async move {
                match context.api.record_share(&report).await {
                    Ok(recorded) => debug!(
                        share = %recorded.share_identifier,
                        already_recorded = recorded.already_recorded,
                        "accepted share reported"
                    ),
                    Err(error) => warn!(%error, "reporting an accepted share failed"),
                }
            });
        }

        _ => {}
    }
}
