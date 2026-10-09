//! The Pearl mining stack, built on llmjob's vendored CUDA core.
//!
//! This module is what `miners/mod.rs` exposes when the `pearl` feature is on.
//! It replaces the hand-rolled `pearl_gpu` / `pearl_pow` / `gpu` stack with a
//! thin Rust front end over `vendor/llmjob/earn/native/` — the source of truth
//! for the search — while keeping the client's own Stratum session and wire
//! encoding, which the TokenMiner proxy already accepts.
//!
//! The pieces, bottom to top:
//!
//!   * [`abi`] — the flat C ABI over the vendored core (`native/pearl/pearl_ffi.h`).
//!   * [`config`] — the mining profile, `config52` and `job_key`.
//!   * [`target`] — share-target arithmetic and the region/tile mapping.
//!   * [`proof`] — a hit's Merkle proofs -> `PlainProof` -> the base64 wire form.
//!   * [`oracle`] — the vendored stack's frozen known-answer vectors.
//!   * [`engine`] — [`engine::GpuMiner`], the search loop the driver drives.
//!
//! The engine is created and driven on one thread (`stratum::spawn_miner`): a
//! CUDA context is bound to the thread that made it.

pub mod abi;
pub mod config;
pub mod engine;
pub mod frontdoor;
pub mod oracle;
pub mod proof;
pub mod target;

// What the rest of the client reaches for. Kept to exactly that: every name
// re-exported here is one `stratum` (or the front door) actually calls, because
// an unused re-export is a warning that hides the next real one.
pub use engine::GpuMiner;
pub use frontdoor::{start, stop};
pub use proof::{encode_plain_proof, verify_share_locally};
pub use target::share_search;
