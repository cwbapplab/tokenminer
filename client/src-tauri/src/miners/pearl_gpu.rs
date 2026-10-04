//! The GPU Pearl search.
//!
//! This is the replacement for the old CPU search. The pipeline it drives, entirely on the device:
//!
//! 1. fill `A[M,K]` and `Bt[N,K]` from a Philox counter-based stream
//! 2. commit to them — a chunk-padded BLAKE3 Merkle root each, then the noise seeds
//! 3. generate the four noise tensors (`EAL`, `EAR`, `EBL`, `EBR`)
//! 4. noise-apply one block at a time into `ApEA` / `BpEB`, which is what the GEMM consumes
//! 5. GEMM + fold, writing a 16-word transcript per 16x16 candidate tile
//! 6. hash each transcript and compare against the share bound on device
//!
//! Only when a tile wins do the winning matrices come back to the host, for proof assembly.
//!
//! The reference for the shape of all of this is Open-Pearl-Miner's `csrc/`, and the shape of the
//! arithmetic is `zk_pow::circuit::chip::compute_jackpot`.

use primitive_types::U256;
use zk_pow::api::proof::IncompleteBlockHeader;
use zk_pow::ffi::plain_proof::PlainProof;

use super::gpu::cuda::CudaBackend;
use super::gpu::GpuBackend;
use super::pearl_mining::PearlMining;

/// How long a search keeps going before handing control back to the caller.
///
/// A job is superseded by the next `mining.notify`, and the caller polls `stop` between attempts,
/// so this only needs to be short enough that a new job is picked up promptly.
const MAX_SECONDS_PER_SEARCH: u64 = 5;

/// Owns the device-side state for one mining attempt and the search loop that drives it.
pub struct GpuMiner {
    backend: CudaBackend,
    mining: PearlMining,
    /// Tiles evaluated since the engine started, for the hashrate the UI reports.
    tiles: u64,
}

impl GpuMiner {
    /// Opens a device and prepares it to mine at this configuration's dimensions.
    ///
    /// Runs the backend self-test first: a device that cannot reproduce `compute_jackpot` or the
    /// BLAKE3 the pool compares against must not be allowed to produce shares.
    pub fn open(mining: PearlMining) -> Result<Self, String> {
        let mut backend = CudaBackend::open(0)?;
        backend
            .self_test()
            .map_err(|error| format!("{} failed its self-test: {error}", backend.name()))?;

        log::info!(
            "pearl gpu: mining on {} (m={} n={} k={} rank={})",
            backend.device().describe(),
            mining.m,
            mining.n,
            mining.k,
            mining.rank
        );

        Ok(Self {
            backend,
            mining,
            tiles: 0,
        })
    }

    pub fn mining(&self) -> &PearlMining {
        &self.mining
    }

    /// Searches `header` for a proof whose jackpot is at or below `bound`.
    ///
    /// Returns `None` when `stop` says the job has been superseded, which is the normal way this
    /// ends: a new `mining.notify` arrives far more often than a share does.
    pub fn search(
        &mut self,
        header: &IncompleteBlockHeader,
        _cert_version: u32,
        _bound: U256,
        _stop: &dyn Fn() -> bool,
    ) -> Result<Option<PlainProof>, String> {
        // The CUDA pipeline is not written yet. See the port plan: device BLAKE3 first (everything
        // else depends on it), then noise + commitment, then the folded GEMM, then this loop.
        let _ = (header, MAX_SECONDS_PER_SEARCH);
        Err(format!(
            "the GPU Pearl mining kernels are not built yet — see the port plan (device: {})",
            self.backend.name()
        ))
    }

    /// Candidate tiles evaluated so far, used to report hashrate.
    pub fn tiles(&self) -> u64 {
        self.tiles
    }
}