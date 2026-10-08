//! The GPU Pearl search.
//!
//! This is the split search path: the fold and the hash are two launches rather than one, which is
//! what lets the search blob use a 4x8 warp tile — the hash state is not live inside it. The chain, in
//! stream order:
//!
//! 1. fill `A[M,K]` and `Bt[N,K]` from a Philox counter-based stream
//! 2. commit to them — a chunk-padded BLAKE3 Merkle root each, then the noise seeds
//! 3. generate the four noise tensors (`EAL`, `EAR`, `EBL`, `EBR`)
//! 4. noise-apply them into `a_sum` and `b_sum`, the summed operands the GEMM consumes
//! 5. fold `a_sum ⊗ b_sumᵀ` into a 16-word transcript per candidate — 64 candidates per 128x128 tile
//! 6. hash each transcript under `dnsA` and compare against the share bound, on device
//! 7. scan the hit flags for the first candidate that cleared the bound
//!
//! Steps 3 through 7 are launches owned by [`super::gpu::pipeline`] over buffers owned by
//! [`super::gpu::bufs`].
//! Steps 1 and 2 stay here because their results cross back to the host: the commitment is what the
//! proof carries, and the proof is what the pool checks.
//!
//! One attempt is one launch per stage over the whole grid, so a hit is readable at the end of an
//! attempt rather than at the end of a batch. That is the latency change this architecture brings: the
//! fused path could report a winner after a batch of tile rows, and this cannot — a grid is one launch
//! and the scan result is one readback. `stop` is therefore checked between attempts, not inside one.
//!
//! The reference for the shape of all of this is the reference miner's `csrc/`, and the shape of the
//! arithmetic is `zk_pow::circuit::chip::compute_jackpot`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaSlice, CudaStream};
use primitive_types::U256;
use zk_pow::api::proof::{
    IncompleteBlockHeader, MiningConfiguration, PublicProofParams, SeedDerivation,
};
use zk_pow::api::sanity_checks::public_params_sanity_check;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof};

use super::gpu::bufs::{SplitBuffers, SplitConfig, CHUNK, TILE_M, TILE_N};
use super::gpu::cuda::CudaBackend;
use super::gpu::GpuBackend;
use super::gpu::fatbin::FatbinKernels;
use super::gpu::pipeline::{self, Hit};
use super::gpu::triton::{HASH_CANDIDATES, TritonKernels};
use super::pearl_mining::{job_key, mining_configuration, seed_derivation_for, PearlMining};

/// How long a search keeps going before handing control back to the caller.
///
/// A job is superseded by the next `mining.notify`, and the caller polls `stop` between attempts, so
/// this only needs to be short enough that a new job is picked up promptly. It is checked between
/// attempts rather than inside one because the whole grid is one launch: there is no point at which a
/// partial grid's results are readable.
const MAX_SECONDS_PER_SEARCH: u64 = 5;

/// Words in one BLAKE3 chaining value, which is the unit the commitment's tree level is held in.
const CV_WORDS: usize = 8;

/// Headroom over the search's own buffers, for the CUDA context, the loaded module and the driver's
/// own per-context allocations.
const DRIVER_SLACK: usize = 128 * 1024 * 1024;

/// Owns the device-side state for one mining configuration and the search loop that drives it.
pub struct GpuMiner {
    backend: CudaBackend,
    /// The same stream the backend launches the fill and the commitment on.
    ///
    /// Held as a clone of the same `Arc` rather than reached through the backend only so the pipeline
    /// module can take it directly — but it is the *same* stream. A second stream would let a noise
    /// launch read a matrix the fill had not finished writing, which is the whole reason the pipeline
    /// is one stream-ordered chain.
    stream: Arc<CudaStream>,
    fatbin: FatbinKernels,
    triton: TritonKernels,
    bufs: SplitBuffers,
    mining: PearlMining,
    /// Candidates evaluated since the engine started, for the hashrate the UI reports.
    ///
    /// Counted as candidates, not as the fused path's 16x16 tiles: at this configuration a tile holds
    /// 64 candidates, so the number is the same work expressed in the unit the search reports.
    tiles: u64,
    /// How many attempts have been drawn, so each one gets its own matrices.
    attempts: u64,
}

impl GpuMiner {
    /// Opens a device and prepares it to mine at this configuration's dimensions.
    ///
    /// Runs the backend self-test first: a device that cannot reproduce `compute_jackpot` or the
    /// BLAKE3 the pool compares against must not be allowed to produce shares. Then it checks the
    /// configuration and the memory, both of which are refused rather than discovered — by a failed
    /// launch, or by an allocation dying several gigabytes into the first job.
    pub fn open(mining: PearlMining) -> Result<Self, String> {
        let mut backend = CudaBackend::open(0)?;
        backend
            .self_test()
            .map_err(|error| format!("{} failed its self-test: {error}", backend.name()))?;

        let config = mining_configuration(&mining)?;
        let split = validate(&mining, &config)?;

        let cc = backend
            .context()
            .compute_capability()
            .map_err(|e| format!("reading the compute capability failed: {e}"))?;

        // Both are load-time gates, not optional extras: the split path has no fallback. A fatbin that
        // was not built, or a device below the image's lowest entry, is a configuration this miner
        // cannot run at all — and it is refused here, before any buffer is allocated.
        let fatbin = FatbinKernels::load(backend.context(), cc)?;
        let triton = TritonKernels::load(backend.context(), cc)?;

        let (m, n, k, rank) = (split.m, split.n, split.k, split.rank);
        let required = footprint(&split) + DRIVER_SLACK;
        let free = backend.free_bytes();
        if free < required {
            return Err(format!(
                "mining at m={m} n={n} k={k} rank={rank} needs {:.1} GiB of device memory and {} \
                 has {:.1} GiB free",
                gib(required as u64),
                backend.device().describe(),
                gib(free as u64)
            ));
        }

        let stream = backend.stream().clone();
        let bufs = SplitBuffers::new(&stream, &split)?;

        log::info!(
            "pearl gpu: mining on {} (m={} n={} k={} rank={}, {:.1} GiB of {:.1} GiB free, \
             {} tiles x {} candidates, fatbin {}.{}, search blob {})",
            backend.device().describe(),
            mining.m,
            mining.n,
            mining.k,
            mining.rank,
            gib(required as u64),
            gib(free as u64),
            split.num_tiles(),
            HASH_CANDIDATES,
            fatbin.cc.0,
            fatbin.cc.1,
            triton.search.arch()
        );

        Ok(Self {
            backend,
            stream,
            fatbin,
            triton,
            bufs,
            mining,
            tiles: 0,
            attempts: 0,
        })
    }

    pub fn mining(&self) -> &PearlMining {
        &self.mining
    }

    /// Searches `header` for a proof whose jackpot is at or below `bound`.
    ///
    /// Returns the proof for the first candidate that clears the bound, and `None` when `stop` says the
    /// job has been superseded or the time slice runs out — the normal way this ends, since a new
    /// `mining.notify` arrives far more often than a share does.
    pub fn search(
        &mut self,
        header: &IncompleteBlockHeader,
        cert_version: u32,
        bound: U256,
        stop: &dyn Fn() -> bool,
    ) -> Result<Option<PlainProof>, String> {
        let derivation = seed_derivation_for(cert_version)?;
        let config = mining_configuration(&self.mining)?;
        let split = validate(&self.mining, &config)?;

        if split != self.bufs.config {
            return Err(format!(
                "the buffers were allocated for {:#?} but this job is {:#?}",
                self.bufs.config, split
            ));
        }

        // Fixed for the job, unlike the attempt number: it is what the noise is derived from, so a
        // new attempt under the same key is a new draw rather than a new search.
        let key = job_key(header, &config);

        // The bound is fixed for the job too, so it goes on the device once rather than per attempt.
        self.bufs.set_pow_target(&self.stream, bound)?;

        let deadline = Instant::now() + Duration::from_secs(MAX_SECONDS_PER_SEARCH);

        loop {
            if stop() {
                return Ok(None);
            }

            let number = self.attempts;
            let jackpot_key = attempt(
                &self.backend,
                &self.stream,
                &self.fatbin,
                &self.triton,
                &mut self.bufs,
                &key,
                derivation,
                number,
            )?;
            self.attempts += 1;

            // The two buffers whose pre-state is part of the answer — see `SplitBuffers::reset_for_attempt`.
            self.bufs.reset_for_attempt(&self.stream)?;

            pipeline::run_search(&self.stream, &self.triton, &self.bufs)?;
            pipeline::run_postpass(&self.stream, &self.fatbin, &self.bufs)?;
            pipeline::run_scan(&self.stream, &self.fatbin, &self.bufs)?;

            // Counted at launch, which is when the work is submitted. The readback below synchronises,
            // so an attempt that never gets that far is still counted — a candidate is hashrate work the
            // device was asked to do either way.
            self.tiles += split.total_candidates() as u64;

            if let Some(hit) = pipeline::read_first_hit(&self.stream, &self.bufs)? {
                return Ok(Some(assemble(
                    &self.backend,
                    &self.bufs,
                    &key,
                    derivation,
                    &hit,
                    &jackpot_key,
                )?));
            }

            // Between attempts: an attempt at production dimensions is a whole grid, and a job
            // superseded half way through one is not worth finishing.
            if stop() || Instant::now() >= deadline {
                return Ok(None);
            }
        }
    }

    /// Candidates evaluated so far, used to report hashrate.
    pub fn tiles(&self) -> u64 {
        self.tiles
    }
}

/// One attempt: draw the signal matrices, commit to them, and derive the summed operands.
///
/// Returns `dnsA`, the key the jackpot transcripts are hashed under. It comes out of the commitment
/// chain rather than the job key, which is the one thing about this pipeline that reads like a mistake
/// and is not.
fn attempt(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    fatbin: &FatbinKernels,
    triton: &TritonKernels,
    bufs: &mut SplitBuffers,
    key: &[u8; 32],
    derivation: SeedDerivation,
    number: u64,
) -> Result<[u8; 32], String> {
    // Only `m` and `n` are needed here: the stages below read `k` and `r` off the config they are
    // handed, so naming them in this function would be a second copy of the shape that could drift.
    let config = bufs.config;
    let (m, n) = (config.m, config.n);

    let (seed_a, seed_b) = fill_seeds(key, number);

    // The matrix, never its padding: the commitment is over the padded bytes, and the padding is
    // zeros, so a fill that ran into it would commit to a matrix the verifier never sees.
    backend.fill_i8(&mut bufs.a, 0, bufs.a_len, seed_a)?;
    backend.fill_i8(&mut bufs.bt, 0, bufs.bt_len, seed_b)?;

    let (b_noise_seed, a_noise_seed) =
        backend.commitment_seeds(&bufs.a, &bufs.bt, *key, m as u32, n as u32, derivation)?;

    // The order here is the order the verifier's `compute_noise_for_indices` destructures its
    // commitment as `(b_seed, a_seed)`. Passing them the other way round produces a valid noise tensor
    // for the wrong side, and every downstream stage composes it perfectly.
    bufs.set_commitment(stream, &a_noise_seed, &b_noise_seed)?;

    pipeline::run_noise_gen(stream, fatbin, bufs)?;
    pipeline::run_noising(stream, triton, bufs)?;

    Ok(a_noise_seed)
}

/// The proof for the candidate that won: the two committed matrices, in Merkle-proof form.
///
/// The matrices come back whole. At production dimensions that is a quarter of a gigabyte over PCIe and
/// a host BLAKE3 tree over it, once per share.
///
/// It is not the right shape regardless: only the leaves the candidate's window covers are ever put on
/// the wire, and the device already builds the whole chunk-chaining-value tree to commit to the
/// matrices during `attempt`, so the siblings are a short walk from where the work already happens.
/// Until that kernel exists this is where a share's time goes.
///
/// The last step is the one that matters. Everything above searched over *this* attempt's matrices; the
/// proof has to commit to *these* matrices, or the pool recomputes the noise from a different pair of
/// roots, gets a different transcript, and rejects a share whose digest was real. Re-deriving `dnsA`
/// from the roots the proof actually carries and comparing it against the key the transcripts were
/// hashed under is what closes that gap — it fails loudly, rather than submitting something that looks
/// like a share.
fn assemble(
    backend: &CudaBackend,
    bufs: &SplitBuffers,
    key: &[u8; 32],
    derivation: SeedDerivation,
    hit: &Hit,
    jackpot_key: &[u8; 32],
) -> Result<PlainProof, String> {
    let config = bufs.config;
    let (m, n, k) = (config.m, config.n, config.k);
    let rank = config.rank;

    // The candidate names two rows inside its tile, and the whole 128-column strip. This is the mapping
    // gate D confirms on the device: the fold is a XOR over the window, so the order of the two rows is
    // unobservable, but which pair is not.
    let a_rows: Vec<usize> = hit.a_rows.to_vec();
    let b_cols: Vec<usize> = (hit.b_col..hit.b_col + TILE_N).collect();

    let a = matrix_proof(backend, &bufs.a, m, k, key, &a_rows)?;
    let bt = matrix_proof(backend, &bufs.bt, n, k, key, &b_cols)?;

    let (_, a_noise_seed) = noise_seeds(key, &a.proof.root, &bt.proof.root, m, n, derivation)?;
    if a_noise_seed != *jackpot_key {
        return Err(format!(
            "the proof for tile ({}, {}) candidate {} commits to roots that derive the jackpot key \
             {a_noise_seed:02x?} rather than the {jackpot_key:02x?} the search hashed under, so it does \
             not describe the candidate that won",
            hit.tile_m, hit.tile_n, hit.candidate % HASH_CANDIDATES
        ));
    }

    Ok(PlainProof {
        m,
        n,
        k,
        noise_rank: rank,
        a,
        bt,
        moe: None,
    })
}

/// One matrix's Merkle proof over the rows a candidate samples.
///
/// The device buffer is copied back whole and handed to the *unmodified* `pearl_blake3::MerkleTree`
/// rather than to something written here: the tree is what the verifier rebuilds the root from, and
/// a second implementation of it in this crate would be a second thing to be wrong.
///
/// No padding is added because none is needed. [`validate`] holds `m` and `n` to whole numbers of
/// 128-row tiles and `k` to a multiple of the 64-byte BLAKE3 message, so every matrix is already a
/// whole number of 1024-byte chunks and `pad_to_chunk_boundary` would be the identity. That is also why
/// the roots agree: the device hashes the buffer and this hashes the bytes that came back out of it,
/// and there is nothing in between for them to disagree about.
fn matrix_proof(
    backend: &CudaBackend,
    buffer: &CudaSlice<i8>,
    rows: usize,
    cols: usize,
    key: &[u8; 32],
    row_indices: &[usize],
) -> Result<MatrixMerkleProof, String> {
    let host = backend.read_i8(buffer)?;
    let bytes: Vec<u8> = host.iter().map(|&value| value as u8).collect();
    drop(host);

    let tree = pearl_blake3::MerkleTree::new(&bytes, *key);
    drop(bytes);

    let leaf_indices =
        pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows, cols));

    Ok(MatrixMerkleProof {
        proof: tree.get_multileaf_proof(&leaf_indices),
        row_indices: row_indices.to_vec(),
    })
}

/// The noise seeds a pair of committed roots implies, re-derived on the host.
///
/// The same chain `zk_pow`'s `PublicProofParams::commitment_hash` runs, so it is that function's
/// inputs that are being checked rather than a reimplementation of it: `bind_roots` is the
/// certificate version's domain separation, and the two digests chain one root to the other.
fn noise_seeds(
    key: &[u8; 32],
    hash_a: &[u8; 32],
    hash_b: &[u8; 32],
    m: usize,
    n: usize,
    derivation: SeedDerivation,
) -> Result<([u8; 32], [u8; 32]), String> {
    let (bound_a, bound_b) = derivation.bind_roots(hash_a, hash_b, m as u32, n as u32);

    let mut chained = Vec::with_capacity(64);
    chained.extend_from_slice(key);
    chained.extend_from_slice(&bound_b);
    let b_noise_seed = pearl_blake3::blake3_digest(&chained, None);

    chained.clear();
    chained.extend_from_slice(&b_noise_seed);
    chained.extend_from_slice(&bound_a);
    let a_noise_seed = pearl_blake3::blake3_digest(&chained, None);

    Ok((b_noise_seed, a_noise_seed))
}

/// The two Philox seeds for one attempt, derived from the job key and the attempt number.
///
/// A and B take different halves of one digest rather than sharing a seed: on the same one with
/// `m == n` the two matrices come out byte-identical, which is a valid candidate and looks like a
/// bug in the log. Deriving from the job key rather than a clock keeps an attempt reproducible from
/// its number, which is what makes a failure reportable at all.
fn fill_seeds(key: &[u8; 32], number: u64) -> (u64, u64) {
    let mut data = Vec::with_capacity(40);
    data.extend_from_slice(key);
    data.extend_from_slice(&number.to_be_bytes());

    let digest = pearl_blake3::blake3_digest(&data, None);
    (
        u64::from_be_bytes(digest[..8].try_into().unwrap()),
        u64::from_be_bytes(digest[8..16].try_into().unwrap()),
    )
}

/// Refuses a configuration the split path cannot run, and returns the shape its buffers are sized from.
///
/// Before the buffers are allocated, not after: every condition here is one that would otherwise
/// surface as a seven-gigabyte allocation followed by a launch error.
fn validate(mining: &PearlMining, config: &MiningConfiguration) -> Result<SplitConfig, String> {
    let (m, n, k) = (mining.m, mining.n, mining.k);
    let rank = mining.rank as usize;

    // The blob's tile shape and the rank it forces. Refused rather than assumed: a partial tile is not
    // a smaller grid, it is a grid whose last tile is never launched, and a rank that is not the
    // checkpoint group is a fold over a different grouping than the verifier's.
    let split = SplitConfig::new(m, n, k, rank)?;

    // The patterns have to be the window a candidate names, not merely a shape with the right cell
    // count: `assemble` reports `rows = {2c, 2c+1}` and `cols = 0..128`, and a pattern that came back as
    // the same cells in a different order would describe a different candidate to the verifier.
    let rows = config.rows_pattern.indices_with_offset(0);
    let cols = config.cols_pattern.indices_with_offset(0);

    if rows != vec![0, 1] {
        return Err(format!(
            "the split path reports a candidate as two consecutive rows, but the rows pattern is \
             {rows:?}"
        ));
    }
    if cols != (0..TILE_N as u32).collect::<Vec<u32>>() {
        return Err(format!(
            "the split path reports a candidate as the whole {TILE_N}-column strip the blob folds, but \
             the cols pattern is {cols:?}"
        ));
    }

    // The verifier's own constraint list, asked once up front rather than reimplemented here.
    //
    // These are not the sort of thing the search would notice going wrong. A `k` that is not a
    // multiple of BLAKE3's message length, a rank that is not a power of two, a `k` under `16r` or
    // under the 1024 the padding assumes — every one of those folds happily, commits happily, and
    // produces a share the verifier rejects at the far end. Checking them here is the difference
    // between a configuration that says so and a miner that runs for hours against a bound nothing
    // can clear. Reimplementing the list instead would be strictly worse: it would be a list that
    // can drift from the one the verifier enforces, which is the failure this whole check exists to
    // prevent.
    //
    // Nothing in that check reads a hash or a header, so the dummy roots and a zeroed header are not
    // placeholders in any meaningful sense — they are inputs it ignores. `t_rows`/`t_cols` are the tile
    // offsets, which run `0 .. m / TILE_M`; every one of them satisfies the `t + max < extent`
    // inequality below, so the zero stands for all of them.
    let params = PublicProofParams::new_dummy(
        IncompleteBlockHeader {
            version: 0,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0,
        },
        SeedDerivation::Legacy,
        *config,
        m as u32,
        n as u32,
        0,
        0,
    );
    public_params_sanity_check(&params).map_err(|e| {
        format!(
            "this mining configuration would not survive the verifier's own sanity check, so \
             every share it produced would be rejected: {e}"
        )
    })?;

    // `m` and `n` are `usize` here and `u32` everywhere past this point, so a value that does not
    // fit would be silently truncated before the sanity check ever saw it — and a truncated `m` of
    // 48 is exactly the `m` the rest of the check approves of. The dimensions are also what the
    // buffers are sized from, so the allocation and the proof would disagree about how big the
    // matrix is.
    for (name, extent) in [("m", m), ("n", n)] {
        if u32::try_from(extent).is_err() {
            return Err(format!(
                "{name} is {extent}, which does not fit in the 32 bits it is carried in"
            ));
        }
    }

    // The postpass and the scan take the candidate count as an `i32`, and the scan's reported index is
    // what `Hit::from_candidate` splits into a tile and a candidate. `m` and `n` fitting in `u32` is
    // not enough: the product `(m / 128) * (n / 128) * 64` is what overflows, and a wrapped index would
    // address the wrong candidate rather than fail. Checked here, before the buffers are sized from the
    // same dimensions.
    let candidates = split.total_candidates() as u64;
    if candidates > u32::MAX as u64 {
        return Err(format!(
            "the grid is {} x {} tiles = {candidates} candidates, which does not fit in the 32-bit \
             index the postpass and the scan are built from",
            m / TILE_M,
            n / TILE_N
        ));
    }

    Ok(split)
}

/// Device memory the search needs, in bytes.
///
/// Stated up front rather than discovered: the alternative is an allocation failing part-way
/// through the first job, with the reason buried in a driver string and the miner left unable to
/// start. The commitment's chunk-chaining-value levels are included because they are allocated and
/// freed inside every attempt, which makes them easy to forget — and they are the one part of the
/// footprint the split path does not own, so they have to be added from the outside.
fn footprint(config: &SplitConfig) -> usize {
    // Two levels of eight-word chaining values per 1024-byte chunk, live at the same time — one level
    // being written while the previous is combined. The commitment builds a tree over each matrix in
    // turn, so the larger of the two sizes the scratch, not `a` alone: at `m < n` the `bt` tree is the
    // bigger one, and a footprint sized from `a` under-counts it by the difference.
    let larger = (config.m.max(config.n) * config.k).div_ceil(CHUNK);
    let cv_scratch = 2 * larger * CV_WORDS * std::mem::size_of::<u32>();

    config.footprint() + cv_scratch
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};

    // Used only by the measurement test below, so they belong here rather than at the top of the file:
    // the non-test build has no use for them and warns about it.
    use crate::miners::gpu::triton::{JACKPOT_SIZE, NOISING_SHARED_BYTES, SEARCH_NOROTL_SHARED_BYTES};
    use crate::miners::gpu::cuda::philox_stream;
    use crate::miners::pearl_pow::verify_share_locally;
    use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
    use zk_pow::circuit::chip::compute_jackpot;
    use zk_pow::circuit::pearl_noise::compute_noise_for_indices;

    /// Certificate version 3, because salted is the current default on the wire. The version is
    /// what picks the derivation, and a search that used the wrong one would still be perfectly
    /// self-consistent — every digest real, every one of them a candidate the pool rejects.
    const CERT_VERSION: u32 = 3;

    /// `validate` hands the configuration to the verifier's own constraint list, so what is being
    /// checked here is that the delegation is wired up and reached before any device work.
    ///
    /// The refused configurations are real ones rather than hypothetical: `k = 256` at `rank = 32` is
    /// under the verifier's `k >= 16r`, and rank 100 is not a power of two. Both were found, before the
    /// delegation was added, not by `validate` but by `verify_share_locally` at the far end of a search
    /// — after the whole pipeline had produced a proof the pool would reject.
    ///
    /// The third refusal is the one only the split path has: a shape that satisfies every verifier
    /// constraint and still cannot run, because the blob truncates a partial tile.
    #[test]
    fn a_configuration_the_verifier_would_refuse_is_refused_before_any_device_work() {
        let narrow = PearlMining {
            k: 256,
            ..test_mining()
        };
        let error = validate(&narrow, &mining_configuration(&narrow).unwrap())
            .expect_err("the verifier requires k >= 16r, and 256 < 16 * 32");
        assert!(
            error.contains("16r"),
            "the message should name the constraint: {error}"
        );

        // The rank has to be one the verifier generates noise for, which is a narrower band than
        // "divides k": `k = 1024` and `rank = 100` satisfy everything the search itself checks.
        let odd_rank = PearlMining {
            rank: 100,
            ..test_mining()
        };
        assert!(
            validate(&odd_rank, &mining_configuration(&odd_rank).unwrap()).is_err(),
            "rank 100 is not a power of two and the verifier would refuse it"
        );

        // A shape the verifier accepts and the blob cannot run: 64 columns off a 128-wide tile is a
        // grid whose last tile is never launched, so the candidates inside it are never hashed. Nothing
        // in the verifier's list says so — `h * w <= 256` is satisfied by a 2x64 pattern.
        let partial = PearlMining {
            n: 192,
            cols_pattern: (0..64).collect(),
            ..test_mining()
        };
        let error = validate(&partial, &mining_configuration(&partial).unwrap())
            .expect_err("n = 192 is not a whole number of 128-column tiles");
        assert!(
            error.contains("truncates"),
            "the message should name the mechanism, not just the shape: {error}"
        );

        // And the configuration everything else here runs on does pass.
        let good = test_mining();
        validate(&good, &mining_configuration(&good).unwrap())
            .expect("the test configuration is one the verifier accepts");
    }

    /// The shipped configuration, opened on the real device and actually searched.
    ///
    /// Every other test here runs at two tiles, which is enough to pin the arithmetic and far too small
    /// to say anything about whether the miner can start: the buffers are sized from `m`, `n` and `k`,
    /// the grid is `(m/128) * (n/128)` tiles, and the attempt allocates about seven gigabytes. None of
    /// that is exercised by a small grid, and a configuration that cannot be allocated is not a subtle
    /// failure — it is the engine refusing to start at all.
    ///
    /// Ignored by default because it wants a device with roughly 7.5 GiB spare and a few seconds of it.
    /// Run with `--ignored`.
    #[test]
    #[ignore = "allocates the full production buffers on the device; run with --ignored"]
    fn the_shipped_default_runs_a_search_on_the_device() {
        let Ok(mut miner) = GpuMiner::open(PearlMining::default()) else {
            // No device, or not one with the memory for it: there is nothing here to run.
            return;
        };

        // A bound every digest clears, so the search returns on its first attempt instead of waiting
        // for a share. The proof it produces still has to be a real one at the real dimensions.
        let started = Instant::now();
        let proof = miner
            .search(&test_header(), CERT_VERSION, U256::MAX, &|| false)
            .expect("the search runs")
            .expect("a bound everything clears is claimed by the first attempt");

        assert_eq!(proof.m, 131_072);
        assert_eq!(proof.n, 131_072);
        assert_eq!(proof.k, 2048);
        assert_eq!(proof.noise_rank, 128);
        // A candidate is two rows of A and the whole 128-column strip of B, and the proof has to say
        // which candidate it found.
        assert_eq!(proof.a.row_indices.len(), 2);
        assert_eq!(proof.bt.row_indices.len(), 128);
        eprintln!(
            "first attempt: {:?}, {} candidates — setup and one grid are the bulk of that, the rest is \
             `assemble`: 2 x {} MiB back over PCIe and both Merkle proofs on the host",
            started.elapsed(),
            miner.tiles(),
            miner.mining.m * miner.mining.k / (1 << 20)
        );

        // And what an attempt is made of, because the number above is an average over a window that
        // contains one whole attempt's setup plus one whole grid. Tuning the wrong one is how a miner
        // spends a week on the fold and keeps the same wall clock.
        //
        // Every stage, not two of them: an earlier breakdown of this measured only the fill and the fold
        // and reported setup at 1%, on a decomposition that silently omitted the commitment chain — the
        // one step that reads a gigabyte and the one step that has to hash it. A breakdown that adds up
        // to less than the thing it decomposes is not a breakdown.
        //
        // Measured here rather than in `attempt` itself, so production carries no timing code. The seeds
        // are wrong and the noise is whatever the last attempt left behind, which is fine: the question
        // is how long each pass takes, not what they produce.
        let GpuMiner {
            backend,
            stream,
            fatbin,
            triton,
            bufs,
            mining,
            ..
        } = &mut miner;
        let (m, n, k) = (mining.m, mining.n, mining.k);
        let rank = mining.rank as usize;

        let started = Instant::now();
        backend.fill_i8(&mut bufs.a, 0, bufs.a_len, 1).expect("the fill runs");
        backend.fill_i8(&mut bufs.bt, 0, bufs.bt_len, 2).expect("the fill runs");
        stream.synchronize().expect("the device drains");
        let fill = started.elapsed();

        let started = Instant::now();
        let (b_noise_seed, a_noise_seed) = backend
            .commitment_seeds(
                &bufs.a,
                &bufs.bt,
                [0x5a; 32],
                m as u32,
                n as u32,
                seed_derivation_for(CERT_VERSION).expect("the certificate version is one we know"),
            )
            .expect("the commitment runs");
        stream.synchronize().expect("the device drains");
        let commit = started.elapsed();

        bufs.set_commitment(stream, &a_noise_seed, &b_noise_seed)
            .expect("the commitment reaches the device");

        let started = Instant::now();
        pipeline::run_noise_gen(stream, fatbin, bufs).expect("the noise draw runs");
        stream.synchronize().expect("the device drains");
        let draw = started.elapsed();

        let started = Instant::now();
        pipeline::run_noising(stream, triton, bufs).expect("the noising runs");
        stream.synchronize().expect("the device drains");
        let noising = started.elapsed();

        // The reset and the fold are timed apart, because together they hide the largest single term in
        // the attempt. `reset_for_attempt` zeroes the transcript, which at this geometry is
        // `tiles * candidates * 16` words -- 4.3 GB, written for a fold the blob then overwrites in
        // full. Lumping it into "fold" is how the reset stays invisible.
        let started = Instant::now();
        bufs.reset_for_attempt(stream).expect("the buffers are reset");
        stream.synchronize().expect("the device drains");
        let reset = started.elapsed();

        let started = Instant::now();
        pipeline::run_search(stream, triton, bufs).expect("the search runs");
        stream.synchronize().expect("the device drains");
        let fold_cold = started.elapsed();

        // And once more, warm, because the first launch of a blob in a process pays its module's first
        // touch. The two numbers are different terms: one is a fixed cost per process, the other is what
        // a steady-state attempt actually pays.
        let started = Instant::now();
        pipeline::run_search(stream, triton, bufs).expect("the search runs");
        stream.synchronize().expect("the device drains");
        let folding = started.elapsed();

        // The postpass is the work the fused kernel did inside the fold, so it is the term this
        // architecture moves out rather than removes: it has to be timed on its own before the split
        // can be called a win or a loss.
        let started = Instant::now();
        pipeline::run_postpass(stream, fatbin, bufs).expect("the postpass runs");
        pipeline::run_scan(stream, fatbin, bufs).expect("the scan runs");
        stream.synchronize().expect("the device drains");
        let hashing = started.elapsed();

        let setup = fill + commit + draw + noising;
        eprintln!(
            "per attempt: fill 2 x {} MiB {fill:?}, commitment over 2 x {} MiB {commit:?}, noise \
             draw {draw:?}, noising 2 x {} MiB {noising:?} -- setup {setup:?}; transcript zero {reset:?} \
             over {} MiB, fold cold {fold_cold:?} then warm {folding:?} ({:.1} TMAC/s), postpass + scan \
             {hashing:?}",
            m * k / (1 << 20),
            m * k / (1 << 20),
            n * k / (1 << 20),
            bufs.config.total_candidates() * JACKPOT_SIZE * 4 / (1 << 20),
            macs_per_attempt(&bufs.config, TILE_M, TILE_N) as f64 / folding.as_secs_f64() / 1e12,
        );

        // And what a share's assembly is made of, since it costs several times a whole grid's fold.
        // Three steps, because they have three different fixes: the readback is bandwidth, the
        // `as u8` conversion is a second host copy of the same gigabyte that nothing needs, and the
        // tree is the hashing itself. Timed on one matrix and doubled for the pair.
        let started = Instant::now();
        let host = backend.read_i8(&bufs.a).expect("the matrix comes back");
        stream.synchronize().expect("the device drains");
        let read = started.elapsed();

        let started = Instant::now();
        let bytes: Vec<u8> = host.iter().map(|&v| v as u8).collect();
        let convert = started.elapsed();

        let started = Instant::now();
        let tree = pearl_blake3::MerkleTree::new(&bytes, [0u8; 32]);
        let hash = started.elapsed();
        eprintln!(
            "per share, per matrix: read {read:?}, `as u8` {convert:?}, Merkle tree {hash:?} \
             (root {:02x?}, {} leaves)",
            tree.root(),
            (m * k).div_ceil(1024),
        );

        // And what the rest of the submit path costs, because every one of these sits between two
        // searches with no candidates folded during it. The reported hashrate divides candidates by wall
        // clock, so all of it is charged to the denominator; a share that takes three seconds to
        // verify is three seconds of not mining.
        //
        // Verified against `U256::MAX` because that is the bound the proof above was found under: the
        // proof is real, so this measures the verifier's actual cost rather than its failure path,
        // which is much cheaper and would flatter the number.
        let started = Instant::now();
        verify_share_locally(&test_header(), CERT_VERSION, &proof, U256::MAX)
            .expect("the proof the first attempt produced verifies");
        let verify = started.elapsed();

        let started = Instant::now();
        let encoded = crate::miners::pearl_mining::encode_plain_proof(&proof, true)
            .expect("the proof encodes");
        let encode = started.elapsed();
        eprintln!(
            "per share: verify {verify:?}, gzip+bincode encode {encode:?} ({} chars) -- \
             assembly and submission, none of which folds a candidate",
            encoded.len(),
        );
    }

    /// MACs one attempt folds: one tile's `tile_m x tile_n` output over `k` columns, times the tiles.
    ///
    /// Not `total_candidates() x tile x k`. The 64 candidates are row pairs *inside* one tile, so the
    /// work of an attempt is the tile's work counted once. Counting candidates as if each were its own
    /// tile inflates every rate by 64 — which is exactly what the first version of this measurement did,
    /// and it reported 11,579,901,293 TMAC/s on a device whose tensor peak is a few hundred.
    fn macs_per_attempt(config: &SplitConfig, tile_m: usize, tile_n: usize) -> usize {
        config.num_tiles() * tile_m * tile_n * config.k
    }

    /// What actually limits the fold, read from the driver rather than inferred from the PTX.
    ///
    /// Two questions, because they point at different knobs and the wrong guess costs a week:
    ///
    /// **How many blocks fit on an SM.** The blob keeps 128 accumulators live per thread — 32 atoms of
    /// `mma.m16n8k32`, which is the 4x8 warp tile. The PTX declares *virtual* registers (`%r<543>`), so
    /// the physical count is not readable from the file; it has to come from the driver. If the physical
    /// count is above half the per-SM register file, one block per SM is forced and the fold runs at four
    /// warps per SM with nothing to hide its latency behind.
    ///
    /// **Whether the fold is fed by DRAM or by its own issue rate.** At production geometry every block
    /// reads a whole 128x2048 A tile and a whole 128x2048 B tile, so the traffic floor is the whole of
    /// `A` plus the whole of `B` divided by the swizzle group's reuse. A grid small enough to sit in L2
    /// has no DRAM traffic after its first pass: if that grid runs much faster than the production grid,
    /// the production number is a bandwidth number and no codegen inside the kernel changes it. If the
    /// two rates are the same, the fold is issue- or latency-bound, and occupancy is the lever.
    ///
    /// The fold here is timed *without* `reset_for_attempt`, because the recorded 224.3 ms lumps the two
    /// together and the reset is a 4.3 GB memset of its own.
    #[test]
    #[ignore = "reads driver attributes and runs the fold on the device; run with --ignored"]
    fn what_limits_the_fold() {
        let Ok(mut miner) = GpuMiner::open(PearlMining::default()) else {
            return;
        };

        let GpuMiner { backend, stream, triton, bufs, mining, .. } = &mut miner;
        let ctx = backend.context();
        let device = |attribute| ctx.attribute(attribute).unwrap_or(-1);

        let regs_per_sm = device(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
        let smem_per_sm = device(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
        let threads_per_sm = device(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        let l2 = device(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        let sms = device(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);

        // The block size the launch actually uses. The driver reports `MAX_THREADS_PER_BLOCK = 256` for
        // both blobs, which is not the block a launch runs: both PTX files declare `.reqntid 128`, and a
        // launch with 256 is rejected. Occupancy has to be computed from the block that runs, so the
        // driver's number is reported alongside rather than used.
        for (name, func, requested_smem, block) in [
            ("search", triton.search.function(), SEARCH_NOROTL_SHARED_BYTES as i32, 128i32),
            ("noising", triton.noising.function(), NOISING_SHARED_BYTES as i32, 128i32),
        ] {
            let regs = func.get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS).unwrap_or(-1);
            let static_smem = func
                .get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
                .unwrap_or(-1);
            let spill = func
                .get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
                .unwrap_or(-1);
            let threads = func
                .get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                .unwrap_or(-1);
            if threads <= 0 || regs <= 0 {
                continue;
            }

            let by_regs = regs_per_sm / (block * regs);
            let by_smem = smem_per_sm / (static_smem + requested_smem);
            let by_threads = threads_per_sm / block;
            let blocks = by_regs.min(by_smem).min(by_threads);
            eprintln!(
                "{name}: {regs} regs/thread, {static_smem} B static smem, {spill} B local (spill), \
                 {requested_smem} B dynamic requested, block {block} (driver says {threads}) -> \
                 {by_regs} blocks by registers, {by_smem} by smem, {by_threads} by threads = \
                 {blocks} block(s)/SM = {} warps/SM",
                blocks * block / 32
            );
        }
        eprintln!(
            "device: {sms} SMs, {regs_per_sm} regs/SM, {smem_per_sm} B smem/SM, \
             {threads_per_sm} threads/SM, L2 {l2} B"
        );

        // The fold alone, at the shipped geometry.
        let started = Instant::now();
        pipeline::run_search(stream, triton, bufs).expect("the fold runs");
        stream.synchronize().expect("the device drains");
        let production = started.elapsed();
        eprintln!(
            "fold alone, production grid: {production:?} = {:.1} TMAC/s ({} tiles)",
            macs_per_attempt(&bufs.config, TILE_M, TILE_N) as f64 / production.as_secs_f64() / 1e12,
            bufs.config.num_tiles()
        );

        // The same kernel on grids small enough that their working set is L2-resident, run repeatedly so
        // the passes after the first read nothing from DRAM. The buffers hold zeros, which is fine: the
        // question is how fast the loop issues, not what it produces.
        //
        // Three sizes, because the first version of this probe changed two variables at once and read the
        // result as one. A grid of 64 blocks runs one block per SM on 64 SMs; a grid of 168 blocks runs
        // two per SM on all 84. Those are different occupancy, and occupancy changes the rate as much as
        // the data source does — so the sizes below are chosen to change one at a time:
        //
        //   64 blocks, L2-resident, 1 block/SM
        //   84 blocks, L2-resident, 1 block/SM on every SM
        //  168 blocks, NOT L2-resident, 2 blocks/SM
        //
        // The last one is the shape the production grid actually runs at, so comparing it against the
        // first two says how much of the production rate is occupancy and how much is the data source.
        let working_set = (TILE_M + TILE_N) * mining.k;
        // `1` is the dispatch baseline: one block is one tile of work, so the queue time at that size is
        // the fixed cost per launch, and the other rows can be read as work plus that constant.
        for tiles in [1usize, 64, 84, 168] {
            if tiles * working_set > l2.max(1) as usize {
                eprintln!(
                    "{tiles} tiles: working set {} B exceeds L2 {l2} B, so this row is DRAM-fed",
                    tiles * working_set
                );
            }
            let small = SplitConfig::new(tiles * TILE_M, TILE_N, mining.k, mining.rank as usize)
                .expect("a row of whole tiles is a configuration the split path accepts");
            let small_bufs = SplitBuffers::new(stream, &small).expect("the small grid allocates");

            // Host cost and device cost separately. Queueing a launch costs the host a fixed amount, and
            // at these grid sizes that amount is the same order as the work: one block is 33.5 M MACs,
            // which at the rate this kernel reaches is around 20 us of device time. So a number read
            // straight off a 20-launch wall clock is mostly host time, and the first version of this
            // probe reported it as device throughput.
            //
            // Warm up first: the first launch on a freshly allocated grid pays the module's first touch,
            // and that is not a property of the grid.
            pipeline::run_search(stream, triton, &small_bufs).expect("the fold runs");
            stream.synchronize().expect("the warm-up drains");

            let started = Instant::now();
            for _ in 0..20 {
                pipeline::run_search(stream, triton, &small_bufs).expect("the fold runs");
            }
            let host = started.elapsed().as_secs_f64() / 20.0;
            stream.synchronize().expect("the queue drains");
            let total = started.elapsed().as_secs_f64() / 20.0;
            let device = total - host;

            let macs = macs_per_attempt(&small, TILE_M, TILE_N) as f64;
            // A grid of one block is busy on one SM, not on 84. Dividing by the device count would report
            // 0.02 TMAC/s for a kernel running at its normal per-SM rate.
            let busy_sms = tiles.min(sms.max(1) as usize);
            eprintln!(
                "  {tiles} blocks: host {:.1} us, device {:.1} us -> {:.2} TMAC/s per busy SM \
                 ({} block(s) per SM, {})",
                host * 1e6,
                device * 1e6,
                macs / device.max(1e-9) / 1e12 / busy_sms as f64,
                tiles.max(1) / busy_sms,
                if tiles * working_set > l2.max(1) as usize { "DRAM-fed" } else { "L2-fed" }
            );
        }
    }

    /// The whole pipeline, against the verifier's own implementation of every step it performs.
    ///
    /// Each stage is already checked on its own, and none of that adds up to this. What is untested
    /// otherwise is the *order* and the *mapping*: that the commitment is taken over the matrix the
    /// search is about to fold, that the noise derived from that commitment is composed into the
    /// operands the fold reads, that the scan's one number resolves to the candidate the host priced,
    /// and that the proof handed back commits to the very matrices that were folded. Every one of those
    /// is a mistake each stage would accept — and two of them (a swapped seed, a swapped operand) are
    /// mistakes that produce perfectly valid proofs of the wrong thing.
    ///
    /// So the reference here re-derives the whole attempt on the host — the same Philox stream, the
    /// same `pearl_blake3` roots, the verifier's own `commitment_hash`, its own `pearl_noise` tensors,
    /// and `compute_jackpot` for the transcript — and the bound handed to the search is exactly the
    /// lowest digest that produces, so only one candidate on the grid can clear it and there is no race
    /// about which. The search then has to return a proof for that candidate, and the pool's own
    /// verifier has to accept it at the bound the search was given.
    ///
    /// The winning candidate is required to be neither the first slot nor in the first tile, so the
    /// tile-and-candidate split is part of what is being checked rather than a zero this grid could
    /// have got right by accident.
    #[test]
    fn a_share_the_gpu_finds_is_one_the_verifier_accepts() {
        let Ok(mut miner) = GpuMiner::open(test_mining()) else {
            // No device: there is nothing to run the loop on, and nothing it could prove here.
            return;
        };

        let (mining, header, config) = test_job();
        let split = SplitConfig::new(mining.m, mining.n, mining.k, mining.rank as usize).unwrap();
        let derivation = seed_derivation_for(CERT_VERSION).expect("a known certificate version");
        let key = job_key(&header, &config);
        assert_eq!(
            key,
            public_params(&header, &config, derivation, &[0; 32], &[0; 32]).job_key(),
            "the search's job key is not the one the verifier derives"
        );

        // Attempts are independent draws, so this is just waiting for one whose cheapest candidate is
        // not the one every indexing gets right by accident.
        let (number, hit, bound) = (0..24u64)
            .find_map(|number| {
                let (hit, digest) = lowest_candidate(&split, &header, &config, derivation, key, number)?;
                (hit.candidate % HASH_CANDIDATES > 0 && hit.tile_linear > 0)
                    .then_some((number, hit, digest))
            })
            .expect("no attempt in twenty-four put its cheapest candidate past the first slot of the first tile");

        miner.attempts = number;
        let proof = miner
            .search(&header, CERT_VERSION, bound, &|| false)
            .expect("the search runs")
            .expect("the bound is the lowest digest on the grid, so a candidate clears it");

        // The proof is for the candidate the host priced. The coordinates only ever reach the proof
        // through here — nothing downstream reads them — so a proof describing a different candidate
        // than the one that won would still verify.
        assert_eq!(proof.a.row_indices, hit.a_rows.to_vec(), "A rows");
        assert_eq!(
            proof.bt.row_indices,
            (hit.b_col..hit.b_col + TILE_N).collect::<Vec<usize>>(),
            "B columns"
        );
        assert_eq!((proof.m, proof.n, proof.k), (mining.m, mining.n, mining.k));
        assert_eq!(proof.noise_rank, mining.rank as usize);

        // And the pool's own verifier accepts it at the bound the search was given. This is the
        // gate every real share passes through, and it is a full re-verification: the rows are
        // pulled back out of the Merkle proofs, the noise is re-derived from the roots the proof
        // commits to, and the transcript is recomputed.
        assert_verifies(&header, &config, &proof, bound);

        // A second call has to draw a second attempt rather than re-searching the one just handed
        // over. The bound names the lowest digest of *that* attempt, so the two proofs cannot be
        // the same share twice.
        let (next_hit, next_bound) =
            lowest_candidate(&split, &header, &config, derivation, key, number + 1)
                .expect("the candidate grid produced nothing");
        let next = miner
            .search(&header, CERT_VERSION, next_bound, &|| false)
            .expect("the second search runs")
            .expect("the second attempt has a candidate under its bound");

        assert_ne!(
            next.a.proof.root, proof.a.proof.root,
            "the second search returned the first attempt's matrices again"
        );
        assert_eq!(next.a.row_indices, next_hit.a_rows.to_vec());
        assert_verifies(&header, &config, &next, next_bound);
    }

    /// Runs the pool's own check over a proof, at a bound the search reported a candidate against.
    ///
    /// The bound and the share target are the same number scaled apart by the verifier's difficulty
    /// factor, so the target is recovered by dividing it back out. Rounding the other way is fine:
    /// the target only has to re-encode to a bound at least this wide, and `share_bound`'s own test
    /// is what pins that.
    fn assert_verifies(
        header: &IncompleteBlockHeader,
        config: &MiningConfiguration,
        proof: &PlainProof,
        bound: U256,
    ) {
        let factor = config.rows_pattern.size() as usize
            * config.cols_pattern.size() as usize
            * config.dot_product_length();
        let target = bound / U256::from(factor as u64);

        verify_share_locally(header, CERT_VERSION, proof, target).unwrap_or_else(|error| {
            panic!(
                "the verifier rejected a proof whose candidate the search priced at {bound:#x}: {error}"
            )
        });
    }

    /// The candidate of `split`'s grid whose jackpot digest is lowest for one attempt, and that digest.
    ///
    /// `key` is the job key and `number` the attempt — the two things that make an attempt's matrices
    /// what they are. This is the verifier's own chain at every step, the commitment included:
    /// `public.commitment_hash` rather than a re-derivation of it, so the reference cannot drift from
    /// what a submitted proof is checked against.
    ///
    /// It is a second reading of the same algorithm, which is why it is not the whole story: a CPU model
    /// that mirrors the kernel agrees with the kernel by construction. The reference that is not a second
    /// reading is a captured third-party share, and this tree has no fixture to replay it against
    /// (`tests/fixtures/` is empty), so this is the strongest thing available here — and it is still a
    /// second reading. What it can catch is a disagreement with the verifier; what it cannot catch is an
    /// algorithm that both sides read the same way.
    fn lowest_candidate(
        split: &SplitConfig,
        header: &IncompleteBlockHeader,
        config: &MiningConfiguration,
        derivation: SeedDerivation,
        key: [u8; 32],
        number: u64,
    ) -> Option<(Hit, U256)> {
        let (m, n, k) = (split.m, split.n, split.k);
        let rank = split.rank;

        let (seed_a, seed_b) = fill_seeds(&key, number);
        let a = philox_stream(0, m * k, seed_a);
        let bt = philox_stream(0, n * k, seed_b);

        // The commitment is over the chunk-padded matrix, which is what the device's buffers carry.
        let padded = |matrix: &[i8]| -> Vec<u8> {
            let bytes: Vec<u8> = matrix.iter().map(|&v| v as u8).collect();
            pearl_blake3::pad_to_chunk_boundary(&bytes)
        };
        let raw_a = pearl_blake3::blake3_digest(&padded(&a), Some(key));
        let raw_b = pearl_blake3::blake3_digest(&padded(&bt), Some(key));

        let public = public_params(header, config, derivation, &raw_a, &raw_b);
        let (b_noise_seed, a_noise_seed) = public.commitment_hash(key);
        let compiled = CompiledPublicParams::from(&public);
        assert_eq!(
            compiled.a_noise_seed(),
            a_noise_seed,
            "the compiled parameters disagree with the commitment chain about the jackpot key"
        );

        let tiles_n = n / TILE_N;
        let mut lowest: Option<(Hit, U256)> = None;

        for tile_linear in 0..split.num_tiles() {
            let tile_m = tile_linear / tiles_n;
            let tile_n = tile_linear % tiles_n;

            for cand in 0..HASH_CANDIDATES {
                let rows = [tile_m * TILE_M + 2 * cand, tile_m * TILE_M + 2 * cand + 1];
                let cols: Vec<usize> = (tile_n * TILE_N..tile_n * TILE_N + TILE_N).collect();

                let window = |matrix: &[i8], rows: &[usize]| -> Vec<Vec<i8>> {
                    rows.iter()
                        .map(|row| matrix[row * k..(row + 1) * k].to_vec())
                        .collect()
                };

                let noise =
                    compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &rows, &cols);
                let transcript =
                    compute_jackpot(&compiled, &window(&a, &rows), &window(&bt, &cols), &noise);
                // Little-endian, because that is the reading the verifier compares against
                // (`check_jackpot_against_nbits`) and the one the postpass walks the bound against.
                let digest =
                    U256::from_little_endian(&compute_jackpot_hash(&transcript, a_noise_seed));

                if lowest.as_ref().is_none_or(|(_, best)| digest < *best) {
                    lowest = Some((
                        Hit::from_candidate(split, tile_linear * HASH_CANDIDATES + cand),
                        digest,
                    ));
                }
            }
        }

        lowest
    }

    /// A configuration small enough to price every candidate on the host, with the production tile shape.
    ///
    /// `rank` is a power of two and a multiple of the 32 bytes one noise hash produces, which is what
    /// `validate_rank` insists on — so not 16, however convenient. The dimensions are otherwise the
    /// smallest that still give several tiles in both directions: a one-tile grid would pass against any
    /// indexing at all, and a single-candidate grid would pass against a scan that always reported
    /// candidate zero.
    ///
    /// Two tiles, one in each direction, at the shape the blobs force: `k = 16r` with `r = 128`.
    fn test_mining() -> PearlMining {
        PearlMining {
            m: 128,
            n: 256,
            k: 2048,
            rank: 128,
            rows_pattern: vec![0, 1],
            cols_pattern: (0..128).collect(),
            gzip: false,
        }
    }

    fn test_header() -> IncompleteBlockHeader {
        IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x11; 32],
            merkle_root: [0x22; 32],
            timestamp: 1_700_000_000,
            nbits: 0x207f_ffff,
        }
    }

    fn test_job() -> (PearlMining, IncompleteBlockHeader, MiningConfiguration) {
        let mining = test_mining();
        let header = test_header();
        let config = mining_configuration(&mining).expect("the test configuration is well formed");
        (mining, header, config)
    }

    /// The public parameters a proof at this job carries, with the two committed roots filled in.
    ///
    /// `new_dummy` supplies the header, the configuration and the tile extents — everything
    /// `job_key` and `commitment_hash` read besides the roots themselves.
    fn public_params(
        header: &IncompleteBlockHeader,
        config: &MiningConfiguration,
        derivation: SeedDerivation,
        hash_a: &[u8; 32],
        hash_b: &[u8; 32],
    ) -> PublicProofParams {
        let mining = test_mining();
        let mut public = PublicProofParams::new_dummy(
            *header,
            derivation,
            *config,
            mining.m as u32,
            mining.n as u32,
            0,
            0,
        );
        public.hash_a = *hash_a;
        public.hash_b = *hash_b;
        public
    }
}
