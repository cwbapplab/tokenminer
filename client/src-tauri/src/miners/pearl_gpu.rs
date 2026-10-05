//! The GPU Pearl search.
//!
//! This is the replacement for the old CPU search. The pipeline it drives, entirely on the device:
//!
//! 1. fill `A[M,K]` and `Bt[N,K]` from a Philox counter-based stream
//! 2. commit to them — a chunk-padded BLAKE3 Merkle root each, then the noise seeds
//! 3. generate the four noise tensors (`EAL`, `EAR`, `EBL`, `EBR`)
//! 4. noise-apply them into `a_sum` and `b_sum`, the summed operands the GEMM consumes
//! 5. fold `a_sum ⊗ b_sumᵀ` into a 16-word transcript per 16x16 candidate tile
//! 6. hash each transcript under `dnsA` and compare against the share bound, on device
//!
//! Steps 5 and 6 are one kernel — `tokenminer_search_grid` folds, hashes, compares and claims in
//! place, one block per tile — and the loop here batches those blocks into regions so there is one
//! host sync per region rather than one per tile. Only when a tile wins do the winning matrices
//! come back to the host, where [`assemble`] turns them into the `PlainProof` that goes on the
//! wire.
//!
//! The reference for the shape of all of this is Open-Pearl-Miner's `csrc/`, and the shape of the
//! arithmetic is `zk_pow::circuit::chip::compute_jackpot`.

use std::time::{Duration, Instant};

use cudarc::driver::CudaSlice;
use primitive_types::U256;
use zk_pow::api::proof::{
    IncompleteBlockHeader, MiningConfiguration, PublicProofParams, SeedDerivation,
};
use zk_pow::api::sanity_checks::public_params_sanity_check;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof};

use super::gpu::cuda::CudaBackend;
use super::gpu::GpuBackend;
use super::pearl_mining::{job_key, mining_configuration, seed_derivation_for, PearlMining};

/// How long a search keeps going before handing control back to the caller.
///
/// A job is superseded by the next `mining.notify`, and the caller polls `stop` between batches, so
/// this only needs to be short enough that a new job is picked up promptly.
const MAX_SECONDS_PER_SEARCH: u64 = 5;

/// A candidate tile is 16 rows of `A` against 16 rows of `Bt`, which is the `wmma` fragment shape
/// and what the grid kernel indexes with.
const TILE: usize = 16;

/// Regions in flight per host sync.
///
/// One region is one row of the tile grid, so a batch is a contiguous run of `REGIONS_PER_BATCH`
/// tile rows and a winner's reported coordinates are its row and column outright — no offsetting on
/// the way back. The trade is latency: a batch has to finish before the flags are readable, and at
/// production dimensions a batch is 524,288 tiles.
const REGIONS_PER_BATCH: usize = 64;

/// BLAKE3's chunk length. A matrix is padded to a whole number of these before it is committed to.
const CHUNK: usize = 1024;

/// Words in one BLAKE3 chaining value, which is the unit the commitment's tree level is held in.
const CV_WORDS: usize = 8;

/// Headroom over the search's own buffers, for the CUDA context, the loaded module and the driver's
/// own per-context allocations.
const DRIVER_SLACK: usize = 128 * 1024 * 1024;

/// The device buffers one configuration needs, allocated once and reused for every attempt.
///
/// The two signal matrices and the two summed ones are separate rather than written in place,
/// because `A` and `Bt` are what the proof commits to: once they have been noised in place the
/// signal is gone, and recovering it would mean an exact-inverse kernel that has to be right for
/// the *next* attempt's commitment, not just for this one's share.
struct Workspace {
    /// `A[M,K]` and `Bt[N,K]`, each padded up to a whole number of BLAKE3 chunks.
    ///
    /// Padded in place rather than in a second buffer so the commitment hashes the same bytes the
    /// proof will. The tail past `a_len` / `bt_len` is zero and stays that way: `alloc_zeros` wrote
    /// it, only `fill_i8` ever writes these buffers, and it is only ever asked for `a_len` elements.
    a: CudaSlice<i8>,
    bt: CudaSlice<i8>,
    /// The matrix extents within the two buffers above, i.e. excluding the padding.
    a_len: usize,
    bt_len: usize,
    /// `EAL[M,R]` and `EBR[N,R]`: the dense noise the sparse pairs index into.
    e_al: CudaSlice<i8>,
    e_br: CudaSlice<i8>,
    /// `EAR` and `EBL`, one column pair per index over `K`.
    ear: CudaSlice<u32>,
    e_bl: CudaSlice<u32>,
    /// `signal + noise`, which is what the fold reads.
    a_sum: CudaSlice<i8>,
    b_sum: CudaSlice<i8>,
    /// One flag word per region in flight and two coordinate words, cleared by each `search_grid`.
    found: CudaSlice<u32>,
    coord: CudaSlice<u32>,
}

impl Workspace {
    fn new(
        backend: &CudaBackend,
        m: usize,
        n: usize,
        k: usize,
        rank: usize,
        regions: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            a: backend.alloc_i8(padded(m * k))?,
            bt: backend.alloc_i8(padded(n * k))?,
            a_len: m * k,
            bt_len: n * k,
            e_al: backend.alloc_i8(m * rank)?,
            e_br: backend.alloc_i8(n * rank)?,
            ear: backend.alloc_u32(k * 2)?,
            e_bl: backend.alloc_u32(k * 2)?,
            a_sum: backend.alloc_i8(m * k)?,
            b_sum: backend.alloc_i8(n * k)?,
            found: backend.alloc_u32(regions)?,
            coord: backend.alloc_u32(regions * 2)?,
        })
    }
}

/// Owns the device-side state for one mining configuration and the search loop that drives it.
pub struct GpuMiner {
    backend: CudaBackend,
    mining: PearlMining,
    workspace: Workspace,
    /// Tile rows per host sync, and the most a single search batch can have in flight.
    ///
    /// A field rather than a constant only so the tests can shrink the grid to a size where the
    /// batching is visible; [`REGIONS_PER_BATCH`] is what ships.
    regions_per_batch: usize,
    /// Tiles evaluated since the engine started, for the hashrate the UI reports.
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
    /// launch, or by an allocation dying two gigabytes into the first job.
    pub fn open(mining: PearlMining) -> Result<Self, String> {
        let mut backend = CudaBackend::open(0)?;
        backend
            .self_test()
            .map_err(|error| format!("{} failed its self-test: {error}", backend.name()))?;

        let config = mining_configuration(&mining)?;
        let regions = validate(&mining, &config)?;

        let (m, n, k) = (mining.m, mining.n, mining.k);
        let rank = mining.rank as usize;
        let required = footprint(m, n, k, rank, regions) + DRIVER_SLACK;
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

        let workspace = Workspace::new(&backend, m, n, k, rank, regions)?;

        log::info!(
            "pearl gpu: mining on {} (m={} n={} k={} rank={}, {:.1} GiB of {:.1} GiB free)",
            backend.device().describe(),
            m,
            n,
            k,
            mining.rank,
            gib(required as u64),
            gib(free as u64)
        );

        Ok(Self {
            backend,
            mining,
            workspace,
            regions_per_batch: regions,
            tiles: 0,
            attempts: 0,
        })
    }

    pub fn mining(&self) -> &PearlMining {
        &self.mining
    }

    /// Searches `header` for a proof whose jackpot is at or below `bound`.
    ///
    /// Returns the proof for the first tile that clears the bound, and `None` when `stop` says the
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
        validate(&self.mining, &config)?;

        // Fixed for the job, unlike the attempt number: it is what the noise is derived from, so a
        // new attempt under the same key is a new draw rather than a new search.
        let key = job_key(header, &config);

        let Self {
            backend,
            mining,
            workspace,
            regions_per_batch,
            tiles,
            attempts,
        } = self;
        let (m, n, k) = (mining.m, mining.n, mining.k);
        let rank = mining.rank as usize;

        // A region is one row of the tile grid, and the batch count follows from how many tile rows
        // fit into a batch of regions.
        let tiles_per_row = n / TILE;
        let batches = tiles_per_row.div_ceil(*regions_per_batch);

        let deadline = Instant::now() + Duration::from_secs(MAX_SECONDS_PER_SEARCH);

        loop {
            if stop() {
                return Ok(None);
            }

            let number = *attempts;
            let jackpot_key = attempt(backend, workspace, &key, m, n, k, rank, derivation, number)?;
            *attempts += 1;

            for batch in 0..batches {
                let first = batch * *regions_per_batch;
                let in_batch = (*regions_per_batch).min(tiles_per_row - first);
                let batch_tiles = in_batch * tiles_per_row;

                backend.search_grid(
                    &workspace.a_sum,
                    &workspace.b_sum,
                    &jackpot_key,
                    bound,
                    &mut workspace.found,
                    &mut workspace.coord,
                    // No transcript read-back: the kernel hashes each transcript and compares the
                    // digest, so the transcripts themselves never have to leave the device.
                    None,
                    first * tiles_per_row,
                    batch_tiles,
                    tiles_per_row,
                    tiles_per_row,
                    k,
                    rank,
                )?;

                // Counted at launch, which is when the work is submitted. The readback below
                // synchronises, so a batch that never gets that far is still counted — a tile is
                // hashedrate work the device was asked to do either way.
                *tiles += batch_tiles as u64;

                if let Some((tile_row, tile_col)) =
                    backend.first_winner(&workspace.found, &workspace.coord, in_batch)?
                {
                    // The coordinates are the grid's own, because the batch's offset went into the
                    // launch rather than being added on here. Nothing downstream reads them but the
                    // Merkle proofs below, so a share that described a different tile than the one
                    // that won would still verify.
                    return Ok(Some(assemble(
                        backend,
                        workspace,
                        &key,
                        derivation,
                        m,
                        n,
                        k,
                        rank,
                        tile_row as usize,
                        tile_col as usize,
                        &jackpot_key,
                    )?));
                }

                // Between batches rather than between attempts: an attempt at production dimensions
                // is long, and a job superseded half way through one is not worth finishing.
                if stop() || Instant::now() >= deadline {
                    return Ok(None);
                }
            }
        }
    }

    /// Candidate tiles evaluated so far, used to report hashrate.
    pub fn tiles(&self) -> u64 {
        self.tiles
    }
}

/// One attempt: draw the signal matrices, commit to them, and derive the summed operands.
///
/// Returns `dnsA`, the key the jackpot transcripts are hashed under. It comes out of the
/// commitment chain rather than the job key, which is the one thing about this pipeline that reads
/// like a mistake and is not.
#[allow(clippy::too_many_arguments)]
fn attempt(
    backend: &CudaBackend,
    workspace: &mut Workspace,
    key: &[u8; 32],
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    derivation: SeedDerivation,
    number: u64,
) -> Result<[u8; 32], String> {
    let (seed_a, seed_b) = fill_seeds(key, number);

    // The matrix, never its padding: the commitment is over the padded bytes, and the padding is
    // zeros, so a fill that ran into it would commit to a matrix the verifier never sees.
    backend.fill_i8(&mut workspace.a, 0, workspace.a_len, seed_a)?;
    backend.fill_i8(&mut workspace.bt, 0, workspace.bt_len, seed_b)?;

    let (b_noise_seed, a_noise_seed) = backend.commitment_seeds(
        &workspace.a,
        &workspace.bt,
        *key,
        m as u32,
        n as u32,
        derivation,
    )?;

    backend.fill_noise(
        &mut workspace.e_al,
        &mut workspace.ear,
        &mut workspace.e_bl,
        &mut workspace.e_br,
        &a_noise_seed,
        &b_noise_seed,
        0,
        0,
        m,
        n,
        k,
        rank,
    )?;

    // `signal + noise` is the summed operand consensus multiplies, and it stays inside INT8 because
    // the signal is in [-64, 63] and the noise is a difference of two [-32, 31] entries. Keeping the
    // two matrices separate is what lets the noise be applied once per attempt rather than per tile.
    backend.noise_apply(
        &workspace.a,
        &workspace.e_al,
        &workspace.ear,
        &mut workspace.a_sum,
        m,
        k,
        rank,
    )?;
    backend.noise_apply(
        &workspace.bt,
        &workspace.e_br,
        &workspace.e_bl,
        &mut workspace.b_sum,
        n,
        k,
        rank,
    )?;

    Ok(a_noise_seed)
}

/// The proof for the tile that won: the two committed matrices, in Merkle-proof form.
///
/// The matrices come back whole. At production dimensions that is a gigabyte over PCIe and a host
/// BLAKE3 tree over it, once per share: measured at ~510ms on the dev box in release (of which
/// ~61ms per matrix is the `as u8` conversion below, which is a second copy of the buffer that
/// nothing needs), against ~1-2s to fold the entire 67.1M-tile grid.
///
/// It is not the right shape regardless: only the sixteen leaves the tile covers are ever put on the
/// wire, and the device already builds the whole chunk-chaining-value tree to commit to the matrices
/// during `attempt`, so the siblings are a short walk from where the work already happens. Until
/// that kernel exists this is where a share's time goes.
///
/// The last step is the one that matters. Everything above searched over *this* attempt's
/// matrices; the proof has to commit to *these* matrices, or the pool recomputes the noise from a
/// different pair of roots, gets a different transcript, and rejects a share whose digest was real.
/// Re-deriving `dnsA` from the roots the proof actually carries and comparing it against the key
/// the transcripts were hashed under is what closes that gap — it fails loudly, rather than
/// submitting something that looks like a share.
#[allow(clippy::too_many_arguments)]
fn assemble(
    backend: &CudaBackend,
    workspace: &Workspace,
    key: &[u8; 32],
    derivation: SeedDerivation,
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    tile_row: usize,
    tile_col: usize,
    jackpot_key: &[u8; 32],
) -> Result<PlainProof, String> {
    // `range(16)` patterns sample sixteen consecutive rows, so a tile is a window and the sampled
    // rows are the ones below the tile rather than a gather.
    let a_rows: Vec<usize> = (tile_row * TILE..(tile_row + 1) * TILE).collect();
    let b_cols: Vec<usize> = (tile_col * TILE..(tile_col + 1) * TILE).collect();

    let a = matrix_proof(backend, &workspace.a, m, k, key, &a_rows)?;
    let bt = matrix_proof(backend, &workspace.bt, n, k, key, &b_cols)?;

    let (_, a_noise_seed) = noise_seeds(key, &a.proof.root, &bt.proof.root, m, n, derivation)?;
    if a_noise_seed != *jackpot_key {
        return Err(format!(
            "the proof for tile ({tile_row}, {tile_col}) commits to roots that derive the jackpot \
             key {a_noise_seed:02x?} rather than the {jackpot_key:02x?} the search hashed under, so \
             it does not describe the candidate that won"
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

/// One matrix's Merkle proof over the rows a tile samples.
///
/// The device buffer is copied back whole and handed to the *unmodified* `pearl_blake3::MerkleTree`
/// rather than to something written here: the tree is what the verifier rebuilds the root from, and
/// a second implementation of it in this crate would be a second thing to be wrong.
///
/// No padding is added because none is needed. [`validate`] holds `m` and `n` to multiples of
/// sixteen and `k` to a multiple of the 64-byte BLAKE3 message, so every matrix is already a whole
/// number of 1024-byte chunks and `pad_to_chunk_boundary` would be the identity. That is also why
/// the roots agree: the device hashes the buffer and this hashes the bytes that came back out of
/// it, and there is nothing in between for them to disagree about.
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

/// Refuses a configuration the grid kernel cannot run, and returns the regions in flight.
///
/// Before the workspace is allocated, not after: every condition here is one that would otherwise
/// surface as a two-gigabyte allocation followed by a launch error.
fn validate(mining: &PearlMining, config: &MiningConfiguration) -> Result<usize, String> {
    let (m, n, k) = (mining.m, mining.n, mining.k);
    let rank = mining.rank as usize;

    if rank == 0 {
        return Err("the noise rank is zero".into());
    }
    for (name, extent) in [("m", m), ("n", n), ("k", k)] {
        if !extent.is_multiple_of(TILE) {
            return Err(format!(
                "{name} is {extent}, which is not a multiple of the {TILE} the fold tile is built \
                 from"
            ));
        }
    }
    if !k.is_multiple_of(rank) {
        return Err(format!("the noise rank {rank} does not divide k {k}"));
    }

    // `m` and `n` are `usize` here and `u32` everywhere past this point, so a value that does not
    // fit would be silently truncated before the sanity check ever saw it — and a truncated `m` of
    // 48 is exactly the `m` the rest of the check approves of. The dimensions are also what the
    // workspace is sized from, so the allocation and the proof would disagree about how big the
    // matrix is.
    for (name, extent) in [("m", m), ("n", n)] {
        if u32::try_from(extent).is_err() {
            return Err(format!(
                "{name} is {extent}, which does not fit in the 32 bits it is carried in"
            ));
        }
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
    // Nothing in that check reads a hash or a header, so the dummy roots and a zeroed header are
    // not placeholders in any meaningful sense — they are inputs it ignores. `t_rows`/`t_cols` are
    // the tile offsets, which run `0 .. m/TILE`; every one of them satisfies the `t + max < extent`
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

    // The grid kernel reads sixteen rows of each operand per tile, so a tile that is not square —
    // or not sixteen by sixteen — has no representation in it at all.
    let tile = (
        config.rows_pattern.size() as usize,
        config.cols_pattern.size() as usize,
    );
    if tile != (TILE, TILE) {
        return Err(format!(
            "the GPU search hashes {TILE}x{TILE} tiles, but this configuration's patterns are \
             {}x{}; change `rowsPattern` and `colsPattern` to 0..16",
            tile.0, tile.1
        ));
    }

    let tiles_per_row = n / TILE;
    if tiles_per_row == 0 {
        return Err(format!("n is {n}, which is fewer than one {TILE}-row tile"));
    }

    // The grid kernel carries every tile coordinate and the tile count in
    // `u32` — `first_tile`, `tiles`, `tiles_per_row`, and the
    // `tr * tiles_per_row + tc` that picks a region flag. `m` and `n`
    // fitting in `u32` is not enough: the product `(m / TILE) * (n / TILE)`
    // is what overflows, and a wrapped index would address the wrong tile or
    // the wrong flag rather than fail. Checked here, before the workspace is
    // sized from the same dimensions.
    let tiles_per_col = m / TILE;
    let total_tiles = (tiles_per_row as u64) * (tiles_per_col as u64);
    if total_tiles > u32::MAX as u64 {
        return Err(format!(
            "the tile grid is {tiles_per_col} x {tiles_per_row} = {total_tiles} tiles, \
             which does not fit in the 32-bit indices the search kernel is built from"
        ));
    }

    Ok(REGIONS_PER_BATCH.min(tiles_per_row))
}

/// Device memory the search needs, in bytes.
///
/// Stated up front rather than discovered: the alternative is an allocation failing part-way
/// through the first job, with the reason buried in a driver string and the miner left unable to
/// start. The commitment's chunk-chaining-value levels are included because they are allocated and
/// freed inside every attempt, which makes them easy to forget and they are 32 MiB at production
/// dimensions.
fn footprint(m: usize, n: usize, k: usize, rank: usize, regions: usize) -> usize {
    let a = padded(m * k);
    let bt = padded(n * k);
    // Two levels of eight-word chaining values per 1024-byte chunk, live at the
    // same time — one level being written while the previous is combined. The
    // commitment builds a tree over each matrix in turn, so the larger of the two
    // sizes the scratch, not `a` alone: at `m < n` the `bt` tree is the bigger
    // one, and a footprint sized from `a` under-counts it by the difference.
    let cv_scratch = 2 * (a.max(bt) / CHUNK) * CV_WORDS * size_of::<u32>();

    a * 2
        + bt * 2
        + m * rank
        + n * rank
        + 2 * k * 2 * size_of::<u32>()
        + regions * 3 * size_of::<u32>()
        + cv_scratch
}

/// The size a matrix occupies in the commitment: its bytes rounded up to a whole BLAKE3 chunk.
fn padded(bytes: usize) -> usize {
    bytes.div_ceil(CHUNK) * CHUNK
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::miners::gpu::cuda::philox_stream;
    use crate::miners::pearl_pow::verify_share_locally;
    use zk_pow::api::proof::PublicProofParams;
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
    /// The refused configuration is a real one rather than a hypothetical: `k = 256` at `rank = 32`
    /// is under the verifier's `k >= 16r`, and it is what this test module ran with before the
    /// delegation was added, where it was found not by `validate` but by `verify_share_locally` at
    /// the far end of a search — after the whole pipeline had produced a proof the pool would
    /// reject.
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

        // And the configuration everything else here runs on does pass.
        let good = test_mining();
        validate(&good, &mining_configuration(&good).unwrap())
            .expect("the test configuration is one the verifier accepts");
    }

    /// The shipped configuration, opened on the real device and actually searched.
    ///
    /// Every other test here runs at 48 x 48, which is enough to pin the arithmetic and far too
    /// small to say anything about whether the miner can start: the workspace is sized from `m`,
    /// `n` and `k`, the grid is `(m/16) * (n/16)` tiles wide, and the attempt allocates about two
    /// gigabytes. None of that is exercised by a small grid, and a configuration that cannot be
    /// allocated is not a subtle failure — it is the engine refusing to start at all.
    ///
    /// Ignored by default because it wants a device with roughly 2.2 GiB spare and a few seconds
    /// of it. Run with `--ignored`.
    #[test]
    #[ignore = "allocates the full production workspace on the device; run with --ignored"]
    fn the_shipped_default_runs_a_search_on_the_device() {
        let Ok(mut miner) = GpuMiner::open(PearlMining::default()) else {
            // No device, or not one with the memory for it: there is nothing here to run.
            return;
        };

        // A bound every digest clears, so the search returns on its first batch instead of waiting
        // for a share. The proof it produces still has to be a real one at the real dimensions.
        let started = Instant::now();
        let proof = miner
            .search(&test_header(), CERT_VERSION, U256::MAX, &|| false)
            .expect("the search runs")
            .expect("a bound everything clears is claimed by the first batch");

        assert_eq!(proof.m, 131_072);
        assert_eq!(proof.n, 131_072);
        assert_eq!(proof.k, 4096);
        assert_eq!(proof.noise_rank, 256);
        // Sixteen rows and columns per tile, and the proof has to say which tile it found.
        assert_eq!(proof.a.row_indices.len(), 16);
        assert_eq!(proof.bt.row_indices.len(), 16);
        eprintln!(
            "first batch: {:?}, {} tiles — setup and one fold are ~23ms of that, the rest is \
             `assemble`: 2 x 512 MiB back over PCIe and both Merkle proofs on the host",
            started.elapsed(),
            miner.tiles()
        );

        // And the sustained rate, which the first batch cannot show: it is paying for two gigabytes
        // of fill, commitment and noise before it folds anything. A bound nothing clears makes the
        // search run for its full time slice instead of returning, which is what the number the
        // launch-geometry work in Phase 8 is measured against.
        //
        // Read it as "this kernel's share of the card", not "this kernel's rate". Two things move it
        // and neither is the kernel: the search saturates the GPU, so the 252W cap puts the SM clock
        // near 2700MHz rather than its 3090MHz boost, and anything else mining the same device takes
        // half of it. On the dev box this number halved from 67.8 to 34.0 TH/s purely because a
        // second `tokenminer-desktop` was started on the same GPU, and moved back the moment that
        // one stopped.
        let before = miner.tiles();
        let started = Instant::now();
        assert!(
            miner
                .search(&test_header(), CERT_VERSION, U256::zero(), &|| false)
                .expect("the search runs")
                .is_none(),
            "no digest is zero, so nothing can clear this bound"
        );
        let elapsed = started.elapsed();
        let tiles = miner.tiles() - before;
        eprintln!(
            "sustained: {} tiles in {:?} — {:.2} TH/s",
            tiles,
            elapsed,
            miner.mining.th_per_second(tiles as f64 / elapsed.as_secs_f64()),
        );

        // Where the time above actually goes, because "sustained" is an average over a window that
        // contains several whole attempts' setup, and setup is a different cost from the fold.
        // Tuning the wrong one is how a miner spends a week on the fold and keeps the same wall
        // clock.
        //
        // All four of `attempt`'s steps, because the first cut of this measured only two of them and
        // reported setup at 1% -- on a breakdown that silently omitted the commitment chain, which
        // is the one step that reads a gigabyte and the one step that has to hash it. A breakdown
        // that adds up to less than the thing it is decomposing is not a breakdown.
        //
        // Measured here rather than in `attempt` itself, so production carries no timing code. The
        // seeds are wrong and the noise is whatever the last attempt left behind, which is fine:
        // the question is how long each pass takes, not what they produce.
        let GpuMiner {
            backend,
            mining,
            workspace,
            ..
        } = &mut miner;
        let (m, n, k) = (mining.m, mining.n, mining.k);
        let rank = mining.rank as usize;

        let started = Instant::now();
        backend
            .fill_i8(&mut workspace.a, 0, workspace.a_len, 1)
            .expect("the fill runs");
        backend
            .fill_i8(&mut workspace.bt, 0, workspace.bt_len, 2)
            .expect("the fill runs");
        // Both launches are asynchronous, so the clock above has to wait for the device or it times
        // the enqueue. Thirteen microseconds for a gigabyte of Philox is not a fast fill.
        backend.synchronize().expect("the device drains");
        let fill = started.elapsed();

        let started = Instant::now();
        let (b_noise_seed, a_noise_seed) = backend
            .commitment_seeds(
                &workspace.a,
                &workspace.bt,
                [0x5a; 32],
                m as u32,
                n as u32,
                seed_derivation_for(CERT_VERSION).expect("the certificate version is one we know"),
            )
            .expect("the commitment runs");
        backend.synchronize().expect("the device drains");
        let commit = started.elapsed();

        let started = Instant::now();
        backend
            .fill_noise(
                &mut workspace.e_al,
                &mut workspace.ear,
                &mut workspace.e_bl,
                &mut workspace.e_br,
                &a_noise_seed,
                &b_noise_seed,
                0,
                0,
                m,
                n,
                k,
                rank,
            )
            .expect("the noise draw runs");
        backend.synchronize().expect("the device drains");
        let draw = started.elapsed();

        let started = Instant::now();
        backend
            .noise_apply(
                &workspace.a,
                &workspace.e_al,
                &workspace.ear,
                &mut workspace.a_sum,
                m,
                k,
                rank,
            )
            .expect("the noise-apply runs");
        backend
            .noise_apply(
                &workspace.bt,
                &workspace.e_br,
                &workspace.e_bl,
                &mut workspace.b_sum,
                n,
                k,
                rank,
            )
            .expect("the noise-apply runs");
        backend.synchronize().expect("the device drains");
        let noise = started.elapsed();

        // What the fold costs for the same grid, so the two are comparable: the tile grid is
        // `(m/16) * (n/16)` tiles, and the tile rate behind the TH/s above is what it has to be
        // divided by.
        let grid_tiles = (m / 16) * (n / 16);
        let folding =
            Duration::from_secs_f64(grid_tiles as f64 / (tiles as f64 / elapsed.as_secs_f64()));
        let setup = fill + commit + draw + noise;
        eprintln!(
            "per attempt: fill 2 x {} MiB {fill:?}, commitment over 2 x {} MiB {commit:?}, noise \
             draw {draw:?}, noise-apply 2 x {} MiB {noise:?} -- setup {setup:?} against a \
             {folding:?} fold, so {:.1}% of an attempt",
            m * k / (1 << 20),
            m * k / (1 << 20),
            n * k / (1 << 20),
            100.0 * setup.as_secs_f64() / (folding + setup).as_secs_f64(),
        );

        // And what a share's assembly is made of, since it costs several times a whole grid's fold.
        // Three steps, because they have three different fixes: the readback is bandwidth, the
        // `as u8` conversion is a second host copy of the same gigabyte that nothing needs, and the
        // tree is the hashing itself. Timed on one matrix and doubled for the pair.
        let started = Instant::now();
        let host = backend
            .read_i8(&workspace.a)
            .expect("the matrix comes back");
        backend.synchronize().expect("the device drains");
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
        // searches with no tiles folded during it. The reported hashrate divides tiles by wall
        // clock, so all of it is charged to the denominator; a share that takes three seconds to
        // verify is three seconds of not mining.
        //
        // Verified against `U256::MAX` because that is the bound the proof above was found under:
        // the proof is real, so this measures the verifier's actual cost rather than its failure
        // path, which is much cheaper and would flatter the number.
        let started = Instant::now();
        verify_share_locally(&test_header(), CERT_VERSION, &proof, U256::MAX)
            .expect("the proof the first batch produced verifies");
        let verify = started.elapsed();

        let started = Instant::now();
        let encoded = crate::miners::pearl_mining::encode_plain_proof(&proof, true)
            .expect("the proof encodes");
        let encode = started.elapsed();
        eprintln!(
            "per share: verify {verify:?}, gzip+bincode encode {encode:?} ({} chars) -- \
             assembly and submission, none of which folds a tile",
            encoded.len(),
        );
    }

    /// The whole pipeline, against the verifier's own implementation of every step it performs.
    ///
    /// Each kernel is already checked on its own, and none of that adds up to this. What is untested
    /// otherwise is the *order*: that the commitment is taken over the matrix the search is about to
    /// fold, that the noise derived from that commitment is composed into the operands the fold
    /// reads, that the batching walks the tile grid once and claims the tile whose digest the
    /// reference says is lowest, and that the proof handed back commits to the very matrices that
    /// were folded. Every one of those is a mistake each kernel would accept — and two of them
    /// (a swapped seed, a swapped operand) are mistakes that produce perfectly valid proofs of the
    /// wrong thing.
    ///
    /// So the reference here re-derives the whole attempt on the host — the same Philox stream, the
    /// same `pearl_blake3` roots, the verifier's own `commitment_hash`, its own `pearl_noise`
    /// tensors, and `compute_jackpot` for the transcript — and the bound handed to the search is
    /// exactly the lowest digest that produces, so only one tile on the grid can clear it and there
    /// is no race about which. The search then has to return a proof for that tile, and the pool's
    /// own verifier has to accept it at the bound the search was given.
    #[test]
    fn a_share_the_gpu_finds_is_one_the_verifier_accepts() {
        let Ok(mut miner) = GpuMiner::open(test_mining()) else {
            // No device: there is nothing to run the loop on, and nothing it could prove here.
            return;
        };

        // One region per batch, so a batch is a single tile row and the grid takes three passes to
        // walk. The kernel's tile index is relative to its batch, so this is what puts the batch's
        // offset on the path the coordinates take back to the proof — with the shipping batch size
        // the whole test grid fits inside one batch, and a search that forgot the offset would
        // still be right here.
        miner.regions_per_batch = 1;

        let (mining, header, config) = test_job();
        let derivation = seed_derivation_for(CERT_VERSION).expect("a known certificate version");
        let key = job_key(&header, &config);
        assert_eq!(
            key,
            public_params(&header, &config, derivation, &[0; 32], &[0; 32]).job_key(),
            "the search's job key is not the one the verifier derives"
        );

        // A first attempt whose cheapest tile is *not* in the leading batch, so the offset is part of
        // what is being checked rather than a zero this grid could have got right by accident.
        // Attempts are independent draws, so this is just waiting for one to come up.
        let (number, (tile_row, tile_col), bound) = (0..24u64)
            .find_map(|number| {
                lowest_tile(&mining, &header, &config, derivation, key, number)
                    .filter(|((row, _), _)| *row >= miner.regions_per_batch)
                    .map(|((row, col), digest)| (number, (row, col), digest))
            })
            .expect("no attempt in twenty-four put its cheapest tile past the leading batch");

        miner.attempts = number;
        let proof = miner
            .search(&header, CERT_VERSION, bound, &|| false)
            .expect("the search runs")
            .expect("the bound is the lowest digest on the grid, so a tile clears it");

        // The proof is for the tile the host priced. The coordinates only ever reach the proof
        // through here — nothing downstream reads them — so a proof describing a different tile
        // than the one that won would still verify.
        assert_eq!(proof.a.row_indices, tile_rows(tile_row), "A rows");
        assert_eq!(proof.bt.row_indices, tile_rows(tile_col), "B columns");
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
        let ((next_row, _), next_bound) =
            lowest_tile(&mining, &header, &config, derivation, key, number + 1)
                .expect("the tile grid produced no candidates");
        let next = miner
            .search(&header, CERT_VERSION, next_bound, &|| false)
            .expect("the second search runs")
            .expect("the second attempt has a tile under its bound");

        assert_ne!(
            next.a.proof.root, proof.a.proof.root,
            "the second search returned the first attempt's matrices again"
        );
        assert_eq!(next.a.row_indices, tile_rows(next_row));
        assert_verifies(&header, &config, &next, next_bound);
    }

    /// Runs the pool's own check over a proof, at a bound the search reported a tile against.
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
                "the verifier rejected a proof whose tile the search priced at {bound:#x}: {error}"
            )
        });
    }

    /// The matrix rows a tile of the grid samples.
    ///
    /// `range(16)` patterns sample sixteen consecutive rows, so a tile is a window of the matrix and
    /// the sampled rows are the ones below it — which is what makes the batching offset visible in
    /// `row_indices` rather than only in the search's own bookkeeping.
    fn tile_rows(tile: usize) -> Vec<usize> {
        (tile * TILE..(tile + 1) * TILE).collect()
    }

    /// The tile of `mining`'s grid whose jackpot digest is lowest for one attempt, and that digest.
    ///
    /// `key` is the job key and `number` the attempt — the two things that make an attempt's matrices
    /// what they are. This is the verifier's own chain at every step, the commitment included:
    /// `public.commitment_hash` rather than a re-derivation of it, so the reference cannot drift from
    /// what a submitted proof is checked against.
    fn lowest_tile(
        mining: &PearlMining,
        header: &IncompleteBlockHeader,
        config: &MiningConfiguration,
        derivation: SeedDerivation,
        key: [u8; 32],
        number: u64,
    ) -> Option<((usize, usize), U256)> {
        let (m, n, k) = (mining.m, mining.n, mining.k);
        let rank = mining.rank as usize;

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

        let mut lowest: Option<((usize, usize), U256)> = None;
        for tile_row in 0..m / TILE {
            for tile_col in 0..n / TILE {
                let rows = tile_rows(tile_row);
                let cols = tile_rows(tile_col);

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
                // (`check_jackpot_against_nbits`) and the one the kernel walks the bound against.
                let digest =
                    U256::from_little_endian(&compute_jackpot_hash(&transcript, a_noise_seed));

                if lowest.as_ref().is_none_or(|(_, best)| digest < *best) {
                    lowest = Some(((tile_row, tile_col), digest));
                }
            }
        }

        lowest
    }

    /// A configuration small enough to price every tile on the host, with the production tile shape.
    ///
    /// `rank` is a power of two and a multiple of the 32 bytes one noise hash produces, which is what
    /// `validate_rank` insists on — so not 16, however convenient. The dimensions are otherwise the
    /// smallest that still give several tiles in both directions: a one-tile grid would pass against
    /// any indexing at all.
    ///
    /// Three tile rows by three, so the batching has three batches to walk and a tile in each — and
    /// `k` is a whole number of the 64-byte messages the verifier requires, which is what makes
    /// every matrix here a whole number of chunks too.
    /// A configuration the verifier accepts, kept as small as it can be so the host re-derivation
    /// in the end-to-end test is cheap.
    ///
    /// `k = 1024` is the floor the verifier sets — below it the 1024-byte matrix padding is not
    /// collision resistant. `rank = 64` is the smallest value the GPU search can fold at on every
    /// architecture: the tensor path stages 64 k and the DP4A path 32, and a `rank` below the staged
    /// width zeroes `steps_per_rank`. Together they sit at exactly `k = 16r`, the same ratio
    /// `k = 4096, r = 256` runs at in production, so the fold walks the same number of rank blocks
    /// per tile dimension at both sizes.
    fn test_mining() -> PearlMining {
        PearlMining {
            m: 48,
            n: 48,
            k: 1024,
            rank: 64,
            rows_pattern: (0..TILE as u32).collect(),
            cols_pattern: (0..TILE as u32).collect(),
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
            TILE as u32,
            TILE as u32,
        );
        public.hash_a = *hash_a;
        public.hash_b = *hash_b;
        public
    }
}
