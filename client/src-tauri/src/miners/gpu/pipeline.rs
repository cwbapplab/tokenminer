//! The launch sequence for the split search path.
//!
//! The fused path folded, hashed and compared in one launch, so its pipeline was one kernel per batch.
//! The split path is a chain of launches on one stream: fill, commitment, noise generation, noising,
//! search, postpass, scan. The fill and the commitment stay in [`crate::miners::pearl_gpu`] because
//! their results cross back to the host; everything below them is stream-ordered device work, which is
//! what this module owns.
//!
//! Two things about the ordering are load-bearing rather than stylistic:
//!
//! * the sparse noise launch runs *after* a zero of its output, on the same stream. The entry point
//!   patches `+1` / `-1` into rows it never clears, so a launch that ran before the zero — or on a
//!   different stream — accumulates onto the previous attempt.
//! * the postpass runs after the search launch finishes, because it reads the transcript the search
//!   wrote. That is the whole point of the split: the search blob can use a 4x8 warp tile only because
//!   the hash state is not live inside it.
//!
//! The search, postpass and scan launches are here too, so the whole attempt is one stream-ordered
//! chain rather than a half that has to be completed somewhere else.
//!
//! The gates below (`check_noise_gen`, `check_noising`, `check_postpass`) run from
//! [`crate::miners::gpu::CudaBackend::self_test`], on every start. They are not optional extras: each
//! is a place where a 1:1 port can silently disagree, and every one of those disagreements produces
//! self-consistent output — a valid noise tensor for the wrong side, a valid transcript for the wrong
//! candidate, a digest that is a real BLAKE3 hash under the wrong key. Nothing downstream can tell.

use std::collections::HashSet;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use primitive_types::U256;

use super::bufs::{SplitBuffers, SplitConfig, SEED_LABEL_A, SEED_LABEL_B, TILE_M, TILE_N};
use super::fatbin::{self, FatbinKernels, FIRST_HIT_SENTINEL, NOISE_BLOCK, POSTPASS_BLOCK, SCAN_BLOCK};
use super::triton::{HASH_CANDIDATES, JACKPOT_SIZE, TritonKernels};

// =============================================================================
//   Launches
// =============================================================================

/// The four noise tensors: dense int8 for `EAL` / `EBR`, sparse for `EAR` / `EBL`.
///
/// The key each side runs under is that side's commitment root, and the seed is the padded label — the
/// same pair the verifier's `generate_uniform_random_matrix` and `generate_permutation_matrix` take.
/// The dense and sparse entry points take the same arguments in the same order; only the count means
/// different things (rows for the dense form, `k` for the sparse), and the grid follows the chunk count
/// each one assumes.
pub fn run_noise_gen(
    stream: &Arc<CudaStream>,
    fatbin: &FatbinKernels,
    bufs: &mut SplitBuffers,
) -> Result<(), String> {
    let config = bufs.config;
    let (m, n, k) = (config.m, config.n, config.k);

    noise_launch(
        stream,
        &fatbin.noise_dense_int8,
        "dense int8 A",
        m as i32,
        &bufs.commit_a,
        &bufs.seed_label_a,
        &bufs.eal,
        fatbin::dense_grid(m),
    )?;
    noise_launch(
        stream,
        &fatbin.noise_dense_int8,
        "dense int8 B",
        n as i32,
        &bufs.commit_b,
        &bufs.seed_label_b,
        &bufs.ebr,
        fatbin::dense_grid(n),
    )?;

    // Before the sparse launch, on the same stream — see the module comment.
    stream
        .memset_zeros(&mut bufs.ear)
        .map_err(|e| format!("zeroing ear failed: {e}"))?;
    stream
        .memset_zeros(&mut bufs.ebl)
        .map_err(|e| format!("zeroing ebl failed: {e}"))?;

    noise_launch(
        stream,
        &fatbin.noise_sparse,
        "sparse A",
        k as i32,
        &bufs.commit_a,
        &bufs.seed_label_a,
        &bufs.ear,
        fatbin::sparse_grid(k),
    )?;
    noise_launch(
        stream,
        &fatbin.noise_sparse,
        "sparse B",
        k as i32,
        &bufs.commit_b,
        &bufs.seed_label_b,
        &bufs.ebl,
        fatbin::sparse_grid(k),
    )?;

    Ok(())
}

/// `signal + noise` for both sides, through the Triton noising blob.
///
/// The blob computes `Out[i, j] = wrap_int8(X[i, j] + sum_r Y[i, r] * Z[j, r])` with `X` and `Out` as
/// `(M, N)`, `Y` as `(M, K_inner)` and `Z` as `(N, K_inner)`. Read against the verifier, `N` here is
/// `k` — the output is the summed operand, one row per signal row and one column per signal column —
/// and `K_inner` is the rank. That is why the launch's `n` argument is `k` and its `k_inner` is `rank`,
/// which is the reverse of what the names suggest.
///
/// With a materialised sparse `Z` the product is exactly two terms: `+Y[i, k0] - Y[i, k1]`, the pair the
/// verifier's `generate_permutation_matrix` produced for that column. So the blob and the verifier
/// compute the same function here, and the sum stays inside `i8` — see the range argument in
/// [`check_noising`].
///
/// `k_inner` has to be 128: the blob stages a full 128-byte row per `cp.async` group, so a smaller rank
/// makes it read past the end of each row.
pub fn run_noising(
    stream: &Arc<CudaStream>,
    triton: &TritonKernels,
    bufs: &SplitBuffers,
) -> Result<(), String> {
    let config = bufs.config;
    let (m, n, k, r) = (config.m, config.n, config.k, config.rank);

    triton.noising.launch(
        stream,
        m as i32,
        k as i32,
        r as i32,
        &bufs.a,
        &bufs.eal,
        &bufs.ear,
        &bufs.a_sum,
    )?;
    triton.noising.launch(
        stream,
        n as i32,
        k as i32,
        r as i32,
        &bufs.bt,
        &bufs.ebr,
        &bufs.ebl,
        &bufs.b_sum,
    )?;

    Ok(())
}

fn noise_launch<T>(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    name: &str,
    count: i32,
    key: &CudaSlice<u32>,
    seed: &CudaSlice<u32>,
    out: &CudaSlice<T>,
    grid: u32,
) -> Result<(), String> {
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (NOISE_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(func);
    launch.arg(&count).arg(key).arg(seed).arg(out);

    unsafe { launch.launch(config) }
        .map_err(|e| format!("launching {name} failed: {e}"))?;
    Ok(())
}

/// The search blob over the summed operands.
///
/// The transcript has to be zeroed before this launch, on the same stream — see
/// [`SplitBuffers::reset_for_attempt`]. At the forced configuration the blob writes all sixteen words of
/// every candidate slot, so the zero is not filling gaps: it is the answer for a fold the verifier never
/// reached, and for a candidate the blob would not visit at all if `k / r` were below 16.
pub fn run_search(
    stream: &Arc<CudaStream>,
    triton: &TritonKernels,
    bufs: &SplitBuffers,
) -> Result<(), String> {
    let (m, n, k) = (bufs.config.m, bufs.config.n, bufs.config.k);

    triton.search.launch(
        stream,
        m as i32,
        n as i32,
        k as i32,
        &bufs.a_sum,
        &bufs.b_sum,
        &bufs.transcripts,
    )
}

/// Hashes every candidate transcript and compares it against the bound, one thread per candidate.
///
/// `pow_key` is the A commitment root — the same key `zk_pow`'s `compute_jackpot_hash` keys the jackpot
/// message with (`commitment_hash.1`, the `a_noise_seed`). The B root never enters the pow hash, so a
/// launch that passed `commit_b` here would still produce self-consistent hits and a self-consistent
/// digest; only a comparison against the verifier's hash would notice, which is what gate A is.
pub fn run_postpass(
    stream: &Arc<CudaStream>,
    fatbin: &FatbinKernels,
    bufs: &SplitBuffers,
) -> Result<(), String> {
    let total = bufs.config.total_candidates() as i32;
    let config = LaunchConfig {
        grid_dim: (fatbin::candidate_grid(bufs.config.total_candidates(), POSTPASS_BLOCK), 1, 1),
        block_dim: (POSTPASS_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&fatbin.blake3_compare);
    launch
        .arg(&bufs.transcripts)
        .arg(&bufs.commit_a)
        .arg(&bufs.pow_target)
        .arg(&bufs.hash)
        .arg(&bufs.hit)
        .arg(&total);

    unsafe { launch.launch(config) }
        .map_err(|e| format!("launching the postpass failed: {e}"))?;
    Ok(())
}

/// Finds the first candidate that cleared the bound.
///
/// `first_hit` has to hold the sentinel before this launch: `atomicMin` can only lower it, so a value
/// left from the previous attempt is a hit the scan cannot undo, and a zero is candidate index zero.
pub fn run_scan(
    stream: &Arc<CudaStream>,
    fatbin: &FatbinKernels,
    bufs: &SplitBuffers,
) -> Result<(), String> {
    let total = bufs.config.total_candidates() as i32;
    let config = LaunchConfig {
        grid_dim: (fatbin::candidate_grid(bufs.config.total_candidates(), SCAN_BLOCK), 1, 1),
        block_dim: (SCAN_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&fatbin.pow_scan_hits);
    launch.arg(&bufs.hit).arg(&total).arg(&bufs.first_hit);

    unsafe { launch.launch(config) }
        .map_err(|e| format!("launching the scan failed: {e}"))?;
    Ok(())
}

/// The scan's result, read once.
pub fn read_first_hit(stream: &Arc<CudaStream>, bufs: &SplitBuffers) -> Result<Option<Hit>, String> {
    let words = stream
        .clone_dtoh(&bufs.first_hit)
        .map_err(|e| format!("reading the scan result failed: {e}"))?;

    let value = words[0];
    if value == FIRST_HIT_SENTINEL {
        return Ok(None);
    }
    Ok(Some(Hit::from_candidate(&bufs.config, value as usize)))
}

/// A winner, resolved to the coordinate `assemble` needs.
///
/// The scan reports one number, so the split between tile and candidate is the whole mapping: the scan
/// cannot say which of the two a shape got wrong, and a square grid cannot tell `tile_linear / tiles_n`
/// from `tile_linear % tiles_n`. Which row pair a candidate index names is the device half — gate D.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub candidate: usize,
    pub tile_linear: usize,
    pub tile_m: usize,
    pub tile_n: usize,
    /// The two A rows the winner's fold covers.
    pub a_rows: [usize; 2],
    /// The first of the 128 B columns the winner's fold covers.
    pub b_col: usize,
}

impl Hit {
    pub fn from_candidate(config: &SplitConfig, candidate: usize) -> Self {
        let tile_linear = candidate / HASH_CANDIDATES;
        let cand = candidate % HASH_CANDIDATES;
        let tiles_n = config.n / TILE_N;
        let tile_m = tile_linear / tiles_n;
        let tile_n = tile_linear % tiles_n;

        Self {
            candidate,
            tile_linear,
            tile_m,
            tile_n,
            a_rows: [tile_m * TILE_M + 2 * cand, tile_m * TILE_M + 2 * cand + 1],
            b_col: tile_n * TILE_N,
        }
    }
}

// =============================================================================
//   Gates, run from `CudaBackend::self_test`
// =============================================================================

/// Gate B: the noise entry points against the verifier.
///
/// The shape is non-square on both sides — `m` and `n` differ from the rank — because the dense output is
/// `(rows, 128)` and the sparse output is `(k, 128)`. On a square shape a kernel that wrote its rows as
/// columns produces the same bytes, so a test built on `128 x 128` cannot tell the two apart.
///
/// The two sides run under different keys, which is what catches a launch that swapped them: the
/// verifier's `compute_noise_for_indices` destructures its commitment as `(b_seed, a_seed)`, the opposite
/// of the order the names suggest, and a test that used one key for both sides could not tell which side
/// it was actually checking.
///
/// What this does *not* pin is the seed labels themselves: both sides of the comparison read the same
/// constant, so nothing in this crate can pin them. A captured third-party share would, because that is
/// the one reference that is not a second reading of the same algorithm — and there is no fixture in this
/// tree yet, so the labels are taken on trust.
pub fn check_noise_gen(stream: &Arc<CudaStream>, fatbin: &FatbinKernels) -> Result<(), String> {
    let config = SplitConfig::new(256, 384, 2048, 128)?;
    let mut bufs = SplitBuffers::new(stream, &config)?;

    let key_a = [0x11u8; 32];
    let key_b = [0x22u8; 32];
    bufs.set_commitment(stream, &key_a, &key_b)?;

    run_noise_gen(stream, fatbin, &mut bufs)?;

    let rank = config.rank;
    let a_rows: Vec<usize> = (0..config.m).collect();
    let b_cols: Vec<usize> = (0..config.n).collect();

    let got = clone_i8(stream, &bufs.eal)?;
    let want = flatten(&zk_pow::circuit::pearl_noise::generate_uniform_random_matrix(
        &SEED_LABEL_A,
        &key_a,
        &a_rows,
        rank,
    ));
    if got != want {
        return Err("the dense A side is not the verifier's EAL".to_string());
    }

    let got = clone_i8(stream, &bufs.ebr)?;
    let want = flatten(&zk_pow::circuit::pearl_noise::generate_uniform_random_matrix(
        &SEED_LABEL_B,
        &key_b,
        &b_cols,
        rank,
    ));
    if got != want {
        return Err("the dense B side is not the verifier's EBR".to_string());
    }

    // The sparse entry point writes a materialised matrix, not pairs, so the comparison expands the
    // verifier's pairs to the same shape: `+1` at `k0`, `-1` at `k1`. The two cannot land on the same
    // cell — `k1 = k0 ^ (1 + mul_hi)` and the operand is never zero — so the order of the two writes
    // does not matter.
    let got = clone_i8(stream, &bufs.ear)?;
    let want = materialise(
        &zk_pow::circuit::pearl_noise::generate_permutation_matrix(
            &SEED_LABEL_A,
            &key_a,
            config.k,
            rank,
        ),
        config.k,
        rank,
    );
    if got != want {
        return Err("the sparse A side is not the verifier's EAR".to_string());
    }

    let got = clone_i8(stream, &bufs.ebl)?;
    let want = materialise(
        &zk_pow::circuit::pearl_noise::generate_permutation_matrix(
            &SEED_LABEL_B,
            &key_b,
            config.k,
            rank,
        ),
        config.k,
        rank,
    );
    if got != want {
        return Err("the sparse B side is not the verifier's EBL".to_string());
    }

    Ok(())
}

/// Gate C: the Triton noising blob against the verifier's noise, end to end.
///
/// The blob computes the dense bilinear form; the verifier computes the same value through sparse pairs.
/// Feeding the blob the materialised sparse matrix makes the two the same function, so a disagreement here
/// is a layout disagreement — an argument order, a stride, or a side swapped — not an arithmetic one.
///
/// The inputs are kept inside the ranges the verifier's own generators produce: signal in `[-64, 63]`,
/// noise in `[-32, 31]`. That is what makes the range argument below checkable rather than asserted:
/// every sum is in `[-127, 126]`, so the blob's `satfinite` mma never saturates, and a disagreement
/// cannot be blamed on saturation. A test that let the sum overflow would measure the saturating clamp,
/// not the layout.
pub fn check_noising(
    stream: &Arc<CudaStream>,
    fatbin: &FatbinKernels,
    triton: &TritonKernels,
) -> Result<(), String> {
    // Non-square on both sides, for the same reason as gate B: `noise_B` takes the sparse matrix in the
    // opposite orientation from `noise_A`, and a square shape cannot tell the two.
    let config = SplitConfig::new(256, 384, 2048, 128)?;
    let (m, n, k, r) = (config.m, config.n, config.k, config.rank);
    let mut bufs = SplitBuffers::new(stream, &config)?;

    let key_a = [0x11u8; 32];
    let key_b = [0x22u8; 32];

    // Signal in the verifier's range, spread across it rather than sampled at a few points.
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 128) as i8) - 64).collect();
    let bt: Vec<i8> = (0..n * k).map(|i| ((i % 128) as i8) - 64).collect();

    // The signal goes through the buffers the pipeline reads, not a copy of them: the point of the check
    // is the launch, so it has to run over the same allocations production uses.
    stream
        .memcpy_htod(&a, &mut bufs.a)
        .map_err(|e| format!("uploading the A signal failed: {e}"))?;
    stream
        .memcpy_htod(&bt, &mut bufs.bt)
        .map_err(|e| format!("uploading the B signal failed: {e}"))?;

    bufs.set_commitment(stream, &key_a, &key_b)?;
    run_noise_gen(stream, fatbin, &mut bufs)?;
    run_noising(stream, triton, &bufs)?;

    // The verifier's commitment tuple is `(b_seed, a_seed)` — the reverse of the order the names
    // suggest — so passing the keys in the obvious order here would compare the A side against the B
    // side's noise and still look like a pass on a single-key check.
    let noise = zk_pow::circuit::pearl_noise::compute_noise_for_indices(
        k,
        r,
        (key_b, key_a),
        &(0..m).collect::<Vec<usize>>(),
        &(0..n).collect::<Vec<usize>>(),
    );

    let got = clone_i8(stream, &bufs.a_sum)?;
    for i in 0..m {
        for t in 0..k {
            let sum = a[i * k + t] as i32 + noise.a[i][t] as i32;
            if !(-127..=126).contains(&sum) {
                return Err(format!("A side sum {sum} left the i8 range"));
            }
            if got[i * k + t] as i32 != sum {
                return Err(format!(
                    "the blob disagrees with the verifier at A row {i}, column {t}"
                ));
            }
        }
    }

    let got = clone_i8(stream, &bufs.b_sum)?;
    for j in 0..n {
        for t in 0..k {
            let sum = bt[j * k + t] as i32 + noise.b[j][t] as i32;
            if !(-127..=126).contains(&sum) {
                return Err(format!("B side sum {sum} left the i8 range"));
            }
            if got[j * k + t] as i32 != sum {
                return Err(format!(
                    "the blob disagrees with the verifier at B row {j}, column {t}"
                ));
            }
        }
    }

    Ok(())
}

/// Gates A and D: the fold and the hash, on the device, against the two references the pool is built
/// from.
///
/// Gate D is the load-bearing one, and it is the one that settles what the blob's checkpoint *is*.
/// `compute_jackpot` accumulates `jackpot[u][v]` across its `ll` loop and never resets it, so its
/// `xored_tile` at group `t` is the XOR of the running sums over groups 1..=t+1 — the same running XOR
/// the blob writes at checkpoint `t`. And at `k = 16r` each `tid` is written exactly once, so
/// `rotate_left(LROT_PER_TILE)` is a no-op on the zero initial value: the no-rotl fold is the verifier's
/// fold, not an approximation of it.
///
/// The operands come from the verifier's own generators, with the signal and the noise kept separate
/// rather than summed by a model written here. That is what makes a disagreement mean the fold or the
/// candidate mapping rather than the noising, which [`check_noising`] already pinned.
///
/// What this cannot pin is the order of the two rows inside a candidate: the fold is a XOR over the whole
/// window, so `{2c, 2c+1}` and `{2c+1, 2c}` give the same value. It pins which pair, not which order —
/// and the pair is what `assemble` reports, so the order does not matter.
///
/// Two tiles, so the candidate index is `tile * 64 + c` and not simply `c`. One tile cannot show that the
/// candidate index names a particular row pair.
pub fn check_postpass(
    stream: &Arc<CudaStream>,
    fatbin: &FatbinKernels,
    triton: &TritonKernels,
) -> Result<(), String> {
    use zk_pow::api::proof::{IncompleteBlockHeader, PublicProofParams, SeedDerivation};
    use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
    use zk_pow::circuit::chip::compute_jackpot;
    use zk_pow::circuit::pearl_noise::compute_noise_for_indices;

    let config = SplitConfig::test_shape();
    let (m, n, k, r) = (config.m, config.n, config.k, config.rank);
    let candidates = config.total_candidates();
    let mut bufs = SplitBuffers::new(stream, &config)?;

    let commit_a = [0x11u8; 32];
    let commit_b = [0x22u8; 32];
    bufs.set_commitment(stream, &commit_a, &commit_b)?;

    // The patterns are the ones the split path reports, so the configuration the gate compares against is
    // the one production would submit — not a shape invented here.
    let mining = crate::miners::pearl_mining::PearlMining {
        m,
        n,
        k,
        rank: r as u16,
        rows_pattern: vec![0, 1],
        cols_pattern: (0..128).collect(),
        gzip: false,
    };
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [0; 32],
        merkle_root: [0; 32],
        timestamp: 0,
        nbits: 0,
    };
    let public = PublicProofParams::new_dummy(
        header,
        SeedDerivation::Salted,
        crate::miners::pearl_mining::mining_configuration(&mining)?,
        m as u32,
        n as u32,
        0,
        0,
    );
    let compiled = CompiledPublicParams::from(&public);

    assert_eq!(compiled.h, 2, "the fold the verifier computes is over a two-row window");
    assert_eq!(compiled.w, 128);
    assert_eq!(compiled.k, k);
    assert_eq!(compiled.r, r);

    // The commitment tuple is `(b_noise_seed, a_noise_seed)` — the reverse of the order the names
    // suggest — taken as the verifier builds it rather than reordered, which is what a check that read it
    // the obvious way would get wrong.
    let rows_all: Vec<usize> = (0..m).collect();
    let cols_all: Vec<usize> = (0..n).collect();
    let noise = compute_noise_for_indices(k, r, compiled.commitment_hash, &rows_all, &cols_all);

    // Signal in the verifier's range, so every operand stays inside i8 and the blob's `satfinite` mma
    // never clamps: a disagreement here cannot be blamed on saturation.
    let secret_a: Vec<Vec<i8>> = (0..m)
        .map(|i| (0..k).map(|t| ((i * k + t) % 128) as i8 - 64).collect())
        .collect();
    let secret_b: Vec<Vec<i8>> = (0..n)
        .map(|j| (0..k).map(|t| ((j * k + t) % 128) as i8 - 64).collect())
        .collect();

    let mut a_sum: Vec<i8> = Vec::with_capacity(m * k);
    for i in 0..m {
        for t in 0..k {
            let sum = secret_a[i][t] as i32 + noise.a[i][t] as i32;
            if !(-128..=127).contains(&sum) {
                return Err(format!("A operand {sum} left i8"));
            }
            a_sum.push(sum as i8);
        }
    }
    let mut b_sum: Vec<i8> = Vec::with_capacity(n * k);
    for j in 0..n {
        for t in 0..k {
            let sum = secret_b[j][t] as i32 + noise.b[j][t] as i32;
            if !(-128..=127).contains(&sum) {
                return Err(format!("B operand {sum} left i8"));
            }
            b_sum.push(sum as i8);
        }
    }

    stream
        .memcpy_htod(&a_sum, &mut bufs.a_sum)
        .map_err(|e| format!("uploading the A operand failed: {e}"))?;
    stream
        .memcpy_htod(&b_sum, &mut bufs.b_sum)
        .map_err(|e| format!("uploading the B operand failed: {e}"))?;

    bufs.reset_for_attempt(stream)?;
    run_search(stream, triton, &bufs)?;

    let got = clone_u32(stream, &bufs.transcripts)?;

    let tiles_m = m / TILE_M;
    let tiles_n = n / TILE_N;
    let mut folds = HashSet::new();
    let mut named_a_different_pair = 0;

    for tile_m in 0..tiles_m {
        for tile_n in 0..tiles_n {
            for c in 0..HASH_CANDIDATES {
                let rows = [tile_m * TILE_M + 2 * c, tile_m * TILE_M + 2 * c + 1];
                let cols: Vec<usize> = (tile_n * TILE_N..tile_n * TILE_N + TILE_N).collect();

                let want = compute_jackpot(
                    &compiled,
                    &window(&secret_a, &rows),
                    &window(&secret_b, &cols),
                    &noise_window(&noise, &rows, &cols),
                );

                let base = (tile_m * tiles_n + tile_n) * HASH_CANDIDATES * JACKPOT_SIZE
                    + c * JACKPOT_SIZE;
                let transcript: [u32; JACKPOT_SIZE] =
                    got[base..base + JACKPOT_SIZE].try_into().unwrap();

                if transcript != want {
                    return Err(format!(
                        "tile ({tile_m}, {tile_n}) candidate {c}: the blob's transcript is not the \
                         verifier's fold for rows {rows:?}"
                    ));
                }

                folds.insert(want);

                // The index has to name *this* pair. If every candidate folded to the same value the
                // comparison above would pass on a transcript that identified nothing at all.
                let next = [rows[0] + 2, rows[1] + 2];
                if next[1] < tile_m * TILE_M + TILE_M {
                    let other = compute_jackpot(
                        &compiled,
                        &window(&secret_a, &next),
                        &window(&secret_b, &cols),
                        &noise_window(&noise, &next, &cols),
                    );
                    if other != want {
                        named_a_different_pair += 1;
                    }
                }
            }
        }
    }

    if folds.len() <= 1 {
        return Err(
            "every candidate folded to the same value, so the comparison identified nothing".to_string(),
        );
    }
    if named_a_different_pair == 0 {
        return Err("the candidate index did not distinguish neighbouring row pairs".to_string());
    }

    // Gate A, on the same transcripts: the hash the pool compares against, and the direction of the walk.
    //
    // `compute_jackpot_hash` is the verifier's own path — the sixteen words as little-endian bytes, keyed
    // with the A commitment root. `blake3_digest` is the same digest with the message built here from the
    // words rather than through the verifier, so the two agree only if the word-to-bytes reading is the
    // verifier's reading. A device that hashed the words as big-endian integers would match neither, and
    // one that hashed the right bytes under the wrong key would match neither either.
    run_postpass(stream, fatbin, &bufs)?;
    let got_hash = clone_u32(stream, &bufs.hash)?;

    for c in 0..candidates {
        let words: [u32; JACKPOT_SIZE] =
            got[c * JACKPOT_SIZE..(c + 1) * JACKPOT_SIZE].try_into().unwrap();
        let want = compute_jackpot_hash(&words, commit_a);

        for i in 0..8 {
            if got_hash[c * 8 + i] != u32::from_le_bytes(want[i * 4..][..4].try_into().unwrap()) {
                return Err(format!(
                    "candidate {c}: word {i} of the device digest is not the verifier's"
                ));
            }
        }

        if pearl_blake3::blake3_digest(&le_bytes(&words), Some(commit_a)) != want {
            return Err(
                "the jackpot message is not the sixteen words read as little-endian bytes".to_string(),
            );
        }
    }

    // Bounds built around candidate 0's own hash. The device's hit flag has to be the verifier's
    // comparison for every candidate, not just for the one the bound was built from: a walk that started
    // anywhere but the leading group would agree on a bound nudged by one about half the time, so the
    // direction has to be pinned by a bound that separates the leading group from a trailing one.
    let hash0: [u32; 8] = got_hash[..8].try_into().unwrap();
    let h0 = U256::from_little_endian(&le_bytes(&hash0));

    for (label, bound, want) in [
        ("equal to candidate 0", h0, true),
        ("one above", h0 + U256::one(), true),
        ("one below", h0 - U256::one(), false),
        ("leading group lowered", h0 - U256([0, 0, 0, 1 << 32]), false),
    ] {
        bufs.set_pow_target(stream, bound)?;
        run_postpass(stream, fatbin, &bufs)?;
        let hit = stream
            .clone_dtoh(&bufs.hit)
            .map_err(|e| format!("reading the hit flags failed: {e}"))?;

        for c in 0..candidates {
            let hash_c = U256::from_little_endian(&le_bytes(&got_hash[c * 8..(c + 1) * 8]));
            if (hit[c] == 1) != (hash_c <= bound) {
                return Err(format!("candidate {c} against a bound {label}"));
            }
        }
        if (hit[0] == 1) != want {
            return Err(format!("candidate 0 against a bound {label}"));
        }
    }

    // The case the two word conventions disagree on. The postpass compares word `i` against word `i`
    // walking from word 7 down — the little-endian reading, where word 7 is the leading group. The fused
    // kernel stored its bound as big-endian words and matched `words[7 - i]`, which pairs the bound's
    // leading group with the digest's trailing one. Raising the leading group and zeroing the trailing one
    // separates them: the little-endian walk accepts, the big-endian reading of the same words rejects.
    let mut pinned = hash0;
    pinned[7] += 1;
    pinned[0] = 0;
    if hash0[7] == u32::MAX || hash0[0] == 0 {
        return Err(format!(
            "this hash cannot separate the two conventions: {hash0:?}"
        ));
    }

    bufs.set_pow_target(stream, U256::from_little_endian(&le_bytes(&pinned)))?;
    run_postpass(stream, fatbin, &bufs)?;
    let hit = stream
        .clone_dtoh(&bufs.hit)
        .map_err(|e| format!("reading the hit flags failed: {e}"))?;
    if hit[0] != 1 {
        return Err("the postpass did not walk from the leading group".to_string());
    }
    if !(U256::from_big_endian(&le_bytes(&hash0)) > U256::from_big_endian(&le_bytes(&pinned))) {
        return Err(
            "this bound does not separate the conventions, so the assertion above measured nothing"
                .to_string(),
        );
    }

    Ok(())
}

/// Gate H: the grid reaches every candidate slot, on a poisoned buffer.
///
/// A zeroed buffer cannot tell an unwritten slot from a slot written as zero, and a single-tile grid
/// cannot tell a launch that covered the whole grid from one that only ever ran tile 0 — which is exactly
/// how the shipped hash loop covered eight tile columns for as long as its guard was always true. The
/// poison is a value the fold is unlikely to produce, so a word still holding it is an unwritten word.
///
/// At the forced configuration the blob writes all sixteen words of all sixty-four candidates in every
/// tile, so nothing may still hold the poison. A false failure here would need the fold to land on the
/// poison value itself.
pub fn check_grid_reaches_second_slot(
    stream: &Arc<CudaStream>,
    triton: &TritonKernels,
) -> Result<(), String> {
    // Six tiles: two tall, three wide. A square grid cannot tell `tile_m` from `tile_n`, and the swizzle
    // group is only exercised when the grid has more tiles than it.
    let config = SplitConfig::new(256, 384, 2048, 128)?;
    let (m, n, k) = (config.m, config.n, config.k);
    let tiles = config.num_tiles();
    let mut bufs = SplitBuffers::new(stream, &config)?;

    let a: Vec<i8> = (0..m * k).map(|i| ((i % 97) as i8) - 48).collect();
    let b: Vec<i8> = (0..n * k).map(|i| ((i % 89) as i8) - 44).collect();
    stream
        .memcpy_htod(&a, &mut bufs.a_sum)
        .map_err(|e| format!("uploading the A operand failed: {e}"))?;
    stream
        .memcpy_htod(&b, &mut bufs.b_sum)
        .map_err(|e| format!("uploading the B operand failed: {e}"))?;

    const POISON: u32 = 0xcccc_cccc;
    let poison: Vec<u32> = vec![POISON; tiles * HASH_CANDIDATES * JACKPOT_SIZE];
    stream
        .memcpy_htod(&poison, &mut bufs.transcripts)
        .map_err(|e| format!("poisoning the transcripts failed: {e}"))?;

    run_search(stream, triton, &bufs)?;
    let got = clone_u32(stream, &bufs.transcripts)?;

    let unwritten: Vec<usize> = (0..got.len()).filter(|i| got[*i] == POISON).collect();
    if !unwritten.is_empty() {
        return Err(format!(
            "{unwritten:?} still hold the poison, so those slots were never written"
        ));
    }

    let written_tiles: Vec<usize> = (0..tiles)
        .filter(|tile| {
            got[tile * HASH_CANDIDATES * JACKPOT_SIZE..(tile + 1) * HASH_CANDIDATES * JACKPOT_SIZE]
                .iter()
                .any(|word| *word != POISON)
        })
        .collect();
    if written_tiles.len() != tiles {
        return Err(format!(
            "the grid covered {} tiles of {tiles}, not every tile",
            written_tiles.len()
        ));
    }
    if !written_tiles.contains(&(tiles - 1)) {
        return Err("the last tile was not reached".to_string());
    }

    Ok(())
}

// =============================================================================
//   Helpers shared by the gates
// =============================================================================

fn clone_i8(stream: &Arc<CudaStream>, slice: &CudaSlice<i8>) -> Result<Vec<i8>, String> {
    stream
        .clone_dtoh(slice)
        .map_err(|e| format!("reading {} back from the device failed: {e}", slice.len()))
}

fn clone_u32(stream: &Arc<CudaStream>, slice: &CudaSlice<u32>) -> Result<Vec<u32>, String> {
    stream
        .clone_dtoh(slice)
        .map_err(|e| format!("reading {} back from the device failed: {e}", slice.len()))
}

/// A candidate's rows of a matrix, as `compute_jackpot` takes them.
fn window(matrix: &[Vec<i8>], indices: &[usize]) -> Vec<Vec<i8>> {
    indices.iter().map(|&i| matrix[i].clone()).collect()
}

/// The verifier's noise restricted to one candidate's window.
///
/// `compute_jackpot` indexes `noise.a[u]` for `u` in `0..h`, so the window has to be the noise for
/// exactly the rows it folds — passing the whole matrix would read the wrong rows for any candidate past
/// the first.
fn noise_window(
    noise: &zk_pow::circuit::pearl_noise::MMSlice,
    rows: &[usize],
    cols: &[usize],
) -> zk_pow::circuit::pearl_noise::MMSlice {
    zk_pow::circuit::pearl_noise::MMSlice {
        a: window(&noise.a, rows),
        b: window(&noise.b, cols),
        routing: vec![],
    }
}

fn le_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn flatten(rows: &[Vec<i8>]) -> Vec<i8> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

fn materialise(pairs: &[[u32; 2]], k: usize, rank: usize) -> Vec<i8> {
    let mut out = vec![0i8; k * rank];
    for (row, &[first, second]) in pairs.iter().enumerate() {
        out[row * rank + first as usize] = 1;
        out[row * rank + second as usize] = -1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    /// Opens a device once and hands the gate its stream, or returns `None` when there is no device to
    /// run on.
    ///
    /// The gates are functions rather than assertions so `self_test` can call them on every start; the
    /// tests below are the thin wrappers that skip when this machine has no CUDA device. One context per
    /// test, not one per helper call: a second `CudaContext::new` on the same device is a second context,
    /// and the buffers a gate allocates belong to whichever one it was handed.
    fn device() -> Option<(Arc<CudaContext>, Arc<CudaStream>, (i32, i32))> {
        let Ok(ctx) = CudaContext::new(0) else {
            return None;
        };
        let Ok(cc) = ctx.compute_capability() else {
            return None;
        };
        let stream = ctx.default_stream();
        Some((ctx, stream, cc))
    }

    #[test]
    #[cfg(feature = "pearl")]
    fn the_noise_entry_points_are_the_verifiers_noise() {
        let Some((ctx, stream, cc)) = device() else {
            return;
        };
        let Ok(fatbin) = FatbinKernels::load(&ctx, cc) else {
            return;
        };
        let (major, minor) = cc;
        eprintln!("gate B on device 0 (compute capability {major}.{minor})");

        check_noise_gen(&stream, &fatbin).expect("gate B");
    }

    #[test]
    #[cfg(feature = "pearl")]
    fn the_triton_noising_blob_agrees_with_the_verifiers_noise() {
        let Some((ctx, stream, cc)) = device() else {
            return;
        };
        let Ok(fatbin) = FatbinKernels::load(&ctx, cc) else {
            return;
        };
        let Ok(triton) = TritonKernels::load(&ctx, cc) else {
            return;
        };
        let (major, minor) = cc;
        eprintln!(
            "gate C on device 0 (compute capability {major}.{minor}): noising blob {}",
            triton.noising.arch()
        );

        check_noising(&stream, &fatbin, &triton).expect("gate C");
    }

    #[test]
    #[cfg(feature = "pearl")]
    fn the_postpass_hash_and_the_norotl_fold_are_the_verifiers() {
        let Some((ctx, stream, cc)) = device() else {
            return;
        };
        let Ok(fatbin) = FatbinKernels::load(&ctx, cc) else {
            return;
        };
        let Ok(triton) = TritonKernels::load(&ctx, cc) else {
            return;
        };
        let (major, minor) = cc;
        eprintln!(
            "gates A + D on device 0 (compute capability {major}.{minor}), search blob {}",
            triton.search.arch()
        );

        check_postpass(&stream, &fatbin, &triton).expect("gates A + D");
    }

    #[test]
    #[cfg(feature = "pearl")]
    fn the_search_grid_reaches_more_than_one_candidate_slot() {
        let Some((ctx, stream, cc)) = device() else {
            return;
        };
        let Ok(triton) = TritonKernels::load(&ctx, cc) else {
            return;
        };
        let (major, minor) = cc;
        eprintln!(
            "gate H on device 0 (compute capability {major}.{minor}), search blob {}",
            triton.search.arch()
        );

        check_grid_reaches_second_slot(&stream, &triton).expect("gate H");
    }

    /// Gate D's negative control: the same blob at a rank its checkpoint group cannot serve.
    ///
    /// The blob takes no rank at all — `BK * REDUCE_EVERY` is a compile-time constant — so it checkpoints
    /// every 128 columns whatever the verifier's fold groups. At `r = 256` the two cover the same columns
    /// only when `chk = 2t + 1`. That is what makes the disagreement checkable as a grouping difference
    /// rather than asserted as one: the blob's checkpoint `2t + 1` has to equal the verifier's tid `t`,
    /// its checkpoint `0` has to disagree with tid `0`, and the verifier's tids 8..15 have to stay zero
    /// while the blob writes sixteen checkpoints.
    ///
    /// The verifier's own `k >= 16r` also rejects `k = 2048, r = 256`, so this is not a configuration the
    /// pool would accept. It is the measurement that says why `SplitConfig::new` refuses it — and why the
    /// refusal is about the grouping, not about a buffer size.
    #[test]
    #[cfg(feature = "pearl")]
    fn the_norotl_fold_is_not_valid_when_the_group_size_differs() {
        use crate::miners::pearl_mining::{mining_configuration, PearlMining};
        use zk_pow::api::proof::{IncompleteBlockHeader, PublicProofParams, SeedDerivation};
        use zk_pow::api::proof_utils::CompiledPublicParams;
        use zk_pow::circuit::chip::compute_jackpot;
        use zk_pow::circuit::pearl_noise::compute_noise_for_indices;

        let Some((ctx, stream, cc)) = device() else {
            return;
        };
        // The same context the stream came from. A second `CudaContext::new` here would resolve the blob
        // against a different context, and the buffers this test allocates on `stream` would belong to the
        // first one — a launch from the second context over memory the first owns.
        let Ok(triton) = TritonKernels::load(&ctx, cc) else {
            return;
        };
        let (major, minor) = cc;
        eprintln!(
            "gate D control on device 0 (compute capability {major}.{minor}), search blob {}",
            triton.search.arch()
        );

        // A shape `SplitConfig::new` refuses, which is the point: the refusal has to be justified by a
        // measurement, not by a rule. So the buffers are allocated directly rather than through it.
        let (m, n, k, r) = (128usize, 256usize, 2048usize, 256usize);
        let tiles = (m / TILE_M) * (n / TILE_N);
        let tiles_n = n / TILE_N;

        let mining = PearlMining {
            m,
            n,
            k,
            rank: r as u16,
            rows_pattern: vec![0, 1],
            cols_pattern: (0..128).collect(),
            gzip: false,
        };
        let header = IncompleteBlockHeader {
            version: 0,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0,
        };
        let public = PublicProofParams::new_dummy(
            header,
            SeedDerivation::Salted,
            mining_configuration(&mining).unwrap(),
            m as u32,
            n as u32,
            0,
            0,
        );
        let compiled = CompiledPublicParams::from(&public);
        assert_eq!(compiled.r, r, "the fold being compared has to be the 256-column one");

        let rows_all: Vec<usize> = (0..m).collect();
        let cols_all: Vec<usize> = (0..n).collect();
        let noise = compute_noise_for_indices(k, r, compiled.commitment_hash, &rows_all, &cols_all);

        let secret_a: Vec<Vec<i8>> = (0..m)
            .map(|i| (0..k).map(|t| ((i * k + t) % 128) as i8 - 64).collect())
            .collect();
        let secret_b: Vec<Vec<i8>> = (0..n)
            .map(|j| (0..k).map(|t| ((j * k + t) % 128) as i8 - 64).collect())
            .collect();

        let mut a_sum: Vec<i8> = Vec::with_capacity(m * k);
        for i in 0..m {
            for t in 0..k {
                let sum = secret_a[i][t] as i32 + noise.a[i][t] as i32;
                assert!((-128..=127).contains(&sum), "A operand {sum} left i8");
                a_sum.push(sum as i8);
            }
        }
        let mut b_sum: Vec<i8> = Vec::with_capacity(n * k);
        for j in 0..n {
            for t in 0..k {
                let sum = secret_b[j][t] as i32 + noise.b[j][t] as i32;
                assert!((-128..=127).contains(&sum), "B operand {sum} left i8");
                b_sum.push(sum as i8);
            }
        }

        let a_dev = stream.clone_htod(&a_sum).unwrap();
        let b_dev = stream.clone_htod(&b_sum).unwrap();
        let t_dev = stream
            .alloc_zeros::<u32>(tiles * HASH_CANDIDATES * JACKPOT_SIZE)
            .unwrap();

        triton.search
            .launch(&stream, m as i32, n as i32, k as i32, &a_dev, &b_dev, &t_dev)
            .unwrap();

        let got = stream.clone_dtoh(&t_dev).unwrap();

        // Candidate 0 of tile 1, so the grid is not the first tile. On a 1 x 2 grid tile_linear 1 is
        // tile (0, 1) — the tile index has to come from the linear index, not from assuming it is the
        // first one, which is what this test itself got wrong the first time it ran.
        let tile_linear = 1;
        let tile_m = tile_linear / tiles_n;
        let tile_n = tile_linear % tiles_n;
        assert_eq!((tile_m, tile_n), (0, 1), "the linear tile index is not the pair it names");

        let rows = [tile_m * TILE_M, tile_m * TILE_M + 1];
        let cols: Vec<usize> = (tile_n * TILE_N..tile_n * TILE_N + TILE_N).collect();
        let want = compute_jackpot(
            &compiled,
            &window(&secret_a, &rows),
            &window(&secret_b, &cols),
            &noise_window(&noise, &rows, &cols),
        );
        let base = tile_linear * HASH_CANDIDATES * JACKPOT_SIZE;

        // Where the two groupings cover the same columns, they agree — which is the running-XOR claim,
        // checked rather than asserted.
        for t in 0..8 {
            assert_eq!(
                got[base + 2 * t + 1],
                want[t],
                "checkpoint {} is not the verifier's tid {t}",
                2 * t + 1
            );
        }

        // And where they do not, they disagree. Checkpoint 0 covers 128 columns; the verifier's first
        // group covers 256.
        assert_ne!(
            got[base], want[0],
            "the blob's first checkpoint matched a 256-column group, which it cannot cover"
        );

        // The verifier never reaches tids 8..15 at this rank, so they stay zero; the blob writes sixteen
        // checkpoints regardless. That is the overrun `SplitConfig::new` refuses.
        for t in 8..16 {
            assert_eq!(want[t], 0, "the verifier's tid {t} should never have been written");
        }
        assert!(
            got[base + 8..base + 16].iter().any(|word| *word != 0),
            "the blob wrote nothing past the verifier's reach, so the overrun was not measured"
        );
    }

    /// The coordinate the scan reports, on a grid that is not square.
    ///
    /// The scan reports one number, so the split between tile and candidate is the whole mapping, and a
    /// square grid cannot tell `tile_linear / tiles_n` from `tile_linear % tiles_n`. Which row pair a
    /// candidate index names is the device half — gate D — and this is the half that pins the arithmetic
    /// `assemble` consumes.
    #[test]
    fn the_candidate_index_maps_to_the_rows_the_emit_claims() {
        let config = SplitConfig::new(256, 384, 2048, 128).unwrap();
        let tiles_n = config.n / TILE_N;
        assert_eq!(tiles_n, 3, "the grid has to be non-square for this to say anything");

        // Candidate 5 * 64 + 7: tile_linear 5, which on a 2 x 3 grid is tile (1, 2) — `5 / 3` and `5 % 3`,
        // not the other way round.
        let hit = Hit::from_candidate(&config, 5 * HASH_CANDIDATES + 7);
        assert_eq!(hit.tile_linear, 5);
        assert_eq!((hit.tile_m, hit.tile_n), (1, 2));
        assert_eq!(hit.a_rows, [TILE_M + 14, TILE_M + 15]);
        assert_eq!(hit.b_col, 2 * TILE_N);

        // The transposed reading names a different tile here, which is why the shape matters: on a square
        // grid the two readings agree and the test cannot tell them apart.
        let transposed = (5 % tiles_n, 5 / tiles_n);
        assert_ne!((hit.tile_m, hit.tile_n), transposed, "a square grid would have made this vacuous");

        // The last candidate the scan can report is the last slot of the last tile. `candidate` is the
        // number the scan writes — the global index — so the slot within the tile has to be taken off it,
        // which is the distinction `read_first_hit` depends on.
        let last = Hit::from_candidate(&config, config.total_candidates() - 1);
        assert_eq!(last.candidate, config.total_candidates() - 1);
        assert_eq!(
            last.candidate % HASH_CANDIDATES,
            HASH_CANDIDATES - 1,
            "the field is the global index, not the slot"
        );
        assert_eq!(last.tile_linear, config.num_tiles() - 1);
        // Hard-coded rather than recomputed from `last`, so this is a second reading: the last slot of the
        // last tile names that tile's last row pair, not the first tile's.
        assert_eq!((last.tile_m, last.tile_n), (1, 2));
        assert_eq!(last.a_rows, [TILE_M + TILE_M - 2, TILE_M + TILE_M - 1]);
        assert_eq!(last.b_col, (tiles_n - 1) * TILE_N);
    }
}
