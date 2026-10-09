//! The device buffers the split search path owns, and the configuration they force.
//!
//! The fused path kept its buffers in [`crate::miners::pearl_gpu::Workspace`] because every stage
//! wrote into the same few arrays and the hash happened inside the search launch. The split path
//! cannot share those arrays: the search blob writes a transcript buffer that a *separate* postpass
//! hashes, so the transcript has to survive from one launch to the next, and three buffers have to be
//! re-zeroed between attempts for reasons that are correctness rather than hygiene — see
//! [`SplitBuffers::reset_for_attempt`].
//!
//! The configuration is not a preference. Three constraints meet at exactly one shape, and
//! [`SplitConfig::new`] refuses anything else rather than launching a grid that computes a different
//! answer. The derivation is recorded with the constants, because the shape looks arbitrary without it.
//!
//! One convention is worth naming before the buffers are read: the postpass compares word `i` of the
//! digest against word `i` of the target, walking from word 7 down. That is the little-endian reading
//! of a uint256 — word 7 is the most significant group — which is the reading the verifier applies to
//! a jackpot hash. The fused kernel stored its bound as *big-endian* words and matched them against
//! `words[7 - i]`; the two conventions are equivalent only while the index and the byte order are
//! independently right, so the split path's target buffer is built in little-endian word order and a
//! test asserts that the walk agrees with `U256` arithmetic at the boundary.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use primitive_types::U256;

use super::cuda::to_words;
use super::fatbin::FIRST_HIT_SENTINEL;
use super::triton::{HASH_CANDIDATES, JACKPOT_SIZE, SEARCH_BLOCK_M, SEARCH_BLOCK_N};

/// The tile the search blob was specialised for: 128 rows of `A` against 128 columns of `Bt`.
///
/// The blob computes `num_pid_m = M / BM` by truncation, so a shape that is not a whole number of
/// these tiles is not a smaller grid — it is a grid whose last partial tile is never launched, and the
/// candidates inside it are silently never hashed.
pub const TILE_M: usize = SEARCH_BLOCK_M;
pub const TILE_N: usize = SEARCH_BLOCK_N;

/// The column group the search blob checkpoints: `BK * REDUCE_EVERY = 64 * 2`.
///
/// Its accumulator is never reset between k-iterations, so each checkpoint is the XOR of the *running*
/// sums, not of one group. That is only the verifier's fold when the checkpoint group equals the fold
/// group, which is what forces the rank below.
pub const CHECKPOINT_GROUP: usize = 128;

/// The rank the split path forces, and the `k` that follows from it.
///
/// The chain, in the order it has to be read:
///
/// 1. The checkpoint group is 128 columns and the fold groups `r` columns per iteration, so `r = 128`.
/// 2. The blob writes one word per checkpoint into a 16-word slot per candidate, so `k / r <= 16`.
///    `zk_pow::api::sanity_checks` requires `k >= 16r`. Together they close the interval: `k = 16r`.
/// 3. At `r = 128` that is `k = 2048`.
///
/// `k / r > 16` is not a different answer, it is a buffer overrun: `chk` runs past the candidate's slot
/// into the next candidate's. `k / r < 16` is not wrong on its own — the verifier's jackpot slots above
/// `k/r` are zero and the pre-zeroed transcript matches them — but the verifier's own `k >= 16r` rules
/// it out, so the equality is the only shape left.
pub const FORCED_RANK: usize = CHECKPOINT_GROUP;
pub const FORCED_K: usize = FORCED_RANK * JACKPOT_SIZE;

/// BLAKE3's chunk length: the matrices are padded to a whole number of these before they are committed
/// to, so the buffers carry the padding the commitment hashes rather than a copy of the matrix.
pub const CHUNK: usize = 1024;

/// The noise generation's seed labels, padded to a full BLAKE3 seed.
///
/// These are the verifier's `SEED_LABEL_A` / `SEED_LABEL_B`. They cannot be pinned by a test that
/// passes the same constant to both sides — the device half of gate B pins the decode and the chunk
/// mapping, not the label. The only reference that could pin the label is a captured third-party share,
/// because that is the one reference here that is not a second reading of the same algorithm — and there
/// is no such fixture in this tree yet (`tests/fixtures/` is empty), so as things stand the labels are
/// taken on trust from the reference. A wrong label is not caught anywhere: it produces a noise tensor
/// that is correct for a different seed, and every downstream stage composes it perfectly.
pub const SEED_LABEL_A: [u8; 32] = padded_label(b"A_tensor");
pub const SEED_LABEL_B: [u8; 32] = padded_label(b"B_tensor");

const fn padded_label(label: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < label.len() {
        out[i] = label[i];
        i += 1;
    }
    out
}

/// A configuration the split path can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub rank: usize,
}

impl SplitConfig {
    /// The reference's default shape: `m = 8192`, `n = 32768`, `k = 2048`, `r = 128`.
    pub fn production() -> Self {
        Self {
            m: 8192,
            n: 32768,
            k: FORCED_K,
            rank: FORCED_RANK,
        }
    }

    /// A small shape the tests can run on a device: two tiles, one in each direction.
    pub fn test_shape() -> Self {
        Self {
            m: 2 * TILE_M,
            n: 2 * TILE_N,
            k: FORCED_K,
            rank: FORCED_RANK,
        }
    }

    /// Checks the configuration against what the blobs and the verifier require, refusing anything else.
    ///
    /// Refused rather than discovered: a grid that truncates a partial tile still launches, still
    /// hashes, and still produces self-consistent candidates the pool rejects. The same is true of a
    /// `k / r` above 16 — the writes land in the next candidate's slot and the hash of a transcript
    /// that was never the verifier's jackpot still looks like a plausible share.
    pub fn new(m: usize, n: usize, k: usize, rank: usize) -> Result<Self, String> {
        if m % TILE_M != 0 {
            return Err(format!(
                "m = {m} is not a whole number of {TILE_M}-row tiles, so the blob's `M / BM` truncates \
                 and the last partial tile is never launched"
            ));
        }
        if n % TILE_N != 0 {
            return Err(format!(
                "n = {n} is not a whole number of {TILE_N}-column tiles, so the blob's `N / BN` \
                 truncates and the last partial tile is never launched"
            ));
        }
        if rank != FORCED_RANK {
            return Err(format!(
                "the split path requires rank {FORCED_RANK} (the blob's checkpoint group), but this is \
                 {rank}: the fold would group {rank} columns while the checkpoint groups \
                 {CHECKPOINT_GROUP}, so the transcript is the XOR of a different grouping than the \
                 verifier's"
            ));
        }
        if k != FORCED_K {
            return Err(format!(
                "the split path requires k = 16 * rank = {FORCED_K}, but this is {k}: k / r = {} \
                 writes {} checkpoints per candidate into a {JACKPOT_SIZE}-word slot, and the verifier \
                 requires k >= 16r",
                k / rank,
                k / rank,
            ));
        }
        Ok(Self { m, n, k, rank })
    }

    /// Tiles in the grid the search launch covers.
    pub fn num_tiles(&self) -> usize {
        (self.m / TILE_M) * (self.n / TILE_N)
    }

    /// Candidates the postpass and the scan have to cover: one per tile slot the blob writes.
    pub fn total_candidates(&self) -> usize {
        self.num_tiles() * HASH_CANDIDATES
    }

    /// Device bytes this configuration's buffers hold, excluding the commitment's scratch.
    pub fn footprint(&self) -> usize {
        let (m, n, k, r) = (self.m, self.n, self.k, self.rank);
        padded(m * k)
            + padded(n * k)
            + 5 * 32 // two commitments, two seed labels, the target
            + m * r
            + n * r
            + 2 * k * r
            + 2 * m * k
            + 2 * n * k
            + self.total_candidates() * (JACKPOT_SIZE * 4 + 1 + 8 * 4)
            + 4
    }
}

/// The buffers, allocated once per configuration and reused for every attempt.
///
/// The two signal matrices and the two summed ones are separate rather than written in place, for the
/// same reason the fused path kept them separate: `A` and `Bt` are what the proof commits to, and once
/// they have been noised in place the signal is gone.
pub struct SplitBuffers {
    pub config: SplitConfig,

    /// `A[M,K]` and `Bt[N,K]`, each padded up to a whole number of BLAKE3 chunks.
    ///
    /// The tail past `a_len` / `bt_len` is zero and stays that way: `alloc_zeros` wrote it, only the
    /// fill ever writes these buffers, and it is only ever asked for the unpadded length. A fill that
    /// ran into the padding would commit to a matrix the verifier never sees.
    pub a: CudaSlice<i8>,
    pub bt: CudaSlice<i8>,
    pub a_len: usize,
    pub bt_len: usize,

    /// The two commitment roots, as eight little-endian words: the key the noise generation runs under
    /// and, for `commit_a`, the `pow_key` the postpass hashes under.
    pub commit_a: CudaSlice<u32>,
    pub commit_b: CudaSlice<u32>,

    /// The noise generation's seed labels, uploaded once.
    pub seed_label_a: CudaSlice<u32>,
    pub seed_label_b: CudaSlice<u32>,

    /// The share bound, in little-endian word order — see the module comment.
    pub pow_target: CudaSlice<u32>,

    /// `EAL[M,R]` and `EBR[N,R]`: the dense noise the sparse pairs index into.
    pub eal: CudaSlice<i8>,
    pub ebr: CudaSlice<i8>,

    /// `EAR` and `EBL`, materialised as dense `(k, R)` matrices: two non-zeros per row, `+1` at `k0`
    /// and `-1` at `k1`.
    pub ear: CudaSlice<i8>,
    pub ebl: CudaSlice<i8>,

    /// `signal + noise`, which is what the search blob reads.
    pub a_sum: CudaSlice<i8>,
    pub b_sum: CudaSlice<i8>,

    /// `(num_tiles, HASH_CANDIDATES, JACKPOT_SIZE)` as u32 — the transcript the postpass hashes.
    pub transcripts: CudaSlice<u32>,

    /// One byte per candidate: the postpass's bound comparison, which the scan then reads.
    pub hit: CudaSlice<u8>,

    /// Eight words per candidate. Written for the checks; production reads only `hit`, but the postpass
    /// entry point does not tolerate a null hash pointer, so the buffer has to exist.
    pub hash: CudaSlice<u32>,

    /// The scan's result: the first candidate index that cleared the bound, or the sentinel.
    pub first_hit: CudaSlice<u32>,
}

impl SplitBuffers {
    /// Allocates every buffer for a configuration.
    ///
    /// Everything starts zeroed, which is the required pre-state for the three buffers that need one:
    /// the sparse noise output, the transcript, and the scan sentinel. The sentinel is the exception —
    /// zero is a candidate index, so it has to be set to the value the scan cannot produce.
    pub fn new(stream: &Arc<CudaStream>, config: &SplitConfig) -> Result<Self, String> {
        let (m, n, k, r) = (config.m, config.n, config.k, config.rank);
        let candidates = config.total_candidates();

        let mut first_hit = allocate(stream, "first_hit", 1)?;
        stream
            .memcpy_htod(&[FIRST_HIT_SENTINEL], &mut first_hit)
            .map_err(|e| format!("setting the scan sentinel failed: {e}"))?;

        Ok(Self {
            config: *config,
            a: allocate(stream, "a", padded(m * k))?,
            bt: allocate(stream, "bt", padded(n * k))?,
            a_len: m * k,
            bt_len: n * k,
            commit_a: allocate(stream, "commit_a", 8)?,
            commit_b: allocate(stream, "commit_b", 8)?,
            seed_label_a: upload(stream, "seed_label_a", &to_words(&SEED_LABEL_A))?,
            seed_label_b: upload(stream, "seed_label_b", &to_words(&SEED_LABEL_B))?,
            pow_target: allocate(stream, "pow_target", 8)?,
            eal: allocate(stream, "eal", m * r)?,
            ebr: allocate(stream, "ebr", n * r)?,
            ear: allocate(stream, "ear", k * r)?,
            ebl: allocate(stream, "ebl", k * r)?,
            a_sum: allocate(stream, "a_sum", m * k)?,
            b_sum: allocate(stream, "b_sum", n * k)?,
            transcripts: allocate(stream, "transcripts", candidates * JACKPOT_SIZE)?,
            hit: allocate(stream, "hit", candidates)?,
            hash: allocate(stream, "hash", candidates * 8)?,
            first_hit,
        })
    }

    /// Puts the two commitment roots on the device as eight words each.
    pub fn set_commitment(
        &mut self,
        stream: &Arc<CudaStream>,
        commit_a: &[u8; 32],
        commit_b: &[u8; 32],
    ) -> Result<(), String> {
        upload_into(stream, "commit_a", &to_words(commit_a), &mut self.commit_a)?;
        upload_into(stream, "commit_b", &to_words(commit_b), &mut self.commit_b)?;
        Ok(())
    }

    /// Puts the share bound on the device in little-endian word order.
    pub fn set_pow_target(&mut self, stream: &Arc<CudaStream>, bound: U256) -> Result<(), String> {
        let mut bytes = [0u8; 32];
        bound.to_little_endian(&mut bytes);
        upload_into(stream, "pow_target", &to_words(&bytes), &mut self.pow_target)
    }

    /// The buffers that must be re-zeroed before every attempt.
    ///
    /// Two, each for a different reason:
    ///
    /// * `transcripts` — the search blob writes only the slots it visits. At `k / r == 16` it visits all
    ///   sixteen, but the verifier's jackpot slots are zero for a fold that never reached them, so the
    ///   buffer's pre-state is part of the answer, not a convenience.
    /// * `first_hit` — `atomicMin` against the sentinel is what makes "no hit" readable as one word. A
    ///   sentinel left from the previous attempt is a hit the scan cannot undo, and a zero is candidate
    ///   index zero.
    ///
    /// `ear` / `ebl` are not here on purpose. The sparse entry point patches `+1` and `-1` into rows it
    /// never clears, so a second attempt accumulates onto the first — but the zero that fixes that has to
    /// be *stream-ordered before the launch that needs it*, which is why it lives in
    /// [`super::pipeline::run_noise_gen`] rather than in a reset a caller could reorder. A reset here
    /// would be a correct fact in the wrong place: nothing would stop the launch from running first.
    ///
    /// `hit` is not in this list because the postpass writes every byte of it.
    pub fn reset_for_attempt(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
        stream
            .memset_zeros(&mut self.transcripts)
            .map_err(|e| format!("zeroing transcripts failed: {e}"))?;

        stream
            .memcpy_htod(&[FIRST_HIT_SENTINEL], &mut self.first_hit)
            .map_err(|e| format!("resetting the scan sentinel failed: {e}"))?;
        Ok(())
    }
}

/// A zeroed device buffer.
///
/// `cudarc` 0.19 has `memset_zeros` but no memset-with-value, which is why the sentinel is written by a
/// host upload rather than by a memset of `0xFFFFFFFF`.
fn allocate<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    name: &str,
    len: usize,
) -> Result<CudaSlice<T>, String> {
    stream
        .alloc_zeros::<T>(len)
        .map_err(|e| format!("allocating {name} ({len} x {}) failed: {e}", std::any::type_name::<T>()))
}

fn upload(stream: &Arc<CudaStream>, name: &str, words: &[u32; 8]) -> Result<CudaSlice<u32>, String> {
    stream
        .clone_htod(words)
        .map_err(|e| format!("uploading {name} failed: {e}"))
}

fn upload_into(
    stream: &Arc<CudaStream>,
    name: &str,
    words: &[u32; 8],
    dst: &mut CudaSlice<u32>,
) -> Result<(), String> {
    stream
        .memcpy_htod(words, dst)
        .map_err(|e| format!("uploading {name} failed: {e}"))
}

fn padded(bytes: usize) -> usize {
    bytes.div_ceil(CHUNK) * CHUNK
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every condition `SplitConfig::new` refuses is one that would otherwise launch, hash and report a
    /// plausible answer. So the test does not just check that a bad shape is refused — it checks that the
    /// shape the *shipped default* uses is refused, because that one is valid everywhere else.
    ///
    /// The verifier side of the same claim (that the split configuration passes `public_params_sanity_check`
    /// and that the difficulty factor halves at it) is `pearl_mining`'s gate E; this is the half that says
    /// the blobs cannot run it.
    #[test]
    fn the_split_path_refuses_a_configuration_the_blobs_cannot_satisfy() {
        // The shape the derivation lands on.
        let ok = SplitConfig::new(8192, 32768, 2048, 128).expect("the forced shape should be accepted");
        assert_eq!(ok, SplitConfig::production());

        // A partial tile is not a smaller grid: `M / BM` truncates, and the candidates inside the
        // truncated tile are never hashed. 64 columns off a 128-wide tile is the smallest case that
        // still leaves a tile to truncate.
        assert!(SplitConfig::new(8192 + 64, 32768, 2048, 128).is_err());
        assert!(SplitConfig::new(8192, 32768 + 64, 2048, 128).is_err());

        // The shipped default: fold-valid for the verifier, but its fold groups 256 columns while the
        // blob checkpoints every 128, so the transcript is a different grouping.
        let default = SplitConfig::new(8192, 32768, 4096, 256);
        assert!(default.is_err(), "rank 256 is not the checkpoint group");

        // k / r above 16 is an overrun, not a different answer.
        let overrun = SplitConfig::new(8192, 32768, 4096, 128).unwrap_err();
        assert!(overrun.contains("32"), "the message should name the overrun: {overrun}");

        // k / r below 16 is not wrong on its own — the verifier's slots above k/r are zero and the
        // pre-zeroed transcript matches them — but the verifier's own `k >= 16r` rules it out, so the
        // equality is the only shape left. The message has to say which rule bit, because the two rules
        // point in opposite directions.
        let short = SplitConfig::new(8192, 32768, 1024, 128).unwrap_err();
        assert!(short.contains("8"), "the message should name the count: {short}");
        assert!(short.contains("16r"), "the message should name the verifier's rule: {short}");
    }

    /// The tile count the launch grid is built from, on a shape that is not square.
    ///
    /// A square grid cannot tell `(m/128) * (n/128)` from `m * n / 128` or from a mapping that reads the
    /// two directions in the opposite order, and the scan's candidate index is `tile * 64 + c` — a
    /// transposed tile count changes which candidate the scan reports, not just how many there are.
    #[test]
    fn the_grid_counts_tiles_the_way_the_scan_indexes_them() {
        let config = SplitConfig::new(384, 256, 2048, 128).unwrap();
        assert_eq!(config.num_tiles(), 3 * 2);
        assert_eq!(config.total_candidates(), 3 * 2 * 64);

        // The scan reports `first_hit / 64` as the tile and `first_hit % 64` as the candidate, so the
        // tile the scan names has to be the one the grid counted.
        let last = config.total_candidates() - 1;
        assert_eq!(last / 64, config.num_tiles() - 1);
        assert_eq!(last % 64, 63);
    }

    /// The word order the postpass compares in, checked against the number the verifier holds.
    ///
    /// The postpass walks from word 7 down and compares word `i` against word `i`, so word 7 has to be
    /// the most significant group — the little-endian reading `U256::from_little_endian` gives a jackpot
    /// hash. The fused kernel stored its bound as big-endian words and matched them against `words[7 - i]`,
    /// which is the same answer only while the index and the byte order are independently right.
    ///
    /// The cases are the ones that rule out a reversed walk: a bound whose leading group is lowered while
    /// every group below it is held equal must reject, and a bound one below must reject at the other end.
    /// A walk that started anywhere but the leading group would see only equality and accept.
    ///
    /// This pins the convention, not the kernel: the model here is the walk the header documents, and the
    /// reference is `U256` arithmetic. The device half is gate A.
    #[test]
    fn the_pow_target_words_are_the_order_the_postpass_walks() {
        // A hash with a small leading group and a large trailing one. That is the shape that makes the
        // two conventions disagree: with a large leading group, a walk that paired the target's leading
        // group with the digest's trailing one would reject for the wrong reason and look correct.
        let hash: [u32; 8] = [
            0xffff_ffff, 0xdead_beef, 0x8bad_f00d, 0xcafe_babe, 1, 2, 3, 0x0000_0001,
        ];
        let hash_bytes: [u8; 32] = words_to_le_bytes(&hash);
        let as_u256 = U256::from_little_endian(&hash_bytes);

        let cases = [
            ("equal", as_u256, true),
            ("one above", as_u256 + U256::one(), true),
            ("one below", as_u256 - U256::one(), false),
            (
                "leading group lowered",
                as_u256 - U256([0, 0, 0, 1 << 32]),
                false,
            ),
        ];

        for (label, bound, should_accept) in cases {
            assert_eq!(
                walk(&hash, &le_words(bound)),
                should_accept,
                "a bound {label} the little-endian walk got wrong"
            );
        }

        // The other convention on the same bound: the big-endian word order the fused kernel used, which
        // pairs the target's leading group with the digest's trailing one. On this hash it accepts a bound
        // the verifier rejects, which is the disagreement the convention has to be checked against — a test
        // that only nudged the bound by one would pass a reversed walk about half the time.
        let lowered = as_u256 - U256([0, 0, 0, 1 << 32]);
        assert!(
            walk(&hash, &be_words(lowered)),
            "the big-endian convention should disagree on the case that pins the walk direction"
        );
    }

    /// The walk the postpass header documents.
    fn walk(hash: &[u32; 8], target: &[u32; 8]) -> bool {
        for i in (0..8).rev() {
            if hash[i] > target[i] {
                return false;
            }
            if hash[i] < target[i] {
                return true;
            }
        }
        true
    }

    fn le_words(value: U256) -> [u32; 8] {
        let mut bytes = [0u8; 32];
        value.to_little_endian(&mut bytes);
        to_words(&bytes)
    }

    fn be_words(value: U256) -> [u32; 8] {
        let mut bytes = [0u8; 32];
        value.to_big_endian(&mut bytes);
        std::array::from_fn(|i| u32::from_be_bytes(bytes[i * 4..][..4].try_into().unwrap()))
    }

    fn words_to_le_bytes(words: &[u32; 8]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, word) in words.iter().enumerate() {
            out[i * 4..][..4].copy_from_slice(&word.to_le_bytes());
        }
        out
    }
}
