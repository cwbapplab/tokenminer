//! The Pearl mining profile and the job_key derivation.
//!
//! A byte-for-byte port of llmjob's `earn/src/shared/miner/pearlhash.js` (itself
//! the C mirror of the reference `MiningConfiguration::to_bytes`). `config52` is
//! hashed with the 76-byte header to derive `job_key`; one wrong byte changes
//! every hash downstream with no error anywhere, so this layout is the single
//! source of truth and is pinned by the `oracle.rs` vectors.

/// Hashed into config52 — protocol-mandated, must match the network exactly.
pub const CONFIG_BYTES: usize = 52;
pub const HEADER_BYTES: usize = 76;
pub const HASH_BYTES: usize = 32;

/// The tiling patterns are protocol constants, not per-profile knobs. Both
/// counts are 16, so the candidate tile is 16x16 = 256 cells — the same cell
/// count as the geometry the client used before this port, which is why the
/// hashrate unit and the share bound do not move.
pub const ROWS_PATTERN: [u32; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27];
pub const COLS_PATTERN: [u32; 16] =
    [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57];

/// The two pattern byte blocks the header carries, asserted equal to
/// `PEARL_ROWS_PATTERN_BYTES` / `PEARL_COLS_PATTERN_BYTES` in pearl_config.h.
const ROWS_PATTERN_BYTES: [u8; 6] = [0, 3, 1, 3, 0, 0];
const COLS_PATTERN_BYTES: [u8; 6] = [0, 1, 3, 7, 0, 0];

/// Seed derivation. Salting commutes with everything but commits `m` and `n`
/// (which are deliberately absent from config52), so it is the live default.
pub const SEED_SALTED: u32 = 0;
pub const SEED_LEGACY: u32 = 1;

/// Operand fill. Constant is what the mainnet profile ships, and the only value
/// this miner uses: it draws every byte of the operand from the profile's
/// constant and stamps the salt over it, which is what the mainnet profile
/// requires. The hashed fill (0) exists in the core for the reference tests and
/// is deliberately absent here.
pub const OPERAND_CONST: u32 = 1;

/// The rank the penalty is measured against.
pub const PENALTY_BASE_RANK: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    pub k: u32,
    pub rank: u16,
    pub mma_type: u16,
    pub m: u32,
    pub n: u32,
    pub seed_derivation: u32,
    pub col_batch: u32,
    pub hash_big_endian: u32,
    pub operand_fill: u32,
    pub rows: [u32; 16],
    pub cols: [u32; 16],
}

/// The mandated mainnet profile, from `pearlhash.js` PROFILE and the core's own
/// `PEARL_MAINNET_PROFILE` (`{2048, 128, 0, 131072, 262144, SALTED, 2048, 0,
/// CONST}`). `m`/`n` are the miner's own and never enter job_key.
pub const PROFILE: Profile = Profile {
    k: 2048,
    rank: 128,
    mma_type: 0,
    m: 131_072,
    n: 262_144,
    seed_derivation: SEED_SALTED,
    col_batch: 2048,
    hash_big_endian: 0,
    operand_fill: OPERAND_CONST,
    rows: ROWS_PATTERN,
    cols: COLS_PATTERN,
};

impl Profile {
    /// The candidate tile's area in cells: `rows * cols`.
    pub fn tile_size(&self) -> u32 {
        self.rows.len() as u32 * self.cols.len() as u32
    }

    pub fn with_seed_derivation(mut self, seed_derivation: u32) -> Self {
        self.seed_derivation = seed_derivation;
        self
    }
}

/// Map a network certificate version to the core's seed derivation.
///
/// Certificate 3 is cert-v3 salted; everything older is legacy. Salting is
/// llmjob's live path and is what the pool expects at cert 3 — if shares are
/// rejected, this is the first flag to flip.
pub fn seed_derivation_for(cert_version: u32) -> u32 {
    if cert_version == 3 {
        SEED_SALTED
    } else {
        SEED_LEGACY
    }
}

/// The 52-byte config block, little-endian:
/// `k u32 @0 | rank u16 @4 | mma_type u16 @6 | rows(6) @8 | cols(6) @14 | MoE(32) @20`.
pub fn build_config52(p: &Profile) -> [u8; CONFIG_BYTES] {
    let mut b = [0u8; CONFIG_BYTES];
    b[0..4].copy_from_slice(&p.k.to_le_bytes());
    b[4..6].copy_from_slice(&p.rank.to_le_bytes());
    b[6..8].copy_from_slice(&p.mma_type.to_le_bytes());
    b[8..14].copy_from_slice(&ROWS_PATTERN_BYTES);
    b[14..20].copy_from_slice(&COLS_PATTERN_BYTES);
    // Bytes 20..51 are the MoE trailer, zero for a standard (non-GROUPED_GEMM) job.
    b
}

/// `job_key = blake3(header76 || config52)`, unkeyed — matching the core
/// (`pearl_host.cu`) and `reference.js`.
pub fn job_key(header: &[u8; HEADER_BYTES], config: &[u8; CONFIG_BYTES]) -> [u8; HASH_BYTES] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(header);
    hasher.update(config);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published pattern bytes and the pattern lists have to agree, or the
    /// kernel and this header hash different configs.
    #[test]
    fn pattern_bytes_match_the_reference_lists() {
        assert_eq!(pattern_to_bytes(&ROWS_PATTERN), ROWS_PATTERN_BYTES);
        assert_eq!(pattern_to_bytes(&COLS_PATTERN), COLS_PATTERN_BYTES);
    }

    /// Ported from pearlhash.js `patternFromList`: recover the (stride, length)
    /// dimensions from an index list so `patternToBytes` can re-emit them.
    fn pattern_from_list(list: &[u32]) -> Vec<(u32, usize)> {
        let mut p: Vec<u32> = list.to_vec();
        let mut shape: Vec<(u32, usize)> = Vec::new();
        while p.len() > 1 {
            let mut found = false;
            for period in 1..p.len() {
                if p.len() % period != 0 {
                    continue;
                }
                let stride = p[period];
                let mut periodic = true;
                let mut i = 0;
                while i + period < p.len() {
                    if p[i] + stride != p[i + period] {
                        periodic = false;
                        break;
                    }
                    i += 1;
                }
                if !periodic {
                    continue;
                }
                // `unshift`: smallest-stride dim first.
                shape.insert(0, (stride, p.len() / period));
                p.truncate(period);
                found = true;
                break;
            }
            assert!(found, "index pattern is not periodic: {list:?}");
        }
        shape
    }

    /// Six bytes: (factor-1, length-1) per dimension, third padded to factor 1.
    ///
    /// The pad carries the **last** dimension's span (`stride * length`), which
    /// is also what the verifier calls the pattern's period — not the product of
    /// all dimensions. Getting that wrong is invisible at the shape level: the
    /// tile is still correct and every hash still agrees, and only the third
    /// six-byte field changes, which is why the pool reads a different pattern
    /// than the miner mined.
    fn pattern_to_bytes(list: &[u32]) -> [u8; 6] {
        let shape = pattern_from_list(list);
        let mut out = [0u8; 6];
        let mut min_stride: u32 = 1;
        let pad = shape.last().map_or(1, |&(stride, length)| stride * length as u32);
        let dims: Vec<(u32, usize)> = {
            let mut d = shape.clone();
            while d.len() < 3 {
                d.push((pad, 1));
            }
            d
        };
        for i in 0..3 {
            let (stride, length) = dims[i];
            let factor = stride / min_stride;
            out[2 * i] = (factor - 1) as u8;
            out[2 * i + 1] = (length - 1) as u8;
            min_stride = stride * length as u32;
        }
        out
    }

    #[test]
    fn config52_layout_is_byte_for_byte() {
        let c = build_config52(&PROFILE);
        assert_eq!(&c[0..4], &2048u32.to_le_bytes());
        assert_eq!(&c[4..6], &128u16.to_le_bytes());
        assert_eq!(&c[6..8], &0u16.to_le_bytes());
        assert_eq!(&c[8..14], &[0, 3, 1, 3, 0, 0]);
        assert_eq!(&c[14..20], &[0, 1, 3, 7, 0, 0]);
        assert!(c[20..52].iter().all(|&x| x == 0), "MoE trailer must be zero");
    }

    #[test]
    fn cert_v3_is_salted() {
        assert_eq!(seed_derivation_for(3), SEED_SALTED);
        assert_eq!(seed_derivation_for(2), SEED_LEGACY);
        assert_eq!(seed_derivation_for(1), SEED_LEGACY);
    }
}
