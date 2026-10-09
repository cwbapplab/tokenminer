//! The client-facing Pearl mining configuration and the hashrate unit.
//!
//! This is the shape the driver reads out of the engine's config JSON and hands
//! to [`super::pearl::GpuMiner`]. It carries the *miner's own* choices — the
//! workload dimensions — and nothing protocol-mandated: the tile patterns, the
//! depth and the rank are protocol constants and live in [`super::pearl::config`],
//! where the core and the verifier both read them.
//!
//! Everything here used to be paired with a hand-rolled GPU stack (`pearl_gpu`,
//! `pearl_pow`, `gpu`). Those are gone; what is left is the configuration and the
//! one conversion that turns a tile rate into the TH/s every other Pearl miner
//! prints.

use zk_pow::api::proof::IncompleteBlockHeader;

/// The mining configuration a client mines with.
///
/// `m` and `n` are the miner's own workload dimensions and are deliberately NOT
/// hashed into `config52`, so changing them does not move any pool-visible
/// number. `k` and `rank` are protocol and are the only pair the fold kernel is
/// built for.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PearlMining {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub rank: u16,
    /// The tile patterns, kept for the hashrate conversion only. The engine
    /// ignores them and uses the protocol constants, so a value here can never
    /// make the miner fold a tile the pool would reject.
    #[serde(default = "default_rows_pattern")]
    pub rows_pattern: Vec<u32>,
    #[serde(default = "default_cols_pattern")]
    pub cols_pattern: Vec<u32>,
    #[serde(default)]
    pub gzip: bool,
}

fn default_rows_pattern() -> Vec<u32> {
    (0..16).collect()
}

fn default_cols_pattern() -> Vec<u32> {
    (0..16).collect()
}

impl Default for PearlMining {
    /// The profile llmjob mines at, which is the one its CUDA core is built for:
    /// `k = 2048`, `rank = 128` (exactly `PENALTY_BASE_RANK`, so no rank penalty
    /// applies), and a 16x16 strided tile.
    ///
    /// `m` and `n` are llmjob's own workload dimensions. They are the one part
    /// not fixed by the protocol: any whole number of 64-row groups is accepted,
    /// and any power-of-two chunk count, subject to the memory the card has.
    fn default() -> Self {
        Self {
            m: 131_072,
            n: 262_144,
            // k = 16 * rank exactly, the ratio the verifier's `k >= 16r` bound and
            // the fold's 16 checkpoints per candidate meet at.
            k: 2048,
            rank: 128,
            rows_pattern: default_rows_pattern(),
            cols_pattern: default_cols_pattern(),
            gzip: false,
        }
    }
}

impl PearlMining {
    /// Tiles per second to TH/s — the unit Open-Pearl-Miner and every other Pearl
    /// miner reports.
    ///
    /// One candidate tile is one `rows_pattern x cols_pattern` output of the full
    /// k-deep GEMM, so it is `h * w * k` MACs, and consensus counts work in MACs:
    /// a TH is 10^12 of them. The reference converts exactly this way and writes
    /// the same constant twice — `TH_PER_MTILE = (1 << 20) / 1e6`, annotated
    /// "1 Mtile/s ~= 1.0486 TH/s", and `TH_PER_REGION = tiles * (1 << 20) / 1e12`.
    /// Its `1 << 20` is just `16 * 16 * 4096` written out, which is why this takes
    /// the depth from `k` instead of hardcoding it.
    ///
    /// This is not a cosmetic rename. A card folding 56 TH/s is folding about
    /// 1.07e8 tiles/s, so publishing the tile rate as the hashrate put our figures
    /// three orders of magnitude from every other Pearl miner's.
    pub fn th_per_second(&self, tiles_per_second: f64) -> f64 {
        let tile = self.rows_pattern.len() as f64 * self.cols_pattern.len() as f64 * self.k as f64;
        tiles_per_second * tile / 1e12
    }
}

/// Parses the 76-byte header hex from a `mining.notify` job.
pub fn parse_header(header_hex: &str) -> Result<IncompleteBlockHeader, String> {
    let bytes = hex::decode(header_hex.trim()).map_err(|e| format!("Invalid job header: {e}"))?;

    IncompleteBlockHeader::from_bytes(&bytes).map_err(|e| format!("Invalid job header: {e}"))
}

/// The smallest matrix pair the verifier accepts, for anything whose subject is
/// not the size.
///
/// The shipped default is 131072 x 262144, and a test that builds a real proof
/// against it would materialise both matrices and a Merkle tree over each on the
/// CPU — minutes per tile. Nothing that wants this fixture depends on the
/// dimensions.
///
/// `m = n = 64` is not a free choice. The verifier lays the tile pattern out in
/// runs of its own period — `stride * length` of the last dimension, 32 for the
/// 16x16 stride-8 tile — and `threads_partition` refuses a dimension that is not
/// a whole number of them. 64 is two periods, and it also clears
/// `t_rows + 27 < m` for the largest offset the pattern can carry.
///
/// `k = 1024` is the floor `public_params_sanity_check` sets for the chunk
/// padding to be collision resistant, and rank 64 is its floor for noise
/// generation, doubled to the smallest value that puts `k = 16r`, the ratio
/// production runs at. Consensus rejects `rank < PENALTY_BASE_RANK` (128)
/// outright, so this is below what a real block could carry regardless — it
/// exists to exercise the container, not the network.
#[cfg(test)]
pub fn small_mining() -> PearlMining {
    PearlMining {
        m: 64,
        n: 64,
        k: 1024,
        rank: 64,
        rows_pattern: default_rows_pattern(),
        cols_pattern: default_cols_pattern(),
        gzip: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At the shipped depth our conversion has to *be* the reference's constant,
    /// not merely the right order of magnitude — a rate off by a power of ten
    /// still reads as a hashrate, and is wrong in the only way nobody notices.
    ///
    /// The reference prints the constant itself: `TH_PER_MTILE = (1 << 20) / 1e6`
    /// (`tests/bench_split.py:20`), where `1 << 20` is `16 * 16 * 4096`. Both
    /// sides divide by the same `1e6`, so agreeing to 12 significant digits is
    /// agreement to the last bit the rounding can express.
    #[test]
    fn one_mtile_per_second_is_the_references_own_tera_hash_constant() {
        // The reference's own shape: `1 << 20` is a statement about 16x16 at k=4096.
        let reference = PearlMining {
            k: 4096,
            ..PearlMining::default()
        };
        let th_per_second = reference.th_per_second(1e6);
        assert!(
            (th_per_second - (1u64 << 20) as f64 / 1e6).abs() < 1e-12,
            "1 Mtile/s at the reference's shape should be {} TH/s, got {th_per_second}",
            (1u64 << 20) as f64 / 1e6
        );

        // And the same rate at the shipped shape: the tile is half the work at
        // half the depth, so the rate is half the constant. Asserting the half is
        // what stops the conversion from being read as a rounding.
        let shipped = PearlMining::default().th_per_second(1e6);
        assert!(
            (shipped - (1u64 << 19) as f64 / 1e6).abs() < 1e-12,
            "1 Mtile/s at the shipped shape should be {} TH/s, got {shipped}",
            (1u64 << 19) as f64 / 1e6
        );
    }

    /// A tile is `h * w * k` MACs, so the same tile rate at twice the GEMM depth
    /// is twice the hashrate. That is what the depth in
    /// [`PearlMining::th_per_second`] buys over hardcoding the reference's
    /// `1 << 20`, which is only right at the depth the reference happens to use.
    #[test]
    fn a_tile_is_counted_as_the_work_the_gemms_depth_implies() {
        let mut deep = PearlMining::default();
        deep.k *= 2;

        let tiles_per_second = 5.0e7;
        let base = PearlMining::default().th_per_second(tiles_per_second);
        assert!(
            (deep.th_per_second(tiles_per_second) - 2.0 * base).abs() < base * 1e-12,
            "twice the depth must be twice the hashrate: {base} then {}",
            deep.th_per_second(tiles_per_second)
        );

        // The magnitude at the shipped shape: 5e7 candidates/s at `16 * 16 * 2048`
        // MACs each is 26.2144 TH/s.
        assert!(
            (base - 26.2144).abs() < 1e-3,
            "5e7 tiles/s at the shipped shape is 26.2144 TH/s, got {base}"
        );
        assert_eq!(PearlMining::default().th_per_second(0.0), 0.0);
    }

    /// A payload from before the tile patterns were carried still deserializes:
    /// the engine reads the patterns from the protocol constants, so an older
    /// config that omits them is not a reason to refuse a session.
    #[test]
    fn a_config_without_the_patterns_still_deserializes() {
        let mining: PearlMining =
            serde_json::from_str(r#"{"m":1024,"n":1024,"k":2048,"rank":128}"#)
                .expect("an older payload should deserialize");
        assert_eq!(mining.rows_pattern.len(), 16);
        assert_eq!(mining.cols_pattern.len(), 16);
        assert!(!mining.gzip);
    }
}
