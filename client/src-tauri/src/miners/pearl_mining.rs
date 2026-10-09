//! Real Pearl proof production.
//!
//! The matrices a Pearl proof commits to are **miner-chosen** (random int8 in `[-64, 64]`); the
//! verifier re-derives the noise from the committed roots and the header, so no model weights are
//! needed.
//!
//! This module owns the mining *configuration* and the wire *encoding*. Searching for a proof lives
//! in [`super::pearl_pow`], which mines against a pool's share target rather than the block
//! difficulty.
//!
//! Serialization is `base64(bincode(PlainProof))` (`PlainProof::to_base64` in `py-pearl-mining`).
//! Some pools put a gzip envelope around it; `gzip` toggles that.

use std::io::Write;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, SeedDerivation};
use zk_pow::ffi::plain_proof::{list_to_pattern, CertificateVersion, PlainProof};

/// The mining configuration a client mines with. The proof carries `m, n, k, rank` and the row
/// indices it sampled; the patterns are re-derived from those indices by the verifier, so any
/// self-consistent configuration is valid — but it must satisfy `sanity_checks`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PearlMining {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub rank: u16,
    pub rows_pattern: Vec<u32>,
    pub cols_pattern: Vec<u32>,
    #[serde(default)]
    pub gzip: bool,
}

impl Default for PearlMining {
    /// The configuration the split search path runs at.
    ///
    /// `k` and `rank` are not a preference here; they are the only pair the search blob can satisfy.
    /// Its accumulator is never reset between k-iterations, so each checkpoint is the XOR of the
    /// *running* sums over a group of `BK * REDUCE_EVERY = 128` columns, and the verifier's fold groups
    /// `r` columns per iteration — so `r` has to be 128. The blob writes 16 words per candidate, so
    /// `k / r <= 16`, and `zk_pow`'s `k >= 16r` closes the interval at `k = 16r = 2048`. The derivation
    /// lives with the constants in [`super::gpu::bufs`], which refuses anything else, and gate D
    /// measures it on the device rather than asserting it.
    ///
    /// The consequence for the pool is stated rather than hidden. A share's difficulty is a function of
    /// `tile_size * dot_product_length`, and the candidate tile is still 256 cells — `2 x 128` instead of
    /// `16 x 16` — while the depth is 2048 instead of 4096, so the adjustment factor halves. The same
    /// hashrate therefore meets the same target at half the work per share, and the share rate at a
    /// given hashrate changes. `rank = 128` is exactly `PENALTY_BASE_RANK`, so no penalty applies.
    ///
    /// `m` and `n` are the one part not forced: any whole number of 128 tiles is accepted. They stay at
    /// the dimensions every measurement in this project was recorded at, but the split path costs memory
    /// the fused path did not — a transcript per candidate, 64 bytes each, so at these dimensions the
    /// transcript buffer alone is 4.3 GB and the whole footprint is about 7.4 GB. The check in
    /// `pearl_gpu::open` refuses up front on a card that cannot hold it, which is the honest outcome:
    /// the fused path hashed inside the launch and never held a transcript per candidate at all.
    fn default() -> Self {
        Self {
            m: 131_072,
            n: 131_072,
            // k = 16 * rank exactly, the tightest the verifier's `k >= 16r` allows and the only value
            // the blob's 16-word candidate slot can hold.
            k: 2048,
            rank: 128,
            // One candidate is a 2 x 128 window inside the blob's 128 x 128 tile, so the rows pattern is
            // the two rows the candidate names and the cols pattern is the whole strip.
            rows_pattern: vec![0, 1],
            cols_pattern: (0..128).collect(),
            gzip: false,
        }
    }
}

impl PearlMining {
    /// Tiles per second to TH/s — the unit Open-Pearl-Miner and every other Pearl miner reports.
    ///
    /// One candidate tile is one `rows_pattern x cols_pattern` output of the full k-deep GEMM, so it is
    /// `h * w * k` MACs, and consensus counts work in MACs: a TH is 10^12 of them. The reference
    /// converts exactly this way and writes the same constant twice — `TH_PER_MTILE = (1 << 20) / 1e6`,
    /// annotated "1 Mtile/s ~= 1.0486 TH/s" (`tests/bench_split.py:20`), and
    /// `TH_PER_REGION = tiles * (1 << 20) / 1e12` (`tests/bench_capi.py:18`). Its `1 << 20` is just
    /// `16 * 16 * 4096` written out, which is why this takes the depth from `k` instead of hardcoding
    /// it: at a k this miner also allows, a tile is twice the work and has to be counted as such.
    ///
    /// At the split path's shape a tile is `2 * 128 * 2048` = 524,288 MACs, so the same tile rate is
    /// half the reference's constant. That is not a discrepancy to paper over — it is the same fact the
    /// difficulty adjustment states, that this configuration prices half the work per candidate. The tile
    /// is read off the patterns rather than hardcoded because the split path's tile *is* the pattern: the
    /// blob folds a 128-column strip and reports 64 candidates inside it, each a 2x128 window, so `h * w`
    /// is the window the kernel actually computed. A hardcoded 16x16 would report the reference's rate
    /// for a shape this kernel no longer runs.
    ///
    /// This is not a cosmetic rename. A card folding 56 TH/s is folding about 1.07e8 tiles/s at this
    /// shape, so publishing the tile rate as the hashrate put our figures three orders of magnitude from
    /// every other Pearl miner's — including the reference's, the only other Pearl miner whose
    /// architecture we can actually check.
    pub fn th_per_second(&self, tiles_per_second: f64) -> f64 {
        let tile = self.rows_pattern.len() as f64 * self.cols_pattern.len() as f64 * self.k as f64;
        tiles_per_second * tile / 1e12
    }
}

pub fn seed_derivation_for(cert_version: u32) -> Result<SeedDerivation, String> {
    Ok(CertificateVersion::try_from(cert_version)
        .map_err(|e| e.to_string())?
        .seed_derivation())
}

/// The key a search is bound to: `BLAKE3(header ‖ configuration)`, unkeyed.
///
/// This is not the jackpot hash key — that is `dnsA`, one link further down the commitment chain.
/// It is the root the verifier derives for itself from the header and the `MiningConfiguration`
/// embedded in the proof, so the two have to agree before any other part of the share means
/// anything. `PublicProofParams::job_key` in `zk-pow` is the reference spelling.
pub fn job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    let mut data = header.to_bytes().to_vec();
    data.extend_from_slice(&config.to_bytes());
    pearl_blake3::blake3_digest(&data, None)
}

/// The smallest matrix pair the verifier accepts, for anything whose subject is not the size.
///
/// The shipped default is 131072 x 131072, and the CUDA self-test builds its operands by mining a
/// real proof through `try_mine_one` — the CPU reference search, which materialises both matrices
/// and a Merkle tree over each. At the default that is minutes per tile, and `self_test` runs from
/// `pearl::start`, so the engine would sit on its start-up check for minutes. Nothing it finds
/// depends on the dimensions.
///
/// The tests need the same thing for the same reason, and most were written against the default
/// only because it was a convenient fixture — which is why they silently became slow when the
/// default moved rather than saying so.
pub fn small_mining() -> PearlMining {
    PearlMining {
        m: 48,
        n: 48,
        // k = 1024 is the floor `public_params_sanity_check` sets for the chunk padding to be
        // collision resistant, and rank 64 its floor for noise generation doubled to the smallest
        // value the GPU search can fold at.
        //
        // The search stages `pow_bk(arch)` k at once — 64 on the tensor path, 32 on Turing's DP4A
        // fold — and a `rank` below the staged width makes `steps_per_rank` zero, which is a device
        // divide by zero rather than a wrong answer. 64 is the smallest value both paths accept, so
        // one fixture checks the same thing on every card. It also puts this at `k = 16r`, the ratio
        // production runs at (4096 / 256), which 32 did not.
        //
        // Consensus does not care either way: the reference rejects `rank < PENALTY_BASE_RANK`
        // (128) outright, so the old 32 was below what a real block could carry regardless.
        k: 1024,
        rank: 64,
        rows_pattern: (0..16).collect(),
        cols_pattern: (0..16).collect(),
        gzip: false,
    }
}

pub fn mining_configuration(mining: &PearlMining) -> Result<MiningConfiguration, String> {
    let (rows_pattern, _) = list_to_pattern(&mining.rows_pattern).map_err(|e| e.to_string())?;
    let (cols_pattern, _) = list_to_pattern(&mining.cols_pattern).map_err(|e| e.to_string())?;

    Ok(MiningConfiguration {
        common_dim: mining.k as u32,
        rank: mining.rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern,
        cols_pattern,
        moe: None,
    })
}

/// Serializes a proof the way the reference does: `base64(bincode(PlainProof))`, optionally gzipped.
pub fn encode_plain_proof(proof: &PlainProof, gzip: bool) -> Result<String, String> {
    let bytes =
        bincode::serialize(proof).map_err(|e| format!("Serializing the proof failed: {e}"))?;
    let bytes = if gzip { gzip_bytes(&bytes)? } else { bytes };

    Ok(STANDARD.encode(bytes))
}

fn gzip_bytes(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(data).map_err(|e| e.to_string())?;

    encoder.finish().map_err(|e| e.to_string())
}

/// Parses the 76-byte header hex from a `mining.notify` job.
pub fn parse_header(header_hex: &str) -> Result<IncompleteBlockHeader, String> {
    let bytes = hex::decode(header_hex.trim()).map_err(|e| format!("Invalid job header: {e}"))?;

    IncompleteBlockHeader::from_bytes(&bytes).map_err(|e| format!("Invalid job header: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zk_pow::api::proof::PublicProofParams;
    use zk_pow::api::sanity_checks;
    use zk_pow::ffi::mine::try_mine_one;

    /// A real header captured from a `mining.notify`.
    const HEADER: &str = "0000002040855504f7a9fc1682784e9b3f1d185a9f2ffb84efa2b81460e2b726ae\
77656dd7a5610c81c03527b58bba629e7f58a84a5a1295776120109f9d2bfad0ef7438f9c4c06a04810018";

    /// At the shipped depth our conversion has to *be* the reference's constant, not merely the
    /// right order of magnitude — a rate off by a power of ten still reads as a hashrate, and is
    /// wrong in the only way nobody notices.
    ///
    /// The reference prints the constant itself: `TH_PER_MTILE = (1 << 20) / 1e6`, annotated
    /// "1 Mtile/s ~= 1.0486 TH/s" (`tests/bench_split.py:20`). Both sides of the comparison below
    /// divide by the same `1e6`, so agreeing to 12 significant digits is agreement to the last bit
    /// the rounding can express.
    #[test]
    fn one_mtile_per_second_is_the_references_own_tera_hash_constant() {
        // The reference's own shape, not the shipped default: `1 << 20` is `16 * 16 * 4096` written out,
        // so the constant is a statement about that shape. Read it at the default, which is now a
        // different tile, and the test would be asserting a coincidence the conversion no longer makes.
        let reference = PearlMining {
            m: 1024,
            n: 1024,
            k: 4096,
            rank: 256,
            rows_pattern: (0..16).collect(),
            cols_pattern: (0..16).collect(),
            gzip: false,
        };
        let th_per_second = reference.th_per_second(1e6);
        assert!(
            (th_per_second - (1u64 << 20) as f64 / 1e6).abs() < 1e-12,
            "1 Mtile/s at the reference's shape should be {} TH/s, got {th_per_second}",
            (1u64 << 20) as f64 / 1e6
        );

        // And the same rate at the shape this miner runs: the tile is half the work, so the rate is half
        // the constant. Asserting the half is what stops the conversion from being read as a rounding.
        let split = PearlMining::default().th_per_second(1e6);
        assert!(
            (split - (1u64 << 19) as f64 / 1e6).abs() < 1e-12,
            "1 Mtile/s at the split shape should be {} TH/s, got {split}",
            (1u64 << 19) as f64 / 1e6
        );
    }

    /// A tile is `h * w * k` MACs, so the same tile rate at twice the GEMM depth is twice the
    /// hashrate. That is what the depth in [`PearlMining::th_per_second`] buys over hardcoding the
    /// reference's `1 << 20`, which is only right at the depth the reference happens to use.
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

        // The magnitude at the shape this miner runs: 5e7 candidates/s at `2 * 128 * 2048` MACs each is
        // 26.2144 TH/s. The same cell count as the old 16x16 tile — 256 cells either way — at half the
        // depth, so half the work per candidate. Asserted rather than assumed, because the two shapes
        // look different and price the same number of cells: only `k` separates them.
        assert!(
            (base - 26.2144).abs() < 1e-3,
            "5e7 tiles/s at the default shape is 26.2144 TH/s, got {base}"
        );

        let mut old_shape = PearlMining::default();
        old_shape.rows_pattern = (0..16).collect();
        old_shape.cols_pattern = (0..16).collect();
        assert!(
            (old_shape.th_per_second(tiles_per_second) - base).abs() < base * 1e-12,
            "the same cell count at the same depth must price the same work: {base} then {}",
            old_shape.th_per_second(tiles_per_second)
        );

        assert_eq!(PearlMining::default().th_per_second(0.0), 0.0);
    }

    /// The shipped default has to pass the verifier's constraint list, checked here directly
    /// rather than through a mined proof.
    ///
    /// `public_params_sanity_check` reads `m`, `n`, `k`, the rank, the patterns and the tile
    /// offsets — it never looks at a hash or a header. So the whole of what it has to say about the
    /// default can be had from the parameters alone, and building a real proof to ask it bought
    /// four minutes of CPU and a gigabyte of RAM per run to re-derive numbers that were never
    /// consulted. The proof-level check is [`a_proof_at_the_default_dimensions_verifies`], ignored
    /// because of exactly that cost.
    #[test]
    fn the_default_mining_configuration_satisfies_the_sanity_checks() {
        let mining = PearlMining::default();
        let config = mining_configuration(&mining).unwrap();

        // The patterns have to survive the conversion to `PeriodicPattern` as *consecutive* indices, not
        // just as the same set of indices — a candidate is a window, and a pattern that came back as the
        // same values spread across the matrix would pass every check below and describe a different
        // candidate. The rows pattern is the two rows a candidate names inside the blob's 128-row tile;
        // the cols pattern is the whole 128-column strip the blob folds for it.
        for (name, pattern, size, period) in [
            ("rows", &config.rows_pattern, 2u32, 2u32),
            ("cols", &config.cols_pattern, 128u32, 128u32),
        ] {
            assert_eq!(
                pattern.size(),
                size,
                "the {name} pattern must span the candidate window the blob reports"
            );
            assert_eq!(
                pattern.indices_with_offset(0),
                (0..size).collect::<Vec<u32>>(),
                "the {name} pattern must be consecutive indices from offset zero"
            );
            // A pattern that slid by its own length instead of by the offset would report the same indices
            // for every candidate, and the proof would describe the first window in the tile rather than
            // the one the blob folded.
            assert_eq!(
                pattern.indices_with_offset(24 * period),
                (24 * period..24 * period + size).collect::<Vec<u32>>(),
                "the {name} pattern must slide by the offset, not by its own length"
            );
            assert!(
                pattern.offset_is_valid(24 * period),
                "a whole number of periods must be a valid offset for the {name} pattern"
            );
        }

        // An offset inside the window is not a valid offset at all, and the two patterns cannot be checked
        // with one number. Both are stride 1, so `offset_is_valid` accepts only whole periods — which is
        // exactly what `assemble` reports, `2 * cand` for the rows and `tile_n * 128` for the cols. For the
        // rows pattern 48 is a whole number of 2-row periods; for the cols pattern the window *is* the
        // period, so 48 is inside it and the verifier would reject a proof that named it. Asserting the
        // rejection is what says the miner's offsets have to be tile-aligned, not merely inside a tile.
        assert!(config.rows_pattern.offset_is_valid(48), "48 is a whole number of 2-row periods");
        assert!(
            !config.cols_pattern.offset_is_valid(48),
            "48 is inside the 128-column window, so it is not a valid offset for the cols pattern"
        );
        assert!(
            config.cols_pattern.offset_is_valid(128),
            "the tile-aligned offset the miner reports must be valid"
        );

        let public = PublicProofParams::new_dummy(
            IncompleteBlockHeader {
                version: 0,
                prev_block: [0; 32],
                merkle_root: [0; 32],
                timestamp: 0,
                nbits: 0,
            },
            SeedDerivation::Salted,
            config,
            mining.m as u32,
            mining.n as u32,
            0,
            0,
        );
        sanity_checks::public_params_sanity_check(&public)
            .expect("the shipped default is a configuration the verifier accepts");

        assert_eq!(public.mining_config.common_dim, mining.k as u32);
        assert_eq!(public.mining_config.rank, mining.rank);
    }

    /// Gate E: the configuration the split search path needs, checked against the verifier rather
    /// than assumed from the reference's defaults.
    ///
    /// Two things here are exercised rather than asserted, because both are the kind of thing a
    /// reasonable-looking assumption gets wrong:
    ///   * `PeriodicPattern::from_list` on a two-element list. Its periodicity loop runs
    ///     `for period in 1..p.len()`, so a two-element list has exactly one candidate period, and the
    ///     pattern is accepted only if that period divides out — which `[0, 1]` does, but `[0, 2]`
    ///     would describe a different shape. `offset_is_valid` is a separate constraint from the shape.
    ///   * the difficulty consequence. The split path's tile is `2 x 128`, not `16 x 16`: the same 256
    ///     cells, but the dot product length is 2048 instead of 4096, so the adjustment factor the pool
    ///     prices work by halves. That is not cosmetic — it changes the share rate at a given hashrate,
    ///     and it is the reason this configuration cannot be adopted quietly.
    #[test]
    #[cfg(feature = "pearl")]
    fn the_mining_configuration_the_split_path_requires_is_the_one_the_verifier_accepts() {
        // Exercised, not assumed: a two-element list is the smallest case the periodicity loop can
        // accept, and it has to round-trip through `to_bytes` to be a pattern a proof can carry.
        let rows = list_to_pattern(&[0, 1]).expect("[0, 1] should be a pattern").0;
        assert_eq!(rows.size(), 2);
        assert_eq!(rows.indices_with_offset(0), vec![0, 1]);
        assert!(rows.is_valid(), "a two-element pattern must round-trip through serialization");
        assert!(rows.offset_is_valid(0), "offset 0 must be valid for the rows pattern");

        let cols = list_to_pattern(&(0..128).collect::<Vec<u32>>())
            .expect("0..128 should be a pattern")
            .0;
        assert_eq!(cols.size(), 128);
        assert_eq!(cols.indices_with_offset(0), (0..128).collect::<Vec<u32>>());
        assert!(cols.is_valid());
        assert!(cols.offset_is_valid(0));

        let split = PearlMining {
            m: 128,
            n: 256, // two tiles, so the grid reaches more than one candidate
            k: 2048,
            rank: 128,
            rows_pattern: vec![0, 1],
            cols_pattern: (0..128).collect(),
            gzip: false,
        };
        let config = mining_configuration(&split).unwrap();
        let public = PublicProofParams::new_dummy(
            IncompleteBlockHeader {
                version: 0,
                prev_block: [0; 32],
                merkle_root: [0; 32],
                timestamp: 0,
                nbits: 0,
            },
            SeedDerivation::Salted,
            config,
            split.m as u32,
            split.n as u32,
            0,
            0,
        );
        sanity_checks::public_params_sanity_check(&public)
            .expect("the split path's configuration must survive the verifier's own check");

        // The derivation, stated as the coincidence it is. `k >= 16r` is the verifier's bound; the blob
        // writes `JACKPOT_SIZE` checkpoints per candidate, one per `r`-sized group, so `k / r` can be at
        // most 16. Both bounds meet at exactly one value when `r = 128`.
        assert_eq!(config.rank as usize, sanity_checks::PENALTY_BASE_RANK);
        assert_eq!(
            config.dot_product_length(),
            2048,
            "the dot product length must be the whole k, not k rounded down to a rank multiple"
        );
        assert_eq!(
            config.dot_product_length() / config.rank as usize,
            zk_pow::circuit::pearl_program::JACKPOT_SIZE,
            "each candidate's checkpoint slots must be written exactly once"
        );

        // The shipped default *is* now this configuration, so the assertion that used to say the default
        // was not runnable has to compare against the shape it replaced instead. Both are fold-valid —
        // 256 cells either way, `k = 16r` either way — so what rules the old shape out is the group size,
        // not the fold: the blob checkpoints 128 columns per group and rank 256 groups twice that.
        assert_eq!(PearlMining::default().rank as usize, 128, "the shipped default is the split configuration");

        let replaced = PearlMining {
            m: split.m,
            n: split.n,
            k: 4096,
            rank: 256,
            rows_pattern: (0..16).collect(),
            cols_pattern: (0..16).collect(),
            gzip: false,
        };
        assert_eq!(
            replaced.k / replaced.rank as usize,
            zk_pow::circuit::pearl_program::JACKPOT_SIZE,
            "the shape this default replaced was fold-valid too, which is why the group size is the only \
             thing that rules it out"
        );

        // The consequence, not hidden: the same 256 cells at half the depth is half the adjustment
        // factor, so the pool's share target is met at half the work per share.
        const NBITS: u32 = 0x1d00ffff;
        let replaced_config = mining_configuration(&replaced).unwrap();
        let split_factor = config.rows_pattern.size() as usize
            * config.cols_pattern.size() as usize
            * config.dot_product_length();
        let replaced_factor = replaced_config.rows_pattern.size() as usize
            * replaced_config.cols_pattern.size() as usize
            * replaced_config.dot_product_length();
        assert_eq!(split_factor, 256 * 2048);
        assert_eq!(replaced_factor, 256 * 4096);
        assert_eq!(
            sanity_checks::extract_difficulty_bound(NBITS, &config),
            sanity_checks::extract_difficulty_bound(NBITS, &replaced_config) / 2,
            "the split path's share difficulty must be exactly half the shape it replaced's"
        );
    }

    /// The proof-level version of the check above: a real proof at the shipped dimensions, parsed
    /// and sanity-checked the way the pool parses it.
    ///
    /// Ignored rather than run, because `try_mine_one` builds two 512 MiB matrices and a Merkle
    /// tree over each on the CPU, which takes minutes in a debug build. Worth running deliberately
    /// after any change to the default, not worth every ordinary test run.
    #[test]
    #[ignore = "mines a 131072x131072 proof on the CPU; run with --ignored"]
    fn a_proof_at_the_default_dimensions_verifies() {
        let mining = PearlMining::default();
        let config = mining_configuration(&mining).unwrap();
        let header = parse_header(HEADER).unwrap();

        let mut rng = rand::rng();

        // `wrong_jackpot_hash = true` inverts the acceptance test, so this yields a proof on the
        // first draw without mining an actual solution — enough to validate the configuration.
        let proof = try_mine_one(
            &mut rng,
            mining.m,
            mining.n,
            mining.k,
            header,
            config,
            None,
            true,
            SeedDerivation::Salted,
        )
        .expect("the search does not error")
        .expect("an inverted acceptance test always returns a proof");

        // parse_proof rebuilds the public params (patterns, job key, committed roots) the way the
        // verifier does — this is the same path the pool takes.
        let (_, public) = proof
            .parse_proof(header, SeedDerivation::Salted)
            .expect("the proof parses");

        sanity_checks::public_params_sanity_check(&public).expect("the configuration is sane");
        assert_eq!(public.mining_config.common_dim, mining.k as u32);
        assert_eq!(public.mining_config.rank, mining.rank);
    }

    #[test]
    fn a_proof_round_trips_through_the_wire_format() {
        // The smallest configuration the verifier accepts, not the shipped default. What is under
        // test is the container — bincode, the `moe: None` discriminant, the gzip envelope — and
        // none of that depends on the dimensions; building a proof at 131072 x 131072 to check it
        // costs minutes and a gigabyte of RAM to re-derive numbers the encoder never reads.
        let mining = small_mining();
        let config = mining_configuration(&mining).unwrap();
        let header = parse_header(HEADER).unwrap();
        let mut rng = rand::rng();

        let proof = try_mine_one(
            &mut rng,
            mining.m,
            mining.n,
            mining.k,
            header,
            config,
            None,
            true,
            SeedDerivation::Salted,
        )
        .unwrap()
        .unwrap();

        let encoded = encode_plain_proof(&proof, false).unwrap();

        // The reference decoder (what a pool would use) reads it straight back.
        let bytes = STANDARD.decode(&encoded).unwrap();
        let parsed = PlainProof::deserialize_compat(&bytes).unwrap();

        assert_eq!(parsed.m, mining.m);
        assert_eq!(parsed.n, mining.n);
        assert_eq!(parsed.k, mining.k);

        // The gzip variant is a different container for the same proof.
        let gzipped = encode_plain_proof(&proof, true).unwrap();
        assert_ne!(gzipped, encoded);
        assert_eq!(
            &STANDARD.decode(&gzipped).unwrap()[..3],
            &[0x1f, 0x8b, 0x08]
        );
    }
}
