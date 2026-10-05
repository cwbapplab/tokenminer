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
    /// The dimensions Open-Pearl-Miner mines at (`python/pool_common.py`), and the ones the GPU
    /// search is built around.
    ///
    /// These are not arbitrary. A share's difficulty is a function of `tile * k`, so a smaller `k`
    /// is not a cheaper configuration, it is a *slower* one — the same proof carries proportionally
    /// less weight in the target, and the pool's difficulty is set for this configuration. The
    /// patterns are `range(16)` for the same reason the GPU search hashes 16x16 tiles: a hash tile
    /// is what consensus prices, and a different shape is a different amount of work per share.
    ///
    /// The old defaults (`1024 x 1024`, `k = 2048`, rank 128, a 2x64 pattern) were the reference
    /// *miner's* configuration, not its production one, and the 2x64 pattern has no representation
    /// in the grid kernel at all — it sampled eight rows at a time.
    fn default() -> Self {
        Self {
            m: 131_072,
            n: 131_072,
            // k = 16 * rank exactly, the tightest the verifier's `k >= 16r` allows.
            k: 4096,
            rank: 256,
            rows_pattern: (0..16).collect(),
            cols_pattern: (0..16).collect(),
            gzip: false,
        }
    }
}

impl PearlMining {
    /// Tiles per second to TH/s — the unit Open-Pearl-Miner and every other Pearl miner reports.
    ///
    /// One candidate tile is one 16x16 output of the full k-deep GEMM, so it is `16 * 16 * k` MACs,
    /// and consensus counts work in MACs: a TH is 10^12 of them. The reference converts exactly this
    /// way, and writes the same constant twice — `TH_PER_MTILE = (1 << 20) / 1e6`, annotated
    /// "1 Mtile/s ~= 1.0486 TH/s" (`tests/bench_split.py:20`), and
    /// `TH_PER_REGION = tiles * (1 << 20) / 1e12` (`tests/bench_capi.py:18`). Its `1 << 20` is
    /// just `16 * 16 * 4096` written out, which is why this takes the depth from `k` instead of
    /// hardcoding it: at a k this miner also allows, a tile is twice the work and has to be counted
    /// as such. At the shipped default the two agree exactly, so the numbers are comparable.
    ///
    /// The 16x16 is not configurable here because it is not this module's choice: the grid kernel
    /// folds 16x16 tiles, the patterns below are `range(16)` because a hash tile must span one, and
    /// the verifier prices a tile as `hash_tile_h * hash_tile_w * dot_product_length` — so this
    /// counts the work the card actually did and it is the work consensus pays for. The tile is
    /// hardcoded rather than read off `rows_pattern.len()` precisely because a configuration with
    /// some other tile is one the kernel does not implement; reading the length would report a
    /// consensus-priced rate for machine throughput that never happened, and report it silently.
    ///
    /// This is not a cosmetic rename. A card folding 56 TH/s is folding about 5.3e7 tiles/s, so
    /// publishing the tile rate as the hashrate put our figures three orders of magnitude from every
    /// other Pearl miner's — including the reference's, the only other Pearl miner whose
    /// architecture we can actually check.
    pub fn th_per_second(&self, tiles_per_second: f64) -> f64 {
        tiles_per_second * (16.0 * 16.0 * self.k as f64) / 1e12
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
        let th_per_second = PearlMining::default().th_per_second(1e6);
        assert!(
            (th_per_second - (1u64 << 20) as f64 / 1e6).abs() < 1e-12,
            "1 Mtile/s should be {} TH/s, got {th_per_second}",
            (1u64 << 20) as f64 / 1e6
        );
    }

    /// A tile is `16 * 16 * k` MACs, so the same tile rate at twice the GEMM depth is twice the
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

        // The magnitude, which is the point of the change: 5e7 tiles/s is ~52 TH/s, so the ~56 TH/s
        // the app reads on the dev box is ~5.3e7 tiles/s — and printing that tile rate as the
        // hashrate put our figures three orders of magnitude from every other Pearl miner's.
        assert!(
            (base - 52.4288).abs() < 1e-3,
            "5e7 tiles/s at the default depth is 52.4288 TH/s, got {base}"
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

        // The patterns have to survive the conversion to `PeriodicPattern` as sixteen consecutive
        // indices, not just as sixteen indices — the tile the GPU folds is sixteen *consecutive*
        // rows, and a pattern that came back as the same sixteen values spread across the matrix
        // would pass every check below and describe a different tile.
        for (name, pattern) in [
            ("rows", &config.rows_pattern),
            ("cols", &config.cols_pattern),
        ] {
            assert_eq!(
                pattern.size(),
                16,
                "the {name} pattern must span a whole 16x16 tile"
            );
            assert_eq!(
                pattern.indices_with_offset(0),
                (0..16).collect::<Vec<u32>>(),
                "the {name} pattern must be sixteen consecutive indices from offset zero"
            );
            assert_eq!(
                pattern.indices_with_offset(48),
                (48..64).collect::<Vec<u32>>(),
                "the {name} pattern must slide by whole tiles, not by sixteen bytes"
            );
        }

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
