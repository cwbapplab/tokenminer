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
    /// Defaults mirror the reference miner's `MinerSettings` (`miner-base/src/miner_base/settings.py`).
    fn default() -> Self {
        let mut cols_pattern = Vec::with_capacity(64);
        for a in 0..32u32 {
            cols_pattern.push(a * 8);
            cols_pattern.push(a * 8 + 1);
        }

        Self {
            m: 1024,
            n: 1024,
            // rank 128 == PENALTY_BASE_RANK, so the consensus rank penalty is neutral; k >= 16*rank.
            k: 2048,
            rank: 128,
            rows_pattern: vec![0, 8],
            cols_pattern,
            gzip: false,
        }
    }
}

pub fn seed_derivation_for(cert_version: u32) -> Result<SeedDerivation, String> {
    Ok(CertificateVersion::try_from(cert_version)
        .map_err(|e| e.to_string())?
        .seed_derivation())
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
    let bytes = bincode::serialize(proof).map_err(|e| format!("Serializing the proof failed: {e}"))?;
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
    use zk_pow::api::sanity_checks;
    use zk_pow::ffi::mine::try_mine_one;

    /// A real header captured from a `mining.notify`.
    const HEADER: &str = "0000002040855504f7a9fc1682784e9b3f1d185a9f2ffb84efa2b81460e2b726ae\
77656dd7a5610c81c03527b58bba629e7f58a84a5a1295776120109f9d2bfad0ef7438f9c4c06a04810018";

    #[test]
    fn the_default_mining_configuration_satisfies_the_sanity_checks() {
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
        let mining = PearlMining::default();
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
        assert_eq!(&STANDARD.decode(&gzipped).unwrap()[..3], &[0x1f, 0x8b, 0x08]);
    }
}
