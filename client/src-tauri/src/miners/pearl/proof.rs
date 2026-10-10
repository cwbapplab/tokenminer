//! Turning a core hit into the bytes a pool will accept.
//!
//! The core hands back both Merkle proofs already, captured on the device at the
//! moment the hit was found. That timing is not an optimisation: the search
//! re-draws its operands every few tens of milliseconds, so a proof read back
//! afterwards belongs to a different matrix than the hash it is supposed to
//! certify. The proof therefore travels WITH the hit, already in flat
//! concatenated buffers ([`super::abi::ProofSide`]).
//!
//! This module unpacks those into a [`PlainProof`] and re-checks them the way
//! llmjob's `shareProof.js` + `plainProof.js` do — the Merkle roots against the
//! job key, the leaf indices against the tile, then the wire encoding — built on
//! the `pearl-blake3` types this client already mines and submits with.
//! [`verify_share_locally`] goes further, to a full `zk-pow` verification, but it
//! is the offline correctness gate — llmjob's `native/probes/verify-hits.js` — and
//! not part of the submit path.
//!
//! The wire form is `base64(bincode(PlainProof))` with fixed-width little-endian
//! integers. A 64-byte transcript is NOT a share — sending one earns
//! `{"code":23,"message":"not a valid PlainProof (tried current and legacy V1
//! formats)"}`; the pool needs the re-checked Merkle proofs over the operand rows
//! the tile touched.

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof};

// Used only by `verify_share_locally`, the offline correctness gate, and the tests.
#[cfg(test)]
use primitive_types::U256;
#[cfg(test)]
use zk_pow::api::proof::{IncompleteBlockHeader, SeedDerivation};
#[cfg(test)]
use zk_pow::api::verify;
#[cfg(test)]
use zk_pow::ffi::plain_proof::CertificateVersion;

// Only `mining_configuration` (test-only) needs these.
#[cfg(test)]
use zk_pow::api::proof::MMAType;
#[cfg(test)]
use zk_pow::ffi::plain_proof::list_to_pattern;

use super::abi::ProofSide;
use super::config::Profile;
use super::target::region_to_tile;
#[cfg(test)]
use super::target::target_to_nbits;

/// One Merkle leaf: the operand bytes one proof chunk carries.
pub const CHUNK_BYTES: usize = pearl_blake3::BLAKE3_CHUNK_LEN;
/// One digest in a proof's sibling list.
pub const DIGEST_BYTES: usize = pearl_blake3::BLAKE3_DIGEST_SIZE;

/// A proof side in the shape the encoder and verifier both take: whole
/// 1024-byte leaves, split out of the core's flat buffer.
#[derive(Debug, Clone)]
pub struct PlainSide {
    pub leaf_data: Vec<[u8; CHUNK_BYTES]>,
    pub leaf_indices: Vec<usize>,
    pub total_leaves: usize,
    pub root: [u8; DIGEST_BYTES],
    pub siblings: Vec<[u8; DIGEST_BYTES]>,
}

/// Split one flat device side into leaves and siblings, mirroring llmjob's
/// `unpackSide`.
///
/// The checks are not redundant with the FFI copy: the FFI copy trusts
/// `leaves_len` and `leaf_indices_len`, and a device short-copy would otherwise
/// be zero-padded into a proof that certifies leaves nobody read. A side with no
/// leaves is not a proof of anything, and a leaf buffer shorter than the indices
/// claim is a truncated copy.
pub fn unpack_side(side: &ProofSide) -> Option<PlainSide> {
    let count = side.leaf_indices.len();
    if count == 0 || side.leaves.len() < count * CHUNK_BYTES {
        return None;
    }

    let leaf_data: Vec<[u8; CHUNK_BYTES]> = (0..count)
        .map(|i| {
            let mut chunk = [0u8; CHUNK_BYTES];
            chunk.copy_from_slice(&side.leaves[i * CHUNK_BYTES..(i + 1) * CHUNK_BYTES]);
            chunk
        })
        .collect();

    // A sibling list that is not a whole number of digests is a truncated copy;
    // `chunks_exact` drops the ragged tail rather than minting a malformed digest.
    let siblings: Vec<[u8; DIGEST_BYTES]> = side
        .siblings
        .chunks_exact(DIGEST_BYTES)
        .map(|d| {
            let mut digest = [0u8; DIGEST_BYTES];
            digest.copy_from_slice(d);
            digest
        })
        .collect();

    Some(PlainSide {
        leaf_data,
        leaf_indices: side.leaf_indices.iter().map(|&i| i as usize).collect(),
        total_leaves: side.total_leaves as usize,
        root: side.root,
        siblings,
    })
}

/// Build a [`PlainProof`] from a hit's two sides and the region it names.
///
/// Returns `None` when the sides cannot be unpacked, when either does not
/// verify against `job_key`, or when its leaf indices are not the ones the
/// hit's region actually touches.
///
/// That last check is the one a Merkle verify cannot make: `verify` shows only
/// that the supplied leaves hash to the root, which is equally true of the WRONG
/// leaves under the same tree. A stale row mask on the device produced exactly
/// that — the fold was right, the hash was right, and the rows it described were
/// a quarter of the way up the matrix, so the pool answered "Failed to extract
/// strip".
pub fn build_share_proof(
    job_key: &[u8; 32],
    profile: &Profile,
    region: u64,
    a: &ProofSide,
    bt: &ProofSide,
) -> Option<PlainProof> {
    let a = unpack_side(a)?;
    let bt = unpack_side(bt)?;

    // Recompute the root from the proof alone, the way the pool does.
    if !side_verifies(job_key, &a) || !side_verifies(job_key, &bt) {
        return None;
    }

    // `region` is the index the core reported, so this is the same tile
    // `regionToTile` names on the JS side.
    let tile = region_to_tile(region, profile);

    // The left operand's rows and the right operand's rows (the B matrix's
    // columns, which are its rows once transposed).
    if a.leaf_indices != row_leaf_indices(&tile.rows, profile.k as usize)
        || bt.leaf_indices != row_leaf_indices(&tile.cols, profile.k as usize)
    {
        return None;
    }

    Some(PlainProof {
        m: profile.m as usize,
        n: profile.n as usize,
        k: profile.k as usize,
        noise_rank: profile.rank as usize,
        a: MatrixMerkleProof::new(
            a.leaf_data,
            a.leaf_indices,
            tile.rows.iter().map(|&r| r as usize).collect(),
            a.total_leaves,
            a.root,
            a.siblings,
        ),
        bt: MatrixMerkleProof::new(
            bt.leaf_data,
            bt.leaf_indices,
            tile.cols.iter().map(|&c| c as usize).collect(),
            bt.total_leaves,
            bt.root,
            bt.siblings,
        ),
        moe: None,
    })
}

/// Does a side's own data hash to the root it claims, under `key`?
fn side_verifies(key: &[u8; 32], side: &PlainSide) -> bool {
    pearl_blake3::MerkleProof {
        leaf_data: side.leaf_data.clone(),
        leaf_indices: side.leaf_indices.clone(),
        total_leaves: side.total_leaves,
        root: side.root,
        siblings: side.siblings.clone(),
    }
    .verify(*key)
}

/// The 1024-byte chunks a set of matrix rows occupies. A row can straddle a
/// chunk boundary, so this is a range per row, not one index.
///
/// `zk_pow`'s `MerkleTree::compute_leaf_indices_from_rows` is the same walk, but
/// it is a method on a tree we do not have here — the device produced the proof
/// — so this re-derives the indices from the shape alone. `cols` is the row
/// width (the operand's `k`).
fn row_leaf_indices(rows: &[u32], cols: usize) -> Vec<usize> {
    let mut set = std::collections::BTreeSet::new();
    for &row in rows {
        let first = (row as usize * cols) / CHUNK_BYTES;
        let last = ((row as usize + 1) * cols - 1) / CHUNK_BYTES;
        for i in first..=last {
            set.insert(i);
        }
    }
    set.into_iter().collect()
}

/// Serializes a proof the way the reference does: `base64(bincode(PlainProof))`,
/// optionally gzipped for a `v2` Stratum session.
pub fn encode_plain_proof(proof: &PlainProof, gzip: bool) -> Result<String, String> {
    let bytes =
        bincode::serialize(proof).map_err(|e| format!("Serializing the proof failed: {e}"))?;
    let bytes = if gzip { gzip_bytes(&bytes)? } else { bytes };

    Ok(STANDARD.encode(bytes))
}

fn gzip_bytes(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Write;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(data).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())
}

/// Re-checks a proof the way the pool will — a full re-verification, the same
/// depth as llmjob's `native/probes/verify-hits.js`.
///
/// Deliberately not a block-difficulty check: no ordinary share meets the block
/// target, so that would reject exactly the proofs we want to send. Instead the
/// share target is re-encoded as compact `nbits` and handed to the real verifier,
/// which recomputes the jackpot from the committed matrices and compares it
/// against that target.
///
/// This is a correctness gate, not part of the submit path. It is an all-core
/// `zk-pow` pass per share, so running it on every hit made the miner's CPU cost
/// track its share rate. The submit path instead does what llmjob's own host does
/// (`pearlMiner._onHit`): a share-bound comparison plus `build_share_proof`'s
/// Merkle and leaf-index checks. The full verifier stays as the offline gate the
/// ignored GPU tests run.
#[cfg(test)]
pub fn verify_share_locally(
    header: &IncompleteBlockHeader,
    cert_version: u32,
    proof: &PlainProof,
    share_target: U256,
) -> Result<(), String> {
    let seed = seed_for(cert_version)?;
    let nbits = target_to_nbits(share_target)?;
    verify::verify_plain_proof(header, proof, Some(nbits), seed).map_err(|e| e.to_string())
}

/// A `zk_pow` `MiningConfiguration` from a profile.
///
/// Test-only: the miner never builds a configuration to search with (it hands
/// the vendored core a flat `PearlProfileFlat`), but a test that wants a real
/// proof has to mint one the way the reference does, and that goes through
/// `zk_pow`'s own `MiningConfiguration`. The patterns are protocol constants;
/// building them through `list_to_pattern` is what the verifier does when it
/// rebuilds the config from a proof's row indices, so the round trip is
/// exercised rather than assumed.
#[cfg(test)]
pub fn mining_configuration(profile: &Profile) -> Result<zk_pow::api::proof::MiningConfiguration, String> {
    let (rows_pattern, _) = list_to_pattern(&profile.rows).map_err(|e| e.to_string())?;
    let (cols_pattern, _) = list_to_pattern(&profile.cols).map_err(|e| e.to_string())?;

    Ok(zk_pow::api::proof::MiningConfiguration {
        common_dim: profile.k,
        rank: profile.rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern,
        cols_pattern,
        moe: None,
    })
}

/// The seed-derivation the network's certificate version mandates.
///
/// Cert 3 is salted (llmjob's live path); everything older is legacy. The single
/// version→derivation mapping lives in `zk_pow`, so this defers to it rather than
/// re-deriving the rule.
#[cfg(test)]
fn seed_for(cert_version: u32) -> Result<SeedDerivation, String> {
    Ok(CertificateVersion::try_from(cert_version)
        .map_err(|e| e.to_string())?
        .seed_derivation())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::miners::pearl::config::PROFILE;

    /// The operand chunks a row set touches are the ones the pre-submit index
    /// check compares against. At `k = 2048` and a 1024-byte chunk each row is
    /// exactly two chunks, and consecutive rows land on consecutive pairs.
    #[test]
    fn row_leaf_indices_cover_every_chunk_a_row_straddles() {
        assert_eq!(row_leaf_indices(&[0], 2048), vec![0, 1]);
        assert_eq!(row_leaf_indices(&[1], 2048), vec![2, 3]);
        // Out-of-order rows still come back as one ascending set.
        assert_eq!(row_leaf_indices(&[3, 0], 2048), vec![0, 1, 6, 7]);
        // Every index distinct and ascending — the property the leaf/data
        // pairing in the encoder relies on.
        let wide = row_leaf_indices(&[0, 1, 2, 3], 2048);
        assert!(wide.windows(2).all(|w| w[0] < w[1]));
    }

    /// A row whose byte range does not divide the chunk size must straddle.
    #[test]
    fn row_leaf_indices_handle_a_row_wider_than_a_chunk() {
        assert_eq!(row_leaf_indices(&[0], 3072), vec![0, 1, 2]);
        assert_eq!(row_leaf_indices(&[1], 3072), vec![3, 4, 5]);
    }

    /// A malformed side is refused rather than zero-padded into a proof: a
    /// short copy off the device would otherwise certify leaves nobody read.
    #[test]
    fn a_short_side_is_not_a_proof() {
        let empty = ProofSide {
            leaf_indices: vec![],
            leaves: vec![],
            siblings: vec![],
            root: [0; 32],
            total_leaves: 0,
        };
        assert!(unpack_side(&empty).is_none());

        let short = ProofSide {
            leaf_indices: vec![0, 1],
            leaves: vec![0u8; CHUNK_BYTES], // one chunk claimed for two indices
            siblings: vec![],
            root: [0; 32],
            total_leaves: 4,
        };
        assert!(unpack_side(&short).is_none());
    }

    /// A side with no siblings is legal: the tile can cover a whole small tree.
    #[test]
    fn an_empty_sibling_list_is_allowed() {
        let side = ProofSide {
            leaf_indices: vec![0],
            leaves: vec![7u8; CHUNK_BYTES],
            siblings: vec![],
            root: [0; 32],
            total_leaves: 1,
        };
        let unpacked = unpack_side(&side).expect("a whole-tree leaf is a valid side");
        assert!(unpacked.siblings.is_empty());
        assert_eq!(unpacked.leaf_data.len(), 1);
    }

    /// The encoding is `base64(bincode(PlainProof))`, and the reference decoder
    /// reads it straight back — the container, the `moe: None` tag and the gzip
    /// envelope. Built from the reference search at the smallest dimensions that
    /// still form a real Merkle tree, so nothing here depends on the shipped
    /// geometry.
    #[test]
    fn a_proof_round_trips_through_the_wire_format() {
        use zk_pow::ffi::mine::try_mine_one;

        let mining = crate::miners::pearl_mining::small_mining();
        let profile = Profile {
            k: mining.k as u32,
            rank: mining.rank,
            m: mining.m as u32,
            n: mining.n as u32,
            ..PROFILE
        };
        let config = mining_configuration(&profile).unwrap();
        let header = IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x11; 32],
            merkle_root: [0x22; 32],
            timestamp: 1_700_000_000,
            nbits: 0,
        };

        let mut rng = rand::rng();
        // `wrong_jackpot_hash = true` inverts the acceptance test, so this yields
        // a proof on the first draw without mining a real solution.
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
        .expect("an inverted acceptance test always returns a proof");

        let encoded = encode_plain_proof(&proof, false).unwrap();
        let bytes = STANDARD.decode(&encoded).unwrap();
        let parsed = PlainProof::deserialize_compat(&bytes).unwrap();

        assert_eq!(parsed.m, mining.m);
        assert_eq!(parsed.n, mining.n);
        assert_eq!(parsed.k, mining.k);
        assert_eq!(parsed.noise_rank, mining.rank as usize);
        assert!(parsed.moe.is_none(), "a dense proof carries the None tag");

        // The gzip variant is a different container for the same proof.
        let gzipped = encode_plain_proof(&proof, true).unwrap();
        assert_ne!(gzipped, encoded);
        assert_eq!(&STANDARD.decode(&gzipped).unwrap()[..3], &[0x1f, 0x8b, 0x08]);
    }
}
