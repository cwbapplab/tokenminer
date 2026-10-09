//! Pearl share-target arithmetic and the local verification gate.
//!
//! `pearl_mining` owns the configuration and the wire encoding, `stratum` speaks the protocol,
//! `pearl_gpu` runs the search. What is left here is the arithmetic that decides *what* counts as a
//! solution, plus the check every candidate must pass before it goes on the wire.
//!
//! The search itself is on the GPU, so there is no host implementation of "find a proof" here.

use primitive_types::U256;
use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration};
use zk_pow::api::verify;
use zk_pow::ffi::plain_proof::PlainProof;

use super::pearl_mining::{mining_configuration, seed_derivation_for, PearlMining};

/// Parses a `mining.notify` share target: a big-endian hex U256.
///
/// Pools do not agree on the difficulty convention — Kryptex sends `2^224 / diff - 1` while
/// HeroMiners and LuckyPool send `floor(0xFFFF * 2^208 / diff)` — so the value is used exactly as
/// received and never recomputed from the `diff` embedded in the job id.
///
/// Shorter-than-256-bit values are left-padded, and Kryptex's targets are not a whole number of
/// bytes, so an odd digit count is normal rather than an error.
pub fn parse_share_target(hex_target: &str) -> Result<U256, String> {
    let trimmed = hex_target.trim();
    let digits = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);

    if digits.is_empty() {
        return Err("The job's share target is empty.".into());
    }
    if !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "The job's share target is not hexadecimal: '{trimmed}'"
        ));
    }
    if digits.len() > 64 {
        return Err(format!(
            "The job's share target is wider than 256 bits: '{trimmed}'"
        ));
    }

    // `hex::decode` needs whole bytes.
    let padded = if digits.len() % 2 == 1 {
        format!("0{digits}")
    } else {
        digits.to_string()
    };

    let bytes =
        hex::decode(&padded).map_err(|e| format!("Invalid share target '{trimmed}': {e}"))?;

    let mut full = [0u8; 32];
    full[32 - bytes.len()..].copy_from_slice(&bytes);

    Ok(U256::from_big_endian(&full))
}

/// The hashed output cells in one candidate tile: `rows_pattern.size() * cols_pattern.size()`.
///
/// Mirrors the private `tile_size` in `zk_pow::api::sanity_checks`, which is not exported.
fn tile_size(config: &MiningConfiguration) -> usize {
    config.rows_pattern.size() as usize * config.cols_pattern.size() as usize
}

/// The jackpot bound a proof must beat for the pool to accept it as a share.
///
/// This is deliberately the **unpenalized** factor, `tile * dot_product_length` — the same one the
/// verifier applies in `extract_difficulty_bound` when it re-checks a submitted share. That is what
/// Open-Pearl-Miner mines against (`hash_tile_h * hash_tile_w * rounded_common_dim`).
///
/// [`zk_pow::api::sanity_checks::penalized_target_bound`] would instead scale by
/// `tile * (dot_product_length / rank) * 128`, which is `128 / rank` times this. At `rank = 256`
/// that is half: a proof clearing that bar also clears the pool's, so it is never *wrong* — it just
/// throws away half the search effort on candidates too hard to ever submit.
///
/// The safety net is [`verify_share_locally`], which runs the real verifier on every candidate
/// before submission. A bound that was somehow too loose costs one rejected share; it can never
/// produce a malformed proof.
pub fn share_bound(target: U256, mining: &PearlMining) -> Result<U256, String> {
    let config = mining_configuration(mining)?;
    let factor = tile_size(&config) * config.dot_product_length();

    // Error rather than saturate: a saturated bound is `U256::MAX`, which every digest satisfies,
    // so the search would report a solution on its first candidate.
    if factor == 0 || target > U256::MAX / factor {
        return Err(format!(
            "No usable share bound for the target {target:#x} at rank {}: the target is too easy \
             for this mining configuration.",
            mining.rank
        ));
    }

    Ok(target * factor)
}

/// Everything the search needs from a `mining.notify`: the target it was given, and the bound to
/// mine against.
pub fn share_search(hex_target: &str, mining: &PearlMining) -> Result<(U256, U256), String> {
    let target = parse_share_target(hex_target)?;
    let bound = share_bound(target, mining)?;

    Ok((target, bound))
}

/// Re-encodes a 256-bit target as Bitcoin's compact `nbits`, rounding up.
///
/// Rounding up matters: a truncated mantissa encodes a *smaller* target, which would make the
/// verification bound tighter than the one the search ran against.
pub fn target_to_nbits(target: U256) -> Result<u32, String> {
    if target.is_zero() {
        return Err("The share target is zero.".into());
    }

    // Bit length gives the exponent (how many bytes the full target needs).
    let bits = 256 - target.leading_zeros();
    let mut exponent = bits.div_ceil(8);

    let mut mantissa = if exponent <= 3 {
        let shift = 8 * (3 - exponent) as usize;
        (target << shift).low_u32()
    } else {
        let shift = 8 * (exponent - 3) as usize;
        let truncated = target >> shift;
        let rounded_up = target != (truncated << shift);

        truncated.low_u32() + u32::from(rounded_up)
    };

    // The compact form carries only a 23-bit mantissa, and `nbits_to_difficulty` reads anything
    // wider as a negative target. Carry into the exponent, rounding up over the discarded byte so
    // the encoded target never lands below the original.
    if mantissa > 0x007f_ffff {
        let dropped = mantissa & 0xff;
        mantissa >>= 8;
        if dropped != 0 {
            mantissa += 1;
        }
        exponent += 1;
    }

    // A 33-byte exponent cannot describe a 256-bit target. Clamp to the largest representable
    // value rather than wrapping; the verifier's bound saturates from there anyway.
    if exponent > 32 {
        exponent = 32;
        mantissa = 0x007f_ffff;
    }

    Ok((exponent << 24) | (mantissa & 0x007f_ffff))
}

/// Re-checks a proof the way the pool will, before it is submitted.
///
/// Deliberately not a block-difficulty check: no ordinary share meets the block target, so that
/// would reject exactly the proofs we want to send. Instead the share target is re-encoded as
/// compact `nbits` and handed to the real verifier, which recomputes the jackpot from the committed
/// matrices and compares it against that target.
///
/// This is the only thing standing between a GPU bug and the wire. It is a full re-verification —
/// the matrices, the noise, the jackpot and the Merkle proof — against the verifier the pool uses,
/// so a proof that passes here is one the pool accepts.
pub fn verify_share_locally(
    header: &IncompleteBlockHeader,
    cert_version: u32,
    proof: &PlainProof,
    share_target: U256,
) -> Result<(), String> {
    let seed = seed_derivation_for(cert_version)?;
    let nbits = target_to_nbits(share_target)?;

    verify::verify_plain_proof(header, proof, Some(nbits), seed).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zk_pow::api::proof::SeedDerivation;
    use zk_pow::api::proof_utils::nbits_to_difficulty;
    use zk_pow::api::sanity_checks::{
        extract_difficulty_bound, penalized_target_bound, PENALTY_BASE_RANK,
    };
    use zk_pow::ffi::mine::try_mine_one;

    fn test_header(nbits: u32) -> IncompleteBlockHeader {
        IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x11; 32],
            merkle_root: [0x22; 32],
            timestamp: 1_700_000_000,
            nbits,
        }
    }

    /// HeroMiners' starting share difficulty (2,097,152), from their documented notify example:
    /// target `0x7fff8 << 184`, compact `0x1a07fff8`.
    const HERO_TARGET: u64 = 0x7fff8;
    const HERO_SHIFT: u32 = 184;
    const HERO_NBITS: u32 = 0x1a07_fff8;

    /// The largest compact target: encodes `0x7fffff << 232`, so any difficulty factor over it
    /// overflows and the verifier saturates to `U256::MAX` — a bound every digest satisfies.
    const TRIVIAL_NBITS: u32 = 0x207f_ffff;

    fn hero_target() -> U256 {
        U256::from(HERO_TARGET) << HERO_SHIFT
    }

    #[test]
    fn parses_a_herominers_share_target() {
        // The pool sends 64 hex characters: 0x7fff8 followed by 46 zero digits.
        let hex = format!("{:0>64}", format!("7fff8{}", "0".repeat(46)));
        assert_eq!(hex.len(), 64);
        assert!(
            hex.starts_with("0000000000000"),
            "left-padded, as the pool sends it"
        );

        let parsed = parse_share_target(&hex).unwrap();
        assert_eq!(parsed, hero_target());

        // Pools differ on whether the value carries a 0x prefix.
        assert_eq!(parse_share_target(&format!("0x{hex}")).unwrap(), parsed);
    }

    #[test]
    fn parses_a_kryptex_share_target_with_an_odd_digit_count() {
        // Kryptex sends 2^224/diff - 1, a 203-bit value: 51 hex digits.
        let odd = "7".to_string() + &"f".repeat(50);
        assert_eq!(odd.len(), 51);

        let parsed = parse_share_target(&odd).unwrap();
        assert_eq!(parsed, parse_share_target(&format!("0{odd}")).unwrap());
        assert_eq!(parsed, (U256::from(1u8) << 203) - U256::from(1u8));
    }

    #[test]
    fn rejects_a_missing_or_malformed_share_target() {
        assert!(parse_share_target("").is_err());
        assert!(parse_share_target("   ").is_err());
        assert!(parse_share_target("not-hex").is_err());
        assert!(parse_share_target(&"f".repeat(65)).is_err());
    }

    #[test]
    fn re_encodes_a_share_target_as_the_documented_compact_nbits() {
        let nbits = target_to_nbits(hero_target()).unwrap();

        assert_eq!(
            nbits, HERO_NBITS,
            "must match the compact value HeroMiners documents"
        );
        assert_eq!(
            nbits_to_difficulty(nbits),
            hero_target(),
            "exact round trip"
        );
    }

    #[test]
    fn compact_nbits_never_encodes_a_target_below_the_original() {
        // Rounding must go up, or the verification bound would be tighter than the search's.
        for shift in [176u32, 180, 184, 190, 200, 203] {
            let target = (U256::from(0x00ab_cdefu32) << shift) + U256::from(0x1234u32);
            let nbits = target_to_nbits(target).unwrap();

            assert!(
                nbits_to_difficulty(nbits) >= target,
                "nbits {nbits:#010x} encodes below the target {target:#x}"
            );
        }
    }

    #[test]
    fn compact_nbits_handles_a_target_near_the_top_of_the_range() {
        // A target this size cannot be represented exactly; the clamp must not wrap to zero.
        let nbits = target_to_nbits(U256::MAX).unwrap();
        assert!(nbits_to_difficulty(nbits) > U256::from(1u8) << 250);
    }

    #[test]
    fn the_search_bound_is_exactly_the_bound_the_pool_verifies_against() {
        // This is the invariant the search depends on. `share_bound` scales the target by
        // `tile * dot_product_length`; `verify_share_locally` re-encodes the same target and hands
        // it to `extract_difficulty_bound`, which applies that identical factor. If these ever
        // drift apart, every share is either wasted (too loose) or silently dropped (too tight).
        let mining = PearlMining::default();
        let config = mining_configuration(&mining).unwrap();

        for target in [
            hero_target(),
            U256::from(1u8) << 200,
            U256::from(1u8) << 220,
        ] {
            let searched = share_bound(target, &mining).unwrap();
            let verified = extract_difficulty_bound(target_to_nbits(target).unwrap(), &config);

            assert_eq!(
                searched, verified,
                "the search bound {searched:#x} and the verification bound {verified:#x} disagree"
            );
        }
    }

    #[test]
    fn the_search_bound_is_not_the_rank_penalised_one() {
        // Guards the 2x regression this replaced: above the base rank the two factors part company,
        // and mining against the penalised one discards that much of the search effort on
        // candidates too hard to ever submit.
        let mining = PearlMining::default();
        let config = mining_configuration(&mining).unwrap();

        // Whatever the rank, `share_bound` applies the verifier's factor: tile * dot_product_length.
        let searched = share_bound(hero_target(), &mining).unwrap();
        assert_eq!(
            U256::from(tile_size(&config) * config.dot_product_length()),
            searched / hero_target()
        );

        // The gap is exactly `rank / PENALTY_BASE_RANK`. The shipped default now sits *at* the base rank,
        // so it is the case where the two agree — asserted, because a default that silently carried a
        // penalty would look like the unpenalised one. The parting needs a rank above the base, so this
        // builds one rather than reading it off the default.
        assert_eq!(
            mining.rank as usize,
            PENALTY_BASE_RANK,
            "the shipped default sits at the base rank, so no penalty applies to it"
        );

        let above = PearlMining {
            rank: 2 * PENALTY_BASE_RANK as u16,
            ..PearlMining::default()
        };
        let above_config = mining_configuration(&above).unwrap();
        let searched_above = share_bound(hero_target(), &above).unwrap();
        let penalised = penalized_target_bound(hero_target(), &above_config).unwrap();

        assert_eq!(
            searched_above,
            penalised * (above.rank as usize / PENALTY_BASE_RANK)
        );
        assert!(
            searched_above > penalised,
            "at rank {} the unpenalised bound should be the wider of the two",
            above.rank
        );
    }

    #[test]
    fn rejects_a_share_target_too_easy_for_the_configuration() {
        // Saturating to U256::MAX here would make the bound universal — every digest would pass and
        // the search would claim a solution on its first candidate.
        let mining = PearlMining::default();
        assert!(share_bound(U256::MAX, &mining).is_err());
    }

    #[test]
    fn a_proof_built_to_the_search_bound_passes_the_pool_side_verification() {
        // Manufactures a fixture rather than mining one: the CPU search is gone, but
        // `try_mine_one` is still the reference implementation and is the honest way to produce a
        // proof to check the accept path against.
        //
        // A trivially easy `nbits` makes the verification bound saturate to U256::MAX, so the very
        // first tile is accepted and the test does not depend on luck.
        //
        // The smallest configuration the verifier accepts, not the shipped default: the default is
        // 131072 x 131072, and `try_mine_one` builds both matrices and a Merkle tree over each on
        // the CPU, which turned this into a twelve-minute test. What is under test is the accept
        // path, which does not read the dimensions.
        let mining = crate::miners::pearl_mining::small_mining();
        let header = test_header(TRIVIAL_NBITS);
        let config = mining_configuration(&mining).unwrap();

        let mut rng = rand::rng();
        let proof = try_mine_one(
            &mut rng,
            mining.m,
            mining.n,
            mining.k,
            header,
            config,
            None,
            false,
            SeedDerivation::Salted,
        )
        .expect("the reference search does not error")
        .expect("a saturated bound accepts the first tile");

        verify_share_locally(&header, 3, &proof, nbits_to_difficulty(TRIVIAL_NBITS))
            .expect("the pool's own check accepts the proof");
    }
}
