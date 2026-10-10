//! The share target arithmetic and the tile/region mapping.
//!
//! A byte-for-byte port of `vendor/llmjob/earn/src/shared/miner/pearlhash.js`'s
//! `shareBound`, `offsetIsValid`, `expandOffset`, `regionToTile` and the
//! little-endian jackpot comparison, reusing the `primitive_types::U256` the
//! `zk-pow` verifier already speaks rather than a fresh bignum.
//!
//! Two endiannesses meet here and they are opposite, which is the single easiest
//! thing to get backwards and the hardest to notice: the *target* off the wire is
//! big-endian, and the *jackpot hash* is read little-endian. Get the second wrong
//! and every "share" the core reports is spurious.

use primitive_types::U256;

use super::config::{Profile, ROWS_PATTERN, COLS_PATTERN, PENALTY_BASE_RANK};

/// Parses a `mining.notify` share target: a big-endian hex U256.
///
/// Pools do not agree on the difficulty convention — Kryptex sends
/// `2^224 / diff - 1` while HeroMiners and LuckyPool send
/// `floor(0xFFFF * 2^208 / diff)` — so the value is used exactly as received and
/// never recomputed from the `diff` embedded in the job id.
///
/// Shorter-than-256-bit values are left-padded, and Kryptex's targets are not a
/// whole number of bytes, so an odd digit count is normal rather than an error.
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

/// The factor a share target is scaled by before a jackpot hash is compared
/// against it: `tile_size * (k / rank) * PENALTY_BASE_RANK`.
///
/// This is llmjob's `penalizedAdjustmentFactor`, and it is deliberately not the
/// same quantity as the consensus difficulty factor `tile_size * k`, even though
/// the two coincide at the mandated rank-128 profile and so are easy to conflate.
/// The protocol scales a *share* bound in proportion to what one attempt costs in
/// MACs, so a hashrate unless this factor is applied every share is `65536x`
/// rarer than the pool intends — which looks exactly like being slow. At the
/// mainnet profile this is `256 * 16 * 128 = 524288`.
pub fn share_bound(target: U256, profile: &Profile) -> Result<U256, String> {
    let factor = penalized_adjustment_factor(profile);

    if factor.is_zero() || target > U256::max_value() / factor {
        return Err(format!(
            "No usable share bound for the target {target:#x} at rank {}: the target is too easy \
             for this mining configuration.",
            profile.rank
        ));
    }

    Ok(target * factor)
}

/// `tile_size * (k / rank) * PENALTY_BASE_RANK`, as a `U256` for scaling and a
/// `u64` for tests that want the plain number.
pub fn penalized_adjustment_factor(profile: &Profile) -> U256 {
    U256::from(penalized_adjustment_factor_u64(profile))
}

pub fn penalized_adjustment_factor_u64(profile: &Profile) -> u64 {
    let tile = profile.tile_size() as u64;
    tile * (profile.k / profile.rank.max(1) as u32) as u64 * PENALTY_BASE_RANK as u64
}

/// Everything the search needs from a `mining.notify`: the target it was given,
/// and the bound to mine against.
pub fn share_search(hex_target: &str, profile: &Profile) -> Result<(U256, U256), String> {
    let target = parse_share_target(hex_target)?;
    let bound = share_bound(target, profile)?;
    Ok((target, bound))
}

/// Does a jackpot hash satisfy a share bound? `int_le(jackpot_hash) <= bound`.
///
/// The hash is read **little-endian**; the bound is the big-endian target already
/// scaled by [`share_bound`]. The core does this on the GPU for throughput, and
/// the host re-checks every hit before submitting, so a core bug or a stale job
/// cannot push a bad share to the pool.
pub fn meets_target(jackpot_hash: &[u8; 32], bound: U256) -> bool {
    U256::from_little_endian(jackpot_hash) <= bound
}

/// Re-encodes a 256-bit target as Bitcoin's compact `nbits`, rounding up.
///
/// This is what [`super::proof::verify_share_locally`] hands the verifier: no
/// ordinary share meets the block target, so the verifier is given the *share*
/// target re-encoded instead. Rounding up matters — a truncated mantissa
/// encodes a *smaller* target, which would make the verification bound tighter
/// than the one the search ran against, and drop shares the pool would take.
#[cfg(test)]
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

    // The compact form carries only a 23-bit mantissa, and `nbits_to_difficulty`
    // reads anything wider as a negative target. Carry into the exponent,
    // rounding up over the discarded byte so the encoded target never lands
    // below the original.
    if mantissa > 0x007f_ffff {
        let dropped = mantissa & 0xff;
        mantissa >>= 8;
        if dropped != 0 {
            mantissa += 1;
        }
        exponent += 1;
    }

    // A 33-byte exponent cannot describe a 256-bit target. Clamp to the largest
    // representable value rather than wrapping; the verifier's bound saturates
    // from there anyway.
    if exponent > 32 {
        exponent = 32;
        mantissa = 0x007f_ffff;
    }

    Ok((exponent << 24) | (mantissa & 0x007f_ffff))
}

/// The i-th offset with `(offset & mask) == 0`: deposit the bits of `i` into the
/// positions the mask leaves free.
pub fn expand_offset(i: u32, mask: u32) -> u32 {
    let mut out = 0u32;
    let mut bit = 1u32;
    let mut rest = i;
    while rest != 0 {
        if mask & bit == 0 {
            if rest & 1 != 0 {
                out |= bit;
            }
            rest >>= 1;
        }
        bit <<= 1;
    }
    out
}

/// The bits of the rows pattern. [`offset_is_valid`] with this mask reduces the
/// verifier's `PeriodicPattern::offset_is_valid` to one AND.
pub const ROWS_MASK: u32 = rows_mask();
pub const COLS_MASK: u32 = cols_mask();

const fn fold_or(list: &[u32]) -> u32 {
    let mut acc = 0u32;
    let mut i = 0;
    while i < list.len() {
        acc |= list[i];
        i += 1;
    }
    acc
}

const fn rows_mask() -> u32 {
    fold_or(&ROWS_PATTERN)
}

const fn cols_mask() -> u32 {
    fold_or(&COLS_PATTERN)
}

/// A tile offset is valid only if it has the pattern's own bits clear. For these
/// patterns the verifier's modulo reduction is exactly `(offset & mask) == 0`.
///
/// Searching an invalid offset is not merely wasteful: the share comes back as
/// "offset N is not valid for pattern". Only 1 in 32 regions qualifies, and the
/// valid tiles partition the grid rather than overlapping. The engine enumerates
/// valid offsets directly through [`expand_offset`], so this is only the rule
/// made checkable in a test.
#[cfg(test)]
pub fn offset_is_valid(offset: u32, mask: u32) -> bool {
    offset & mask == 0
}

/// A region index -> the tile it names. Row and column offsets are enumerated
/// over valid offsets only, so this is dense in the submittable space.
pub fn region_to_tile(region: u64, profile: &Profile) -> Tile {
    let rows_valid = profile.m / profile.rows.len() as u32;
    let cols_valid = profile.n / profile.cols.len() as u32;
    let row_off = expand_offset((region % rows_valid as u64) as u32, ROWS_MASK);
    let col_off = expand_offset(((region / rows_valid as u64) % cols_valid as u64) as u32, COLS_MASK);
    Tile {
        row_off,
        col_off,
        // The pattern's bits are clear in a valid offset, so this is an OR.
        rows: profile.rows.iter().map(|r| row_off | r).collect(),
        cols: profile.cols.iter().map(|c| col_off | c).collect(),
    }
}

/// The rows and columns a region names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tile {
    pub row_off: u32,
    pub col_off: u32,
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
}

/// How many regions one salt's operand draw offers: the number of valid tiles.
pub fn regions_per_salt(profile: &Profile) -> u64 {
    (profile.m / profile.rows.len() as u32) as u64
        * (profile.n / profile.cols.len() as u32) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::miners::pearl::config::PROFILE;

    /// HeroMiners' documented starting share target: `0x7fff8 << 184`, compact
    /// `0x1a07fff8`.
    const HERO_TARGET: u64 = 0x7fff8;
    const HERO_SHIFT: u32 = 184;
    const HERO_NBITS: u32 = 0x1a07_fff8;

    fn hero_target() -> U256 {
        U256::from(HERO_TARGET) << HERO_SHIFT
    }

    #[test]
    fn parses_a_herominers_share_target() {
        let hex = format!("{:0>64}", format!("7fff8{}", "0".repeat(46)));
        assert_eq!(hex.len(), 64);
        assert!(
            hex.starts_with("0000000000000"),
            "left-padded, as the pool sends it"
        );

        let parsed = parse_share_target(&hex).unwrap();
        assert_eq!(parsed, hero_target());
        assert_eq!(parse_share_target(&format!("0x{hex}")).unwrap(), parsed);
    }

    #[test]
    fn parses_a_kryptex_share_target_with_an_odd_digit_count() {
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

    /// The factor is the protocol's `tile * (k/rank) * 128`, and at the mainnet
    /// profile it is 524288 — the same value the JS core's own comment states.
    #[test]
    fn the_share_factor_is_the_penalized_one() {
        assert_eq!(penalized_adjustment_factor_u64(&PROFILE), 524_288);
        assert_eq!(
            penalized_adjustment_factor(&PROFILE),
            U256::from(524_288u64)
        );
    }

    /// The bound is `target * factor`, exactly, with no saturation: a target that
    /// would overflow is refused rather than clamped to `U256::MAX`, which every
    /// digest satisfies and would report a solution on the first candidate.
    #[test]
    fn the_bound_is_the_target_scaled_and_refuses_to_saturate() {
        let bound = share_bound(hero_target(), &PROFILE).unwrap();
        assert_eq!(bound, hero_target() * U256::from(524_288u64));
        assert!(U256::from(524_288u64) > U256::from(1u8));
        // Refusal, not saturation.
        assert!(share_bound(U256::MAX, &PROFILE).is_err());
    }

    /// Re-encoding a target as compact `nbits` must reproduce the value a pool
    /// documents, and round-trip through the verifier's own decoder exactly.
    #[test]
    fn re_encodes_a_share_target_as_the_documented_compact_nbits() {
        use zk_pow::api::proof_utils::nbits_to_difficulty;

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

    /// Rounding must go up, or the verification bound would be tighter than the
    /// search's and drop shares the pool would accept.
    #[test]
    fn compact_nbits_never_encodes_a_target_below_the_original() {
        use zk_pow::api::proof_utils::nbits_to_difficulty;

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
        use zk_pow::api::proof_utils::nbits_to_difficulty;

        // A target this size cannot be represented exactly; the clamp must not wrap to zero.
        let nbits = target_to_nbits(U256::MAX).unwrap();
        assert!(nbits_to_difficulty(nbits) > U256::from(1u8) << 250);
    }

    #[test]
    fn compact_nbits_refuses_a_zero_target() {
        assert!(target_to_nbits(U256::zero()).is_err());
    }

    /// The jackpot hash is read little-endian. Reversing it is the classic silent
    /// bug, so both readings are pinned: byte 0 is the least significant.
    #[test]
    fn the_jackpot_hash_is_compared_little_endian() {
        let mut hash = [0u8; 32];
        hash[0] = 0x01; // least significant
        assert_eq!(U256::from_little_endian(&hash), U256::from(1u8));
        assert_eq!(
            U256::from_big_endian(&hash),
            U256::from(1u8) << 248,
            "reading it big-endian would put the same byte at the top"
        );

        // A small bound admits a hash whose LE value is small and rejects one whose LE value is
        // large, regardless of how the bytes are arranged in memory.
        let mut big = [0u8; 32];
        big[31] = 0xff; // most significant under LE
        assert!(meets_target(&hash, U256::from(1u8)));
        assert!(!meets_target(&big, U256::from(1u8)));
    }

    /// The masks reduce the verifier's offset rule to one AND, and the pattern's
    /// values are exactly the bits it is allowed to set.
    #[test]
    fn the_offset_masks_are_the_patterns_own_bits() {
        // The braces in the message are doubled: the message goes through
        // `format!`, where a bare `{` starts a placeholder.
        assert_eq!(ROWS_MASK, 0b0001_1011, "rows {{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27}}");
        assert_eq!(COLS_MASK, 0b0011_1001, "cols {{0,1,8,9,16,17,24,25,32,33,40,41,48,49,56,57}}");
        // Every pattern index keeps its own bits clear in the mask, or a valid
        // offset could not be formed by OR-ing one in.
        for &r in &ROWS_PATTERN {
            assert_eq!(r & ROWS_MASK, r, "row {r} is not a subset of the rows mask");
        }
        for &c in &COLS_PATTERN {
            assert_eq!(c & COLS_MASK, c, "col {c} is not a subset of the cols mask");
        }
    }

    /// `expand_offset` deposits index bits into the free positions, so the valid
    /// offsets it enumerates are exactly those with the mask's bits clear, in
    /// ascending order, and the first few are the pattern itself.
    ///
    /// The stride-8 pattern leaves bits 2 and 5 free, so the second offset is 4
    /// and the third is 32 — not 16, which is what the old contiguous 0..15
    /// pattern produced. A wrong expectation here is a wrong *offset*, and the
    /// pool rejects a share whose offset does not match its pattern.
    #[test]
    fn expand_offset_enumerates_exactly_the_valid_offsets() {
        assert_eq!(expand_offset(0, ROWS_MASK), 0);
        assert_eq!(expand_offset(1, ROWS_MASK), 4, "bit 2 is the lowest free one");
        assert_eq!(expand_offset(2, ROWS_MASK), 32, "then bit 5");
        assert_eq!(expand_offset(3, ROWS_MASK), 36);

        // The mask has four bits set, so 2^4 = 16 offsets have it clear — the
        // free bits are 2, 5, 6 and 7, so they run up to 4 + 32 + 64 + 128 = 228.
        // Only the first two land inside the pattern's own 32-row span; the rest
        // sit in later spans, which is why the operand is staged in span-sized
        // runs rather than as one block per offset.
        let mut seen = Vec::new();
        for i in 0..16u32 {
            let offset = expand_offset(i, ROWS_MASK);
            assert!(offset_is_valid(offset, ROWS_MASK));
            assert!(
                ROWS_PATTERN.iter().all(|&r| offset | r == offset + r),
                "offset {offset} overlaps the pattern's own bits, so the tile is not a partition"
            );
            assert!(!seen.contains(&offset));
            seen.push(offset);
        }
        assert_eq!(seen.len(), 16, "the mask's four free bits name 16 offsets");
        assert_eq!(seen.iter().copied().max(), Some(228));
    }

    /// `regionToTile` is dense in the submittable space: the first `rows_valid`
    /// regions sweep the row offset and hold the column offset at zero, and the
    /// region index names a distinct valid tile.
    #[test]
    fn region_to_tile_sweeps_rows_then_columns() {
        let p = Profile {
            m: 128,
            n: 128,
            ..PROFILE
        };
        let rows_valid = p.m / p.rows.len() as u32; // 8
        assert_eq!(rows_valid, 8);

        let first = region_to_tile(0, &p);
        assert_eq!(first.row_off, 0);
        assert_eq!(first.col_off, 0);
        assert_eq!(first.rows, ROWS_PATTERN.to_vec());

        // Region 1 is the next VALID row offset, not row 1: the pattern's bits
        // are skipped, so it is offset 4.
        assert_eq!(region_to_tile(1, &p).row_off, 4);
        assert_eq!(region_to_tile(1, &p).col_off, 0);

        // Region rows_valid wraps into the column axis, and lands on the first
        // VALID column offset. The cols mask 0b111001 leaves bit 1 free, so that
        // is 2 — not 4 (the rows pattern's answer) and not 8 (the stride).
        let wrapped = region_to_tile(rows_valid as u64, &p);
        assert_eq!(wrapped.row_off, 0);
        assert_eq!(wrapped.col_off, 2);

        assert_eq!(regions_per_salt(&p), 64);
    }

    /// Distinct regions name distinct tiles — the property the whole search relies
    /// on to not repeat work within a salt.
    #[test]
    fn distinct_regions_name_distinct_tiles() {
        let p = Profile {
            m: 512,
            n: 512,
            ..PROFILE
        };
        let n = regions_per_salt(&p) as usize;
        let mut tiles = Vec::new();
        for region in 0..n as u64 {
            let tile = region_to_tile(region, &p);
            assert!(offset_is_valid(tile.row_off, ROWS_MASK));
            assert!(offset_is_valid(tile.col_off, COLS_MASK));
            assert!(!tiles.contains(&(tile.row_off, tile.col_off)));
            tiles.push((tile.row_off, tile.col_off));
        }
        assert_eq!(tiles.len(), n);
    }
}
