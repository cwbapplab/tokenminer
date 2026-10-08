//! The fatbin `build.rs` produced, embedded in the binary.
//!
//! One image for every card, built by `build.rs::build_fatbin` with multi-arch gencode. This is the
//! difference from [`cubins`], and it is the reason there is one image here instead of one per arch:
//! `cuModuleLoadData` reads the image's own per-arch entries and picks the SASS that matches the
//! device, falling back to the `compute_86` PTX when none matches. So the list in `build.rs` is not
//! the set of GPUs this runs on the way the cubin list is — a card above `sm_86` runs through the
//! driver JIT, and a card below it has nothing to load at all, because PTX forward-compatibility runs
//! from older to newer only.
//!
//! `cudarc` reaches this through `Ptx::from_binary`, which routes to `cuModuleLoadData` — the same
//! entry point that accepts a cubin or PTX text. `cuModuleLoadFatBinary` is not needed: the driver
//! parses the fatbin header itself. Unlike PTX, the image is not read as a C string, so no trailing
//! NUL is required.
//!
//! Every symbol here is unmangled, which is a deliberate deviation from the reference. Its table mixes
//! `pearl_*` names with Itanium-mangled names for kernels at file scope in their headers. A mangled
//! name in a symbol table is a string that breaks when the signature changes and nothing checks it —
//! the lookup fails at run time, not at compile time. The test
//! [`the_symbol_table_matches_the_entry_points`] reads the entry points themselves and asserts the
//! names agree, so a rename on either side fails here rather than on a device.
//!
//! Build policy: a machine with no CUDA toolkit produces an empty image and the crate still builds, so
//! presence has to be checked, never assumed. But this image is the only search path, so an empty
//! image is a miner that never starts — which is why `build.rs` panics on a toolchain that is present
//! and cannot compile, rather than warning.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;

/// The embedded image. Empty when `nvcc` was not available at build time.
pub const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pearl_gemm.fatbin"));

/// The lowest compute capability this image can run on.
///
/// The image carries `compute_86` PTX as its JIT fallback, so a device below `sm_86` has no entry to
/// load. `cubins` lists `sm_75`; that arch cannot be served from this image.
pub const MIN_COMPUTE_CAPABILITY: (i32, i32) = (8, 6);

// =============================================================================
//   Symbols
// =============================================================================

pub mod symbols {
    pub const NOISE_GEN_DENSE_INT8_R128: &str = "pearl_noise_gen_dense_int8_R128";
    pub const NOISE_GEN_DENSE_FP16_R128: &str = "pearl_noise_gen_dense_fp16_R128";
    pub const NOISE_GEN_SPARSE_R128: &str = "pearl_noise_gen_sparse_R128";
    pub const BLAKE3_COMPARE: &str = "pearl_blake3_compare_kernel";
    pub const POW_SCAN_HITS: &str = "pearl_pow_scan_hits_kernel";
}

/// Every symbol the launch layer looks up, in the order the pipeline uses them.
pub const ALL_SYMBOLS: &[&str] = &[
    symbols::NOISE_GEN_DENSE_INT8_R128,
    symbols::NOISE_GEN_DENSE_FP16_R128,
    symbols::NOISE_GEN_SPARSE_R128,
    symbols::BLAKE3_COMPARE,
    symbols::POW_SCAN_HITS,
];

// =============================================================================
//   Launch geometry, as the entry points declare it
// =============================================================================

/// Block size for the three noise entry points.
pub const NOISE_BLOCK: u32 = 128;

/// Block size for the postpass and the scan.
pub const POSTPASS_BLOCK: u32 = 256;
pub const SCAN_BLOCK: u32 = 256;

/// The value `first_hit` must hold before every scan launch. `atomicMin` against it means "no hit"
/// survives as itself, so the host reads one word and decides.
pub const FIRST_HIT_SENTINEL: u32 = 0xFFFFFFFF;

/// Grid for a dense noise launch: one thread per 32-byte chunk of a `rows x 128` output.
pub fn dense_grid(rows: usize) -> u32 {
    ceil_div(ceil_div(rows * 128, 32), NOISE_BLOCK as usize) as u32
}

/// Grid for a sparse noise launch: one thread per 8-row chunk of a `k x 128` output.
pub fn sparse_grid(k: usize) -> u32 {
    ceil_div(ceil_div(k + 7, 8), NOISE_BLOCK as usize) as u32
}

/// Grid for a launch that runs one thread per candidate.
pub fn candidate_grid(candidates: usize, block: u32) -> u32 {
    ceil_div(candidates, block as usize) as u32
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

// =============================================================================
//   Load
// =============================================================================

/// The five entry points, resolved against the loaded image on a device.
pub struct FatbinKernels {
    pub noise_dense_int8: CudaFunction,
    pub noise_dense_fp16: CudaFunction,
    pub noise_sparse: CudaFunction,
    pub blake3_compare: CudaFunction,
    pub pow_scan_hits: CudaFunction,
    /// The compute capability the image was loaded on, which for a card above `sm_86` is not the arch
    /// the code was compiled for — it is the arch the driver JIT'd the `compute_86` PTX onto.
    pub cc: (i32, i32),
}

impl FatbinKernels {
    /// Loads the image and resolves every symbol.
    ///
    /// Resolving all five up front is the point: a symbol that does not exist fails here, at load,
    /// with the name in the message. A launch layer that looked symbols up lazily would fail mid-
    /// pipeline, on the first attempt that reached that stage.
    pub fn load(ctx: &Arc<CudaContext>, cc: (i32, i32)) -> Result<Self, String> {
        if FATBIN.is_empty() {
            return Err("no fatbin was embedded: `nvcc` was not available at build time, so the split \
                        search path has no kernels to run. Install the CUDA toolkit and rebuild."
                .to_string());
        }

        let (major, minor) = cc;
        if major * 10 + minor < MIN_COMPUTE_CAPABILITY.0 * 10 + MIN_COMPUTE_CAPABILITY.1 {
            return Err(format!(
                "the fatbin cannot run on {major}.{minor}: its lowest entry is sm_86 and its PTX \
                 fallback is compute_86, which the driver cannot JIT backwards. This image is not a \
                 substitute for the per-arch cubins on older cards."
            ));
        }

        let module = ctx
            .load_module(Ptx::from_binary(FATBIN.to_vec()))
            .map_err(|e| format!("loading the fatbin failed: {e}"))?;

        Ok(Self {
            noise_dense_int8: resolve(&module, symbols::NOISE_GEN_DENSE_INT8_R128)?,
            noise_dense_fp16: resolve(&module, symbols::NOISE_GEN_DENSE_FP16_R128)?,
            noise_sparse: resolve(&module, symbols::NOISE_GEN_SPARSE_R128)?,
            blake3_compare: resolve(&module, symbols::BLAKE3_COMPARE)?,
            pow_scan_hits: resolve(&module, symbols::POW_SCAN_HITS)?,
            cc,
        })
    }

    /// True when this machine has an image to load.
    pub fn available() -> bool {
        !FATBIN.is_empty()
    }
}

fn resolve(module: &Arc<CudaModule>, symbol: &str) -> Result<CudaFunction, String> {
    module
        .load_function(symbol)
        .map_err(|e| format!("the fatbin has no `{symbol}`: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the entry points themselves, not a copy of their names.
    ///
    /// The symbol table is a list of strings and the entry points are a source file; nothing in the
    /// build connects the two. This is the check that makes a rename on either side fail.
    #[test]
    fn the_symbol_table_matches_the_entry_points() {
        let source = include_str!("kernels/extern_c_shims.inc");
        for symbol in ALL_SYMBOLS {
            let declaration = format!("extern \"C\" __global__ void {symbol}(");
            assert!(
                source.contains(&declaration),
                "`{symbol}` is in the symbol table but is not an entry point in \
                 extern_c_shims.inc; the lookup would fail at run time"
            );
        }

        // The reverse direction: an entry point that is not in the table is unreachable.
        for line in source.lines() {
            if let Some(rest) = line.strip_prefix("extern \"C\" __global__ void ") {
                let name = rest.split('(').next().unwrap();
                assert!(
                    ALL_SYMBOLS.contains(&name),
                    "extern_c_shims.inc declares `{name}`, which the symbol table does not list"
                );
            }
        }
    }

    /// The geometry the launch layer computes has to be the geometry the entry points assume.
    ///
    /// Asserted at the point where the two disagree loudly: a grid one block short leaves the last
    /// chunk unwritten, and a grid one block long is harmless because the entry points guard on the
    /// chunk index. So the test uses a size that is not a multiple of the block — a test built on a
    /// round number cannot tell a `div_ceil` from a plain division.
    #[test]
    fn the_launch_grids_cover_every_chunk() {
        // Dense: rows * 128 / 32 chunks. 40 chunks at rows = 10, 164 at rows = 41 — the second is one
        // chunk past two blocks' worth, so a plain division would give 1 and drop chunk 160.
        assert_eq!(dense_grid(10), 1);
        assert_eq!(dense_grid(41), 2);

        // Sparse: (k + 7) / 8 chunks. 257 chunks at k = 2049 is one past two blocks' worth.
        assert_eq!(sparse_grid(257), 1); // 33 chunks
        assert_eq!(sparse_grid(2049), 3); // 257 chunks

        assert_eq!(candidate_grid(257, POSTPASS_BLOCK), 2);
        assert_eq!(candidate_grid(256, POSTPASS_BLOCK), 1);
    }

    /// The sentinel is the value the host reads to mean "no hit", so it has to be the value the scan
    /// cannot produce. `atomicMin` over candidate indices can produce any index, so the sentinel has to
    /// be above every possible index — which is why it is `u32::MAX` and not, say, `total`.
    #[test]
    fn the_no_hit_sentinel_is_above_every_candidate_index() {
        assert_eq!(FIRST_HIT_SENTINEL, u32::MAX);
        for candidates in [64, 4096, 1_048_576] {
            assert!(FIRST_HIT_SENTINEL > candidates);
        }
    }

    // ---------------------------------------------------------------------
    //   Gate B, host half: the noise decode arithmetic
    // ---------------------------------------------------------------------

    /// The reference's signed decode and `zk_pow`'s masked decode have to be the same function on
    /// every byte, not just on the bytes a test happens to sample.
    ///
    /// `((int8(b) + 128) % 64) - 32` is signed arithmetic; `(b & 63) - 32` is unsigned. They agree only
    /// because `+128` lands the value in `[0, 191]` before the modulo — for a byte whose `int8` reading
    /// is negative the two expressions take different routes, and a language whose `%` follows the
    /// sign of the dividend would disagree on half the range. A test that sampled ten bytes could pass
    /// on a decode that was wrong for the negative half.
    ///
    /// The verifier's two constants (`UNIFORM_NOISE_RANGE - 1`, `ZERO_POINT_TRANSLATION`) are private
    /// in `zk_pow`, so they are restated as literals here. What this proves is the identity between the
    /// two expressions; what pins the kernel itself is the device half of gate B, which runs the entry
    /// point and compares its output against `generate_uniform_random_matrix`.
    #[test]
    #[cfg(feature = "pearl")]
    fn the_dense_decode_is_the_verifiers_decode_for_every_byte_value() {
        for b in 0..=255u8 {
            let reference = ((b as i8) as i32 + 128) % 64 - 32;
            let verifier = ((b & 63) as i8) as i32 - 32;
            assert_eq!(reference, verifier, "byte {b} decodes two different ways");
        }
    }

    /// The reference's comment says `k1` "may be exactly `R` when `mul_hi(R-1, u) = R-1`". It cannot,
    /// for any `R`: `mul_hi(a, u) <= a - 1` whenever `u < 2^32`, because `a * u < a * 2^32`. So
    /// `1 + mul_hi(R-1, u) <= R - 1`, and XORing that with a `k0 < R` cannot reach `R`.
    ///
    /// That is why `zk_pow`'s `generate_permutation_matrix` has no modulo at all while the reference's
    /// kernel does. They are the same function, and the modulo is dead code that reads like a
    /// correctness guard — asserted here at the point where the bound is tight, over every rank the
    /// verifier accepts, so the claim is checked rather than inherited from the comment.
    #[test]
    #[cfg(feature = "pearl")]
    fn the_sparse_pair_needs_no_modulo_at_any_rank_the_verifier_accepts() {
        for rank in [32, 64, 128, 256, 512, 1024] {
            // Tight at the maximum: this is the only `u` that can come close to the claimed bound.
            assert_eq!(
                zk_pow::circuit::pearl_noise::mul_hi_u32((rank - 1) as u32, u32::MAX),
                (rank - 2) as u32,
                "the bound should be tight at rank {rank}"
            );

            for u in 0..65_536u32 {
                let k0 = u & (rank as u32 - 1);
                let raw = k0 ^ (1 + zk_pow::circuit::pearl_noise::mul_hi_u32((rank - 1) as u32, u));
                assert!(raw < rank as u32, "rank {rank}, u {u}: k1 = {raw}");
                assert_eq!(raw % rank as u32, raw, "the modulo changed the answer at rank {rank}");
            }
        }
    }

    /// End-to-end on real hardware: the image loads and every symbol resolves.
    ///
    /// Passes trivially where there is no NVIDIA device or the build had no CUDA toolkit. On a machine
    /// with a supported GPU this is the test that catches a stale image, a toolchain that cannot emit
    /// the arch, or a symbol that drifted out of the table.
    #[test]
    fn the_fatbin_symbols_resolve_on_the_device() {
        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        let Ok(cc) = ctx.compute_capability() else {
            return;
        };
        let (major, minor) = cc;
        if FATBIN.is_empty() {
            eprintln!("no fatbin embedded (no CUDA toolkit at build time); nothing to assert");
            return;
        }

        eprintln!(
            "resolving {} symbols on device 0 (compute capability {major}.{minor}): {}",
            ALL_SYMBOLS.len(),
            ALL_SYMBOLS.join(", ")
        );
        let kernels = FatbinKernels::load(&ctx, cc)
            .unwrap_or_else(|error| panic!("the fatbin failed to load: {error}"));
        let _ = kernels;
    }
}
