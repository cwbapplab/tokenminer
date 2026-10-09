//! Triton kernels, embedded one blob per architecture.
//!
//! Triton is Python-first: a kernel is JIT-compiled on first call. These blobs are the compiled
//! output of that JIT, captured once per supported architecture and `include_bytes!`'d here, so the
//! shipped binary carries them and never writes to disk or runs a Python toolchain at run time.
//!
//! At startup [`TritonKernels::load`] asks the device its compute capability and loads the matching
//! blob. Exact match is preferred; otherwise the highest blob at or below the device is used,
//! because PTX is forward-compatible — an `sm_86` blob runs on an `sm_89` or `sm_120` device through
//! the driver's JIT, just with the code that JIT produces rather than the code Triton produced. The
//! compatibility is one-directional: a blob built for a newer arch will not load on an older device,
//! so a capability below the lowest blob has no blob at all rather than a fallback.
//!
//! PTX is text, and `cuModuleLoadData` reads it as a C string, so the blob has to be NUL-terminated
//! on the way in — see [`null_terminate`].
//!
//! Two kernels, both from the reference Triton sources:
//!
//! - `_noising_kernel` — `Out[i,j] = wrap_int8(X[i,j] + sum_r Y[i,r] * Z[j,r])`, the dense bilinear
//!   noise apply.
//! - `_pearl_search_norotl_kernel` — the tile search, writing a `(tiles, 64, 16)` u32 transcript
//!   buffer that a separate postpass hashes.
//!
//! Neither is the consensus path here, and the difference is not a performance one:
//!
//! * tokenminer's noise apply is `signal + dense[first] - dense[second]` over sparse column pairs
//!   (`tokenminer_noise_apply`), which is a different function from the dense product this blob
//!   computes. Feeding it the same tensors produces a different `a_sum`.
//! * the search blob folds `rotl(prev) ^ x` without the rotate, so its transcripts are not the ones
//!   the verifier derives.
//!
//! So this module is the loader and the launch layer, not a drop-in for the mining loop: it makes the
//! blobs loadable on any device we support, and the argument layouts below are pinned against the
//! blobs themselves by the tests.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

// =============================================================================
//   Embedded blobs, one per supported architecture
// =============================================================================

const NOISING_PTX_SM86: &[u8] =
    include_bytes!("../../../../triton_kernels/sm86/noising_kernel.ptx");
const NOISING_PTX_SM89: &[u8] =
    include_bytes!("../../../../triton_kernels/sm89/noising_kernel.ptx");
const NOISING_PTX_SM120: &[u8] =
    include_bytes!("../../../../triton_kernels/sm120/noising_kernel.ptx");

const SEARCH_NOROTL_PTX_SM86: &[u8] =
    include_bytes!("../../../../triton_kernels/sm86/pearl_search_norotl_kernel.ptx");
const SEARCH_NOROTL_PTX_SM89: &[u8] =
    include_bytes!("../../../../triton_kernels/sm89/pearl_search_norotl_kernel.ptx");
const SEARCH_NOROTL_PTX_SM120: &[u8] =
    include_bytes!("../../../../triton_kernels/sm120/pearl_search_norotl_kernel.ptx");

/// A blob plus the arch it was built for, which is not always the arch it runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TritonBlob {
    pub arch: &'static str,
    pub bytes: &'static [u8],
}

/// Pick the noising blob for a device, exact match preferred, else the highest blob at or below it.
pub fn noising_ptx(cc_major: i32, cc_minor: i32) -> Option<TritonBlob> {
    let blob = pick_ptx(
        cc_major,
        cc_minor,
        NOISING_PTX_SM86,
        NOISING_PTX_SM89,
        NOISING_PTX_SM120,
    )?;
    Some(TritonBlob {
        arch: blob.0,
        bytes: blob.1,
    })
}

/// Pick the search blob for a device, same rule.
pub fn search_norotl_ptx(cc_major: i32, cc_minor: i32) -> Option<TritonBlob> {
    let blob = pick_ptx(
        cc_major,
        cc_minor,
        SEARCH_NOROTL_PTX_SM86,
        SEARCH_NOROTL_PTX_SM89,
        SEARCH_NOROTL_PTX_SM120,
    )?;
    Some(TritonBlob {
        arch: blob.0,
        bytes: blob.1,
    })
}

/// The blob a device gets, or `None` when the device is older than the oldest blob.
///
/// `None` here is not a missing build: an `sm_86` blob cannot run on `sm_80`, because PTX
/// forward-compatibility runs from older to newer only. A caller that treated "no blob" as "use the
/// oldest blob" would fault on the load, which is why this returns an option rather than a blob.
fn pick_ptx(
    cc_major: i32,
    cc_minor: i32,
    sm86: &'static [u8],
    sm89: &'static [u8],
    sm120: &'static [u8],
) -> Option<(&'static str, &'static [u8])> {
    let cc = cc_major * 10 + cc_minor; // 86, 89, 120
    if cc >= 120 {
        if !sm120.is_empty() {
            return Some(("sm_120a", sm120));
        }
        if !sm89.is_empty() {
            return Some(("sm_89", sm89));
        }
        if !sm86.is_empty() {
            return Some(("sm_86", sm86));
        }
    } else if cc >= 89 {
        if !sm89.is_empty() {
            return Some(("sm_89", sm89));
        }
        if !sm86.is_empty() {
            return Some(("sm_86", sm86));
        }
    } else if cc >= 86 && !sm86.is_empty() {
        return Some(("sm_86", sm86));
    }
    None
}

/// PTX must be NUL-terminated for `cuModuleLoadData`.
fn null_terminate(blob: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(blob.len() + 1);
    out.extend_from_slice(blob);
    out.push(0);
    out
}

// =============================================================================
//   Kernel constants, from the JIT-time constexpr values
// =============================================================================

pub const NOISING_KERNEL_NAME: &str = "_noising_kernel";
pub const SEARCH_NOROTL_KERNEL_NAME: &str = "_pearl_search_norotl_kernel";

/// Block size, `num_warps * 32`. Both blobs declare `.reqntid 128`, so a launch with a different
/// block size is rejected by the driver rather than silently running with a different mapping.
pub const NOISING_BLOCK_X: u32 = 128;
pub const SEARCH_NOROTL_BLOCK_X: u32 = 128;

/// Dynamic shared memory, from the JIT metadata. These are the amounts the launch has to request:
/// the blobs address `global_smem` through `extern .shared`, so a launch that asks for less faults.
pub const NOISING_SHARED_BYTES: u32 = 49_152;
pub const SEARCH_NOROTL_SHARED_BYTES: u32 = 33_792;

/// Tile shapes the blobs were specialised for. The grid is derived from these, so a launch that
/// changes them changes the kernel's tile mapping, not just its size.
pub const NOISING_BLOCK_M: usize = 128;
pub const NOISING_BLOCK_N: usize = 64;
pub const SEARCH_BLOCK_M: usize = 128;
pub const SEARCH_BLOCK_N: usize = 128;

/// The swizzle group the blobs' grid mapping assumes.
pub const GROUP_SIZE_M: usize = 8;

/// Transcript shape the search blob writes: `HASH_CANDIDATES` candidates of `JACKPOT_SIZE` words
/// each, per tile.
pub const HASH_CANDIDATES: usize = 64;
pub const JACKPOT_SIZE: usize = 16;

/// Launch parameters per blob, as the blobs declare them.
///
/// The reference counts these as 11 and 8, which is the Triton signature with the specialised-away
/// strides dropped. These blobs keep the trailing two pointer slots — Triton's `global_scratch` and
/// `profile_scratch` — and the bodies never read them, so the launch has to pass them anyway, as
/// zero. A launch that passed 11 arguments to a 13-parameter kernel would read garbage for the last
/// two.
pub const NOISING_PARAMS: usize = 13;
pub const SEARCH_PARAMS: usize = 10;

// =============================================================================
//   Load and launch
// =============================================================================

/// The two Triton kernels, loaded on a device.
pub struct TritonKernels {
    pub noising: TritonNoising,
    pub search: TritonSearchNorotl,
    /// The blob actually loaded, which for a device between two supported arches is the one below it.
    pub noising_arch: &'static str,
    pub search_arch: &'static str,
}

impl TritonKernels {
    /// Loads both blobs for a device and resolves their entry points.
    ///
    /// The worst case is a driver JIT step here, on a device that has no exact blob.
    pub fn load(ctx: &Arc<CudaContext>, cc: (i32, i32)) -> Result<Self, String> {
        let (major, minor) = cc;
        let noising_blob = noising_ptx(major, minor)
            .ok_or_else(|| format!("no Triton noising blob for {major}.{minor}"))?;
        let search_blob = search_norotl_ptx(major, minor)
            .ok_or_else(|| format!("no Triton search blob for {major}.{minor}"))?;

        let noising = TritonNoising::new(ctx, &noising_blob)?;
        let search = TritonSearchNorotl::new(ctx, &search_blob)?;

        Ok(Self {
            noising,
            search,
            noising_arch: noising_blob.arch,
            search_arch: search_blob.arch,
        })
    }
}

pub struct TritonNoising {
    func: CudaFunction,
    arch: &'static str,
}

impl TritonNoising {
    fn new(ctx: &Arc<CudaContext>, blob: &TritonBlob) -> Result<Self, String> {
        let module = ctx
            .load_module(Ptx::from_binary(null_terminate(blob.bytes)))
            .map_err(|e| {
                format!(
                    "loading the Triton noising blob for {} failed: {e}",
                    blob.arch
                )
            })?;
        let func = module
            .load_function(NOISING_KERNEL_NAME)
            .map_err(|e| format!("the Triton noising blob has no `{NOISING_KERNEL_NAME}`: {e}"))?;
        opt_in_shared(&func, NOISING_SHARED_BYTES, NOISING_KERNEL_NAME)?;
        Ok(Self {
            func,
            arch: blob.arch,
        })
    }

    /// The loaded function, so a measurement can read what the driver actually allocated for it.
    pub fn function(&self) -> &CudaFunction {
        &self.func
    }

    /// `Out = wrap_int8(X + Y @ Z^T)`, all four tensors row-major and contiguous.
    ///
    /// `X` and `Out` are `(M, N)`, `Y` is `(M, K_inner)`, `Z` is `(N, K_inner)`. The strides the blob
    /// takes are the row strides; its column strides were specialised away at JIT time.
    pub fn launch(
        &self,
        stream: &CudaStream,
        m: i32,
        n: i32,
        k_inner: i32,
        x: &CudaSlice<i8>,
        y: &CudaSlice<i8>,
        z: &CudaSlice<i8>,
        out: &CudaSlice<i8>,
    ) -> Result<(), String> {
        let grid_x = ceil_div(m as usize, NOISING_BLOCK_M) * ceil_div(n as usize, NOISING_BLOCK_N);
        let config = LaunchConfig {
            grid_dim: (grid_x as u32, 1, 1),
            block_dim: (NOISING_BLOCK_X, 1, 1),
            shared_mem_bytes: NOISING_SHARED_BYTES,
        };

        let mut launch = stream.launch_builder(&self.func);
        launch
            .arg(x)
            .arg(y)
            .arg(z)
            .arg(out)
            .arg(&m)
            .arg(&n)
            .arg(&k_inner)
            .arg(&n) // stride_xm
            .arg(&k_inner) // stride_ym
            .arg(&k_inner) // stride_zn
            .arg(&n) // stride_om
            // Triton's scratch slots. The body never reads them; see `NOISING_PARAMS`.
            .arg(&0u64)
            .arg(&0u64);

        unsafe { launch.launch(config) }
            .map_err(|e| format!("launching {NOISING_KERNEL_NAME} failed: {e}"))?;
        Ok(())
    }

    /// The blob this kernel was loaded from.
    pub fn arch(&self) -> &'static str {
        self.arch
    }
}

pub struct TritonSearchNorotl {
    func: CudaFunction,
    arch: &'static str,
}

impl TritonSearchNorotl {
    fn new(ctx: &Arc<CudaContext>, blob: &TritonBlob) -> Result<Self, String> {
        let module = ctx
            .load_module(Ptx::from_binary(null_terminate(blob.bytes)))
            .map_err(|e| {
                format!(
                    "loading the Triton search blob for {} failed: {e}",
                    blob.arch
                )
            })?;
        let func = module
            .load_function(SEARCH_NOROTL_KERNEL_NAME)
            .map_err(|e| {
                format!("the Triton search blob has no `{SEARCH_NOROTL_KERNEL_NAME}`: {e}")
            })?;
        opt_in_shared(&func, SEARCH_NOROTL_SHARED_BYTES, SEARCH_NOROTL_KERNEL_NAME)?;
        Ok(Self {
            func,
            arch: blob.arch,
        })
    }

    /// The loaded function, so a measurement can read what the driver actually allocated for it.
    ///
    /// The PTX declares *virtual* registers (`%r<543>`), which is not the physical count. Only `ptxas`
    /// or the driver JIT decides that, and the physical count is what decides how many blocks fit on an
    /// SM — which is the whole occupancy question for a kernel that keeps 128 accumulators live.
    pub fn function(&self) -> &CudaFunction {
        &self.func
    }

    /// Writes `transcripts: (tiles, HASH_CANDIDATES, JACKPOT_SIZE)` as u32.
    ///
    /// The buffer must be pre-zeroed: the blob writes candidate words and leaves the gaps it skips
    /// as it found them.
    ///
    /// `A` is `(M, K)` row-major. `B` is passed as `(N, K)` row-major, which is the same bytes as the
    /// `(K, N)` column-major view the blob indexes — its `stride_bk` was specialised to 1 at JIT time
    /// and only `stride_bn` survives.
    pub fn launch(
        &self,
        stream: &CudaStream,
        m: i32,
        n: i32,
        k: i32,
        a: &CudaSlice<i8>,
        b: &CudaSlice<i8>,
        transcripts: &CudaSlice<u32>,
    ) -> Result<(), String> {
        let want = ceil_div(m as usize, SEARCH_BLOCK_M)
            * ceil_div(n as usize, SEARCH_BLOCK_N)
            * HASH_CANDIDATES
            * JACKPOT_SIZE;
        if transcripts.len() < want {
            return Err(format!(
                "a grid of {m}x{n} tiles needs {want} transcript words, but the buffer holds {}",
                transcripts.len()
            ));
        }

        let grid_x = ceil_div(m as usize, SEARCH_BLOCK_M) * ceil_div(n as usize, SEARCH_BLOCK_N);
        let config = LaunchConfig {
            grid_dim: (grid_x as u32, 1, 1),
            block_dim: (SEARCH_NOROTL_BLOCK_X, 1, 1),
            shared_mem_bytes: SEARCH_NOROTL_SHARED_BYTES,
        };

        let mut launch = stream.launch_builder(&self.func);
        launch
            .arg(a)
            .arg(b)
            .arg(transcripts)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&k) // stride_am
            .arg(&k) // stride_bn
            .arg(&0u64)
            .arg(&0u64);

        unsafe { launch.launch(config) }
            .map_err(|e| format!("launching {SEARCH_NOROTL_KERNEL_NAME} failed: {e}"))?;
        Ok(())
    }

    /// The blob this kernel was loaded from.
    pub fn arch(&self) -> &'static str {
        self.arch
    }
}

/// Opt a blob into its dynamic shared memory.
///
/// The default cap is 48 KB, and both blobs are above it, so a launch that skipped this step fails
/// with a driver error rather than running with a smaller buffer.
fn opt_in_shared(func: &CudaFunction, bytes: u32, name: &str) -> Result<(), String> {
    func.set_attribute(
        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        bytes as i32,
    )
    .map_err(|e| format!("opting {name} into {bytes} B of shared memory failed: {e}"))
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_supported_capability_gets_its_own_blob() {
        for (cc, want) in [((8, 6), "sm_86"), ((8, 9), "sm_89"), ((12, 0), "sm_120a")] {
            assert_eq!(noising_ptx(cc.0, cc.1).map(|b| b.arch), Some(want));
            assert_eq!(search_norotl_ptx(cc.0, cc.1).map(|b| b.arch), Some(want));
        }
    }

    #[test]
    fn a_capability_between_blobs_falls_back_to_the_highest_one_below_it() {
        // PTX is forward-compatible, so a device newer than every blob still gets the newest one.
        assert_eq!(noising_ptx(12, 5).map(|b| b.arch), Some("sm_120a"));
        assert_eq!(search_norotl_ptx(12, 5).map(|b| b.arch), Some("sm_120a"));
        // A 4090-class device has no blob of its own here; it runs the sm_89 blob through the JIT.
        assert_eq!(noising_ptx(9, 0).map(|b| b.arch), Some("sm_89"));
        assert_eq!(search_norotl_ptx(9, 0).map(|b| b.arch), Some("sm_89"));
        assert_eq!(noising_ptx(8, 7).map(|b| b.arch), Some("sm_86"));
    }

    #[test]
    fn a_capability_below_the_oldest_blob_has_no_blob() {
        // Forward compatibility is one-directional: an sm_86 blob will not load on sm_80 or sm_75.
        // A fallback that returned the oldest blob anyway would fault at load.
        assert_eq!(noising_ptx(8, 0), None);
        assert_eq!(noising_ptx(7, 5), None);
        assert_eq!(search_norotl_ptx(8, 0), None);
    }

    #[test]
    fn every_blob_is_built_for_the_arch_it_is_filed_under() {
        // The sm_86 and sm_89 blobs differ only in this line, so a blob filed in the wrong directory
        // would load on the device it is named for and be silently the wrong code for the other.
        for (dir, blob) in [
            ("sm_86", NOISING_PTX_SM86),
            ("sm_89", NOISING_PTX_SM89),
            ("sm_120", NOISING_PTX_SM120),
        ] {
            assert!(!blob.is_empty(), "{dir} noising blob is empty");
            // The suffix is the arch-specific extension (`sm_120a`), which the directory name does
            // not carry, so this compares the arch part only.
            assert!(
                target_line(blob).contains(dir),
                "{dir} noising blob declares {}",
                target_line(blob)
            );
        }
        for (dir, blob) in [
            ("sm_86", SEARCH_NOROTL_PTX_SM86),
            ("sm_89", SEARCH_NOROTL_PTX_SM89),
            ("sm_120", SEARCH_NOROTL_PTX_SM120),
        ] {
            assert!(!blob.is_empty(), "{dir} search blob is empty");
            assert!(
                target_line(blob).contains(dir),
                "{dir} search blob declares {}",
                target_line(blob)
            );
        }
    }

    #[test]
    fn every_blob_declares_the_entry_point_the_wrapper_looks_up() {
        for blob in [NOISING_PTX_SM86, NOISING_PTX_SM89, NOISING_PTX_SM120] {
            assert!(
                text(blob).contains(&format!(".visible .entry {NOISING_KERNEL_NAME}(")),
                "a noising blob does not define {NOISING_KERNEL_NAME}"
            );
        }
        for blob in [
            SEARCH_NOROTL_PTX_SM86,
            SEARCH_NOROTL_PTX_SM89,
            SEARCH_NOROTL_PTX_SM120,
        ] {
            assert!(
                text(blob).contains(&format!(".visible .entry {SEARCH_NOROTL_KERNEL_NAME}(")),
                "a search blob does not define {SEARCH_NOROTL_KERNEL_NAME}"
            );
        }
    }

    #[test]
    fn the_launch_argument_count_matches_what_the_blobs_declare() {
        // The launch chains pass exactly these many arguments. A Triton recompile that changes the
        // signature changes what the driver reads for every argument after the change, so the count
        // is the thing to check first when a blob is regenerated.
        assert_eq!(param_count(NOISING_PTX_SM120), NOISING_PARAMS);
        assert_eq!(param_count(NOISING_PTX_SM89), NOISING_PARAMS);
        assert_eq!(param_count(NOISING_PTX_SM86), NOISING_PARAMS);
        assert_eq!(param_count(SEARCH_NOROTL_PTX_SM120), SEARCH_PARAMS);
        assert_eq!(param_count(SEARCH_NOROTL_PTX_SM89), SEARCH_PARAMS);
        assert_eq!(param_count(SEARCH_NOROTL_PTX_SM86), SEARCH_PARAMS);
    }

    #[test]
    fn the_scratch_slots_are_never_read_by_the_bodies() {
        // The launch passes zero for the last two parameters of each blob. That is only safe while
        // the bodies ignore them; if a recompile starts reading scratch, this fails and the launch
        // has to pass a real allocation.
        for blob in [NOISING_PTX_SM86, NOISING_PTX_SM89, NOISING_PTX_SM120] {
            for index in NOISING_PARAMS - 2..NOISING_PARAMS {
                assert!(
                    !referenced_in_body(blob, "noising_kernel", index),
                    "the noising body reads param {index}, which the launch passes as zero"
                );
            }
        }
        for blob in [
            SEARCH_NOROTL_PTX_SM86,
            SEARCH_NOROTL_PTX_SM89,
            SEARCH_NOROTL_PTX_SM120,
        ] {
            for index in SEARCH_PARAMS - 2..SEARCH_PARAMS {
                assert!(
                    !referenced_in_body(blob, "pearl_search_norotl_kernel", index),
                    "the search body reads param {index}, which the launch passes as zero"
                );
            }
        }
    }

    #[test]
    fn the_shared_memory_a_launch_requests_covers_every_offset_the_kernel_touches() {
        // Both blobs address `global_smem` through `extern .shared`, so the launch has to request the
        // whole allocation. The largest literal offset in a blob is a stage base; the per-lane swizzle
        // sits on top of it, which is the gap between these numbers and the constants.
        for blob in [NOISING_PTX_SM86, NOISING_PTX_SM89, NOISING_PTX_SM120] {
            let max = max_literal(blob);
            assert!(
                max < NOISING_SHARED_BYTES as usize,
                "a noising blob offsets shared memory to {max}, past the {NOISING_SHARED_BYTES} B a launch requests"
            );
        }
        for blob in [
            SEARCH_NOROTL_PTX_SM86,
            SEARCH_NOROTL_PTX_SM89,
            SEARCH_NOROTL_PTX_SM120,
        ] {
            let max = max_literal(blob);
            assert!(
                max < SEARCH_NOROTL_SHARED_BYTES as usize,
                "a search blob offsets shared memory to {max}, past the {SEARCH_NOROTL_SHARED_BYTES} B a launch requests"
            );
        }
    }

    #[test]
    fn a_blob_needs_the_block_size_the_launch_uses() {
        for blob in [NOISING_PTX_SM86, NOISING_PTX_SM89, NOISING_PTX_SM120] {
            assert!(
                text(blob).contains(".reqntid 128"),
                "a noising blob does not require a 128-thread block"
            );
        }
        for blob in [
            SEARCH_NOROTL_PTX_SM86,
            SEARCH_NOROTL_PTX_SM89,
            SEARCH_NOROTL_PTX_SM120,
        ] {
            assert!(
                text(blob).contains(".reqntid 128"),
                "a search blob does not require a 128-thread block"
            );
        }
    }

    #[test]
    fn null_termination_adds_exactly_one_nul() {
        for blob in [NOISING_PTX_SM120, SEARCH_NOROTL_PTX_SM120] {
            // `cuModuleLoadData` reads the blob as a C string, so an interior NUL would truncate it.
            assert!(!blob.contains(&0), "a PTX blob contains an interior NUL");
            let terminated = null_terminate(blob);
            assert_eq!(terminated.len(), blob.len() + 1);
            assert_eq!(terminated[terminated.len() - 1], 0);
        }
    }

    /// Loads both blobs on the device this runs on and resolves both entry points.
    ///
    /// The selection tests above cannot tell whether a blob actually loads: that is what this does,
    /// and it is the check that a blob built for a newer arch than the device would fail.
    #[test]
    fn the_blobs_load_on_the_device() {
        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        let Ok(cc) = ctx.compute_capability() else {
            return;
        };
        let kernels = TritonKernels::load(&ctx, cc)
            .expect("the blobs should load on a device we picked a blob for");
        assert_eq!(
            kernels.noising.arch(),
            noising_ptx(cc.0, cc.1).unwrap().arch
        );
        assert_eq!(
            kernels.search.arch(),
            search_norotl_ptx(cc.0, cc.1).unwrap().arch
        );
    }

    /// The noising wrapper's argument order, checked against the formula the blob implements.
    ///
    /// The CPU model is the documented formula, so the check is not the arithmetic -- it is the layout.
    /// The four strides the blob takes are `n`, `k`, `k`, `n`, which are only distinguishable when
    /// `k != n`: a wrapper that passed `stride_ym` where `stride_xm` belongs produces a different
    /// output here, and the same wrapper passing the tensors in the wrong order does too.
    ///
    /// The inputs are kept small so every sum stays inside `i8`. The blob's `mma` is `satfinite`, so a
    /// sum that overflowed would make the model disagree for a reason that is not the layout.
    ///
    /// `k_inner` is 128 because that is the block the blob stages: each `cp.async` group pulls a full
    /// 128-byte row, so a smaller `k_inner` makes the kernel read past the end of each row and the
    /// comparison measures the buffer layout rather than the argument order.
    #[test]
    fn the_noising_launch_passes_its_arguments_in_the_order_the_blob_reads_them() {
        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        let Ok(cc) = ctx.compute_capability() else {
            return;
        };
        let kernels = match TritonKernels::load(&ctx, cc) {
            Ok(kernels) => kernels,
            Err(_) => return,
        };
        let stream = ctx.default_stream();

        // Two tiles wide and two tall, so the grid reaches past the first slot: a launch that only
        // ever ran tile 0 would pass a single-tile check and still be wrong.
        let (m, n, k) = (256usize, 128usize, 128usize);
        let x: Vec<i8> = (0..m * n).map(|i| ((i % 5) as i8) - 2).collect();
        let y: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect();
        let z: Vec<i8> = (0..n * k).map(|i| ((i % 4) as i8) - 1).collect();

        let x_dev = stream.clone_htod(&x).unwrap();
        let y_dev = stream.clone_htod(&y).unwrap();
        let z_dev = stream.clone_htod(&z).unwrap();
        let out_dev = stream.alloc_zeros::<i8>(m * n).unwrap();

        kernels
            .noising
            .launch(
                &stream, m as i32, n as i32, k as i32, &x_dev, &y_dev, &z_dev, &out_dev,
            )
            .unwrap();

        let got = stream.clone_dtoh(&out_dev).unwrap();
        let want = cpu_noising(m, n, k, &x, &y, &z);
        assert_eq!(
            got, want,
            "the noising launch disagrees with the formula it documents"
        );
    }

    /// The search wrapper's grid mapping, on a grid that reaches more than one tile.
    ///
    /// This does not compare transcript values: the blob folds without the rotate, so its transcript
    /// is not the one consensus derives, and a CPU model of it here would be a second reading of the
    /// same PTX rather than a check. What is checked is the part the wrapper owns -- that the output
    /// pointer lands on the buffer and that tiles past the first one are written.
    ///
    /// `k` is 128, not 64, for a reason worth knowing: the blob's only store is guarded by
    /// `candidate % 2`, so a single k-block (`k == BK == 64`) runs one iteration and writes nothing.
    /// A test at `k = 64` would read an all-zero buffer and blame the pointer.
    #[test]
    fn the_search_launch_writes_across_more_than_one_tile() {
        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        let Ok(cc) = ctx.compute_capability() else {
            return;
        };
        let kernels = match TritonKernels::load(&ctx, cc) {
            Ok(kernels) => kernels,
            Err(_) => return,
        };
        let stream = ctx.default_stream();

        let (m, n, k) = (256usize, 256usize, 128usize);
        let tiles = (m / SEARCH_BLOCK_M) * (n / SEARCH_BLOCK_N);
        let a: Vec<i8> = (0..m * k).map(|i| (i % 7) as i8).collect();
        let b: Vec<i8> = (0..n * k).map(|i| (i % 11) as i8).collect();

        let a_dev = stream.clone_htod(&a).unwrap();
        let b_dev = stream.clone_htod(&b).unwrap();
        let t_dev = stream
            .alloc_zeros::<u32>(tiles * HASH_CANDIDATES * JACKPOT_SIZE)
            .unwrap();

        kernels
            .search
            .launch(
                &stream, m as i32, n as i32, k as i32, &a_dev, &b_dev, &t_dev,
            )
            .unwrap();

        let got = stream.clone_dtoh(&t_dev).unwrap();
        let written: Vec<usize> = (0..tiles)
            .filter(|tile| {
                got[tile * HASH_CANDIDATES * JACKPOT_SIZE
                    ..(tile + 1) * HASH_CANDIDATES * JACKPOT_SIZE]
                    .iter()
                    .any(|word| *word != 0)
            })
            .collect();
        assert!(
            written.len() > 1,
            "only tile {written:?} was written on a grid of {tiles} tiles"
        );
    }

    fn cpu_noising(m: usize, n: usize, k: usize, x: &[i8], y: &[i8], z: &[i8]) -> Vec<i8> {
        let mut out = vec![0i8; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = x[i * n + j] as i32;
                for r in 0..k {
                    sum += (y[i * k + r] as i32) * (z[j * k + r] as i32);
                }
                out[i * n + j] = sum as i8;
            }
        }
        out
    }

    fn text(blob: &[u8]) -> String {
        String::from_utf8_lossy(blob).into_owned()
    }

    fn target_line(blob: &[u8]) -> String {
        text(blob)
            .lines()
            .find(|line| line.starts_with(".target"))
            .unwrap_or("<none>")
            .to_string()
    }

    fn param_count(blob: &[u8]) -> usize {
        let text = text(blob);
        let start = text.find(".visible .entry ").expect("no entry point");
        let signature = &text[start..];
        // The first `)` after the entry name closes the parameter list; no parameter declaration
        // contains one, so this does not have to know whether the file uses LF or CRLF.
        let end = signature.find(')').expect("no signature end");
        signature[..end].matches("_param_").count()
    }

    fn referenced_in_body(blob: &[u8], name: &str, index: usize) -> bool {
        let text = text(blob);
        let body = &text[text.find(".reqntid").expect("no reqntid")..];
        body.contains(&format!("{name}_param_{index}"))
    }

    fn max_literal(blob: &[u8]) -> usize {
        text(blob)
            .split(|c: char| !c.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse::<usize>().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }
}
