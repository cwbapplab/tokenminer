//! CUDA backend: prebuilt cubins, loaded through the CUDA driver API.
//!
//! cudarc owns the driver plumbing (`dynamic-loading` means we never link against CUDA at build
//! time and need no toolkit at run time). We own architecture selection, because a cubin only loads
//! on the compute capability it was compiled for.

use std::sync::Arc;

use cudarc::driver::{
    sys::CUdevice_attribute, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
#[cfg(feature = "pearl")]
use primitive_types::U256;

use super::{cubins, DeviceInfo, GpuBackend};

#[cfg(feature = "pearl")]
use crate::miners::pearl_mining::{mining_configuration, small_mining, PearlMining};

/// Words in the Pearl jackpot, mirroring `JACKPOT_SIZE` in the consensus code.
#[cfg(feature = "pearl")]
pub const JACKPOT_WORDS: usize = 16;

/// The probe kernels are single-threaded by design, so pin the geometry rather than letting
/// `LaunchConfig::for_num_elems` hand us a 1024-thread block (whose threads would race on the
/// result, making the check nondeterministic).
const ONE_THREAD: LaunchConfig = LaunchConfig {
    grid_dim: (1, 1, 1),
    block_dim: (1, 1, 1),
    shared_mem_bytes: 0,
};

/// Threads per block for the BLAKE3 kernels: one thread per chunk or per tree node, each a handful
/// of independent compressions, so occupancy comes from the grid rather than from within a block.
const B3_BLOCK: u32 = 256;

/// A grid of `B3_BLOCK` threads per block with enough blocks for `items` work items.
///
/// Every one of these kernels grid-strides, so capping the grid is a bound on launch size only —
/// a matrix commits correctly whether or not it fits in one wave.
fn grid_for(items: usize) -> LaunchConfig {
    let blocks = items.div_ceil(B3_BLOCK as usize).min(65_535);

    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (B3_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// BLAKE3 constants, mirroring `blake3.cuh`. The IV is what an unkeyed hash is keyed with.
#[cfg(feature = "pearl")]
const B3_IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

#[cfg(feature = "pearl")]
const B3_CHUNK_LEN: usize = 1024;
#[cfg(feature = "pearl")]
const B3_WORDS: usize = 8;
#[cfg(feature = "pearl")]
const B3_KEYED_HASH: u32 = 16;

/// Bytes one noise hash yields: the reference draws its tensors 32 bytes at a time.
#[cfg(feature = "pearl")]
const NOISE_HASH_LEN: usize = 32;

/// The seed labels the noise tensors are hashed under, built the way `padded_seed_label` builds
/// them: the eight ASCII bytes, then zeroes to fill the 32-byte seed.
#[cfg(feature = "pearl")]
pub(crate) const SEED_LABEL_A: [u8; 32] = seed_label(*b"A_tensor");
#[cfg(feature = "pearl")]
pub(crate) const SEED_LABEL_B: [u8; 32] = seed_label(*b"B_tensor");

#[cfg(feature = "pearl")]
const fn seed_label(label: [u8; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < label.len() {
        out[i] = label[i];
        i += 1;
    }
    out
}

/// The four noise tensors as a check gets them back: A's dense rows, A's and B's sparse pairs, and
/// B's dense rows.
#[cfg(feature = "pearl")]
type NoiseTensors = (Vec<i8>, Vec<[u32; 2]>, Vec<[u32; 2]>, Vec<i8>);

/// Which contiguous run of rows a check wants out of each dense tensor: the first row and the count.
#[cfg(feature = "pearl")]
#[derive(Clone, Copy)]
struct NoiseWindow {
    a: (u32, usize),
    b: (u32, usize),
}

/// Thirty-two bytes as eight little-endian words, the form BLAKE3 chains on the device.
#[cfg(feature = "pearl")]
fn to_words(bytes: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
}

/// The `sm_` target a device runs, or `None` when we ship no image for it.
///
/// A cubin is architecture-specific: it will not load on a different compute capability, and there
/// is no PTX fallback yet, so this is an exact mapping rather than a range.
fn arch_for(compute_capability: (i32, i32)) -> Option<&'static str> {
    Some(match compute_capability {
        (7, 5) => "sm_75",
        (8, 6) | (8, 7) => "sm_86",
        (8, 9) => "sm_89",
        (9, 0) => "sm_90a",
        (12, 0) => "sm_120a",
        _ => return None,
    })
}

/// The value `__CUDA_ARCH__` holds for an arch, used to confirm which image the driver loaded.
fn arch_code(arch: &str) -> Option<u32> {
    Some(match arch {
        "sm_75" => 750,
        "sm_86" => 860,
        "sm_89" => 890,
        "sm_90a" => 900,
        "sm_120a" => 1200,
        _ => return None,
    })
}

pub struct CudaBackend {
    /// Kept alive because the module and stream are owned by the context's lifetime.
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    arch: &'static str,
    device: DeviceInfo,
}

impl CudaBackend {
    pub fn open(ordinal: usize) -> Result<Self, String> {
        if !cubins::any_embedded() {
            return Err(
                "this build has no CUDA cubins embedded (the build machine had no nvcc). Install \
                 the CUDA toolkit and rebuild."
                    .to_string(),
            );
        }

        let ctx = CudaContext::new(ordinal)
            .map_err(|e| format!("opening CUDA device {ordinal} failed: {e}"))?;

        let (major, minor) = ctx.compute_capability().map_err(|e| {
            format!("reading the compute capability of device {ordinal} failed: {e}")
        })?;

        let arch = arch_for((major, minor)).ok_or_else(|| {
            format!(
                "device {ordinal} reports compute capability {major}.{minor}, which has no embedded \
                 cubin (embedded: {})",
                cubins::embedded_arches().join(", ")
            )
        })?;

        let image = cubins::kernels_cubin(arch)
            .ok_or_else(|| format!("no embedded cubin for {arch}: it was not built"))?;

        let module = ctx
            .load_module(Ptx::from_binary(image.to_vec()))
            .map_err(|e| format!("loading the {arch} cubin failed: {e}"))?;

        let name = ctx.name().unwrap_or_else(|_| "NVIDIA GPU".to_string());
        let multiprocessors = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(0);

        let device = DeviceInfo {
            backend: "cuda".to_string(),
            name,
            compute_capability: format!("{major}.{minor}"),
            arch: arch.to_string(),
            multiprocessors,
        };

        Ok(Self {
            stream: ctx.default_stream(),
            _ctx: ctx,
            module,
            arch,
            device,
        })
    }

    fn function(&self, entry: &str) -> Result<CudaFunction, String> {
        self.module.load_function(entry).map_err(|e| {
            format!(
                "the {arch} cubin has no `{entry}` entry point: {e}",
                arch = self.arch
            )
        })
    }

    /// Device memory still free, in bytes.
    ///
    /// Asked before the search allocates anything rather than discovered by the first allocation
    /// failing: an out-of-memory part-way through a job leaves the miner wedged and the reason
    /// buried in a driver string, where refusing up front with a number is actionable.
    #[cfg(feature = "pearl")]
    pub fn free_bytes(&self) -> usize {
        self._ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0)
    }

    /// A zeroed `i8` buffer, for a matrix or a noise tensor.
    #[cfg(feature = "pearl")]
    pub fn alloc_i8(&self, len: usize) -> Result<CudaSlice<i8>, String> {
        self.stream
            .alloc_zeros::<i8>(len)
            .map_err(|e| format!("allocating {len} i8 failed: {e}"))
    }

    /// A zeroed `u32` buffer, for the sparse column pairs and the per-batch result arrays.
    #[cfg(feature = "pearl")]
    pub fn alloc_u32(&self, len: usize) -> Result<CudaSlice<u32>, String> {
        self.stream
            .alloc_zeros::<u32>(len)
            .map_err(|e| format!("allocating {len} u32 failed: {e}"))
    }

    /// Copies a device `i8` buffer back to the host.
    ///
    /// Only ever for the signal matrices, and only once a tile has won: at production dimensions
    /// these are half a gigabyte each, so this is the search's one real round trip.
    #[cfg(feature = "pearl")]
    pub fn read_i8(&self, slice: &CudaSlice<i8>) -> Result<Vec<i8>, String> {
        let mut host = vec![0i8; slice.len()];
        self.stream.memcpy_dtoh(slice, &mut host[..]).map_err(|e| {
            format!(
                "copying {} bytes back from the device failed: {e}",
                slice.len()
            )
        })?;

        Ok(host)
    }

    /// Runs `tokenminer_arch_probe` and checks that the image on the device is the one we built.
    fn check_arch(&self) -> Result<(), String> {
        let expected = arch_code(self.arch)
            .ok_or_else(|| format!("no expected __CUDA_ARCH__ for {}", self.arch))?;

        let probe = self.function("tokenminer_arch_probe")?;
        let mut stamp = self
            .stream
            .alloc_zeros::<u32>(2)
            .map_err(|e| format!("allocating device memory failed: {e}"))?;

        let mut launch = self.stream.launch_builder(&probe);
        launch.arg(&mut stamp);
        unsafe { launch.launch(ONE_THREAD) }
            .map_err(|e| format!("launching tokenminer_arch_probe failed: {e}"))?;

        let host = self
            .stream
            .clone_dtoh(&stamp)
            .map_err(|e| format!("reading the probe result back failed: {e}"))?;

        if host[0] != expected {
            return Err(format!(
                "the {arch} cubin reports __CUDA_ARCH__ {got} instead of {expected}: the wrong image \
                 was loaded",
                arch = self.arch,
                got = host[0]
            ));
        }
        if host[1] != 0 {
            return Err(format!(
                "the probe kernel wrote thread index {} where 0 was expected",
                host[1]
            ));
        }

        Ok(())
    }

    /// Runs `tokenminer_i8_dot` and compares against the same sum computed on the host.
    fn check_i8_dot(&self) -> Result<(), String> {
        let k = 4096usize;
        let a: Vec<i8> = (0..k).map(|i| ((i % 127) as i8) - 63).collect();
        let b: Vec<i8> = (0..k).map(|i| (((i * 7) % 113) as i8) - 56).collect();
        let expected: i32 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| i32::from(x) * i32::from(y))
            .sum();

        let a_dev = self
            .stream
            .clone_htod(&a)
            .map_err(|e| format!("copying the left operand failed: {e}"))?;
        let b_dev = self
            .stream
            .clone_htod(&b)
            .map_err(|e| format!("copying the right operand failed: {e}"))?;
        let mut out = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("allocating the result failed: {e}"))?;

        let kernel = self.function("tokenminer_i8_dot")?;
        let k_arg = k as i32;

        let mut launch = self.stream.launch_builder(&kernel);
        launch.arg(&a_dev).arg(&b_dev).arg(&mut out).arg(&k_arg);
        unsafe { launch.launch(ONE_THREAD) }
            .map_err(|e| format!("launching tokenminer_i8_dot failed: {e}"))?;

        let host = self
            .stream
            .clone_dtoh(&out)
            .map_err(|e| format!("reading the dot product back failed: {e}"))?;

        if host[0] != expected {
            return Err(format!(
                "the INT8 dot product returned {} but the host computed {expected}",
                host[0]
            ));
        }

        Ok(())
    }

    /// Runs `tokenminer_i8_gemm` and compares every output against a CPU INT8 matmul.
    ///
    /// This is the check that matters for mining: it exercises the tensor-core path that the
    /// mining kernel is built on, and it fails loudly on the first disagreement rather than
    /// reporting a match.
    fn check_i8_gemm(&self) -> Result<(), String> {
        const M: usize = 64;
        const N: usize = 64;
        const K: usize = 512;
        const TILE: usize = 16;

        // Magnitudes chosen so the 32-bit accumulators cannot overflow.
        let a: Vec<i8> = (0..M * K).map(|i| (((i * 31) % 127) as i8) - 63).collect();
        let b: Vec<i8> = (0..K * N).map(|i| (((i * 17) % 113) as i8) - 56).collect();

        let expected = cpu_i8_gemm(&a, &b, M, N, K);

        let a_dev = self
            .stream
            .clone_htod(&a)
            .map_err(|e| format!("copying A failed: {e}"))?;
        let b_dev = self
            .stream
            .clone_htod(&b)
            .map_err(|e| format!("copying B failed: {e}"))?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<i32>(M * N)
            .map_err(|e| format!("allocating C failed: {e}"))?;

        let kernel = self.function("tokenminer_i8_gemm")?;
        let (m, n, k) = (M as i32, N as i32, K as i32);

        // One block per 16x16 output tile; `wmma` needs at least a full warp.
        let config = LaunchConfig {
            grid_dim: ((N / TILE) as u32, (M / TILE) as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&m)
            .arg(&n)
            .arg(&k);
        unsafe { launch.launch(config) }
            .map_err(|e| format!("launching tokenminer_i8_gemm failed: {e}"))?;

        let got = self
            .stream
            .clone_dtoh(&c_dev)
            .map_err(|e| format!("reading the GEMM result back failed: {e}"))?;

        if let Some(index) = got
            .iter()
            .zip(&expected)
            .position(|(got, want)| got != want)
        {
            return Err(format!(
                "the INT8 GEMM disagrees at row {}, column {}: device {} but the host computed {}",
                index / N,
                index % N,
                got[index],
                expected[index]
            ));
        }

        Ok(())
    }

    /// Computes the canonical Pearl jackpot for one tile pair on the device.
    ///
    /// `secret_a` is `h * k` and `secret_b` is `w * k`, both row-major `i8`, with `noise_a` /
    /// `noise_b` shaped to match. This has to reproduce
    /// [`zk_pow::circuit::chip::compute_jackpot`] word for word, which is the test that guards it.
    #[cfg(feature = "pearl")]
    #[allow(clippy::too_many_arguments)]
    pub fn jackpot_tile(
        &self,
        secret_a: &[i8],
        noise_a: &[i8],
        secret_b: &[i8],
        noise_b: &[i8],
        k: usize,
        rank: usize,
        h: usize,
        w: usize,
    ) -> Result<[u32; JACKPOT_WORDS], String> {
        let threads = validate_tile(
            secret_a.len(),
            noise_a.len(),
            secret_b.len(),
            noise_b.len(),
            k,
            rank,
            h,
            w,
        )?;

        let secret_a_dev = self
            .stream
            .clone_htod(secret_a)
            .map_err(|e| format!("copying secret_a failed: {e}"))?;
        let noise_a_dev = self
            .stream
            .clone_htod(noise_a)
            .map_err(|e| format!("copying noise_a failed: {e}"))?;
        let secret_b_dev = self
            .stream
            .clone_htod(secret_b)
            .map_err(|e| format!("copying secret_b failed: {e}"))?;
        let noise_b_dev = self
            .stream
            .clone_htod(noise_b)
            .map_err(|e| format!("copying noise_b failed: {e}"))?;
        let mut jackpot_dev = self
            .stream
            .alloc_zeros::<u32>(JACKPOT_WORDS)
            .map_err(|e| format!("allocating the jackpot failed: {e}"))?;

        let kernel = self.function("tokenminer_jackpot")?;
        let (k_arg, rank_arg, h_arg, w_arg) = (k as i32, rank as i32, h as i32, w as i32);

        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads, 1, 1),
            // The h*w cell sums plus one partial per thread.
            shared_mem_bytes: ((h * w + threads as usize) * std::mem::size_of::<u32>()) as u32,
        };

        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(&secret_a_dev)
            .arg(&noise_a_dev)
            .arg(&secret_b_dev)
            .arg(&noise_b_dev)
            .arg(&mut jackpot_dev)
            .arg(&k_arg)
            .arg(&rank_arg)
            .arg(&h_arg)
            .arg(&w_arg);
        unsafe { launch.launch(config) }
            .map_err(|e| format!("launching tokenminer_jackpot failed: {e}"))?;

        let host = self
            .stream
            .clone_dtoh(&jackpot_dev)
            .map_err(|e| format!("reading the jackpot back failed: {e}"))?;

        let mut jackpot = [0u32; JACKPOT_WORDS];
        jackpot.copy_from_slice(&host[..JACKPOT_WORDS]);

        Ok(jackpot)
    }

    /// The jackpot transcript from the summed operands, on the tensor cores.
    ///
    /// `a_sum` is `h * k` and `b_sum` is `w * k`, both row-major `i8`, each entry already
    /// `signal + noise`. That is the shape consensus multiplies and the shape the GEMM can take
    /// directly: `signal` is in [-64, 63] and the noise is a difference of two entries of a
    /// [-32, 31] dense row, so the sums stay in [-127, 126] and are still valid INT8 operands.
    ///
    /// `h` and `w` must be multiples of 16 — the fragment shape `wmma` gives us. A job whose tile
    /// pattern is not (the current default is 2x64) goes through [`Self::jackpot_tile`] instead;
    /// both are held against `compute_jackpot`.
    #[cfg(feature = "pearl")]
    pub fn jackpot_fold(
        &self,
        a_sum: &[i8],
        b_sum: &[i8],
        k: usize,
        rank: usize,
        h: usize,
        w: usize,
    ) -> Result<[u32; JACKPOT_WORDS], String> {
        if h == 0 || w == 0 || !h.is_multiple_of(16) || !w.is_multiple_of(16) {
            return Err(format!(
                "the fold GEMM needs a tile whose h and w are multiples of 16, got {h}x{w}"
            ));
        }
        if a_sum.len() != h * k || b_sum.len() != w * k {
            return Err(format!(
                "the fold GEMM wants {h}x{k} and {w}x{k} operands, got {} and {}",
                a_sum.len(),
                b_sum.len()
            ));
        }
        if rank == 0 || !k.is_multiple_of(rank) {
            return Err(format!("rank {rank} does not divide k {k}"));
        }

        let a_dev = self
            .stream
            .clone_htod(a_sum)
            .map_err(|e| format!("copying a_sum failed: {e}"))?;
        let b_dev = self
            .stream
            .clone_htod(b_sum)
            .map_err(|e| format!("copying b_sum failed: {e}"))?;
        let mut jackpot_dev = self
            .stream
            .alloc_zeros::<u32>(JACKPOT_WORDS)
            .map_err(|e| format!("allocating the jackpot failed: {e}"))?;

        let kernel = self.function("tokenminer_jackpot_fold")?;
        let (k_arg, rank_arg, h_arg, w_arg) = (k as i32, rank as i32, h as i32, w as i32);

        // One warp: `wmma` needs a full one, and the whole block works on a single 16x16 tile.
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut jackpot_dev)
            .arg(&k_arg)
            .arg(&rank_arg)
            .arg(&h_arg)
            .arg(&w_arg);
        unsafe { launch.launch(config) }
            .map_err(|e| format!("launching tokenminer_jackpot_fold failed: {e}"))?;

        let host = self
            .stream
            .clone_dtoh(&jackpot_dev)
            .map_err(|e| format!("reading the jackpot back failed: {e}"))?;

        let mut jackpot = [0u32; JACKPOT_WORDS];
        jackpot.copy_from_slice(&host[..JACKPOT_WORDS]);

        Ok(jackpot)
    }

    /// Blocks until everything already launched on this backend's stream has finished.
    ///
    /// Every kernel launch here is asynchronous, so a host-side clock around one measures how long
    /// the launch took to enqueue rather than how long the device worked. That is not a rounding
    /// error: the fill and the noise-apply at production dimensions move a gigabyte between them,
    /// and timing them without this reports tens of microseconds for the pair.
    ///
    /// An event rather than `cuStreamSynchronize` because cudarc 0.19 does not expose one. The
    /// event is created per call, which costs a few microseconds and only matters to a caller
    /// doing this in a loop, which nothing here does.
    ///
    /// Test-only, because production never needs it: the one place a host wait is unavoidable is
    /// the read-back after a batch, and that is a `memcpy_dtoh`, which is already a barrier.
    #[cfg(test)]
    pub fn synchronize(&self) -> Result<(), String> {
        let event = self
            ._ctx
            .new_event(None)
            .map_err(|e| format!("creating a synchronisation event failed: {e}"))?;
        event
            .record(&self.stream)
            .map_err(|e| format!("recording a synchronisation event failed: {e}"))?;
        event
            .synchronize()
            .map_err(|e| format!("waiting on the device failed: {e}"))
    }

    /// The coordinates of the first region in `0..regions` that claimed, if any.
    ///
    /// Reads the flag array first and the coordinate array only when something won, so the common
    /// case — a batch that found nothing — costs one small transfer rather than two. The region
    /// index is not reported: a batch is a contiguous run of tiles, and the caller knows which one it
    /// launched, so the row and column are the whole answer.
    #[cfg(feature = "pearl")]
    pub fn first_winner(
        &self,
        found: &CudaSlice<u32>,
        coord: &CudaSlice<u32>,
        regions: usize,
    ) -> Result<Option<(u32, u32)>, String> {
        if found.len() < regions || coord.len() < regions * 2 {
            return Err(format!(
                "reading a winner needs {regions} flag words and {} coordinate words, but the \
                 buffers hold {} and {}",
                regions * 2,
                found.len(),
                coord.len()
            ));
        }

        // The whole arrays: `CudaSlice` cannot be sliced in this version of cudarc, so a sub-range
        // read is not available and the destination has to be at least as long as the source. The
        // buffers are sized for the largest batch the miner will ever run, so they are usually
        // longer than this one — a few hundred bytes either way, read on the path where a share is
        // about to be assembled anyway. Only the first `regions` entries are this batch's.
        let mut flags = vec![0u32; found.len()];
        self.stream
            .memcpy_dtoh(found, &mut flags[..])
            .map_err(|e| format!("reading the found flags back failed: {e}"))?;

        let Some(region) = flags[..regions].iter().position(|f| *f != 0) else {
            return Ok(None);
        };

        let mut coords = vec![0u32; coord.len()];
        self.stream
            .memcpy_dtoh(coord, &mut coords[..])
            .map_err(|e| format!("reading the coordinates back failed: {e}"))?;

        Ok(Some((coords[region * 2], coords[region * 2 + 1])))
    }

    /// The search kernel's block geometry, mirrored from `POW_BLOCK_*` in `kernels/pearl.cu`.
    ///
    /// These have to agree with the kernel because the host derives `gridDim.x` from them: a block
    /// owns a `POW_BLOCK_ROWS` x `POW_BLOCK_COLS` rectangle of the tile grid, and the kernel reads
    /// `blockIdx.x` as `(x / cols_per_strip, x % cols_per_strip)`. Change one side without the other
    /// and the failure is silent in the worst way -- too few blocks and part of the batch is never
    /// searched at all, and nothing reports it.
    ///
    /// `16 x 12`, chosen by sweeping the rectangle at production geometry against the real kernel
    /// (`kernels/bench_real.cu` + `sweep_cols.bat`, 8192 tiles per row, one launch per point):
    ///
    /// ```text
    ///   16 x 4   97.1 TH/s
    ///   16 x 8  130.3 TH/s   (the previous shipped value)
    ///   16 x 10 135.5 TH/s
    ///   16 x 12 137.1 TH/s   <- shipped
    ///   16 x 14 106.2 TH/s   (not a multiple of 8: breaks the ldmatrix B-fragment layout)
    ///   16 x 16  17.1 TH/s   (96 KB of shared memory incl. the static transcript buffer)
    /// ```
    ///
    /// Wider columns amortise the A operand across more tiles, which is the dominant traffic: a 16x12
    /// block moves 597 GB per full grid against 768 GB for 16x8, a 22% reduction that buys 5.4%.
    ///
    /// The step from 12 to 16 collapses rather than continuing, and bandwidth does not explain it.
    /// 16x16 fits (80 KB dynamic + 16 KB static `sT` = 96 KB against a 101,376 B opt-in limit) and
    /// should be *faster* on traffic alone, so something else dominates at 16 columns — most likely
    /// the `sT` transcript buffer reaching 16 KB, which halves the blocks an SM can hold against
    /// shared memory. 12 columns keeps the whole footprint at 82 KB.
    ///
    /// Every number above is a single launch of `tokenminer_search_grid` itself. A standalone
    /// reconstruction of the same loop (`bench_ceiling.cu`) predicted 8x8 would be ~20% *faster*; the
    /// real kernel measured it 44% slower. Geometry here is measured on the shipped kernel only.
    #[cfg(feature = "pearl")]
    const POW_BLOCK_ROWS: usize = 16;
    #[cfg(feature = "pearl")]
    const POW_BLOCK_COLS: usize = 12;
    /// One warp per tile row of a block's rectangle.
    #[cfg(feature = "pearl")]
    const POW_THREADS: u32 = (Self::POW_BLOCK_ROWS * 32) as u32;
    /// Stages of k in flight. Mirrors `POW_STAGES`.
    #[cfg(feature = "pearl")]
    const POW_STAGES: usize = 2;
    #[cfg(feature = "pearl")]
    const POW_TRANSCRIPT_WORDS: usize = 16;
    #[cfg(feature = "pearl")]
    const POW_TILES_PER_BLOCK: usize = Self::POW_BLOCK_ROWS * Self::POW_BLOCK_COLS;
    /// The transcript buffer: one 16-word row per warp, per tile column of the block rectangle.
    /// Mirrors `POW_SMEM_T_BYTES`. Part of the dynamic request because `POW_SMEM_T_DYNAMIC` puts
    /// `sT` in the same allocation as the staged operands.
    #[cfg(feature = "pearl")]
    const POW_SMEM_T_BYTES: usize =
        Self::POW_BLOCK_ROWS * Self::POW_BLOCK_COLS * Self::POW_TRANSCRIPT_WORDS * 4;

    /// k staged at once, which is also the mma instruction's k. `k` and `rank` must both be multiples
    /// of it. Mirrors `POW_BK`.
    ///
    /// **This is a function, not a constant, because the kernel stages differently by architecture.**
    /// The split mirrors the kernel's own `POW_DP4A`, which is `__CUDA_ARCH__ < 800`: Turing has no
    /// int8 tensor cores and folds with DP4A, and that path's staging array doubles with the stage
    /// width, so at 64 it spills 76 bytes against 20 at 32. Every architecture with tensor cores
    /// stages 64, which is worth **+77%** at production geometry -- see the note beside `POW_BK` in
    /// `kernels/pearl.cu` for the measurement and the stall breakdown that explains it.
    ///
    /// This mirrors the cubin that is actually loaded, so it has to be derived from `arch` rather
    /// than fixed at compile time: the same host runs either fold depending on the card it finds.
    #[cfg(feature = "pearl")]
    fn pow_bk(arch: &str) -> usize {
        // `arch_code` is the exact `__CUDA_ARCH__` value the cubin was compiled for, so this is the
        // same predicate the kernel used rather than a second place to keep the arch table in sync.
        // A cubin loaded for an arch `arch_code` does not know cannot have been loaded at all --
        // `arch_for` and the cubin lookup both reject it earlier.
        match arch_code(arch) {
            Some(code) if code < 800 => 32,
            _ => 64,
        }
    }

    /// The staged row stride, padding each row past its `POW_BK` bytes to keep the fragment loads off
    /// the same banks. Mirrors `POW_SMEM_STRIDE`.
    #[cfg(feature = "pearl")]
    fn pow_smem_stride(arch: &str) -> usize {
        Self::pow_bk(arch) + 16
    }

    /// Shared memory the search kernel's staged A and B buffers take, from `POW_SMEM_A_BYTES` and
    /// `POW_SMEM_B_BYTES` in the kernel, **plus the transcript buffer**.
    ///
    /// The transcript buffer (`POW_SMEM_T_BYTES`: one 16-word row per warp per tile column) used to
    /// be `static __shared__`, which meant it was allocated by the driver *on top* of this request
    /// and had to be deliberately left out — the kernel read `sT` out of the static allocation. It
    /// now lives at the end of the same dynamic block (`POW_SMEM_T_DYNAMIC`), so it is part of this
    /// number and must be added here or the launch requests too little and the transcript writes
    /// run off the end of the allocation.
    ///
    /// The two totals differ by exactly the transcript, which is the whole reason `POW_SMEM_BYTES`
    /// exists separately in the kernel. At the shipped 16x12 that is 70,656 + 12,288 = 82,944 bytes,
    /// against a 101,376 byte per-block opt-in limit.
    #[cfg(feature = "pearl")]
    fn pow_smem_dynamic_bytes(arch: &str) -> usize {
        Self::POW_STAGES
            * (Self::POW_BLOCK_ROWS + Self::POW_BLOCK_COLS)
            * 16
            * Self::pow_smem_stride(arch)
            + Self::POW_SMEM_T_BYTES
    }

    /// Raises the kernel's dynamic shared-memory ceiling to what [`Self::search_grid`] requests.
    ///
    /// A block gets 48 KiB of shared memory without asking. Above that the launch fails with
    /// `cudaErrorInvalidValue` until the function's `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
    /// is set. `16 x 8` stages 36,864 bytes of A and B and would fit under the default, so this call is
    /// currently raising a ceiling it does not need -- it is here because the *next* geometry over the
    /// line should not silently fail to launch, and because the dynamic-shared rewrite makes the
    /// request explicit where it used to be implied by static allocation.
    ///
    /// Called on every launch rather than once at module load. The attribute is a property of the
    /// loaded image, not of any argument, so caching it would be tidier -- but the cost is one
    /// `cuFuncSetAttribute` against a search that folds 67 million tiles, and keeping the call next to
    /// the `shared_mem_bytes` it authorises means the two cannot drift apart. Move it to load time if
    /// it ever shows up in a profile; do not move it away from the launch without moving the constant.
    ///
    /// Idempotent, and a failure here is fatal rather than a fallback: a kernel that cannot get its
    /// shared memory will not launch, and launching it anyway would report the driver error with
    /// nothing pointing at this.
    #[cfg(feature = "pearl")]
    fn pow_opt_in_shared_memory(func: &CudaFunction, arch: &str) -> Result<(), String> {
        use cudarc::driver::sys::CUfunction_attribute_enum;

        let want = Self::pow_smem_dynamic_bytes(arch) as i32;
        func.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            want,
        )
        .map_err(|e| {
            format!(
                "asking for {want} bytes of shared memory per block failed: {e}. A device that \
                 cannot grant it cannot run this geometry"
            )
        })
    }

    /// Searches a batch of 16x16 tiles on the device: fold, hash, bound compare, and claim.
    ///
    /// `a_sum` is the noise-applied `A` as `m_tiles * 16` rows of `k` and `b_sum` the same for `Bt`;
    /// the tile grid is walked linearly and cut into `tiles`/`tiles_per_row`-shaped regions, each with
    /// its own `found` and `coord` slot. `key` is `dnsA` and `bound` is the 256-bit search bound.
    ///
    /// Exactly one tile per region can win, which is what the `atomicCAS` in the kernel enforces: the
    /// rest of a region's qualifying tiles are discarded, because they would produce the same share.
    /// A batch of regions in flight therefore cannot have one region's win overwrite another's, and
    /// the caller can tell which region won without reading back anything but `found`.
    ///
    /// Returns the number of regions in the batch that were won.
    ///
    /// `found` is cleared before the launch, so a caller can reuse one pair of buffers across a
    /// whole search without clearing them itself.
    ///
    /// `transcript_out` is `Some` only for [`Self::check_fold`], which reads every tile's transcript
    /// back to compare against the reference fold. Production passes `None`: hashing the transcript
    /// and comparing the digest already pins the same value, and a 512 KiB read-back per batch to
    /// learn it a second way is not worth the PCIe round trip.
    #[cfg(feature = "pearl")]
    #[allow(clippy::too_many_arguments)]
    pub fn search_grid(
        &self,
        a_sum: &CudaSlice<i8>,
        b_sum: &CudaSlice<i8>,
        key: &[u8; 32],
        bound: U256,
        found: &mut CudaSlice<u32>,
        coord: &mut CudaSlice<u32>,
        transcript_out: Option<&mut CudaSlice<u32>>,
        first_tile: usize,
        tiles: usize,
        tiles_per_row: usize,
        tiles_per_region: usize,
        k: usize,
        rank: usize,
    ) -> Result<usize, String> {
        if rank == 0 || !k.is_multiple_of(rank) {
            return Err(format!("rank {rank} does not divide k {k}"));
        }
        // The fold runs once per staged k-step, and `rank` says how often. A `rank` that is not a whole
        // number of them never fires the fold at all, so every transcript comes back zero and the
        // search reports nothing -- no error, no wrong share, just no mining.
        //
        // The staged width is architecture-dependent (see [`Self::pow_bk`]), so this has to test the
        // width the loaded cubin actually uses: a `rank` below the tensor path's 64 is legal on
        // Turing's 32 and illegal here, and `steps_per_rank` being zero is a device-side divide by
        // zero rather than anything this could catch on the way in.
        let pow_bk = Self::pow_bk(self.arch);
        if !k.is_multiple_of(pow_bk) || !rank.is_multiple_of(pow_bk) {
            return Err(format!(
                "k {k} and rank {rank} both have to be multiples of the {pow_bk} the fold stages at \
                 once on {arch}",
                arch = self.arch
            ));
        }
        if tiles_per_row == 0 || tiles_per_region == 0 {
            return Err(format!(
                "a grid of {tiles_per_row} tiles per row cut into {tiles_per_region}-tile regions \
                 has a zero-sized extent"
            ));
        }
        if tiles == 0 || !tiles.is_multiple_of(tiles_per_region) {
            return Err(format!(
                "a batch of {tiles} tiles has to be a whole number of {tiles_per_region}-tile regions"
            ));
        }
        // The batch has to start on a tile-row boundary, or `tile / tiles_per_row` lands in the
        // middle of one and the reported row is a row the operands were not read from.
        if !first_tile.is_multiple_of(tiles_per_row) {
            return Err(format!(
                "a batch starting at tile {first_tile} does not start on a row of a grid {tiles_per_row} \
                 tiles wide"
            ));
        }
        // And it has to *be* whole tile rows. The kernel recovers the batch's first tile row by
        // dividing out `tiles_per_row`, so a partial row would put every tile after it one row off —
        // reading operand rows the grid does not have while reporting coordinates that look right.
        if !tiles.is_multiple_of(tiles_per_row) {
            return Err(format!(
                "a batch of {tiles} tiles is not a whole number of {tiles_per_row}-tile rows, so it \
                 does not start and end on row boundaries"
            ));
        }

        let regions = tiles / tiles_per_region;
        if found.len() < regions || coord.len() < regions * 2 {
            return Err(format!(
                "a batch of {regions} regions needs {} flag words and {} coordinate words, but the \
                 buffers hold {} and {}",
                regions,
                regions * 2,
                found.len(),
                coord.len()
            ));
        }

        let key_words = to_words(key);
        let key_dev = self
            .stream
            .clone_htod(&key_words)
            .map_err(|e| format!("copying the search key failed: {e}"))?;

        // The 256-bit target as eight big-endian words, most significant group first. The kernel walks
        // them from index 0 and matches each against `words[7 - i]`, which is the digest group of
        // the same significance: the digest is compared as `from_little_endian`, so its last group
        // is its leading one. Getting the index and the byte order independently right is the whole
        // job — a kernel that walks the bound from the top and the digest from the same end, or one
        // that reverses each group as well, compares the target's most significant group against
        // the digest's least significant one and accepts roughly half of every candidate.
        //
        // Taken from the big-endian bytes rather than by re-slicing `to_little_endian`'s output:
        // that output is already little-endian *bytes*, so decoding it as words swaps each
        // four-byte group a second time.
        let mut bound_bytes = [0u8; 32];
        bound.to_big_endian(&mut bound_bytes);
        let bound_words: [u32; 8] = std::array::from_fn(|i| {
            u32::from_be_bytes(bound_bytes[i * 4..][..4].try_into().unwrap())
        });
        let bound_dev = self
            .stream
            .clone_htod(&bound_words)
            .map_err(|e| format!("copying the bound failed: {e}"))?;

        let kernel = self.function("tokenminer_search_grid")?;
        // Before the launch, not after: a geometry that needs more than the default 48 KiB fails at
        // launch time with a driver code that names nothing about shared memory.
        Self::pow_opt_in_shared_memory(&kernel, self.arch)?;
        let first_arg = first_tile as u32;
        let tiles_arg = tiles as u32;
        let row_arg = tiles_per_row as u32;
        let region_arg = tiles_per_region as u32;
        let (k_arg, rank_arg) = (k as i32, rank as i32);

        // Cleared here rather than left to the caller: the search reuses one pair of buffers for
        // every batch of every attempt, and a stale flag would report a batch that found nothing as
        // a win. A stale *coordinate* is harmless, because only a region whose flag is set is read.
        self.stream
            .memset_zeros(&mut *found)
            .map_err(|e| format!("clearing the found flags failed: {e}"))?;

        // A block owns a `POW_BLOCK_ROWS` x `POW_BLOCK_COLS` rectangle of the tile grid, not a tile,
        // and the kernel maps `blockIdx.x` onto that rectangle as `(x / cols_per_strip,
        // x % cols_per_strip)`. Both divisions are ceilings, so a batch whose tile count is not a
        // multiple of either still gets a trailing rectangle that is merely short. Getting this wrong
        // is silent: too few blocks and part of the batch is never searched, too many and the surplus
        // blocks read past the grid.
        let rows_in_batch = tiles / tiles_per_row;
        let cols_per_strip = tiles_per_row.div_ceil(Self::POW_BLOCK_COLS);
        let blocks = rows_in_batch.div_ceil(Self::POW_BLOCK_ROWS) * cols_per_strip;

        // The optional transcript buffer is written unconditionally by the kernel, so its size is the
        // grid's tile count whether or not the rectangles cover every tile.
        if let Some(buffer) = transcript_out.as_ref() {
            let want = blocks * Self::POW_TILES_PER_BLOCK * Self::POW_TRANSCRIPT_WORDS;
            if buffer.len() < want {
                return Err(format!(
                    "a grid of {blocks} blocks needs {want} transcript words, but the buffer holds {}",
                    buffer.len()
                ));
            }
        }

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            // One warp per tile row of the rectangle, each warp holding that row's sixteen columns.
            block_dim: (Self::POW_THREADS, 1, 1),
            // The staged A and B buffers are `extern __shared__`, sized by `pow_opt_in_shared_memory`
            // above. The transcript buffer is still `static __shared__`, so it is not in this number.
            shared_mem_bytes: Self::pow_smem_dynamic_bytes(self.arch) as u32,
        };

        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(a_sum)
            .arg(b_sum)
            .arg(&key_dev)
            .arg(&bound_dev)
            .arg(&mut *found)
            .arg(&mut *coord);
        // A null pointer rather than an empty slice, because the kernel tests the pointer to decide
        // whether to write transcripts at all and an empty slice still has an address. cudarc has no
        // `Option` argument impl, but `u64` is a device-representable scalar, so passing a zero word
        // is how a null device pointer is spelled here.
        match transcript_out {
            Some(buffer) => {
                launch.arg(&*buffer);
            }
            None => {
                launch.arg(&0u64);
            }
        }
        launch
            .arg(&first_arg)
            .arg(&tiles_arg)
            .arg(&row_arg)
            .arg(&region_arg)
            .arg(&k_arg)
            .arg(&rank_arg);
        unsafe { launch.launch(config) }
            .map_err(|e| format!("launching tokenminer_search_grid failed: {e}"))?;

        // The whole array, since `CudaSlice` cannot be sliced in this version of cudarc. It is a few
        // words — one per region in flight — so this is not the synchronisation it looks like.
        let flags = self
            .stream
            .clone_dtoh(&*found)
            .map_err(|e| format!("reading the found flags back failed: {e}"))?;

        Ok(flags.iter().take(regions).filter(|f| **f != 0).count())
    }

    /// One tile of the search grid, for the checks that pin the fold, hash and bound compare against
    /// the reference. The degenerate one-tile, one-region case of [`Self::search_grid`], so the check
    /// covers the kernel the search actually runs rather than a copy of it.
    ///
    /// A 1x1 tile grid is short on both axes of a block rectangle, which is the point: this is the
    /// shape that exercises the kernel's short-rectangle paths on every run of `check_bound`.
    #[cfg(feature = "pearl")]
    #[allow(clippy::too_many_arguments)]
    pub fn search_tile(
        &self,
        a_sum: &CudaSlice<i8>,
        b_sum: &CudaSlice<i8>,
        key: &[u8; 32],
        bound: U256,
        found: &mut CudaSlice<u32>,
        coord: &mut CudaSlice<u32>,
        k: usize,
        rank: usize,
    ) -> Result<bool, String> {
        Ok(self.search_grid(
            a_sum, b_sum, key, bound, found, coord, None, 0, 1, 1, 1, k, rank,
        )? > 0)
    }

    /// Keyed BLAKE3 of the 64-byte jackpot message, mirroring `compute_jackpot_hash`.
    ///
    /// Doing this on the device is what lets a candidate be rejected without a round trip: the
    /// mining loop compares the digest against the bound and only surfaces a winner.
    ///
    /// No kernel of its own — a 64-byte message is a single chunk, which is [`Self::blake3`]. The
    /// same routine hashes the matrix commitments, so the two cannot drift apart.
    #[cfg(feature = "pearl")]
    pub fn jackpot_hash(&self, message: &[u8; 64], key: &[u8; 32]) -> Result<[u8; 32], String> {
        self.blake3(&message[..], Some(*key))
    }

    /// BLAKE3 over a buffer already resident on the device, keyed or not.
    ///
    /// This is the digest the commitment chain is built from. A Pearl proof commits to each matrix
    /// with the keyed BLAKE3 of its 1024-byte-padded bytes, and the two noise seeds chain off those
    /// digests — so this has to agree with `pearl_blake3` byte for byte, not merely be *a* hash.
    /// It does: `MerkleTree::root()` walks the standard BLAKE3 chunk-CV tree, and so does this.
    ///
    /// Takes device memory because that is the form the search will have: the matrices are filled
    /// on the device and never come back unless they win a share.
    #[cfg(feature = "pearl")]
    pub fn blake3_device(
        &self,
        data: &CudaSlice<u8>,
        key: Option<[u8; 32]>,
    ) -> Result<[u8; 32], String> {
        let len = data.len();
        self.blake3_device_bytes(data, len, key)
    }

    /// The same hash over a device buffer of any element type, with its byte length given explicitly.
    ///
    /// The commitment is over the *bytes* of a matrix, and the matrices the search builds are `i8`,
    /// so this is the form the search hashes them through. Byte length rather than element count is
    /// the argument that matters: a commitment over `M x K` int8 is over `M * K` bytes, and reading
    /// it as `len` elements of a wider type is how that gets silently wrong.
    #[cfg(feature = "pearl")]
    pub fn blake3_device_bytes<T>(
        &self,
        data: &CudaSlice<T>,
        len: usize,
        key: Option<[u8; 32]>,
    ) -> Result<[u8; 32], String> {
        if len > u32::MAX as usize {
            return Err(format!(
                "BLAKE3 input of {len} bytes exceeds the kernel's 32-bit length argument"
            ));
        }

        // A key enters BLAKE3 as the chaining value of the first eight words plus a flag; an
        // unkeyed hash is the IV with no flag, so there is nothing else to distinguish the modes.
        let key_words: [u32; 8] = match key {
            Some(key) => to_words(&key),
            None => B3_IV,
        };
        let base_flags = if key.is_some() { B3_KEYED_HASH } else { 0 };

        let key_dev = self
            .stream
            .clone_htod(&key_words)
            .map_err(|e| format!("copying the hash key failed: {e}"))?;
        let mut digest_dev = self
            .stream
            .alloc_zeros::<u32>(B3_WORDS)
            .map_err(|e| format!("allocating the digest failed: {e}"))?;

        if len <= B3_CHUNK_LEN {
            // One chunk: the chunk is the root, which is the path the reference takes for a short
            // input and the one a 64-byte jackpot message takes.
            let kernel = self.function("tokenminer_blake3_root_chunk")?;
            let len_arg = len as u32;

            let mut launch = self.stream.launch_builder(&kernel);
            launch
                .arg(data)
                .arg(&len_arg)
                .arg(&key_dev)
                .arg(&base_flags)
                .arg(&mut digest_dev);
            unsafe { launch.launch(ONE_THREAD) }
                .map_err(|e| format!("launching tokenminer_blake3_root_chunk failed: {e}"))?;
        } else {
            let chunks = len.div_ceil(B3_CHUNK_LEN);
            let mut front = self
                .stream
                .alloc_zeros::<u32>(chunks * B3_WORDS)
                .map_err(|e| format!("allocating the chunk chaining values failed: {e}"))?;
            let mut back = self
                .stream
                .alloc_zeros::<u32>(chunks * B3_WORDS)
                .map_err(|e| format!("allocating the tree level failed: {e}"))?;

            let len_arg = len as u32;
            let cvs = self.function("tokenminer_blake3_chunk_cvs")?;
            let mut launch = self.stream.launch_builder(&cvs);
            launch
                .arg(data)
                .arg(&len_arg)
                .arg(&key_dev)
                .arg(&base_flags)
                .arg(&mut front);
            unsafe { launch.launch(grid_for(chunks)) }
                .map_err(|e| format!("launching tokenminer_blake3_chunk_cvs failed: {e}"))?;

            // One level at a time, each halving the node count, until two are left. An odd node is
            // carried up by the kernel unchanged, which is what makes a non-power-of-two leaf count
            // hash to the reference's value.
            let combine = self.function("tokenminer_blake3_combine")?;
            let mut count = chunks;
            while count > 2 {
                let mut launch = self.stream.launch_builder(&combine);
                launch
                    .arg(&front)
                    .arg(&count)
                    .arg(&key_dev)
                    .arg(&base_flags)
                    .arg(&0u32)
                    .arg(&mut back);
                unsafe { launch.launch(grid_for(count.div_ceil(2))) }
                    .map_err(|e| format!("launching tokenminer_blake3_combine failed: {e}"))?;

                std::mem::swap(&mut front, &mut back);
                count = count.div_ceil(2);
            }

            // The last pair is the root, so it is the one node hashed with ROOT set.
            let mut launch = self.stream.launch_builder(&combine);
            launch
                .arg(&front)
                .arg(&count)
                .arg(&key_dev)
                .arg(&base_flags)
                .arg(&1u32)
                .arg(&mut digest_dev);
            unsafe { launch.launch(grid_for(1)) }
                .map_err(|e| format!("hashing the BLAKE3 tree root failed: {e}"))?;
        }

        let host = self
            .stream
            .clone_dtoh(&digest_dev)
            .map_err(|e| format!("reading the digest back failed: {e}"))?;

        let mut digest = [0u8; 32];
        for (word, bytes) in host.iter().zip(digest.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&word.to_le_bytes());
        }

        Ok(digest)
    }

    /// BLAKE3 over host memory: [`Self::blake3_device`] with the input copied across first.
    #[cfg(feature = "pearl")]
    pub fn blake3(&self, data: &[u8], key: Option<[u8; 32]>) -> Result<[u8; 32], String> {
        let device = self
            .stream
            .clone_htod(data)
            .map_err(|e| format!("copying the BLAKE3 input failed: {e}"))?;

        self.blake3_device(&device, key)
    }

    /// Fills `rows x rank` bytes of dense noise (EAL or EBR) starting at row `first_row`.
    ///
    /// Every entry is one keyed BLAKE3 digest's worth of data, so this is the shape the search wants
    /// it in: A's noise is a full `M x R` tensor and B's a full `N x R` one, both written straight
    /// into the buffer the GEMM will read, never touched on the host. `first_row` lets a region ask
    /// for the slice of rows it is working on.
    ///
    /// Consensus values, so this reproduces `generate_uniform_random_matrix` exactly; `check_noise`
    /// holds the two to each other.
    #[cfg(feature = "pearl")]
    pub fn fill_noise_dense(
        &self,
        out: &mut CudaSlice<i8>,
        seed: &[u8; 32],
        key: &[u8; 32],
        first_row: u32,
        rows: usize,
        rank: usize,
    ) -> Result<(), String> {
        validate_rank(rank)?;
        if rows * rank > out.len() {
            return Err(format!(
                "dense noise wants {rows} rows of {rank} ({}) but the buffer holds {}",
                rows * rank,
                out.len()
            ));
        }
        if rows == 0 {
            return Ok(());
        }

        let (seed_dev, key_dev) = self.noise_words(seed, key)?;
        let (first_row_arg, rows_arg, rank_arg) = (first_row, rows as u32, rank as u32);

        let kernel = self.function("tokenminer_noise_dense")?;
        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(&seed_dev)
            .arg(&key_dev)
            .arg(&first_row_arg)
            .arg(&rows_arg)
            .arg(&rank_arg)
            .arg(out);
        unsafe { launch.launch(grid_for(rows * rank / NOISE_HASH_LEN)) }
            .map_err(|e| format!("launching tokenminer_noise_dense failed: {e}"))?;

        Ok(())
    }

    /// Fills `count` sparse noise pairs (EAR or EBL) starting at hash index `first_index`.
    ///
    /// Each pair is two column indices into a dense row of length `rank`: a `+1` at the first and a
    /// `-1` at the second. Composing the two tensors is two loads and a subtract per entry, which is
    /// why the reference never materialises a dense `M x K` noise matrix either.
    ///
    /// One BLAKE3 digest yields eight pairs, so `count` pairs cost `count / 8`
    /// hashes: the kernel writes all eight per thread rather than eight threads
    /// hashing the same message and each keeping a single four-byte slot.
    #[cfg(feature = "pearl")]
    pub fn fill_noise_perm(
        &self,
        out: &mut CudaSlice<u32>,
        seed: &[u8; 32],
        key: &[u8; 32],
        first_index: u32,
        count: usize,
        rank: usize,
    ) -> Result<(), String> {
        validate_rank(rank)?;
        if count * 2 > out.len() {
            return Err(format!(
                "sparse noise wants {count} pairs ({}) but the buffer holds {} words",
                count * 2,
                out.len()
            ));
        }
        if count == 0 {
            return Ok(());
        }

        let (seed_dev, key_dev) = self.noise_words(seed, key)?;
        let (first_index_arg, count_arg, rank_arg) = (first_index, count as u32, rank as u32);

        let kernel = self.function("tokenminer_noise_perm")?;
        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(&seed_dev)
            .arg(&key_dev)
            .arg(&first_index_arg)
            .arg(&count_arg)
            .arg(&rank_arg)
            .arg(out);
        // One hash yields eight pairs, so the grid is an eighth of the pair
        // count — `tokenminer_noise_perm` writes all eight per thread.
        unsafe { launch.launch(grid_for(count.div_ceil(8))) }
            .map_err(|e| format!("launching tokenminer_noise_perm failed: {e}"))?;

        Ok(())
    }

    /// Fills a signal matrix from the Philox-4x32-10 stream, `i8` in `[-64, 63]`.
    ///
    /// `offset` is the element index of `out[0]` within the whole matrix, which is what makes the
    /// stream counter-based rather than sequential: the search fills `Bt` as `N x K` directly and
    /// writes a column block at a time, and the bytes have to be the ones a full-matrix fill would
    /// have produced there. `check_fill` pins that.
    #[cfg(feature = "pearl")]
    pub fn fill_i8(
        &self,
        out: &mut CudaSlice<i8>,
        offset: usize,
        numel: usize,
        seed: u64,
    ) -> Result<(), String> {
        if numel > out.len() {
            return Err(format!(
                "the fill wants {numel} bytes but the buffer holds {}",
                out.len()
            ));
        }
        if numel == 0 {
            return Ok(());
        }

        // The grid covers `numel` elements; `offset` goes to the kernel as the absolute index of
        // `out[0]`, which is what makes the counters — and therefore the bytes — agree with a
        // full-matrix fill.
        let kernel = self.function("tokenminer_fill_i8")?;
        let (offset_arg, numel_arg, seed_arg) = (offset as i64, numel as i64, seed);

        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(&*out)
            .arg(&offset_arg)
            .arg(&numel_arg)
            .arg(&seed_arg);

        // Four bytes per thread.
        let threads = 256usize;
        let blocks = numel.div_ceil(threads * 4);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { launch.launch(config) }
            .map_err(|e| format!("launching tokenminer_fill_i8 failed: {e}"))?;

        Ok(())
    }

    /// `out = signal + noise` for one row window, where the noise is the composed sparse pair form.
    ///
    /// This is the summed operand the fold GEMM takes, so it is where the `[-127, 126]` range
    /// argument is actually cashed out: the kernel adds in `i32` and stores `i8`, which is only
    /// lossless because of it. `check_noise_apply` holds the result against `compute_noise`.
    #[cfg(feature = "pearl")]
    #[allow(clippy::too_many_arguments)]
    pub fn noise_apply(
        &self,
        signal: &CudaSlice<i8>,
        dense: &CudaSlice<i8>,
        pairs: &CudaSlice<u32>,
        out: &mut CudaSlice<i8>,
        rows: usize,
        k: usize,
        rank: usize,
    ) -> Result<(), String> {
        validate_rank(rank)?;
        if rows == 0 || k == 0 {
            return Ok(());
        }
        if signal.len() < rows * k || out.len() < rows * k {
            return Err(format!(
                "the noise-apply wants {rows}x{k} operands but the buffers hold {} and {}",
                signal.len(),
                out.len()
            ));
        }
        if dense.len() < rows * rank {
            return Err(format!(
                "the noise-apply wants {rows}x{rank} of dense noise but the buffer holds {}",
                dense.len()
            ));
        }
        if pairs.len() < k * 2 {
            return Err(format!(
                "the noise-apply wants {k} pairs but the buffer holds {} words",
                pairs.len()
            ));
        }
        if !k.is_multiple_of(rank) {
            return Err(format!("rank {rank} does not divide k {k}"));
        }

        let kernel = self.function("tokenminer_noise_apply")?;
        let (rows_arg, k_arg, rank_arg) = (rows as u32, k as u32, rank as u32);

        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(signal)
            .arg(dense)
            .arg(pairs)
            .arg(&*out)
            .arg(&rows_arg)
            .arg(&k_arg)
            .arg(&rank_arg);
        unsafe { launch.launch(grid_for(rows * k)) }
            .map_err(|e| format!("launching tokenminer_noise_apply failed: {e}"))?;

        Ok(())
    }

    /// Uploads the two eight-word inputs both noise kernels take.
    #[cfg(feature = "pearl")]
    fn noise_words(
        &self,
        seed: &[u8; 32],
        key: &[u8; 32],
    ) -> Result<(CudaSlice<u32>, CudaSlice<u32>), String> {
        let seed_dev = self
            .stream
            .clone_htod(&to_words(seed))
            .map_err(|e| format!("copying the noise seed failed: {e}"))?;
        let key_dev = self
            .stream
            .clone_htod(&to_words(key))
            .map_err(|e| format!("copying the noise key failed: {e}"))?;

        Ok((seed_dev, key_dev))
    }

    /// Checks the device BLAKE3 tree against the reference it has to agree with.
    ///
    /// The reference is `pearl_blake3`: the commitment chain runs `blake3_digest` over the padded
    /// matrices, the proof carries `MerkleTree` leaves, and the pool verifies with both. They are
    /// the same value — the reference's tree *is* BLAKE3 tree hashing — so this checks the digest
    /// against both spellings, at the leaf counts where a tree changes shape.
    #[cfg(feature = "pearl")]
    fn check_blake3_tree(&self) -> Result<(), String> {
        let key: [u8; 32] = std::array::from_fn(|i| (i * 37 + 11) as u8);

        // Lengths chosen to cross every boundary the implementation has: empty, a partial block, one
        // and two whole blocks inside a chunk, a partial block again, the short-input root path at
        // exactly one chunk, and then tree shapes — two chunks (the smallest multi-chunk tree), three
        // and five (the odd-node carry-up), sixty-four (a whole segment), sixty-five (across it), and
        // a thousand (five levels).
        let lengths = [
            0,
            1,
            63,
            64,
            128,
            65,
            1023,
            B3_CHUNK_LEN,
            B3_CHUNK_LEN + 1,
            2 * B3_CHUNK_LEN,
            2 * B3_CHUNK_LEN + 1,
            3 * B3_CHUNK_LEN,
            5 * B3_CHUNK_LEN,
            64 * B3_CHUNK_LEN,
            65 * B3_CHUNK_LEN,
            1000 * B3_CHUNK_LEN,
        ];

        for len in lengths {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();

            for key in [Some(key), None] {
                let want = pearl_blake3::blake3_digest(&data, key);
                let got = self
                    .blake3(&data, key)
                    .map_err(|error| format!("{len} bytes: {error}"))?;

                if got != want {
                    return Err(format!(
                        "the device BLAKE3 of {len} bytes is {got:?}, not the reference's \
                         {want:?} (key {})",
                        key.is_some()
                    ));
                }

                // The spelling the commitment actually uses. `MerkleTree` is keyed-only — an
                // unkeyed hash is the IV with no KEYED_HASH flag, which a key argument cannot
                // express — so this half of the check is the keyed path. It also has no answer for
                // empty input, which a matrix commitment never is.
                if let Some(key) = key.filter(|_| !data.is_empty()) {
                    let root = pearl_blake3::MerkleTree::new(&data, key).root();
                    if got != root {
                        return Err(format!(
                            "the device BLAKE3 of {len} bytes is {got:?}, not the Merkle root \
                             the proof commits to ({root:?})"
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Checks both jackpot kernels against the canonical implementation, on operands from real
    /// proofs.
    ///
    /// This is the check that decides whether the backend may mine: the reference is
    /// `zk_pow::circuit::chip::compute_jackpot` — the function the verifier itself uses — so a
    /// mismatch here means the GPU would produce proofs the network rejects.
    ///
    /// Two shapes, because there are two kernels and the tile shape is not ours to choose. The
    /// default configuration is 2x64, which `wmma` cannot hold, and the reference miner's
    /// production tile is 16x16 (`HT = 16`), which it can. Checking only one would leave half the
    /// search paths unguarded, and the fold is the one the miner will actually run.
    #[cfg(feature = "pearl")]
    fn check_jackpot(&self) -> Result<(), String> {
        // Not the shipped default, and deliberately so. `check_one_jackpot` gets its operands by
        // mining a real proof through `try_mine_one`, the CPU reference search, which builds both
        // matrices and a Merkle tree over each. At 131072 x 131072 that is minutes per tile — and
        // this runs from `pearl::start`, so the engine would sit on its start-up self-test for
        // minutes. Nothing it would find depends on the size.

        // The shape the search actually folds: sixteen consecutive rows against sixteen consecutive
        // columns, which `tokenminer_jackpot_fold` takes as a single fragment pair.
        let square = small_mining();

        // The probe's own tile — two rows against sixty-four columns — so the expanded path is
        // checked on a shape the folded one cannot be. It was the reference miner's default before
        // the search moved to `range(16)` patterns, and the grid kernel has no representation for
        // it at all, which is exactly why it is worth keeping on the expansion.
        //
        // `n` has to clear the pattern: its highest index is 249, and the verifier requires the
        // tile's offset plus that index to stay inside the matrix.
        let wide = PearlMining {
            n: 256,
            rows_pattern: vec![0, 8],
            cols_pattern: (0..32u32)
                .flat_map(|step| [step * 8, step * 8 + 1])
                .collect(),
            ..small_mining()
        };

        self.check_one_jackpot(&wide, JackpotPath::Expanded)?;
        self.check_one_jackpot(&square, JackpotPath::Fold)
    }

    /// One tile shape, end to end: mine a proof at it, take the real committed matrices and the
    /// real noise out of it, and hold whichever jackpot kernel covers that shape against
    /// `compute_jackpot`.
    #[cfg(feature = "pearl")]
    fn check_one_jackpot(&self, mining: &PearlMining, path: JackpotPath) -> Result<(), String> {
        use zk_pow::api::proof::{IncompleteBlockHeader, SeedDerivation};
        use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
        use zk_pow::circuit::chip::compute_jackpot;
        use zk_pow::circuit::pearl_noise::compute_noise;
        use zk_pow::ffi::mine::try_mine_one;

        let config = mining_configuration(mining)?;
        let header = IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x11; 32],
            merkle_root: [0x22; 32],
            timestamp: 1_700_000_000,
            nbits: 0x207f_ffff,
        };

        // The largest compact target. The verifier's difficulty factor scales it past 2^256, so the
        // bound saturates and the first tile drawn wins: nothing is actually mined here, this check
        // only needs a real proof to take its committed matrices and noise from.
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
        .map_err(|e| format!("producing a proof for the jackpot check failed: {e}"))?
        .ok_or("the saturated bound should have yielded a proof on the first tile")?;

        let (private, public) = proof
            .parse_proof(header, SeedDerivation::Salted)
            .map_err(|e| format!("parsing the proof failed: {e}"))?;

        let compiled = CompiledPublicParams::from(&public);
        let noise = compute_noise(&compiled);
        let want = compute_jackpot(&compiled, &private.s_a, &private.s_b, &noise);

        let h = private.s_a.len();
        let w = private.s_b.len();
        let k = private.s_a.first().map(Vec::len).unwrap_or(0);
        let rank = public.mining_config.rank as usize;

        if noise.a.len() != h || noise.b.len() != w {
            return Err(format!(
                "the noise is {}xA / {}xB but the tile is {h}x{w}",
                noise.a.len(),
                noise.b.len()
            ));
        }

        let flatten = |rows: &[Vec<i8>]| -> Vec<i8> { rows.iter().flatten().copied().collect() };
        let secret_a = flatten(&private.s_a);
        let noise_a = flatten(&noise.a);
        let secret_b = flatten(&private.s_b);
        let noise_b = flatten(&noise.b);

        let got = match path {
            JackpotPath::Expanded => {
                self.jackpot_tile(&secret_a, &noise_a, &secret_b, &noise_b, k, rank, h, w)?
            }
            JackpotPath::Fold => {
                let a_sum = sum_operands(&secret_a, &noise_a, h, k, "A")?;
                let b_sum = sum_operands(&secret_b, &noise_b, w, k, "B")?;

                self.jackpot_fold(&a_sum, &b_sum, k, rank, h, w)?
            }
        };

        if got != want {
            return Err(format!(
                "the {path} device jackpot {got:?} does not match the verifier's {want:?}"
            ));
        }

        // And the hash the pool actually compares against, also on the device. Once is enough — it
        // is the same message either kernel's output produces.
        let mut message = [0u8; 64];
        for (i, word) in want.iter().enumerate() {
            message[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let key = compiled.a_noise_seed();

        let want_hash = compute_jackpot_hash(&want, key);
        let got_hash = self.jackpot_hash(&message, &key)?;

        if got_hash != want_hash {
            return Err(format!(
                "the device jackpot hash {got_hash:?} does not match the verifier's {want_hash:?}"
            ));
        }

        Ok(())
    }

    /// Checks the device noise tensors against `zk_pow::circuit::pearl_noise`, the reference the
    /// verifier itself uses.
    ///
    /// All four are checked, not just the composed result: a wrong tensor can still compose into the
    /// right answer for a tile the check happens not to cover, and EAL/EBL and EAR/EBL fail for
    /// different reasons (the `byte & 63` draw and the sparse permutation respectively). The
    /// composition is then checked too, on the host, against `compute_noise_for_indices` — so the
    /// device is pinned to the tensors and the pairing is pinned to the reference, with neither
    /// silently standing in for the other.
    #[cfg(feature = "pearl")]
    fn check_noise(&self) -> Result<(), String> {
        use zk_pow::api::proof::{
            IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern,
            PublicProofParams, SeedDerivation,
        };
        use zk_pow::circuit::pearl_noise::{
            compute_noise_for_indices, generate_permutation_matrix, generate_uniform_random_matrix,
        };

        // The tile patterns scatter rows (a winning tile's rows are 800, 808, 864, 872, not 800…803),
        // but the search never asks for a scattered window: it generates the whole matrices and
        // indexes the tile's rows out of them. So the window checked here is contiguous and starts
        // well above row 0, which is what makes `first_row` matter — a kernel that hashed from zero
        // would agree on row 0 and be wrong everywhere the miner actually runs.
        let header = IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x11; 32],
            merkle_root: [0x22; 32],
            timestamp: 1_700_000_000,
            nbits: 0x207f_ffff,
        };
        let config = MiningConfiguration {
            common_dim: 512,
            rank: 128,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[
                0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105,
            ])
            .unwrap(),
            moe: None,
        };
        let mut public = PublicProofParams::new_dummy(
            header,
            SeedDerivation::Salted,
            config,
            1024,
            1024,
            800,
            300,
        );
        public.hash_a = [0xAA; 32];
        public.hash_b = [0xBB; 32];

        let k = public.common_dim();
        let rank = public.mining_config.rank as usize;
        let job_key = [0x5Au8; 32];
        let (b_noise_seed, a_noise_seed) = public.commitment_hash(job_key);

        let a_rows: Vec<usize> = (800..840).collect();
        let b_cols: Vec<usize> = (300..364).collect();

        let (e_al, ear, e_bl, e_br) = self.noise_tensors(
            &a_noise_seed,
            &b_noise_seed,
            NoiseWindow {
                a: (a_rows[0] as u32, a_rows.len()),
                b: (b_cols[0] as u32, b_cols.len()),
            },
            k,
            rank,
        )?;

        let want_al = generate_uniform_random_matrix(&SEED_LABEL_A, &a_noise_seed, &a_rows, rank);
        let want_ear = generate_permutation_matrix(&SEED_LABEL_A, &a_noise_seed, k, rank);
        let want_bl = generate_permutation_matrix(&SEED_LABEL_B, &b_noise_seed, k, rank);
        let want_br = generate_uniform_random_matrix(&SEED_LABEL_B, &b_noise_seed, &b_cols, rank);

        compare_i8(&e_al, &flatten_i8(&want_al), "EAL")?;
        compare_i8(&e_br, &flatten_i8(&want_br), "EBR")?;
        compare_perm(&ear, &want_ear, "EAR")?;
        compare_perm(&e_bl, &want_bl, "EBL")?;

        // And the composition, which is a two-term difference per entry rather than a product.
        let reference =
            compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &a_rows, &b_cols);

        for (row, want) in reference.a.iter().enumerate() {
            for (l, value) in want.iter().enumerate() {
                let got =
                    e_al[row * rank + ear[l][0] as usize] - e_al[row * rank + ear[l][1] as usize];
                if got != *value {
                    return Err(format!(
                        "the composed noise A row {row} column {l} is {got} but the reference has \
                         {value}"
                    ));
                }
            }
        }

        for (col, want) in reference.b.iter().enumerate() {
            for (l, value) in want.iter().enumerate() {
                let got =
                    e_br[col * rank + e_bl[l][0] as usize] - e_br[col * rank + e_bl[l][1] as usize];
                if got != *value {
                    return Err(format!(
                        "the composed noise B column {col} column {l} is {got} but the reference has \
                         {value}"
                    ));
                }
            }
        }

        Ok(())
    }

    /// The four noise tensors, on the device and back again.
    ///
    /// A convenience for the checks only: the search fills these straight into the buffers it will
    /// read, so it never pays for the round trip.
    #[cfg(feature = "pearl")]
    fn noise_tensors(
        &self,
        a_noise_seed: &[u8; 32],
        b_noise_seed: &[u8; 32],
        window: NoiseWindow,
        k: usize,
        rank: usize,
    ) -> Result<NoiseTensors, String> {
        let mut e_al = self
            .stream
            .alloc_zeros::<i8>(window.a.1 * rank)
            .map_err(|e| format!("allocating EAL failed: {e}"))?;
        let mut ear = self
            .stream
            .alloc_zeros::<u32>(k * 2)
            .map_err(|e| format!("allocating EAR failed: {e}"))?;
        let mut e_bl = self
            .stream
            .alloc_zeros::<u32>(k * 2)
            .map_err(|e| format!("allocating EBL failed: {e}"))?;
        let mut e_br = self
            .stream
            .alloc_zeros::<i8>(window.b.1 * rank)
            .map_err(|e| format!("allocating EBR failed: {e}"))?;

        self.fill_noise(
            &mut e_al,
            &mut ear,
            &mut e_bl,
            &mut e_br,
            a_noise_seed,
            b_noise_seed,
            window.a.0,
            window.b.0,
            window.a.1,
            window.b.1,
            k,
            rank,
        )?;

        let e_al = self
            .stream
            .clone_dtoh(&e_al)
            .map_err(|e| format!("reading EAL back failed: {e}"))?;
        let e_br = self
            .stream
            .clone_dtoh(&e_br)
            .map_err(|e| format!("reading EBR back failed: {e}"))?;
        let ear = pairs(
            &self
                .stream
                .clone_dtoh(&ear)
                .map_err(|e| format!("reading EAR back failed: {e}"))?,
        );
        let e_bl = pairs(
            &self
                .stream
                .clone_dtoh(&e_bl)
                .map_err(|e| format!("reading EBL back failed: {e}"))?,
        );

        Ok((e_al, ear, e_bl, e_br))
    }

    /// Fills the four noise tensors into buffers the caller already owns.
    ///
    /// This is the layout [`Self::noise_tensors`] reads back and `check_noise` holds against
    /// `generate_uniform_random_matrix` and `generate_permutation_matrix`, kept in one place so the
    /// search cannot order the four tensors differently from the check that verifies them: `EAR` and
    /// `EBL` are one column pair per index over `k` — not `rank * k` of them — and `EAL` / `EBR` are
    /// the dense `rows * rank` matrices the pairs index into.
    #[cfg(feature = "pearl")]
    #[allow(clippy::too_many_arguments)]
    pub fn fill_noise(
        &self,
        e_al: &mut CudaSlice<i8>,
        ear: &mut CudaSlice<u32>,
        e_bl: &mut CudaSlice<u32>,
        e_br: &mut CudaSlice<i8>,
        a_noise_seed: &[u8; 32],
        b_noise_seed: &[u8; 32],
        a_first_row: u32,
        b_first_row: u32,
        a_rows: usize,
        b_rows: usize,
        k: usize,
        rank: usize,
    ) -> Result<(), String> {
        self.fill_noise_dense(e_al, &SEED_LABEL_A, a_noise_seed, a_first_row, a_rows, rank)?;
        self.fill_noise_perm(ear, &SEED_LABEL_A, a_noise_seed, 0, k, rank)?;
        self.fill_noise_perm(e_bl, &SEED_LABEL_B, b_noise_seed, 0, k, rank)?;
        self.fill_noise_dense(e_br, &SEED_LABEL_B, b_noise_seed, b_first_row, b_rows, rank)?;
        Ok(())
    }

    /// Checks the Philox fill against a host transcription, and pins the property the search needs.
    ///
    /// Two separate things are wrong in different ways. A wrong Philox — a transposed round, a
    /// mistyped multiplier — produces plausible bytes in range that simply are not the stream, and
    /// the symptom would be a miner that produces *internally* consistent proofs (the commitment
    /// covers whatever we filled) but whose jobs are worth nothing. The reference comparison catches
    /// that. And a wrong *offset* — counting the counter from the window instead of the matrix —
    /// would make every window disagree with the commitment the next window was taken against.
    #[cfg(feature = "pearl")]
    fn check_fill(&self) -> Result<(), String> {
        const K: usize = 512;
        const SEED: u64 = 0x0123_4567_89ab_cdef;

        // A whole matrix, and a window of it filled at an offset — the two must agree, which is the
        // only reason the offset argument exists.
        let mut whole = self
            .stream
            .alloc_zeros::<i8>(K)
            .map_err(|e| format!("allocating the fill target failed: {e}"))?;
        self.fill_i8(&mut whole, 0, K, SEED)?;

        let whole = self
            .stream
            .clone_dtoh(&whole)
            .map_err(|e| format!("reading the fill back failed: {e}"))?;

        for (index, (&got, want)) in whole.iter().zip(philox_stream(0, K, SEED)).enumerate() {
            if got != want {
                return Err(format!(
                    "the device fill has {got} at element {index} but the reference stream has {want}"
                ));
            }
        }

        // The window, filled at a non-zero offset that is not a multiple of the per-thread chunk, so
        // the threads straddle the boundary. An offset bug would not be caught by an aligned one.
        const OFFSET: usize = 3;
        const WINDOW: usize = 300;
        let mut window = self
            .stream
            .alloc_zeros::<i8>(WINDOW)
            .map_err(|e| format!("allocating the window target failed: {e}"))?;
        self.fill_i8(&mut window, OFFSET, WINDOW, SEED)?;

        let window = self
            .stream
            .clone_dtoh(&window)
            .map_err(|e| format!("reading the window back failed: {e}"))?;

        for (index, (&got, want)) in window
            .iter()
            .zip(whole[OFFSET..OFFSET + WINDOW].iter())
            .enumerate()
        {
            if got != *want {
                return Err(format!(
                    "the window filled at offset {OFFSET} has {got} at element {index} but the whole \
                     matrix has {want} there"
                ));
            }
        }

        // And the range the fold GEMM's operand arithmetic rests on. The signal is drawn in
        // [-64, 63] precisely so that adding the noise cannot leave INT8.
        match whole.iter().min() {
            Some(min) if *min < -64 => {
                return Err(format!(
                    "the fill produced {min}, below the -64 the operand range needs"
                ));
            }
            _ => {}
        }
        match whole.iter().max() {
            Some(max) if *max > 63 => {
                return Err(format!(
                    "the fill produced {max}, above the 63 the operand range needs"
                ));
            }
            _ => {}
        }

        Ok(())
    }

    /// Checks the noise-apply against the reference noise, on a window.
    ///
    /// The device form is the summed operand consensus multiplies, so it is checked against
    /// `compute_noise_for_indices` composed with the same signal — not against a hand-computed
    /// expectation, because the range argument is the thing that matters and only the reference can
    /// produce the actual extremes. A window at a non-zero row offset, for the same
    /// `first_row`-vs-window reason `check_noise` uses one.
    #[cfg(feature = "pearl")]
    fn check_noise_apply(&self) -> Result<(), String> {
        use zk_pow::api::proof::{
            IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern,
            PublicProofParams, SeedDerivation,
        };
        use zk_pow::circuit::pearl_noise::compute_noise_for_indices;

        let header = IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x11; 32],
            merkle_root: [0x22; 32],
            timestamp: 1_700_000_000,
            nbits: 0x207f_ffff,
        };
        let config = MiningConfiguration {
            common_dim: 512,
            rank: 128,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9]).unwrap(),
            moe: None,
        };
        let mut public = PublicProofParams::new_dummy(
            header,
            SeedDerivation::Salted,
            config,
            1024,
            1024,
            800,
            300,
        );
        public.hash_a = [0xAA; 32];
        public.hash_b = [0xBB; 32];

        let k = public.common_dim();
        let rank = public.mining_config.rank as usize;
        let job_key = [0x5Au8; 32];
        let (b_noise_seed, a_noise_seed) = public.commitment_hash(job_key);

        let rows = 24usize;
        let a_rows: Vec<usize> = (800..800 + rows).collect();
        let b_cols: Vec<usize> = (300..300 + rows).collect();

        let (e_al, ear, e_bl, e_br) = self.noise_tensors(
            &a_noise_seed,
            &b_noise_seed,
            NoiseWindow {
                a: (a_rows[0] as u32, a_rows.len()),
                b: (b_cols[0] as u32, b_cols.len()),
            },
            k,
            rank,
        )?;

        // Signal matrices drawn from the fill, so the operand ranges under test are the real ones
        // rather than a synthetic spread that might never reach the extremes.
        let signal_a = philox_stream(0, rows * k, 1);
        let signal_b = philox_stream(0, rows * k, 2);

        let (got_a, got_b) = self.applied_operands(
            &signal_a, &e_al, &ear, &signal_b, &e_br, &e_bl, rows, k, rank,
        )?;

        // One call, because the two halves are not a symmetric pair: the seed tuple is ordered
        // `(b, a)` and the index lists `(a_rows, b_cols)`, and swapping them to "get the B side"
        // silently builds a different tensor rather than the transposed one.
        let want =
            compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &a_rows, &b_cols);

        for (label, got, want, signal) in [
            ("A", &got_a, &want.a, &signal_a),
            ("B", &got_b, &want.b, &signal_b),
        ] {
            for (row, reference) in want.iter().enumerate() {
                for (l, &noise) in reference.iter().enumerate() {
                    let sum = signal[row * k + l] as i32 + noise as i32;
                    if !(-128..=127).contains(&sum) {
                        return Err(format!(
                            "the composed noise for {label} row {row} column {l} is {noise}, which \
                             puts the signal {signal:?} outside INT8 — the operand range argument \
                             for the single-GEMM fold no longer holds"
                        ));
                    }
                    if got[row * k + l] != sum as i8 {
                        return Err(format!(
                            "the device sum for {label} row {row} column {l} is {} but the \
                             reference gives {sum}",
                            got[row * k + l]
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Runs `tokenminer_noise_apply` twice and returns both results.
    ///
    /// A convenience for the check only — the search writes straight into the buffer the GEMM will
    /// read, so it never pays for the round trip.
    #[cfg(feature = "pearl")]
    #[allow(clippy::too_many_arguments)]
    fn applied_operands(
        &self,
        signal_a: &[i8],
        e_al: &[i8],
        ear: &[[u32; 2]],
        signal_b: &[i8],
        e_br: &[i8],
        e_bl: &[[u32; 2]],
        rows: usize,
        k: usize,
        rank: usize,
    ) -> Result<(Vec<i8>, Vec<i8>), String> {
        let run = |signal: &[i8], dense: &[i8], perm: &[[u32; 2]], what: &str| {
            let signal_dev = self
                .stream
                .clone_htod(signal)
                .map_err(|e| format!("copying the {what} signal failed: {e}"))?;
            let dense_dev = self
                .stream
                .clone_htod(dense)
                .map_err(|e| format!("copying the {what} dense noise failed: {e}"))?;
            let flat: Vec<u32> = perm.iter().flat_map(|pair| pair.iter().copied()).collect();
            let perm_dev = self
                .stream
                .clone_htod(&flat)
                .map_err(|e| format!("uploading the {what} pairs failed: {e}"))?;

            let mut out = self
                .stream
                .alloc_zeros::<i8>(rows * k)
                .map_err(|e| format!("allocating the {what} output failed: {e}"))?;
            self.noise_apply(&signal_dev, &dense_dev, &perm_dev, &mut out, rows, k, rank)?;

            self.stream
                .clone_dtoh(&out)
                .map_err(|e| format!("reading the {what} output back failed: {e}"))
        };

        Ok((
            run(signal_a, e_al, ear, "A")?,
            run(signal_b, e_br, e_bl, "B")?,
        ))
    }

    /// Checks the search kernel's accept decision against the reference path, at the boundary.
    ///
    /// This is the one check that cannot be satisfied by comparing digests, because what is being
    /// verified is a *comparison*. The digest is computed on the device too, so the check runs the
    /// whole tail of the pipeline — fold, hash, compare — and compares the accept/reject decision
    /// against `U256::from_little_endian(digest) <= bound`, which is what
    /// `sanity_checks::check_jackpot_against_nbits` means.
    ///
    /// The boundary cases are the point. A word order that walks from the wrong end passes every
    /// ordinary candidate and fails only where the leading words are equal — which is exactly where
    /// high-difficulty candidates live. So the bound is set to the true digest, to the digest plus
    /// one, to the digest less one, and to one whose leading group alone is lowered, and each has to
    /// come out the way integer arithmetic says: accept, accept, reject, reject.
    #[cfg(feature = "pearl")]
    fn check_bound(&self) -> Result<(), String> {
        use zk_pow::api::proof::{IncompleteBlockHeader, SeedDerivation};
        use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
        use zk_pow::circuit::chip::compute_jackpot;
        use zk_pow::circuit::pearl_noise::compute_noise;
        use zk_pow::ffi::mine::try_mine_one;

        // A 16x16 tile at the smallest accepted size, because that is the only shape the
        // tensor-core search handles and the check should not pay for the shipped default.
        let mining = small_mining();
        let config = mining_configuration(&mining)?;
        let header = IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x11; 32],
            merkle_root: [0x22; 32],
            timestamp: 1_700_000_000,
            nbits: 0x207f_ffff,
        };

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
        .map_err(|e| format!("producing a proof for the bound check failed: {e}"))?
        .ok_or("the saturated bound should have yielded a proof on the first tile")?;

        let (private, public) = proof
            .parse_proof(header, SeedDerivation::Salted)
            .map_err(|e| format!("parsing the proof failed: {e}"))?;

        let compiled = CompiledPublicParams::from(&public);
        let noise = compute_noise(&compiled);
        let want = compute_jackpot(&compiled, &private.s_a, &private.s_b, &noise);

        let h = private.s_a.len();
        let w = private.s_b.len();
        let k = private.s_a.first().map(Vec::len).unwrap_or(0);
        let rank = public.mining_config.rank as usize;

        let a_sum = sum_operands(&flatten_i8(&private.s_a), &flatten_i8(&noise.a), h, k, "A")?;
        let b_sum = sum_operands(&flatten_i8(&private.s_b), &flatten_i8(&noise.b), w, k, "B")?;
        let key = compiled.a_noise_seed();

        // What the pool would compute for this tile, and the reading of it the pool compares against:
        // `sanity_checks::check_jackpot_against_nbits` applies `U256::from_little_endian`, so the
        // digest's first byte is its *least* significant one. Big-endian here would still be a real
        // digest and would still pass the boundary cases below against a kernel that agrees with it —
        // it just would not be the number the verifier holds.
        let digest = compute_jackpot_hash(&want, key);
        let as_u256 = U256::from_little_endian(&digest);

        // The rule is `digest <= bound`, so the rejects are the bounds *below* the digest and the
        // accepts are the ones at or above it. Equal and one above accept; one below and one whose
        // *leading* group of four bytes is smaller while every group below it is held equal must
        // reject.
        //
        // The last case is the one that pins the walk direction. Subtracting `2^224` lowers the
        // leading group, leaving groups 1 through 7 untouched; a comparison that starts anywhere but
        // group 0 sees only equality and accepts. Subtracting one does the mirror job at the other
        // end, and the two together rule out a reversed walk as well as a truncated one. Merely
        // nudging the bound by one is not enough on its own: the digest's leading groups are
        // arbitrary, so a walk that ignored them would pass it about half the time.
        //
        // Neither case pins the *order* within a group, though, and that is what a reversed index
        // gets wrong: a kernel that reversed each digest word but not the index walks the bound's
        // most significant group against the digest's least significant one, which passes all four
        // of these while disagreeing with the verifier on roughly every candidate. The re-keying
        // sweep at the end of this check is the other half.
        let cases = [
            ("equal", as_u256, true),
            ("one above", as_u256 + U256::one(), true),
            ("one below", as_u256 - U256::one(), false),
            (
                "leading group lowered",
                as_u256 - U256([0, 0, 0, 1 << 32]),
                false,
            ),
        ];

        for (label, bound, should_accept) in cases {
            let won = self.run_search_tile(&a_sum, &b_sum, &key, bound, k, rank)?;
            if won != should_accept {
                return Err(format!(
                    "a bound {label} the digest ({digest:x?}) accepted a candidate the reference \
                     says should {}",
                    if should_accept { "accept" } else { "reject" }
                ));
            }
        }

        // And the widest possible bound must accept, since it is the saturated one the search uses
        // when the pool hands out an easy target.
        let won = self.run_search_tile(&a_sum, &b_sum, &key, U256::MAX, k, rank)?;
        if !won {
            return Err("the device rejected a candidate that clears the maximum bound".into());
        }

        // Which end of the digest is which. The cases above hold the bound a step away from the
        // digest in *its own* reading, so they cannot tell a correct compare from one that reads
        // the digest the other way round — under the wrong reading every bound there is still just
        // "equal", "one away" or "one group away" from a number the device also derives.
        //
        // So take the same tile and re-key it. The transcript is unchanged, the digest is entirely
        // new, and its two readings land on unrelated numbers. One bound then separates them: the
        // smaller of the two, which a device reading little-endian accepts when it is `le` and
        // rejects when it is `be`, and a device reading big-endian always rejects. Both halves of
        // the compare are pinned by the pair — which is the half the boundary cases cannot reach.
        //
        // The tile above is drawn from `rand::rng()`, so *which* way each of these digests reads is
        // a fresh coin flip on every run, and a sweep of eight has a one-in-128 chance of landing
        // entirely on one side. That is not an acceptable failure mode here: `self_test` runs from
        // `pearl::start`, so a check that trips one run in a hundred means the engine refuses to
        // mine, on correct code, for no reason a user could act on. Sixty-four rounds puts the
        // chance below 2^-63 — short of this by a wide margin would reintroduce the flake, so widen
        // it rather than trim it.
        const READING_ROUNDS: u8 = 64;
        let mut saw_be_above = false;
        let mut saw_be_below = false;
        for round in 0..READING_ROUNDS {
            let mut reading_key = key;
            reading_key[0] = round;
            let other = compute_jackpot_hash(&want, reading_key);
            let other_le = U256::from_little_endian(&other);
            let other_be = U256::from_big_endian(&other);
            if other_le == other_be {
                continue;
            }

            // (`bound`, whether the verifier's reading accepts it, which side this digest was on).
            let (bound, should_accept, seen) = if other_be > other_le {
                (other_le, true, &mut saw_be_above)
            } else {
                (other_be, false, &mut saw_be_below)
            };
            *seen = true;

            if self.run_search_tile(&a_sum, &b_sum, &reading_key, bound, k, rank)? != should_accept
            {
                return Err(format!(
                    "the digest ({other:x?}) reads {other_le:#x} little-endian and {other_be:#x} \
                     big-endian, and a bound of {bound:#x} should {}, but the device says the \
                     other; it is not reading the digest in the order the verifier compares it in",
                    if should_accept { "accept" } else { "reject" }
                ));
            }
        }
        if !(saw_be_above && saw_be_below) {
            return Err(format!(
                "{READING_ROUNDS} keys did not produce digests on both sides (big-endian above: \
                 {}, below: {}), \
                 so the digest byte order is only checked in one direction",
                saw_be_above, saw_be_below
            ));
        }

        Ok(())
    }

    /// One `search_tile` launch with freshly zeroed `found` / `coord` buffers, for the checks.
    #[cfg(feature = "pearl")]
    fn run_search_tile(
        &self,
        a_sum: &[i8],
        b_sum: &[i8],
        key: &[u8; 32],
        bound: U256,
        k: usize,
        rank: usize,
    ) -> Result<bool, String> {
        let a_dev = self
            .stream
            .clone_htod(a_sum)
            .map_err(|e| format!("copying the A operand failed: {e}"))?;
        let b_dev = self
            .stream
            .clone_htod(b_sum)
            .map_err(|e| format!("copying the B operand failed: {e}"))?;

        // One region, so `found` and `coord` are a single word and two words. `search_grid` clears
        // the flag itself; these are fresh allocations anyway.
        let mut found = self
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| format!("allocating found failed: {e}"))?;
        let mut coord = self
            .stream
            .alloc_zeros::<u32>(2)
            .map_err(|e| format!("allocating coord failed: {e}"))?;

        self.search_tile(&a_dev, &b_dev, key, bound, &mut found, &mut coord, k, rank)
    }

    /// The tile grid the search actually runs, over more than one block.
    ///
    /// [`Self::check_bound`] pins the fold, the hash and the compare, but it launches a single block,
    /// so the arithmetic that turns a linear tile index into a row, a column and the sixteen rows
    /// and sixteen columns that block reads is covered only if something launches more than one. An
    /// off-by-one there is the worst kind of wrong: every digest stays well-formed and every one of
    /// them clears the bound the pool checks, so the share still verifies — it just describes a
    /// different candidate than the coordinates say, and nothing downstream can tell.
    #[cfg(feature = "pearl")]
    fn check_search_grid(&self) -> Result<(), String> {
        // Three tile rows by two, and a region per tile row, so the regions and the tile rows are
        // different partitions of the same grid. A region size that lined up with the tile row would
        // let a swap between the two go unnoticed.
        //
        // `k` and `rank` are the smallest the search kernel accepts: it stages `pow_bk(arch)` k at a time
        // and folds once per rank, so a `rank` that is not a multiple of the staged width never folds
        // at all and every transcript comes back zero. The staged width is 64 on the tensor path and
        // 32 on Turing's DP4A fold, so 64 is the width both accept — and `k = 128, rank = 64` still
        // gives two rank blocks, so the fold's running-totals behaviour is exercised and not just its
        // first iteration.
        //
        // A 3x2 grid is also short on both axes of the kernel's 16x8 block rectangle, so this runs the
        // short-rectangle paths on every invocation.
        let (m_tiles, n_tiles, k, rank) = (3usize, 2usize, 128usize, 64usize);
        let tile = 16usize;
        let tiles_per_region = n_tiles;
        let tiles = m_tiles * n_tiles;

        // In [-64, 63]: the range the noise-applied operands really occupy, so the fold sees inputs
        // the search could produce rather than a degenerate constant.
        //
        // A multiplicative mix, not `i * 37`. With a row stride of `k` a tile's first index is a
        // multiple of `16 * k`, and `1024 * 37` is a multiple of 128, so a plain linear generator
        // hands every tile byte-identical operands: every digest ties, every region wins, and the
        // check cannot tell which tile the grid picked.
        let signal = |i: usize, salt: u64| -> i8 {
            let mixed = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(salt);
            (((mixed >> 33) & 0x7F) as i8) - 64
        };

        // Not a constant: a key of all one bytes would still exercise every line, but a digest that
        // came out unchanged across a wrong key would be easier to miss.
        let key = [0x5a; 32];

        // Each tile's digest, computed one tile at a time through the checked fold. The grid then has
        // to find the same tile the per-tile path priced, which is the whole claim.
        let digest_of = |a: &[i8], b: &[i8], tile_row: usize, tile_col: usize| -> Result<U256, String> {
            let tile_a = &a[tile_row * tile * k..][..tile * k];
            let tile_b = &b[tile_col * tile * k..][..tile * k];
            let words = self.jackpot_fold(tile_a, tile_b, k, rank, tile, tile)?;
            Ok(U256::from_little_endian(
                &self.jackpot_hash(&transcript_bytes(&words), &key)?,
            ))
        };

        // **The operands are searched for, not pinned.** The offset check at the bottom of this
        // function is only meaningful when the grid's lowest digest lands off tile row 0, since row 0
        // is where a batch that ignored `first_tile` would look anyway. That depends on the digests,
        // so it depends on `k`, `rank` and the staged width -- which is to say it moves whenever the
        // geometry does, and a salt chosen once goes stale silently: the check stops testing the
        // offset and starts testing nothing, while still reading as a pass.
        //
        // So try salts until one puts the lowest candidate somewhere other than row 0. Which tile wins
        // is a hash, so this is a coin flip per attempt and a handful of tries is ample; the whole
        // loop is a few folds of a `k = 128` tile, and it runs on start-up only.
        //
        // The salt has to reach `signal`'s *sampled* bits to do anything: it takes 33..40 of
        // `i * GOLDEN + salt`, so adding a small number to the salt perturbs only the low bits and
        // leaves every output byte identical. The attempt is therefore folded in through a multiply,
        // which moves it across the whole word. A first attempt at this searched `11 + 2 * attempt`
        // and all 32 tries produced the same digests -- which is how the loop below was found to be
        // inert at all, since 32 failures at two-in-three odds is not something a working search does.
        let mix = |base: u64, attempt: u64| -> u64 {
            base.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(attempt.wrapping_mul(0xD6E8_FEB8_6659_FD93))
        };
        let mut chosen: Option<(Vec<i8>, Vec<i8>, Vec<((usize, usize), U256)>)> = None;
        for attempt in 0..32u64 {
            let a = (0..m_tiles * tile * k)
                .map(|i| signal(i, mix(11, attempt)))
                .collect::<Vec<i8>>();
            let b = (0..n_tiles * tile * k)
                .map(|i| signal(i, mix(29, attempt)))
                .collect::<Vec<i8>>();

            let mut priced: Vec<((usize, usize), U256)> = Vec::with_capacity(tiles);
            for tile_row in 0..m_tiles {
                for tile_col in 0..n_tiles {
                    priced.push(((tile_row, tile_col), digest_of(&a, &b, tile_row, tile_col)?));
                }
            }

            let lowest = priced
                .iter()
                .min_by_key(|(_, digest)| *digest)
                .ok_or("the tile grid produced no candidates to search")?;
            if lowest.0 .0 != 0 {
                chosen = Some((a, b, priced));
                break;
            }
        }
        let (a, b, priced) = chosen.ok_or_else(|| {
            format!(
                "no salt out of 32 put the tile grid's lowest candidate off tile row 0 in a \
                 {m_tiles}x{n_tiles} grid, so the batch-offset check below cannot be given \
                 operands it can distinguish"
            )
        })?;

        let a_dev = self
            .stream
            .clone_htod(&a)
            .map_err(|e| format!("copying A failed: {e}"))?;
        let b_dev = self
            .stream
            .clone_htod(&b)
            .map_err(|e| format!("copying B failed: {e}"))?;

        let regions = tiles / tiles_per_region;
        let mut found = self
            .stream
            .alloc_zeros::<u32>(regions)
            .map_err(|e| format!("allocating found failed: {e}"))?;
        let mut coord = self
            .stream
            .alloc_zeros::<u32>(regions * 2)
            .map_err(|e| format!("allocating coord failed: {e}"))?;
        let read_back = |found: &CudaSlice<u32>,
                         coord: &CudaSlice<u32>|
         -> Result<(Vec<u32>, Vec<u32>), String> {
            let mut flags = vec![0u32; found.len()];
            let mut coords = vec![0u32; coord.len()];
            self.stream
                .memcpy_dtoh(found, &mut flags[..])
                .map_err(|e| format!("reading the found flags back failed: {e}"))?;
            self.stream
                .memcpy_dtoh(coord, &mut coords[..])
                .map_err(|e| format!("reading the coordinates back failed: {e}"))?;
            Ok((flags, coords))
        };

        // Two runs, because there are two separate claims to pin and only one of them can be tested
        // by making a single tile qualify.
        //
        // The grid claims whichever qualifying tile reaches the `atomicCAS` first, so a bound can only
        // ever isolate the tile with the *lowest* digest — every tile below that bound qualifies too,
        // and the winner among them is a race. That is enough to pin how the tile grid is cut into
        // regions, since a region is a whole tile row and the minimum's row has a known index. It is
        // not enough to pin every tile's coordinates, so the second run puts one tile in each region
        // and lets them all qualify at once: then each region's report is about a tile the host
        // already knows the index of.
        let lowest = priced
            .iter()
            .min_by_key(|(_, digest)| *digest)
            .ok_or("the tile grid produced no candidates to search")?;
        let ((min_row, min_col), min_digest) = *lowest;

        let won = self.search_grid(
            &a_dev,
            &b_dev,
            &key,
            min_digest,
            &mut found,
            &mut coord,
            None,
            0,
            tiles,
            n_tiles,
            tiles_per_region,
            k,
            rank,
        )?;
        if won != 1 {
            return Err(format!(
                "a bound at the lowest tile ({min_row}, {min_col})'s digest won {won} of {regions} \
                 regions, not 1"
            ));
        }
        let (flags, coords) = read_back(&found, &coord)?;
        let claiming = flags.iter().position(|f| *f != 0).unwrap_or(0);
        let want_region = min_row * n_tiles / tiles_per_region;
        if claiming != want_region {
            return Err(format!(
                "region {claiming} claimed tile ({min_row}, {min_col}), which is in region \
                 {want_region}"
            ));
        }
        let got = (
            coords[claiming * 2] as usize,
            coords[claiming * 2 + 1] as usize,
        );
        if got != (min_row, min_col) {
            return Err(format!(
                "the grid reported tile {got:?} for the lowest candidate, priced at \
                 ({min_row}, {min_col})"
            ));
        }

        // One tile per region, everything qualifying. Every tile is now the only tile in its region,
        // so the coordinates each region reports are its own — which pins the linear-tile-to-row-and-
        // column arithmetic for the whole grid rather than for one lucky corner of it.
        let mut solo_found = self
            .stream
            .alloc_zeros::<u32>(tiles)
            .map_err(|e| format!("allocating found failed: {e}"))?;
        let mut solo_coord = self
            .stream
            .alloc_zeros::<u32>(tiles * 2)
            .map_err(|e| format!("allocating coord failed: {e}"))?;
        let won = self.search_grid(
            &a_dev,
            &b_dev,
            &key,
            U256::MAX,
            &mut solo_found,
            &mut solo_coord,
            None,
            0,
            tiles,
            n_tiles,
            1,
            k,
            rank,
        )?;
        if won != tiles {
            return Err(format!(
                "with every tile in its own region and an unbounded target, {won} of {tiles} \
                 claimed"
            ));
        }
        let (flags, coords) = read_back(&solo_found, &solo_coord)?;
        for ((tile_row, tile_col), _) in &priced {
            let tile = tile_row * n_tiles + tile_col;
            if flags[tile] == 0 {
                return Err(format!(
                    "tile ({tile_row}, {tile_col}) did not claim its own region"
                ));
            }
            let got = (coords[tile * 2] as usize, coords[tile * 2 + 1] as usize);
            if got != (*tile_row, *tile_col) {
                return Err(format!(
                    "region {tile} reported tile {got:?} for the candidate at ({tile_row}, \
                     {tile_col})"
                ));
            }
        }

        // And a bound nothing reaches leaves every region untouched, so the flags above mean "this
        // candidate cleared" rather than "something ran". Zero is beaten only by a digest of exactly
        // zero, which is not a thing BLAKE3 produces.
        let won = self.search_grid(
            &a_dev,
            &b_dev,
            &key,
            U256::zero(),
            &mut found,
            &mut coord,
            None,
            0,
            tiles,
            n_tiles,
            tiles_per_region,
            k,
            rank,
        )?;
        if won != 0 {
            return Err(format!(
                "an unreachable bound won {won} of {regions} regions, so the flags do not mean what \
                 the check assumes"
            ));
        }

        // And the batch offset, which none of the three runs above can see: all three start at the top
        // of the grid. A kernel that ignored `first_tile` would resolve a linear tile index to a row
        // the batch never reached while indexing the operands by that same row — re-searching the
        // first row on every batch and reporting entirely plausible coordinates for it.
        //
        // Catching that needs the batch pointed somewhere an offset-ignoring kernel would not look,
        // and a bound that nothing else clears. Both come from one tile: the grid's lowest digest,
        // which is unique, and lives in some row R. A kernel that searched row 0 instead finds no
        // qualifying tile there and claims nothing. The flip side is that this is blind when R is 0,
        // because then the offset-ignoring kernel finds the same tile and reports the same coordinate
        // for the wrong reason — so that case is ruled out above, by choosing operands that do not
        // put the winner there, rather than by failing here and asking for a hand edit.
        let ((target_row, target_col), target_digest) = *priced
            .iter()
            .min_by_key(|(_, digest)| *digest)
            .ok_or("the tile grid produced no candidates to search")?;
        // One tile row per region, so the batch is exactly the row it was pointed at.
        let won = self.search_grid(
            &a_dev,
            &b_dev,
            &key,
            target_digest,
            &mut found,
            &mut coord,
            None,
            target_row * n_tiles,
            n_tiles,
            n_tiles,
            n_tiles,
            k,
            rank,
        )?;
        if won != 1 {
            return Err(format!(
                "a batch of one tile row pointed at row {target_row} claimed {won} regions for the \
                 grid's lowest candidate, which is in that row alone"
            ));
        }
        let (flags, coords) = read_back(&found, &coord)?;
        let claiming = flags.iter().position(|f| *f != 0).unwrap_or(0);
        let got = (
            coords[claiming * 2] as usize,
            coords[claiming * 2 + 1] as usize,
        );
        if got != (target_row, target_col) {
            return Err(format!(
                "a batch of one tile row pointed at row {target_row} reported tile {got:?}, which is \
                 not in the row it was pointed at"
            ));
        }

        Ok(())
    }

    /// The search kernel's fold, against the reference fold, tile by tile.
    ///
    /// `check_jackpot` pins `tokenminer_jackpot_fold` against `compute_jackpot`, and `check_search_grid`
    /// pins the search kernel's *digests* against that same fold. Between them the value is covered --
    /// but only through a BLAKE3 of it, on one 3x2 grid, so a failure reports "the search found the
    /// wrong tile" with nothing to say about which word of which transcript diverged.
    ///
    /// This compares the sixteen words directly, across grid shapes chosen to be ragged against the
    /// kernel's block rectangle. That coverage is not decorative: a wrong staging bound, a short
    /// rectangle and a missed ceiling division all fold most of a grid correctly and only diverge on
    /// the tiles past the ragged edge, which a digest comparison at the grid's lowest candidate would
    /// never reach.
    ///
    /// Read back through the kernel's own `transcript_out`, so there is no second implementation of
    /// the grid's indexing to disagree about — the tiles are addressed the way the kernel addresses
    /// them, and what is being compared is the arithmetic.
    #[cfg(feature = "pearl")]
    fn check_fold(&self) -> Result<(), String> {
        // Off a block multiple on at least one axis, and `rank` a whole number of the staged k so the
        // fold fires more than once. The last shape is a batch that does not start at the top of the
        // grid, which is the only one that can see an operand row read from the wrong place.
        //
        // `rank` is 64 rather than 32 because the staged k is architecture-dependent — 32 on Turing's
        // DP4A fold, 64 on the tensor path (see `pow_bk`) — and a `rank` below the staged width is
        // rejected outright rather than mis-folded. 64 is the width both paths accept, so these
        // shapes check the same thing on every card. Production runs at `rank = 256`.
        let shapes: [(usize, usize, usize, usize, usize); 4] = [
            (11, 6, 128, 64, 0),
            (16, 16, 128, 64, 0),
            (19, 3, 256, 128, 0),
            (24, 6, 256, 128, 16),
        ];

        // A multiplicative mix, not `i * 37`: with a row stride of `k` a tile's first index is a
        // multiple of `16 * k`, and `1024 * 37` is a multiple of 128, so a plain linear generator hands
        // every tile byte-identical operands — every transcript ties and the comparison proves nothing.
        let signal = |i: usize, salt: u64| -> i8 {
            let mixed = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(salt);
            (((mixed >> 33) & 0x7F) as i8) - 64
        };

        for (m_tiles, n_tiles, k, rank, first_row) in shapes {
            let tile = 16usize;
            let a = (0..(first_row + m_tiles) * tile * k)
                .map(|i| signal(i, 11))
                .collect::<Vec<i8>>();
            let b = (0..n_tiles * tile * k)
                .map(|i| signal(i, 29))
                .collect::<Vec<i8>>();

            let a_dev = self
                .stream
                .clone_htod(&a)
                .map_err(|e| format!("copying A failed: {e}"))?;
            let b_dev = self
                .stream
                .clone_htod(&b)
                .map_err(|e| format!("copying B failed: {e}"))?;

            // What the reference fold says each tile's transcript is.
            let mut want = vec![0u32; m_tiles * n_tiles * Self::POW_TRANSCRIPT_WORDS];
            for tile_row in 0..m_tiles {
                for tile_col in 0..n_tiles {
                    let tile_a = &a[(first_row + tile_row) * tile * k..][..tile * k];
                    let tile_b = &b[tile_col * tile * k..][..tile * k];
                    let words = self.jackpot_fold(tile_a, tile_b, k, rank, tile, tile)?;
                    let at = (tile_row * n_tiles + tile_col) * Self::POW_TRANSCRIPT_WORDS;
                    want[at..at + Self::POW_TRANSCRIPT_WORDS].copy_from_slice(&words);
                }
            }

            let rows_in_batch = m_tiles;
            let cols_per_strip = n_tiles.div_ceil(Self::POW_BLOCK_COLS);
            let blocks = rows_in_batch.div_ceil(Self::POW_BLOCK_ROWS) * cols_per_strip;
            let words_per_block = Self::POW_TILES_PER_BLOCK * Self::POW_TRANSCRIPT_WORDS;

            // One region for the whole batch. The bound is zero, which no digest clears, so nothing
            // claims and nothing can early-out: every tile is folded.
            let mut found = self
                .stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| format!("allocating found failed: {e}"))?;
            let mut coord = self
                .stream
                .alloc_zeros::<u32>(2)
                .map_err(|e| format!("allocating coord failed: {e}"))?;
            let mut transcripts = self
                .stream
                .alloc_zeros::<u32>(blocks * words_per_block)
                .map_err(|e| format!("allocating transcripts failed: {e}"))?;

            // Poisoned, so a word the kernel never writes is distinguishable from one it wrote as
            // zero — without this, a tile that is never reached reads as sixteen correct zeroes.
            let poison = vec![0xcccc_ccccu32; blocks * words_per_block];
            self.stream
                .memcpy_htod(&poison, &mut transcripts)
                .map_err(|e| format!("poisoning the transcript buffer failed: {e}"))?;

            let key = [0x5au8; 32];
            self.search_grid(
                &a_dev,
                &b_dev,
                &key,
                U256::zero(),
                &mut found,
                &mut coord,
                Some(&mut transcripts),
                first_row * n_tiles,
                m_tiles * n_tiles,
                n_tiles,
                // One region, so no tile's claim can retire another block's work mid-fold.
                m_tiles * n_tiles,
                k,
                rank,
            )?;

            let mut got = vec![0u32; blocks * words_per_block];
            self.stream
                .memcpy_dtoh(&transcripts, &mut got)
                .map_err(|e| format!("reading the transcripts back failed: {e}"))?;

            // The block writes its rectangle row-major, and `blockIdx.x` runs strips of tile rows
            // outer and column blocks inner — the same order the kernel maps them in.
            let mut checked = 0usize;
            for strip in 0..rows_in_batch.div_ceil(Self::POW_BLOCK_ROWS) {
                for col_block in 0..cols_per_strip {
                    let block = strip * cols_per_strip + col_block;
                    for r in 0..Self::POW_BLOCK_ROWS {
                        let tile_row = strip * Self::POW_BLOCK_ROWS + r;
                        if tile_row >= rows_in_batch {
                            continue;
                        }
                        for c in 0..Self::POW_BLOCK_COLS {
                            let tile_col = col_block * Self::POW_BLOCK_COLS + c;
                            if tile_col >= n_tiles {
                                continue;
                            }
                            let from = block * words_per_block
                                + (r * Self::POW_BLOCK_COLS + c) * Self::POW_TRANSCRIPT_WORDS;
                            let at = (tile_row * n_tiles + tile_col) * Self::POW_TRANSCRIPT_WORDS;
                            for w in 0..Self::POW_TRANSCRIPT_WORDS {
                                if got[from + w] != want[at + w] {
                                    return Err(format!(
                                        "a {m_tiles}x{n_tiles} grid at tile row {first_row} folded \
                                         tile ({tile_row}, {tile_col}) word {w} as {:08x}, and the \
                                         reference fold says {:08x}",
                                        got[from + w], want[at + w]
                                    ));
                                }
                            }
                            checked += 1;
                        }
                    }
                }
            }
            if checked != m_tiles * n_tiles {
                return Err(format!(
                    "compared {checked} of {} tiles in a {m_tiles}x{n_tiles} grid",
                    m_tiles * n_tiles
                ));
            }
        }

        Ok(())
    }

    /// Checks the commitment chain against `PublicProofParams::commitment_hash`.
    ///
    /// Both derivations, because the certificate version chooses between them and comes off the
    /// wire: under `Salted` the two roots are salted with their row and column counts before the
    /// chain runs, and getting that wrong produces a self-consistent miner that submits every share
    /// late. The reference chain is `compute_commitment_hash_with_offsets` from `mine.rs`, which is
    /// private, so it is re-derived here from the same public pieces it uses.
    #[cfg(feature = "pearl")]
    fn check_commitment(&self) -> Result<(), String> {
        use zk_pow::api::proof::SeedDerivation;

        // Shapes chosen to cross a chunk boundary in both matrices, since the commitment is the
        // tree over the padded bytes and not the hash of the bare matrix.
        let (m, n, k) = (200usize, 96usize, 640usize);

        // Entries in [-64, 63], the range the miner's own matrices use: the commitment is taken
        // over real signal values, not over anything the search could not produce.
        let matrix = |rows: usize, cols: usize, salt: usize| -> Vec<i8> {
            (0..rows * cols)
                .map(|i| ((i * 31 + salt * 7) % 128) as i8 - 64)
                .collect()
        };
        let a = matrix(m, k, 0);
        let bt = matrix(n, k, 1);

        // What the proof carries is the chunk-padded matrix, not the matrix.
        let padded_a = pad_to_chunk(&a);
        let padded_bt = pad_to_chunk(&bt);

        let a_dev = self
            .stream
            .clone_htod(&padded_a)
            .map_err(|e| format!("copying A failed: {e}"))?;
        let bt_dev = self
            .stream
            .clone_htod(&padded_bt)
            .map_err(|e| format!("copying B failed: {e}"))?;

        let job_key = [0x33u8; 32];

        for derivation in [SeedDerivation::Legacy, SeedDerivation::Salted] {
            // The reference chain: raw roots, salted or not, then two unkeyed hashes.
            let raw_a = pearl_blake3::blake3_digest(&padded_a, Some(job_key));
            let raw_b = pearl_blake3::blake3_digest(&padded_bt, Some(job_key));
            let (hash_a, hash_b) = derivation.bind_roots(&raw_a, &raw_b, m as u32, n as u32);

            let mut chain = Vec::with_capacity(64);
            chain.extend_from_slice(&job_key);
            chain.extend_from_slice(&hash_b);
            let b_noise_seed = pearl_blake3::blake3_digest(&chain, None);

            let mut chain = Vec::with_capacity(64);
            chain.extend_from_slice(&b_noise_seed);
            chain.extend_from_slice(&hash_a);
            let a_noise_seed = pearl_blake3::blake3_digest(&chain, None);

            let (got_b, got_a) = self
                .commitment_seeds(&a_dev, &bt_dev, job_key, m as u32, n as u32, derivation)
                .map_err(|error| format!("{derivation:?}: {error}"))?;

            if got_b != b_noise_seed || got_a != a_noise_seed {
                return Err(format!(
                    "{derivation:?}: the device chain gives B {got_b:?} and A {got_a:?}, but the \
                     reference gives B {b_noise_seed:?} and A {a_noise_seed:?}"
                ));
            }
        }

        Ok(())
    }

    /// The noise seeds for a pair of device-resident matrices, all on the device.
    ///
    /// Same four hashes as `compute_commitment_hash_with_offsets`, on matrices that never leave the
    /// device. The two 32-byte roots and the two 32-byte seeds do cross back, because they are
    /// kernel arguments to the noise generators and 128 bytes per attempt is not worth a bespoke
    /// orchestration kernel; every byte that actually *is* the commitment is hashed where it lives.
    ///
    /// Generic over the element type because the search's matrices are `i8` and the commitment is
    /// over their bytes: a `CudaSlice<i8>` of `M * K` elements hashes as `M * K` *bytes*, which is
    /// only true while `T` is one byte wide. The buffers are allocated chunk-padded, so the
    /// element count the buffer carries is already the padded byte length the reference hashes —
    /// which is the padding the commitment is taken over, not a copy of the matrix.
    #[cfg(feature = "pearl")]
    pub fn commitment_seeds<T>(
        &self,
        a: &CudaSlice<T>,
        bt: &CudaSlice<T>,
        job_key: [u8; 32],
        m: u32,
        n: u32,
        derivation: zk_pow::api::proof::SeedDerivation,
    ) -> Result<([u8; 32], [u8; 32]), String> {
        // The buffer length doubles as the byte length below, which only means the same thing for a
        // one-byte element. For anything wider it would hash a quarter of the matrix and return a
        // root that looks entirely plausible.
        if std::mem::size_of::<T>() != 1 {
            return Err(format!(
                "a commitment is over the bytes of a matrix, so its buffer has to be one byte per \
                 element, but this one is {} bytes",
                std::mem::size_of::<T>()
            ));
        }

        let raw_a = self.blake3_device_bytes(a, a.len(), Some(job_key))?;
        let raw_b = self.blake3_device_bytes(bt, bt.len(), Some(job_key))?;

        let (hash_a, hash_b) = derivation.bind_roots(&raw_a, &raw_b, m, n);

        let mut chain = Vec::with_capacity(64);
        chain.extend_from_slice(&job_key);
        chain.extend_from_slice(&hash_b);
        let b_noise_seed = self.blake3(&chain, None)?;

        let mut chain = Vec::with_capacity(64);
        chain.extend_from_slice(&b_noise_seed);
        chain.extend_from_slice(&hash_a);
        let a_noise_seed = self.blake3(&chain, None)?;

        Ok((b_noise_seed, a_noise_seed))
    }
}

/// The largest dynamic shared-memory request a launch may make without opting in.
const MAX_DYNAMIC_SHARED_BYTES: usize = 48 * 1024;

/// Checks the job key against `PublicProofParams::job_key`.
///
/// Free-standing rather than a backend method because it touches no device state: it is four bytes
/// of framing deciding which matrix the noise comes from, and being wrong still produces a
/// beautifully self-consistent miner — every digest a real BLAKE3 of a real transcript, every one of
/// them a candidate the pool rejects. Being pure host code also means it runs on a machine with no
/// GPU instead of skipping with the rest of them.
///
/// `new_dummy` supplies only the header and configuration `job_key` reads, so the reference here is
/// the same function the verifier calls on a parsed proof.
#[cfg(feature = "pearl")]
fn check_job_key() -> Result<(), String> {
    use zk_pow::api::proof::{
        IncompleteBlockHeader, MMAType, MiningConfiguration, PublicProofParams, SeedDerivation,
    };
    use zk_pow::ffi::plain_proof::list_to_pattern;

    // A real header, so `to_bytes` is exercised on the same 76-byte layout the wire carries.
    const HEADER: &str = "0000002040855504f7a9fc1682784e9b3f1d185a9f2ffb84efa2b81460e2b726ae\
                          77656dd7a5610c81c03527b58bba629e7f58a84a5a1295776120109f9d2bfad0ef7438f9c4c06a04810018";
    let header = IncompleteBlockHeader::from_bytes(
        &hex::decode(HEADER).map_err(|e| format!("the check's header is not hex: {e}"))?,
    )
    .map_err(|e| format!("the check's header does not parse: {e}"))?;

    // The 16x16 pattern the production configuration uses, so the configuration bytes being hashed
    // are a real one rather than a degenerate list.
    let pattern = |indices: Vec<u32>| {
        list_to_pattern(&indices)
            .map(|(pattern, _)| pattern)
            .map_err(|e| e.to_string())
    };
    let config = MiningConfiguration {
        common_dim: 4096,
        rank: 256,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: pattern((0..16u32).collect())?,
        cols_pattern: pattern((0..16u32).collect())?,
        moe: None,
    };

    let want = PublicProofParams::new_dummy(
        header,
        SeedDerivation::Salted,
        config,
        131072,
        131072,
        16,
        16,
    )
    .job_key();

    let got = crate::miners::pearl_mining::job_key(&header, &config);

    if got != want {
        return Err(format!(
            "the job key is {got:?}, but the verifier derives {want:?} from the same header and \
             configuration"
        ));
    }

    Ok(())
}

/// The 64-byte message consensus hashes: the sixteen transcript words as little-endian bytes.
///
/// The same serialisation `tokenminer_search_grid` does before its BLAKE3 call, kept here so a check
/// that predicts a tile's digest cannot quietly disagree with the kernel about what it hashed.
#[cfg(feature = "pearl")]
fn transcript_bytes(jackpot: &[u32; JACKPOT_WORDS]) -> [u8; 64] {
    let mut message = [0u8; 64];
    for (word, chunk) in jackpot.iter().zip(message.chunks_exact_mut(4)) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    message
}

/// Pads to a whole number of 1024-byte chunks, the shape a commitment is taken over.
#[cfg(feature = "pearl")]
fn pad_to_chunk(matrix: &[i8]) -> Vec<u8> {
    pearl_blake3::pad_to_chunk_boundary(&matrix.iter().map(|&v| v as u8).collect::<Vec<u8>>())
}

#[cfg(feature = "pearl")]
fn flatten_i8(rows: &[Vec<i8>]) -> Vec<i8> {
    rows.iter().flatten().copied().collect()
}

#[cfg(feature = "pearl")]
fn pairs(words: &[u32]) -> Vec<[u32; 2]> {
    words
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect()
}

#[cfg(feature = "pearl")]
fn compare_i8(got: &[i8], want: &[i8], what: &str) -> Result<(), String> {
    match got.iter().zip(want).position(|(got, want)| got != want) {
        Some(index) => Err(format!(
            "{what} disagrees at entry {index}: the device has {} but the reference has {}",
            got[index], want[index]
        )),
        None => Ok(()),
    }
}

#[cfg(feature = "pearl")]
fn compare_perm(got: &[[u32; 2]], want: &[[u32; 2]], what: &str) -> Result<(), String> {
    match got.iter().zip(want).position(|(got, want)| got != want) {
        Some(index) => Err(format!(
            "{what} disagrees at entry {index}: the device has {:?} but the reference has {want:?}",
            got[index]
        )),
        None => Ok(()),
    }
}

/// Rejects a noise rank the reference would reject, or the kernels would mis-index.
/// Which of the two jackpot kernels a tile shape goes to.
#[cfg(feature = "pearl")]
#[derive(Clone, Copy)]
enum JackpotPath {
    /// The four-term scalar expansion, for tile shapes `wmma` cannot hold.
    Expanded,
    /// The single summed-operand GEMM, for tiles whose h and w are multiples of 16.
    Fold,
}

#[cfg(feature = "pearl")]
impl std::fmt::Display for JackpotPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JackpotPath::Expanded => f.write_str("expanded"),
            JackpotPath::Fold => f.write_str("folded"),
        }
    }
}

/// Adds the signal and the noise in `i32` and returns them as the `i8` the tensor cores take.
///
/// Consensus multiplies the sums, and the sums are in `[-127, 126]` — the noise is a difference of
/// two entries of a `[-32, 31]` dense row, so it is at worst 63 either way, against a signal already
/// in `[-64, 63]`. That is what lets this be one INT8 GEMM instead of the four-way expansion, so the
/// bound is checked rather than assumed: an out-of-range sum means the operand ranges above are
/// wrong, and silently truncating would turn that into a wrong jackpot that still passes the
/// comparison against `compute_jackpot` only by accident.
#[cfg(feature = "pearl")]
fn sum_operands(
    signal: &[i8],
    noise: &[i8],
    rows: usize,
    k: usize,
    what: &str,
) -> Result<Vec<i8>, String> {
    if signal.len() != rows * k || noise.len() != rows * k {
        return Err(format!(
            "the summed operand {what} wants {rows}x{k} of each half, got {} and {}",
            signal.len(),
            noise.len()
        ));
    }

    let mut out = Vec::with_capacity(signal.len());
    for (index, (&s, &n)) in signal.iter().zip(noise).enumerate() {
        let sum = s as i32 + n as i32;
        if !(-128..=127).contains(&sum) {
            return Err(format!(
                "{what} signal + noise at entry {index} is {sum}, which does not fit the INT8 \
                 tensor-core operand the fold GEMM needs"
            ));
        }
        out.push(sum as i8);
    }

    Ok(out)
}

/// One byte of the reference Philox-4x32-10 stream at absolute index `index`.
///
/// Transcribed from Random123 rather than from our own kernel, so a transcription slip in the device
/// version does not get mirrored here — the point of a check that both sides came from the same
/// author is nil. Byte `n` depends only on `(n, seed)`, which is the property `check_fill` leans on.
#[cfg(feature = "pearl")]
fn philox_i8(index: usize, seed: u64) -> i8 {
    fn round(ctr: &mut [u32; 4], key: &mut [u32; 2]) {
        let p0 = u64::from(ctr[0]) * 0xD251_1F53;
        let p1 = u64::from(ctr[2]) * 0xCD9E_8D57;
        let next = [
            ((p1 >> 32) as u32) ^ ctr[1] ^ key[0],
            p1 as u32,
            ((p0 >> 32) as u32) ^ ctr[3] ^ key[1],
            p0 as u32,
        ];
        ctr.copy_from_slice(&next);

        let p = u64::from(key[0]) * 0x9E37_79B9;
        let next_key = [((p >> 32) as u32) ^ key[1] ^ 0x9E37_79B9, p as u32];
        key.copy_from_slice(&next_key);
    }

    let idx = index as u64;
    let (lo, hi) = (idx as u32, (idx >> 32) as u32);
    let mut ctr = [lo, hi, hi ^ lo, 0];
    let mut key = [seed as u32, (seed >> 32) as u32];

    for _ in 0..10 {
        round(&mut ctr, &mut key);
    }

    // Lane `index % 4` of the counter — the byte at this index, independent of its neighbours.
    ((ctr[index % 4] & 0x7F) as i32 - 64) as i8
}

/// `numel` bytes of the reference stream starting at element `from`.
///
/// Public within the crate because the search's end-to-end check has to draw the same matrices on
/// the host to know which tile should win.
#[cfg(feature = "pearl")]
pub(crate) fn philox_stream(from: usize, numel: usize, seed: u64) -> Vec<i8> {
    (from..from + numel).map(|i| philox_i8(i, seed)).collect()
}

#[cfg(feature = "pearl")]
fn validate_rank(rank: usize) -> Result<(), String> {
    if rank == 0 || !rank.is_power_of_two() {
        return Err(format!("noise rank {rank} is not a power of two"));
    }
    if !rank.is_multiple_of(NOISE_HASH_LEN) {
        return Err(format!(
            "noise rank {rank} is not a multiple of {NOISE_HASH_LEN}, so it does not divide into \
             whole hashes"
        ));
    }

    Ok(())
}

/// Validates a tile before it reaches the kernel, returning the thread count to launch.
///
/// Kept out of the launch so it can be tested without a CUDA device: every rejection here is a
/// caller bug, and letting one through would surface as an opaque driver error instead.
#[allow(clippy::too_many_arguments)]
fn validate_tile(
    secret_a_len: usize,
    noise_a_len: usize,
    secret_b_len: usize,
    noise_b_len: usize,
    k: usize,
    rank: usize,
    h: usize,
    w: usize,
) -> Result<u32, String> {
    if k == 0 || h == 0 || w == 0 {
        return Err(format!("empty tile: k={k} h={h} w={w}"));
    }
    if rank == 0 || !k.is_multiple_of(rank) {
        return Err(format!("rank {rank} does not divide k {k}"));
    }
    // The noise generator this mirrors needs a rank that is a whole number of BLAKE3 digests.
    if !rank.is_multiple_of(32) {
        return Err(format!("rank {rank} is not a multiple of 32"));
    }

    for (name, len, rows) in [
        ("secret_a", secret_a_len, h),
        ("noise_a", noise_a_len, h),
        ("secret_b", secret_b_len, w),
        ("noise_b", noise_b_len, w),
    ] {
        if len != rows * k {
            return Err(format!(
                "{name} is {len} bytes, expected {} for {rows} rows of {k}",
                rows * k
            ));
        }
    }

    // The block reduction halves the thread count each step, so it must be a power of two.
    let threads = w.next_power_of_two();
    if threads > 1024 {
        return Err(format!(
            "tile width {w} needs {threads} threads, past the 1024 limit"
        ));
    }

    let shared = (h * w + threads) * std::mem::size_of::<u32>();
    if shared > MAX_DYNAMIC_SHARED_BYTES {
        return Err(format!(
            "a {h}x{w} tile needs {shared} bytes of shared memory, over the \
             {MAX_DYNAMIC_SHARED_BYTES} limit"
        ));
    }

    Ok(threads as u32)
}

/// The reference the device must match: `C[row][col] = sum_k A[row][k] * B[k][col]`.
fn cpu_i8_gemm(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for kk in 0..k {
                acc += i32::from(a[row * k + kk]) * i32::from(b[kk * n + col]);
            }
            c[row * n + col] = acc;
        }
    }

    c
}

impl GpuBackend for CudaBackend {
    fn name(&self) -> String {
        format!("cuda/{}", self.arch)
    }

    fn device(&self) -> &DeviceInfo {
        &self.device
    }

    fn self_test(&mut self) -> Result<(), String> {
        self.check_arch()?;
        self.check_i8_dot()?;
        self.check_i8_gemm()?;
        #[cfg(feature = "pearl")]
        self.check_blake3_tree()?;
        #[cfg(feature = "pearl")]
        check_job_key()?;
        #[cfg(feature = "pearl")]
        self.check_commitment()?;
        #[cfg(feature = "pearl")]
        self.check_noise()?;
        #[cfg(feature = "pearl")]
        self.check_fill()?;
        #[cfg(feature = "pearl")]
        self.check_noise_apply()?;
        #[cfg(feature = "pearl")]
        self.check_jackpot()?;
        #[cfg(feature = "pearl")]
        self.check_fold()?;
        #[cfg(feature = "pearl")]
        self.check_bound()?;
        #[cfg(feature = "pearl")]
        self.check_search_grid()?;

        Ok(())
    }

    fn shutdown(&mut self) {
        // Every handle here is refcounted and released on drop, so there is nothing to tear down
        // beyond letting the backend fall out of scope.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_arch_has_a_known_cuda_arch_code() {
        // The self-test would otherwise fail on a device it should support, with a confusing
        // "no expected __CUDA_ARCH__" error.
        for arch in cubins::KERNEL_CUBINS.iter().map(|cubin| cubin.arch) {
            assert!(
                arch_code(arch).is_some(),
                "{arch} has no __CUDA_ARCH__ mapping"
            );
            assert!(
                arch_for(compute_capability_of(arch)).is_some(),
                "{arch} cannot be reached from any compute capability"
            );
        }
    }

    fn compute_capability_of(arch: &str) -> (i32, i32) {
        match arch {
            "sm_75" => (7, 5),
            "sm_86" => (8, 6),
            "sm_89" => (8, 9),
            "sm_90a" => (9, 0),
            "sm_120a" => (12, 0),
            other => panic!("no compute capability for {other}"),
        }
    }

    #[test]
    fn an_unlisted_compute_capability_has_no_arch() {
        assert_eq!(
            arch_for((8, 0)),
            None,
            "sm_80 has no cubin, so it must not be claimed"
        );
        assert_eq!(arch_for((11, 0)), None);
    }

    #[test]
    fn the_reference_gemm_computes_a_hand_checked_case() {
        // Guards the reference itself: if this were wrong, a correct kernel would look broken.
        let a: [i8; 4] = [1, 2, 3, 4];
        let b: [i8; 4] = [5, 6, 7, 8];

        // A is 1x4, B is 4x1 -> C = 1*5 + 2*6 + 3*7 + 4*8 = 70.
        let c = cpu_i8_gemm(&a, &b, 1, 1, 4);
        assert_eq!(c, vec![70]);
    }

    /// The jackpot kernel must reproduce the canonical implementation word for word.
    ///
    /// The comparison lives in the backend's self-test, because a backend that cannot compute the
    /// jackpot correctly must not be allowed to mine; this is the focused entry point for it.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_jackpot_kernel_matches_the_canonical_implementation() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        backend
            .check_jackpot()
            .unwrap_or_else(|error| panic!("the jackpot kernel disagrees with consensus: {error}"));
    }

    /// The search's job key must be the one the verifier derives for itself.
    ///
    /// No device needed, and deliberately so: this is the one link in the chain that nothing else
    /// checks, because a wrong key yields digests that are internally perfect and externally
    /// worthless.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_job_key_is_the_one_the_verifier_derives() {
        check_job_key().unwrap_or_else(|error| panic!("{error}"));
    }

    /// The device BLAKE3 tree must match the reference, because every commitment and every noise
    /// seed is built from it.
    ///
    /// `MerkleTree::root()` is BLAKE3 tree hashing over 1024-byte chunk leaves, and so is
    /// `blake3_digest` over the same padded bytes — the two spellings in the reference agree, and
    /// so must ours. Checked at the leaf counts where the tree changes shape.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_device_blake3_tree_matches_the_reference() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        backend.check_blake3_tree().unwrap_or_else(|error| {
            panic!("the device BLAKE3 disagrees with pearl-blake3: {error}")
        });
    }

    /// The device commitment chain must match the reference, because the noise seeds it produces
    /// are what the noise tensors are hashed under.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_device_commitment_chain_matches_the_reference() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        backend.check_commitment().unwrap_or_else(|error| {
            panic!("the device commitment chain disagrees with consensus: {error}")
        });
    }

    /// The device noise tensors must match `zk_pow::circuit::pearl_noise` exactly.
    ///
    /// A single wrong entry here is a proof the network rejects, and it would not show up as a
    /// crash — only as shares that never get accepted.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_device_noise_matches_the_reference() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        backend
            .check_noise()
            .unwrap_or_else(|error| panic!("the device noise disagrees with consensus: {error}"));
    }

    /// The accept decision must be an integer comparison of the digest against the bound, checked at
    /// the boundary where a reversed word order would first go wrong.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_device_accepts_exactly_the_candidates_that_clear_the_bound() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        backend
            .check_bound()
            .unwrap_or_else(|error| panic!("the device accept decision is wrong: {error}"));
    }

    /// A batch of tiles must resolve each block's row and column, because a wrong one still produces
    /// digests that clear the pool's bound.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_search_grid_claims_the_tile_whose_digest_clears_the_bound() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        backend
            .check_search_grid()
            .unwrap_or_else(|error| panic!("the search grid indexed the wrong tile: {error}"));
    }

    /// The Philox fill must be the reference stream, and a window of it must match a whole-matrix
    /// fill, because the search fills windows and commits to the whole.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_device_fill_is_the_reference_stream() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        backend
            .check_fill()
            .unwrap_or_else(|error| panic!("the device fill is not the reference stream: {error}"));
    }

    /// The summed operand is what consensus multiplies, so it must match the reference noise added
    /// to the signal — and must stay inside INT8, which is what licenses the single-GEMM fold.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_device_noise_apply_matches_the_reference_noise() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        backend.check_noise_apply().unwrap_or_else(|error| {
            panic!("the device noise-apply disagrees with consensus: {error}")
        });
    }

    /// The device BLAKE3 must match the reference, because the pool gates acceptance on it.
    ///
    /// `compute_jackpot_hash` is standard keyed BLAKE3 over the sixteen little-endian jackpot words,
    /// and the message is a single chunk, so this is one compression with the ROOT flag.
    #[cfg(feature = "pearl")]
    #[test]
    fn the_device_blake3_matches_the_reference() {
        let Ok(backend) = CudaBackend::open(0) else {
            return;
        };

        let message: [u8; 64] = std::array::from_fn(|i| (i * 7 + 3) as u8);
        let key: [u8; 32] = std::array::from_fn(|i| (i * 11 + 5) as u8);

        let want = pearl_blake3::blake3_digest(&message, Some(key));
        let got = backend
            .jackpot_hash(&message, &key)
            .expect("the hash kernel runs");

        assert_eq!(got, want, "the device digest must match the reference");
    }

    #[test]
    fn the_expanded_tile_validates_and_reports_its_thread_count() {
        // The shape of the *probe's* four-way expansion path, not the shipped default's. This is
        // `jackpot_expanded`, the self-test that pins the probe against `compute_jackpot`; the
        // search reads summed operands through `pearl.cu` instead and never comes here.
        assert_eq!(
            validate_tile(2 * 2048, 2 * 2048, 64 * 2048, 64 * 2048, 2048, 128, 2, 64),
            Ok(64)
        );
    }

    #[test]
    fn a_tile_width_that_is_not_a_power_of_two_rounds_up() {
        assert_eq!(
            validate_tile(2 * 256, 2 * 256, 48 * 256, 48 * 256, 256, 128, 2, 48),
            Ok(64)
        );
    }

    #[test]
    fn rejects_tiles_the_kernel_cannot_run() {
        // A rank that does not divide k, and one that is not a whole number of BLAKE3 digests.
        assert!(validate_tile(2 * 2048, 2 * 2048, 64 * 2048, 64 * 2048, 2048, 100, 2, 64).is_err());
        assert!(validate_tile(2 * 2048, 2 * 2048, 64 * 2048, 64 * 2048, 2048, 16, 2, 64).is_err());

        // Operands whose length does not match the tile shape.
        assert!(validate_tile(
            2 * 2048 - 1,
            2 * 2048,
            64 * 2048,
            64 * 2048,
            2048,
            128,
            2,
            64
        )
        .is_err());
        assert!(validate_tile(
            2 * 2048,
            2 * 2048,
            64 * 2048 + 7,
            64 * 2048,
            2048,
            128,
            2,
            64
        )
        .is_err());

        // Degenerate shapes.
        assert!(validate_tile(0, 0, 0, 0, 2048, 128, 0, 64).is_err());
        assert!(validate_tile(2048, 2048, 64 * 2048, 64 * 2048, 2048, 128, 1, 0).is_err());

        // A width needing more than one block's worth of threads.
        assert!(validate_tile(2048, 2048, 4096 * 2048, 4096 * 2048, 2048, 128, 1, 4096).is_err());

        // A tile whose cell sums do not fit the shared-memory budget: 64x1024 cells is 256 KB.
        assert!(validate_tile(
            64 * 2048,
            64 * 2048,
            1024 * 2048,
            1024 * 2048,
            2048,
            128,
            64,
            1024
        )
        .is_err());
    }
}
