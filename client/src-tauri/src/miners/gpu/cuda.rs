//! CUDA backend: prebuilt cubins, loaded through the CUDA driver API.
//!
//! cudarc owns the driver plumbing (`dynamic-loading` means we never link against CUDA at build
//! time and need no toolkit at run time). We own architecture selection, because a cubin only loads
//! on the compute capability it was compiled for.
//!
//! What is left here after the split: the probes, the fill, the commitment, and the two jackpot
//! kernels. The search itself is not here — it is the Triton blob launched by
//! [`super::pipeline`], and the hash that used to live inside it is the fatbin postpass.
//!
//! `tokenminer_jackpot_fold` is the only device-side implementation of the *rotating* fold, and the blob
//! is equivalent to it only under the `k = 16r` coincidence — each jackpot slot is written exactly once,
//! so the rotate is a no-op on the zero it starts from. The equivalence is asserted by the postpass gate,
//! with a negative control at a group size where it stops holding. What `check_jackpot` pins is the
//! rotating kernel against the verifier at a tile shape the blob cannot launch (16x16); at that shape
//! `k/r` is 16 as well, so it pins the value rather than the rotate.

use std::sync::Arc;

use cudarc::driver::{
    sys::CUdevice_attribute, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use super::{cubins, DeviceInfo, GpuBackend};
use super::{fatbin, pipeline, triton};

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


/// Thirty-two bytes as eight little-endian words, the form BLAKE3 chains on the device.
#[cfg(feature = "pearl")]
pub(crate) fn to_words(bytes: &[u8; 32]) -> [u32; 8] {
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

    /// The stream every launch runs on.
    ///
    /// Public because the split path's launch layer is a separate module: it drives the fatbin entry
    /// points and the Triton blobs over buffers owned by [`super::bufs`], and those launches have to be
    /// stream-ordered with the fill and the commitment, which happen here. Two streams would let a noise
    /// launch read a matrix the fill had not finished writing.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The context the buffers and the loaded modules belong to.
    ///
    /// Public for the same reason as the stream: the fatbin and the Triton blobs are loaded against a
    /// context, and the buffers the pipeline launches over are allocated on it. A module loaded against
    /// a second context would resolve its symbols against that context's device, and a buffer allocated
    /// there would not be valid in a launch driven from this one.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self._ctx
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

/// Pads to a whole number of 1024-byte chunks, the shape a commitment is taken over.
#[cfg(feature = "pearl")]
fn pad_to_chunk(matrix: &[i8]) -> Vec<u8> {
    pearl_blake3::pad_to_chunk_boundary(&matrix.iter().map(|&v| v as u8).collect::<Vec<u8>>())
}

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
        self.check_fill()?;
        #[cfg(feature = "pearl")]
        self.check_jackpot()?;

        // `check_fold` is gone with the kernel it read: it compared the fused search kernel's transcripts
        // against `tokenminer_jackpot_fold`, and the fused kernel is deleted. Its coverage is now the
        // postpass gate below, which compares the blob's transcripts against the verifier's fold and is the
        // only place that says the blob's no-rotl form is a coincidence of `k = 16r` rather than the general
        // fold — the negative control at a group size where it stops holding is what makes that claim mean
        // something.
        //
        // `tokenminer_jackpot_fold` stays because `check_jackpot` pins it against the verifier at a tile
        // shape the blob cannot represent (16x16). Stated precisely: at that shape `k/r` is also 16, so the
        // rotate is a no-op there too and that check pins the *value*, not the rotate. Nothing on the device
        // currently proves the rotating kernel rotates; the blob and the rotating kernel agree because both
        // agree with the verifier at `k = 16r`, and the disagreement that would separate them is only
        // reachable at `k/r != 16`, which the blob cannot launch.

        // The split path's gates, on every start. Each is a place where a 1:1 port can disagree while
        // still producing output that looks self-consistent: a real noise tensor generated for the wrong
        // side, a real BLAKE3 digest under the wrong key, a valid fold for the wrong candidate. Nothing
        // downstream can tell, so the disagreement has to be caught here rather than in a share the pool
        // rejects.
        //
        // Loading is itself a gate, not a lookup: a symbol that the build did not emit fails here with
        // its name in the message, before any attempt runs.
        let cc = self.context().compute_capability().map_err(|e| format!("{e}"))?;
        let fatbin = fatbin::FatbinKernels::load(self.context(), cc)?;
        let triton = triton::TritonKernels::load(self.context(), cc)?;

        pipeline::check_noise_gen(self.stream(), &fatbin)?;
        pipeline::check_noising(self.stream(), &fatbin, &triton)?;
        pipeline::check_postpass(self.stream(), &fatbin, &triton)?;
        // Separate from the others because it asserts something they cannot: that the grid the launch
        // computes reaches more than the first candidate slot. A shape where every gate passes on tile 0,
        // candidate 0 is a pass that proves nothing about the grid arithmetic.
        pipeline::check_grid_reaches_second_slot(self.stream(), &triton)?;

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
