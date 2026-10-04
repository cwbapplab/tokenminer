//! CUDA backend: prebuilt cubins, loaded through the CUDA driver API.
//!
//! cudarc owns the driver plumbing (`dynamic-loading` means we never link against CUDA at build
//! time and need no toolkit at run time). We own architecture selection, because a cubin only loads
//! on the compute capability it was compiled for.

use std::sync::Arc;

use cudarc::driver::{
    sys::CUdevice_attribute, CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use super::{cubins, DeviceInfo, GpuBackend};

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

        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| format!("reading the compute capability of device {ordinal} failed: {e}"))?;

        let arch = arch_for((major, minor)).ok_or_else(|| {
            format!(
                "device {ordinal} reports compute capability {major}.{minor}, which has no embedded \
                 cubin (embedded: {})",
                cubins::embedded_arches().join(", ")
            )
        })?;

        let image = cubins::probe_cubin(arch)
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

        if let Some(index) = got.iter().zip(&expected).position(|(got, want)| got != want) {
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

    /// Keyed BLAKE3 of the 64-byte jackpot message, mirroring `compute_jackpot_hash`.
    ///
    /// Doing this on the device is what lets a candidate be rejected without a round trip: the
    /// mining loop compares the digest against the bound and only surfaces a winner.
    #[cfg(feature = "pearl")]
    pub fn jackpot_hash(&self, message: &[u8; 64], key: &[u8; 32]) -> Result<[u8; 32], String> {
        let message_dev = self
            .stream
            .clone_htod(&message[..])
            .map_err(|e| format!("copying the jackpot message failed: {e}"))?;
        let key_dev = self
            .stream
            .clone_htod(&key[..])
            .map_err(|e| format!("copying the key failed: {e}"))?;
        let mut digest_dev = self
            .stream
            .alloc_zeros::<u8>(32)
            .map_err(|e| format!("allocating the digest failed: {e}"))?;

        let kernel = self.function("tokenminer_jackpot_hash")?;
        let mut launch = self.stream.launch_builder(&kernel);
        launch.arg(&message_dev).arg(&key_dev).arg(&mut digest_dev);
        unsafe { launch.launch(ONE_THREAD) }
            .map_err(|e| format!("launching tokenminer_jackpot_hash failed: {e}"))?;

        let host = self
            .stream
            .clone_dtoh(&digest_dev)
            .map_err(|e| format!("reading the digest back failed: {e}"))?;

        let mut digest = [0u8; 32];
        digest.copy_from_slice(&host[..32]);

        Ok(digest)
    }

    /// Checks the jackpot kernel against the canonical implementation, on operands from a real
    /// proof.
    ///
    /// This is the check that decides whether the backend may mine: the reference is
    /// `zk_pow::circuit::chip::compute_jackpot` — the function the verifier itself uses — so a
    /// mismatch here means the GPU would produce proofs the network rejects.
    #[cfg(feature = "pearl")]
    fn check_jackpot(&self) -> Result<(), String> {
        use crate::miners::pearl_mining::{mining_configuration, PearlMining};
        use zk_pow::api::proof::{IncompleteBlockHeader, SeedDerivation};
        use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
        use zk_pow::circuit::chip::compute_jackpot;
        use zk_pow::circuit::pearl_noise::compute_noise;
        use zk_pow::ffi::mine::try_mine_one;

        let mining = PearlMining::default();
        let config = mining_configuration(&mining)?;
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

        let got = self.jackpot_tile(
            &flatten(&private.s_a),
            &flatten(&noise.a),
            &flatten(&private.s_b),
            &flatten(&noise.b),
            k,
            rank,
            h,
            w,
        )?;

        if got != want {
            return Err(format!("the device jackpot {got:?} does not match the verifier's {want:?}"));
        }

        // And the hash the pool actually compares against, also on the device.
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
}

/// The largest dynamic shared-memory request a launch may make without opting in.
const MAX_DYNAMIC_SHARED_BYTES: usize = 48 * 1024;

/// Validates a tile before it reaches the kernel, returning the thread count to launch.
///
/// Kept out of the launch so it can be tested without a CUDA device: every rejection here is a
/// caller bug, and letting one through would surface as an opaque driver error instead.
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
    if rank == 0 || k % rank != 0 {
        return Err(format!("rank {rank} does not divide k {k}"));
    }
    // The noise generator this mirrors needs a rank that is a whole number of BLAKE3 digests.
    if rank % 32 != 0 {
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
        return Err(format!("tile width {w} needs {threads} threads, past the 1024 limit"));
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
        self.check_jackpot()?;

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
        for arch in cubins::PROBE_CUBINS.iter().map(|cubin| cubin.arch) {
            assert!(arch_code(arch).is_some(), "{arch} has no __CUDA_ARCH__ mapping");
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
        assert_eq!(arch_for((8, 0)), None, "sm_80 has no cubin, so it must not be claimed");
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
    fn a_default_pearl_tile_validates_and_reports_its_thread_count() {
        // h=2, w=64, k=2048, rank=128: the shape `PearlMining::default()` produces.
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
        assert!(validate_tile(2 * 2048 - 1, 2 * 2048, 64 * 2048, 64 * 2048, 2048, 128, 2, 64).is_err());
        assert!(validate_tile(2 * 2048, 2 * 2048, 64 * 2048 + 7, 64 * 2048, 2048, 128, 2, 64).is_err());

        // Degenerate shapes.
        assert!(validate_tile(0, 0, 0, 0, 2048, 128, 0, 64).is_err());
        assert!(validate_tile(2048, 2048, 64 * 2048, 64 * 2048, 2048, 128, 1, 0).is_err());

        // A width needing more than one block's worth of threads.
        assert!(
            validate_tile(2048, 2048, 4096 * 2048, 4096 * 2048, 2048, 128, 1, 4096).is_err()
        );

        // A tile whose cell sums do not fit the shared-memory budget: 64x1024 cells is 256 KB.
        assert!(
            validate_tile(64 * 2048, 64 * 2048, 1024 * 2048, 1024 * 2048, 2048, 128, 64, 1024)
                .is_err()
        );
    }
}
