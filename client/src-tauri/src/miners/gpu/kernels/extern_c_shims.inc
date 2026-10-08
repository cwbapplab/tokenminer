// extern_c_shims.inc
//
// The kernel entry points the Rust launch layer looks up by name in `fatbin.rs::symbols`. Included by
// `kernels_only.cu` after the headers, so each entry point is a thin `extern "C"` wrapper over the
// header's `__device__` work — one copy of the algorithm, no body duplication.
//
// Deliberate deviation from the reference: every symbol here is unmangled. The reference's symbol table
// mixes unmangled `pearl_*` names with Itanium-mangled names for the kernels that live at file scope in
// their headers (`_ZN5pearl4sm8015triton_postpass21blake3_compare_kernelEPKjS3_S3_PjPhi`). A mangled
// name in a symbol table is a string that breaks when the signature changes, and nothing checks it —
// the lookup fails at runtime, not at compile time. Names here match the reference's `symbols`
// constants wherever a `pearl_*` name exists, so the two tables can be diffed.
//
// Signatures are 1:1 with the reference's shims so the launch layer's argument lists are the same.
// The launch geometry (block size and grid formula) is recorded per entry point below, because the
// reference's `launch_*` helpers are not ported — they are C++ functions the Rust side cannot call.
// The Rust launch layer computes these same grids, and a test asserts it computes them the same way.

// Block 128. Grid = ceil((rows * 128 / 32) / 128).
extern "C" __global__ void pearl_noise_gen_dense_int8_R128(
    int rows,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ seed,
    int8_t* __restrict__ out) {
  constexpr int R = 128;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= (rows * R) / 32) return;

  uint32_t key_r[8], seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }
  pearl::sm80::noise_gen::dense_int8_chunk<R>(chunk, key_r, seed_r, out);
}

// Block 128. Grid = ceil((rows * 128 / 32) / 128). Same geometry as the int8 form.
extern "C" __global__ void pearl_noise_gen_dense_fp16_R128(
    int rows,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ seed,
    int32_t scale_factor,
    __half* __restrict__ out) {
  constexpr int R = 128;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= (rows * R) / 32) return;

  uint32_t key_r[8], seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }
  pearl::sm80::noise_gen::dense_fp16_chunk<R>(chunk, key_r, seed_r, scale_factor, out);
}

// Block 128. Grid = ceil(((k + 7) / 8) / 128). Output must be zeroed before every launch: this
// entry point patches two non-zeros per row and never clears the previous attempt's.
extern "C" __global__ void pearl_noise_gen_sparse_R128(
    int k,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ seed,
    int8_t* __restrict__ out_r_major) {
  constexpr int R = 128;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= (k + 7) / 8) return;

  uint32_t key_r[8], seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }
  pearl::sm80::noise_gen::sparse_chunk<R>(chunk, key_r, seed_r, k, out_r_major);
}

// Block 256. Grid = ceil(total / 256). `g_first_hit_idx` must be 0xFFFFFFFF before every launch.
extern "C" __global__ void pearl_pow_scan_hits_kernel(
    const uint8_t* __restrict__ d_hit,
    int total,
    uint32_t* __restrict__ g_first_hit_idx) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;
  pearl::sm80::pow_scan_emit::scan_one(tid, d_hit, g_first_hit_idx);
}

// Block 256. Grid = ceil(total_threads / 256). The reference's symbol for this one is mangled; this
// is the unmangled equivalent, following the `pearl_*` convention of the rest of the table.
extern "C" __global__ void pearl_blake3_compare_kernel(
    const uint32_t* __restrict__ d_transcripts,   // (num_tiles * HASH_CANDIDATES, 16)
    const uint32_t* __restrict__ pow_key,         // (8,)
    const uint32_t* __restrict__ pow_target,      // (8,)
    uint32_t* __restrict__ d_hash,                // (num_tiles * HASH_CANDIDATES * 8)
    uint8_t* __restrict__ d_hit,                  // (num_tiles * HASH_CANDIDATES)
    int total_threads) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_threads) return;
  pearl::sm80::triton_postpass::compare_one(
      idx, d_transcripts, pow_key, pow_target, d_hash, d_hit);
}
