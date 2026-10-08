// noise_generation_sm80.cuh
//
// Ported from the reference miner's `csrc/pearl_gemm/noise_generation_sm80.cuh`.
//
// The algorithm is: one keyed Blake3 compress per 32-byte output chunk, then decode the 32 digest
// bytes into int8 noise. No CUTLASS, no swizzled smem, no vectorised stores — the reference itself is
// the correctness-first version of upstream's ~760 LOC of Tensor/Copy abstractions.
//
// Every compress uses the single-block-keyed flags (KEYED_HASH | CHUNK_START | CHUNK_END | ROOT),
// counter = 0, block_len = 64. The message is 8 u32 zeros followed by the 32-byte seed, with one
// position patched per upstream's rule:
//   * dense matrices: message_u32[0]  = chunk_idx + 1
//   * sparse matrices: message_u32[1] = chunk_idx + 1
// Key, seed and decoding rule are identical between the two. The slot is the only difference, so a
// test that hashes a square shape cannot tell dense from sparse — gate B uses a non-square shape.
//
// Two parity facts against `zk_pow` (the verifier, which is the pool's truth):
//   * The decode `((int8(b) + 128) % 64) - 32` equals `zk_pow`'s `(b & 63) - 32` for all 256 byte
//     values. `NOISE_RANGE = 64`, `ZERO_POINT_TRANSLATION = 32`. Asserted host-side in
//     `the_dense_and_sparse_noise_decode_matches_the_reference`.
//   * The sparse pair `k0 = u & (R-1)`, `k1 = k0 ^ (1 + mul_hi_u32(R-1, u))` is `zk_pow`'s
//     `generate_permutation_matrix`, whose consumer (`matvec_sparse_perm`) reads the pair as
//     `vec[first] - vec[second]`.
//
//     The reference's comment says the `% R` matters because `k1` "may be exactly R when
//     `mul_hi(R-1, u) = R-1`". It cannot, for any R: `mul_hi(a, u) <= a - 1` whenever `u < 2^32`, so
//     `1 + mul_hi(R-1, u) <= R - 1` and `k1` is already in range. That is why `zk_pow` has no modulo
//     at all — the two are the same function, and the guard here is dead code that reads like a
//     correctness rule. It is kept (it costs nothing and cannot change the answer) and asserted rather
//     than inherited: `the_sparse_pair_needs_no_modulo_at_any_rank_the_verifier_accepts`.
//
// The sparse kernel patches `+1` / `-1` into a buffer it does not clear, so the host must zero the
// output before every launch: a second attempt would accumulate onto the first.
//
// Two deviations from the reference, both deliberate:
//   * The per-chunk work is a `__device__` function, not a `__global__` kernel. The reference has both
//     a kernel and a duplicated copy of the same body in `extern_c_shims.inc`, because a `__global__`
//     cannot call a `__global__`. Here the entry point in the `.inc` calls this, so there is one copy
//     of the algorithm and the two cannot drift.
//   * The reference's `launch_*` host helpers are not ported: they are C++ functions the Rust launch
//     layer cannot call, so they would be dead code that can disagree with the grid Rust actually
//     computes. The launch geometry is recorded in `extern_c_shims.inc` and asserted by a test.
//   * The reference's `transpose_kr_kernel` is not ported. It exists to derive K-major sparse variants
//     for its own layout; the Triton noising blob reads the R-major sparse output directly, so nothing
//     in this path consumes a transposed buffer.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "blake3_sm80.cuh"

namespace pearl::sm80::noise_gen {

namespace detail {

// One keyed Blake3 compress (single-block-keyed) into an 8-u32 chaining value. `key` is 32 bytes
// copied into the chaining value, `message` is the 64-byte block. Both are register-resident on entry.
__device__ __forceinline__ void keyed_compress(const uint32_t key[8],
                                               uint32_t message[16],
                                               uint32_t cv[8]) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key[i];
  pearl::sm80::blake3::compress_msg_block_u32(
      message, cv, pearl::sm80::blake3::make_single_block_keyed_params());
}

// 8 u32 zeros followed by the seed (8 u32), then patch position [slot].
__device__ __forceinline__ void make_message(const uint32_t seed_u32[8],
                                             uint32_t out[16], int slot,
                                             uint32_t chunk_idx_plus_one) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = 0u;
  #pragma unroll
  for (int i = 0; i < 8; ++i) out[8 + i] = seed_u32[i];
  out[slot] = chunk_idx_plus_one;
}

// Upper 32 bits of (a * b), a and b uint32.
__device__ __forceinline__ uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
  return __umulhi(a, b);
}

}  // namespace detail

// =============================================================================
//   Dense noise (EAL / EBR and their fp16 variants)
// =============================================================================
//
// One thread per 32-byte chunk. R must be a multiple of 32, so each chunk lives entirely within one
// row of the output. True for the only ranks the reference supports (64 and 128), and for the rank
// this path forces (128).

template <int R>
__device__ inline void dense_int8_chunk(int chunk,
                                        const uint32_t key[8],
                                        const uint32_t seed[8],
                                        int8_t* __restrict__ out) {
  static_assert(R % 32 == 0, "R must be multiple of 32");

  uint32_t msg[16];
  detail::make_message(seed, msg, /*slot=*/0,
                       /*chunk_idx_plus_one=*/static_cast<uint32_t>(chunk + 1));

  uint32_t cv[8];
  detail::keyed_compress(key, msg, cv);

  // Decode 32 bytes (= 8 u32) -> 32 int8 in [-32, 32) and store contiguously.
  const int row = (chunk * 32) / R;
  const int col0 = (chunk * 32) % R;
  int8_t* out_row = out + row * R + col0;

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t w = cv[i];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      int8_t hb = static_cast<int8_t>(w & 0xff);
      w >>= 8;
      // Reference: ((int32(hb) + 128) % 64) - 32. With NOISE_ABS_MAX = 128 and
      // PERM_IDXS_PER_COL = 2, NOISE_RANGE = 64.
      int32_t v = (static_cast<int32_t>(hb) + 128) % 64 - 32;
      out_row[i * 4 + b] = static_cast<int8_t>(v);
    }
  }
}

template <int R>
__device__ inline void dense_fp16_chunk(int chunk,
                                        const uint32_t key[8],
                                        const uint32_t seed[8],
                                        int32_t scale_factor,
                                        __half* __restrict__ out) {
  static_assert(R % 32 == 0, "R must be multiple of 32");

  uint32_t msg[16];
  detail::make_message(seed, msg, /*slot=*/0,
                       static_cast<uint32_t>(chunk + 1));

  uint32_t cv[8];
  detail::keyed_compress(key, msg, cv);

  const int row = (chunk * 32) / R;
  const int col0 = (chunk * 32) % R;
  __half* out_row = out + row * R + col0;

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t w = cv[i];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      int8_t hb = static_cast<int8_t>(w & 0xff);
      w >>= 8;
      int32_t v = (static_cast<int32_t>(hb) + 128) % 64 - 32;
      float scaled = static_cast<float>(v * scale_factor);
      out_row[i * 4 + b] = __float2half(scaled);
    }
  }
}

// =============================================================================
//   Sparse noise (EAR_R_major / EBL_R_major; the K-major form is the transpose)
// =============================================================================
//
// Each chunk produces indices for 8 rows of a (k, R) int8 matrix; each row has exactly two non-zeros,
// +1 at k0 and -1 at k1. R is a power of 2 so `& (R-1)` is a valid modulus.
//
// Note the orientation: the sparse buffers are (k, R) — k-major rows — while the dense buffers are
// (rows, R). `noise_B` takes EAR/EBL transposed relative to `noise_A`, so the two sides are not
// interchangeable.

template <int R>
__device__ inline void sparse_chunk(int chunk,
                                    const uint32_t key[8],
                                    const uint32_t seed[8],
                                    int k,
                                    int8_t* __restrict__ out_r_major) {
  static_assert((R & (R - 1)) == 0, "R must be a power of 2");

  uint32_t msg[16];
  detail::make_message(seed, msg, /*slot=*/1,
                       static_cast<uint32_t>(chunk + 1));

  uint32_t cv[8];
  detail::keyed_compress(key, msg, cv);

  const int k_base = chunk * 8;

  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int row = k_base + j;
    if (row >= k) return;
    uint32_t u = cv[j];
    uint32_t k0 = u & static_cast<uint32_t>(R - 1);
    uint32_t k1 = k0 ^ (1u + detail::mul_hi_u32(static_cast<uint32_t>(R - 1), u));
    // The reference does `% R` even though k0 < R; k1 may be exactly R when
    // `mul_hi(R-1, u) = R-1` and `1 + (R-1) = R`. Modulo handles it. For R = 128
    // the modulo is a no-op, which is asserted host-side rather than assumed.
    k1 = k1 % R;
    int8_t* out_row = out_r_major + row * R;
    // out is zero-initialised by the host before launch, every attempt.
    out_row[k0] = 1;
    out_row[k1] = -1;
  }
}

}  // namespace pearl::sm80::noise_gen
