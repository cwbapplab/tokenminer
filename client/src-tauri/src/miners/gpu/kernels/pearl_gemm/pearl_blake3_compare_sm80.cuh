// pearl_blake3_compare_sm80.cuh
//
// Ported from the reference miner's `csrc/pearl_gemm/pearl_blake3_compare_sm80.cuh`.
//
// The postpass for the Triton search blob. One thread per candidate:
//   1. keyed Blake3 compress(transcript, pow_key)
//   2. uint256 compare hash vs pow_target
// writing per-candidate hash + hit byte that the scan then reads.
//
// This kernel is the reason the search blob can use a 4x8 warp tile: the hash is not in the search
// launch, so the accumulator registers are not competing with the hash state. tokenminer's fused
// kernel kept the hash inline and capped at 4x6 for that reason.
//
// The compare walks from word 7 down, which is the uint256 little-endian order — word 7 is the most
// significant. `zk_pow` compares in the same order, so a hit here is a hit the pool accepts. Gate A
// pins this against `zk_pow::api::proof_utils::compute_jackpot_hash` and against `pearl_blake3`, both
// of which are the pool's truth, and asserts the boundary behaviour (equal prefix, then a word above
// or below the target).
//
// The per-candidate work is a `__device__` function so the entry point in `extern_c_shims.inc` calls
// the same code rather than duplicating it. The reference's `launch_blake3_compare` is not ported —
// the Rust launch layer computes the grid, and a C++ helper nothing calls can disagree with it.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "blake3_sm80.cuh"

namespace pearl::sm80::triton_postpass {

__device__ inline void compare_one(int idx,
                                   const uint32_t* __restrict__ d_transcripts,
                                   const uint32_t* __restrict__ pow_key,
                                   const uint32_t* __restrict__ pow_target,
                                   uint32_t* __restrict__ d_hash,
                                   uint8_t* __restrict__ d_hit) {
  uint32_t msg[16];
  uint32_t cv[8];

  // Load transcript + pow_key.
  const uint32_t* transcript = d_transcripts + idx * 16;
  #pragma unroll
  for (int i = 0; i < 16; ++i) msg[i] = transcript[i];
  #pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = pow_key[i];

  // Compress (single-block keyed Blake3).
  pearl::sm80::blake3::compress_msg_block_u32(
      msg, cv, pearl::sm80::blake3::make_single_block_keyed_params());

  // uint256 LE compare: hit if cv (as uint256 little-endian) <= pow_target.
  bool hit = true;
  #pragma unroll
  for (int i = 7; i >= 0; --i) {
    const uint32_t hi = cv[i];
    const uint32_t ti = pow_target[i];
    if (hi > ti) { hit = false; break; }
    if (hi < ti) {               break; }
  }

  // Writeback. `d_hash` is always a real buffer in this path — the entry point does not tolerate a
  // null, so a caller that wants to skip the hash still has to pass one.
  uint32_t* out_hash = d_hash + idx * 8;
  #pragma unroll
  for (int i = 0; i < 8; ++i) out_hash[i] = cv[i];
  d_hit[idx] = hit ? 1u : 0u;
}

}  // namespace pearl::sm80::triton_postpass
