// pow_scan_emit_sm80.cuh
//
// Ported from the reference miner's `csrc/pearl_gemm/pow_scan_emit_sm80.cuh`, scan half only.
//
// The reference's emit half writes a 640-byte `HostSignalHeader` for its own wire protocol. tokenminer
// already builds a `PlainProof` from the winner coordinate, and that path is verified against captured
// third-party shares, so the mining path takes only the scan result and keeps `pearl_gpu::assemble`.
// The emit kernel is deliberately not ported: porting it would mean porting a header format nothing
// in this repo reads.
//
// One thread per byte of `d_hit`; hits are rare so atomicMin contention is negligible.
//
// Two things the caller owns:
//   * `g_first_hit_idx` must be reset to `FIRST_HIT_SENTINEL` before every launch. cudarc 0.19 has no
//     memset-with-value, so this is a `cuMemsetD8Async` through the sys API or a host upload.
//   * `total` must be the number of candidates the search blob actually wrote. The reference's
//     `launch_pow_scan_and_emit_triton_paired` comment says `hash_candidates = 128` while the blob's
//     `HC` is 64; one of the two is wrong. Buffer sizing here follows 64, which is what the blob
//     declares, and the scan grid follows `total` rather than a hardcoded candidate count so a wrong
//     constant cannot silently read past the buffer.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pearl::sm80::pow_scan_emit {

// The value the caller sets before every scan. `atomicMin` against it means "no hit" survives as
// itself, so the host reads one word and decides.
constexpr uint32_t FIRST_HIT_SENTINEL = 0xFFFFFFFFu;

__device__ inline void scan_one(int tid,
                                const uint8_t* __restrict__ d_hit,
                                uint32_t* __restrict__ g_first_hit_idx) {
  if (d_hit[tid]) {
    atomicMin(g_first_hit_idx, static_cast<uint32_t>(tid));
  }
}

}  // namespace pearl::sm80::pow_scan_emit
