// The Pearl mining kernels.
//
// What is left here is device BLAKE3 in tree mode and the Philox fill. The search path's tensor-core
// work, the noise generators and the fold are no longer in this file: the split path takes the noise
// generators from the fatbin (`pearl_gemm/noise_generation_sm80.cuh`) and the fold from the Triton
// blob, and the fused entry points that used to do both in one launch are gone.
//
// The commitment a proof carries is a Merkle root over a whole matrix, so the thing that has to be
// right on the device is a hash of many chunks, not a hash of one 64-byte message. The tree is walked
// exactly as `pearl_blake3::MerkleTree` walks it: one chaining value per 1024-byte chunk, pairs
// combined upward, an unpaired node carried up unchanged, and the last pair promoted with ROOT.
// `check_blake3_tree` holds that against the reference.
//
// The dense and sparse noise decodes are held against `zk_pow::circuit::pearl_noise` by
// `check_noise_gen`; they are consensus values, not hashing, which is why the check is against the
// reference rather than against a host model of the same code.

#ifndef __CUDACC__
#error "pearl.cu must be compiled by nvcc"
#endif

// No `<mma.h>`: the tensor-core work reaches the tensor cores through `mma.sync` PTX directly, which is
// both the int8 form and the only one whose accumulator layout is fixed by the ISA rather than left
// opaque. `probe.cu` still uses `wmma`, for the correctness-anchor fold.

#include "blake3.cuh"

// One chaining value per chunk of `data`, in words. `out` holds `ceil(len / 1024) * 8` words.
//
// A thread per chunk: a chunk is 16 compressions, which is enough work to hide the launch, and one
// thread per chunk means no shared memory and no reduction. The grid-stride loop is only there so
// that a buffer larger than the grid still hashes.
extern "C" __global__ void tokenminer_blake3_chunk_cvs(
        const unsigned char* data,
        unsigned int len,
        const unsigned int* key_words,
        unsigned int base_flags,
        unsigned int* out) {
    const unsigned int chunks = (len + B3_CHUNK_LEN - 1) / B3_CHUNK_LEN;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < chunks;
         i += stride) {
        const unsigned int start = (unsigned int)i * B3_CHUNK_LEN;
        const unsigned int n = (len - start < B3_CHUNK_LEN) ? (len - start) : B3_CHUNK_LEN;

        unsigned int words[16];
        b3_chunk_cv(data + start, n, i, key_words, base_flags, 0u, words);

        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            out[i * 8 + w] = words[w];
        }
    }
}

// One tree level: combines `count` chaining values into `ceil(count / 2)` parents.
//
// An odd node at the end is carried up unchanged, which is what keeps a non-power-of-two leaf count
// hashing to the same value as the reference. That can only happen on a non-final level — the final
// level always has exactly two children — so `promote_root` is never set here with an odd count.
extern "C" __global__ void tokenminer_blake3_combine(
        const unsigned int* in,
        unsigned int count,
        const unsigned int* key_words,
        unsigned int base_flags,
        unsigned int promote_root,
        unsigned int* out) {
    const unsigned int parents = (count + 1) / 2;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < parents;
         i += stride) {
        const unsigned int left = (unsigned int)i * 2;
        const unsigned int right = left + 1;

        if (right < count) {
            unsigned int words[16];
            b3_parent_cv(&in[left * 8], &in[right * 8], key_words, base_flags,
                         promote_root ? B3_ROOT : 0u, words);
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                out[i * 8 + w] = words[w];
            }
        } else {
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                out[i * 8 + w] = in[left * 8 + w];
            }
        }
    }
}

// The root of an input of at most one chunk: the chunk itself, promoted.
//
// This is the shape the reference takes when `data.len() <= CHUNK_LEN` — it hashes the bytes
// directly rather than building a chunk-CV tree — and it is the path a single-chunk commitment and
// the 64-byte jackpot message both take. One thread; at most 16 compressions.
//
// The digest comes out as eight chaining-value words, like every other output in this file. The
// host serialises them little-endian; that is exactly the byte order the reference emits, because a
// BLAKE3 output *is* the little-endian image of these words.
extern "C" __global__ void tokenminer_blake3_root_chunk(
        const unsigned char* data,
        unsigned int len,
        const unsigned int* key_words,
        unsigned int base_flags,
        unsigned int* digest) {
    unsigned int words[16];
    b3_chunk_cv(data, len, 0ull, key_words, base_flags, B3_ROOT, words);

    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        digest[w] = words[w];
    }
}

// Philox-4x32-10, the counter-based stream the reference fills the signal matrices from.
//
// Counter-based rather than stateful, and that is the whole reason to use it here: output element
// `n` is a pure function of `(n, seed)`, so a window of `A` can be regenerated without replaying
// everything before it. The search walks row blocks constantly, and a stateful stream would force
// the walk to be sequential.
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u

__device__ __forceinline__ void philox_round(unsigned int ctr[4], unsigned int key[2]) {
    const unsigned long long p0 = (unsigned long long)ctr[0] * PHILOX_M0;
    const unsigned long long p1 = (unsigned long long)ctr[2] * PHILOX_M1;

    const unsigned int lo0 = (unsigned int)p0;
    const unsigned int hi0 = (unsigned int)(p0 >> 32);
    const unsigned int lo1 = (unsigned int)p1;
    const unsigned int hi1 = (unsigned int)(p1 >> 32);

    // The counter is rotated rather than written in place: the low word of each product lands in a
    // different lane from the product it came from, and getting that rotation wrong still produces
    // plausible-looking bytes.
    const unsigned int out[4] = {
        hi1 ^ ctr[1] ^ key[0],
        lo1,
        hi0 ^ ctr[3] ^ key[1],
        lo0,
    };
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        ctr[i] = out[i];
    }
}

__device__ __forceinline__ void philox_key_round(unsigned int key[2]) {
    const unsigned long long p = (unsigned long long)key[0] * PHILOX_W0;

    const unsigned int hi = (unsigned int)(p >> 32);
    const unsigned int lo = (unsigned int)p;
    key[0] = hi ^ key[1] ^ PHILOX_W0;
    key[1] = lo;
}

__device__ __forceinline__ void philox_10(unsigned int ctr[4], unsigned int key[2]) {
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        philox_round(ctr, key);
        philox_key_round(key);
    }
}

/// Fills `numel` signed bytes with values in `[-64, 63]` — `(word & 0x7F) - 64`.
///
/// Each byte gets its own counter, keyed by its *absolute* element index, where `base_index` is
/// where `out[0]` sits in the whole matrix. Byte `n` is then a pure function of `(n, seed)` alone —
/// no grid position, no block alignment, no dependency on what was generated before it. That is
/// what lets the search fill a column block on its own and still commit to the same bytes a
/// whole-matrix fill would have produced there.
///
/// Four counters per thread is four times the work of the shared-key form, and it is the only way to
/// get that property: batching several elements into one counter makes each element's bytes depend
/// on which thread group it happened to land in, so a window starting off a group boundary
/// disagrees with a whole fill. The extra Philox rounds are nothing next to the commitment hash over
/// the same bytes.
///
/// The reference generates sixteen values per thread from four counters sharing an evolving key.
/// Nothing in consensus depends on *which* random values the signal matrices hold — the commitment
/// is a hash of whatever was generated, and the verifier recomputes from the matrices carried in the
/// proof — so the stream is ours to choose, and only a position-independent one makes a
/// region-walking search possible.
///
/// `A` and `B` get `seed` and `seed + 1` from one attempt counter, as the reference passes them.
extern "C" __global__ void tokenminer_fill_i8(
        signed char* out,
        long long base_index,
        long long numel,
        unsigned long long seed) {
    const long long at = (long long)blockIdx.x * blockDim.x * 4 + (long long)threadIdx.x * 4;
    if (at >= numel) {
        return;
    }

    const unsigned int seed_lo = (unsigned int)(seed & 0xffffffffull);
    const unsigned int seed_hi = (unsigned int)(seed >> 32);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (at + i >= numel) {
            break;
        }

        const unsigned long long absolute = (unsigned long long)(base_index + at + i);
        const unsigned int lo = (unsigned int)(absolute & 0xffffffffull);
        const unsigned int hi = (unsigned int)(absolute >> 32);

        unsigned int ctr[4] = {lo, hi, hi ^ lo, 0u};
        unsigned int key[2] = {seed_lo, seed_hi};
        philox_10(ctr, key);

        // The lane is chosen by the *absolute* index, not by `i`. They coincide when `base_index` is
        // a multiple of four and disagree otherwise, which is exactly the window case that matters:
        // using `i` here makes a window fill disagree with a whole-matrix fill everywhere.
        const unsigned int lane = (unsigned int)(absolute & 3ull);
        out[at + i] = (signed char)((ctr[lane] & 0x7fu) - 64u);
    }
}
