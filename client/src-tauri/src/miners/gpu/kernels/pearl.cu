// The Pearl mining kernels.
//
// Phase one of the port: device BLAKE3 in tree mode, and the noise tensors it seeds. Everything else
// the search needs — the summed-operand GEMM, the jackpot fold, the bound compare — is keyed off a
// hash, and the commitment a proof carries is a Merkle root over a whole matrix. So the first thing
// that has to be right on the device is a hash of many chunks, not a hash of one 64-byte message.
//
// The noise generators reproduce `zk_pow::circuit::pearl_noise` exactly, down to the 1-based hash
// index and the `byte & 63` draw. They are separate kernels rather than part of the BLAKE3 file
// because they are consensus values, not hashing: `check_noise` holds them against the reference.
//
// The tree is walked exactly as `pearl_blake3::MerkleTree` walks it: one chaining value per
// 1024-byte chunk, pairs combined upward, an unpaired node carried up unchanged, and the last pair
// promoted with ROOT. `check_blake3_tree` holds that against the reference.

#ifndef __CUDACC__
#error "pearl.cu must be compiled by nvcc"
#endif

// No `<mma.h>`: the search kernel reaches the tensor cores through `mma.sync` PTX directly, which is
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

// The dense half of the noise: EAL (M x R) and EBR (N x R).
//
// Entry (i, j) is `(byte & 63) - 32`, where the byte is byte `j % 32` of the keyed hash of block
// `i * (rank / 32) + j / 32` — the reference draws 32 bytes at a time and indexes those hashes by
// *global* element position, so row `i` owns hashes `[i * R/32, (i+1) * R/32)` and no neighbour
// shares one. `first_row` shifts that window, which is how the search asks for one row block.
//
// `rank` must be a multiple of 32 (the reference's `noise_rank % BLAKE3_DIGEST_SIZE == 0`); the
// host checks it. One thread per 32 entries.
extern "C" __global__ void tokenminer_noise_dense(
        const unsigned int* seed_words,
        const unsigned int* key_words,
        unsigned int first_row,
        unsigned int rows,
        unsigned int rank,
        signed char* out) {
    const unsigned int groups = rank / 32u;
    const unsigned long long total = (unsigned long long)rows * groups;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += stride) {
        const unsigned int row = first_row + (unsigned int)(i / groups);
        const unsigned int group = (unsigned int)(i % groups);

        unsigned char message[64];
        b3_noise_message(message, seed_words, row * groups + group, 0u);

        unsigned int words[16];
        b3_chunk_cv(message, 64u, 0ull, key_words, B3_KEYED_HASH, B3_ROOT, words);

        unsigned char digest[32];
        b3_store_words(digest, words);

        // Hashed at the absolute row, written at the window's — `out` holds only the `rows` rows
        // asked for, not everything below them. `i` counts 32-byte groups across the window, so the
        // write offset is `i * 32`.
        signed char* row_out = out + (long long)i * 32;
        #pragma unroll
        for (unsigned int k = 0; k < 32; ++k) {
            row_out[k] = (signed char)((digest[k] & 63u) - 32u);
        }
    }
}

// The sparse half of the noise: EAR (K pairs) and EBL (K pairs).
//
// Each entry is two column indices into a dense row of length `rank` — a `+1` at the first and a
// `-1` at the second, so composing with a dense row is a two-term difference, not a matrix product.
// One hash yields eight pairs, taken as eight little-endian `u32`s.
//
// `first_index` shifts the hash window, for the same reason `first_row` does above.
extern "C" __global__ void tokenminer_noise_perm(
        const unsigned int* seed_words,
        const unsigned int* key_words,
        unsigned int first_index,
        unsigned int count,
        unsigned int rank,
        unsigned int* out) {
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count;
         i += stride) {
        const unsigned int hash_index = first_index + (unsigned int)(i / 8u);
        const unsigned int slot = (unsigned int)(i % 8u);

        unsigned char message[64];
        b3_noise_message(message, seed_words, hash_index, 1u);

        unsigned int words[16];
        b3_chunk_cv(message, 64u, 0ull, key_words, B3_KEYED_HASH, B3_ROOT, words);

        unsigned char digest[32];
        b3_store_words(digest, words);

        const unsigned int word = (unsigned int)digest[slot * 4]
                                | ((unsigned int)digest[slot * 4 + 1] << 8)
                                | ((unsigned int)digest[slot * 4 + 2] << 16)
                                | ((unsigned int)digest[slot * 4 + 3] << 24);

        const unsigned int first = word & (rank - 1u);
        const unsigned int second = first ^ (1u + b3_mul_hi_u32(rank - 1u, word));

        out[i * 2] = first;
        out[i * 2 + 1] = second;
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

// The sixteen-word transcript, mirroring `JACKPOT_SIZE` in the consensus code. Same value as in
// `probe.cu`; duplicated rather than shared because `probe.cu` and `pearl.cu` are separate
// translation units compiled into separate cubins, and neither includes the other.
#define POW_TRANSCRIPT_WORDS 16
#define POW_HASH_ROT 13

// --- Block geometry ---------------------------------------------------------------------------
//
// A block owns a `POW_BLOCK_ROWS` x `POW_BLOCK_COLS` rectangle of the tile grid rather than a single
// tile, and that is why this kernel is not as memory-bound as its naive form. One tile per block
// re-reads both of its operands whole: sixteen rows of `k` from each side, so `32 * k` bytes moved
// for `16 * 16 * k` multiply-accumulates. A rectangle amortises each side across the other axis --
// `16 * k * (1/cols + 1/rows)` bytes per tile.
//
// **This kernel is latency-bound at low occupancy, not bandwidth-bound.** That took two corrections
// to establish, and an earlier version of this comment asserted the opposite — twice, confidently,
// and wrong both times. The arithmetic below is recorded because it is what made it look true.
//
// The arithmetic: a block reads `ROWS*16*k` bytes of A and `COLS*16*k` of B, and there are
// `(tiles/ROWS) * (tiles/COLS)` blocks per grid, so a full 8192x8192-tile grid moves ~597 GB from
// DRAM to read 1 GB of operands. Over the 539 ms the 16x8 fold took, that is 1.53 TB/s against the
// 5080's ~1.79 TB/s peak — about 85%, which reads like a bandwidth wall.
//
// The measurement that refutes it: shrinking the tile grid so the operands fit inside the 64 MB L2
// does *not* speed the kernel up. Throughput *falls*, because there are then too few blocks to fill
// 84 SMs:
// ```text
//   16384 tiles/row  138.2 TH/s     operand A = 1.0 GiB
//    8192 tiles/row  138.8 TH/s     operand A = 512 MiB
//    4096 tiles/row  134.9 TH/s     operand A = 256 MiB  (fits L2)
//    2048 tiles/row  126.6 TH/s     operand A = 128 MiB
//    1024 tiles/row  116.9 TH/s
//     512 tiles/row   97.3 TH/s
// ```
// If DRAM were the constraint the smaller grids would be faster — they are strictly less traffic,
// and from 4096 down they fit in L2 entirely. They are slower.
//
// The profiler settles it, and says something neither the arithmetic nor the grid sweep could:
// DRAM throughput is **56.2%** and the Tensor(INT) pipe is **57.6%**, with Nsight naming both
// "not a bottleneck", while **68.8% of cycles have no eligible warp** because occupancy is 33% and
// register-capped. The wall is latency at 4 warps per scheduler. That is why every traffic-shaped
// idea tried measured as no effect or worse (L2 swizzle, `cp.async.ca`), and why the one lever that
// did work — widening columns — helped only 5%.
//
// Ablations that price the rest of the loop, all on the real kernel at production geometry:
//   - the per-tile BLAKE3 and bound compare cost **0.6%** (compile with `-DPOW_SKIP_HASH`).
//   - the same `mma` count with the `ldmatrix` operand loads removed runs at **228 TH/s**. Over half
//     the instructions issued per `mma` are fragment loads, and an issue/latency-bound kernel feels
//     that more than a bandwidth-bound one would.
//   - `cp.async.ca` (cache in L1) is **worse** than `.cg`: 122.4 against 130.7 at 16x8, and 113.4
//     against 137.0 at 16x12.
//   - an L2-locality swizzle of the block index (`POW_L2_GROUP`) changes nothing: 130.1 at group 1
//     against 130.6 at 2 and 130.5 at 32.
//   - stage depth: s2 = 138.8 beats s1 = 131.1; s3 does not fit at 16x12 (118,272 B against a
//     101,376 B limit) and is slower at geometries where it does.
//   - staged width: `POW_BK` 64 = 138.8, 32 = 72.0, 128 does not fit. The 32 -> 64 step is the
//     historical +77%, and it holds.
//
// `POW_SMEM_STRIDE` pads every staged row, so a block spends more shared memory than the data needs.
// That is not an occupancy lever at one block per SM — both the register file and shared memory are
// already the binding limit, and the profiler puts them at the same place. A stride of 32 would need
// an XOR swizzle to keep the fragment loads off the same banks; at 64 that swizzle is not optional,
// because an unpadded 64 is `0 (mod 32)` and every `ldmatrix` would cost four times what it should.
//
// Overridable so the benchmark harness can A/B a geometry without editing this file. The shipped
// values are the defaults; nothing sets these in the real build.
//
// **Why 16 x 12 and not something with better occupancy.** On sm_120 this kernel runs at ONE block
// per SM, and the obvious fix — get two resident — is measurably worse. Two blocks/SM needs both
// `regs * threads <= 32768` and total shared memory `<= 51200` B, and the accumulator alone costs
// `ROWS * COLS * 2 * 4 * 32` registers per block. That admits only small rectangles, and small
// rectangles pay for it in operand traffic. Measured on the real kernel at production geometry:
//
// ```text
//   16x12  138.7 TH/s   1 block/SM    <- shipped
//   16x8   130.3 TH/s   1 block/SM
//    8x8    74.5 TH/s   2 blocks/SM   (regs=120, 45 KB)
//   12x8    94.1 TH/s   2 blocks/SM
//    8x10   72.8 TH/s   2 blocks/SM
//   16x4    97.1 TH/s   2 blocks/SM
// ```
//
// Doubled occupancy loses, by a wide margin. The reason is registers, not traffic — see the profile
// below — but either way the small rectangles that *can* reach two blocks/SM all measure slower.
//
// **What the profiler says** (Nsight Compute `--set full`, sm_120a, 16x12, production geometry).
// These are the numbers that decide whether this stops here:
// ```text
//   Registers Per Thread        128        exactly the __launch_bounds__(THREADS, 1) ceiling
//   Block Limit Registers         1        -> a second block per SM will not fit
//   Block Limit Shared Mem        1        (shared memory agrees; both are binding)
//   Theoretical Occupancy     33.3%        16 warps/SM of a possible 48
//   Eligible Warps/Scheduler   0.55        of 4.00 active; "No Eligible" on 68.8% of cycles
//   Warp Cycles Per Issued Inst 12.80      of which 4.7 is math-pipe throttle (36.6%)
//   Local Memory (spill)       7.04 MB     100% overhead
//   Compute (SM) Throughput    57.6%       Tensor (INT) pipe — explicitly "not a bottleneck"
//   DRAM Throughput            56.2%       532 GB/s
// ```
// Tensor at 58% and not the bottleneck, DRAM at 56% and not the bottleneck, and 69% of cycles with
// no eligible warp at 4 warps/scheduler: the kernel is **latency-bound at low occupancy**, and the
// occupancy is register-bound. Tensor(INT) at 57.6% is the ceiling being pressed, not the wall.
//
// **Why 2 blocks/SM is not reachable from here.** It needs `regs * threads <= 32768`, i.e. 64
// registers per lane. The accumulator alone is `POW_BLOCK_COLS * 2 * 4 = 8 * COLS` registers per
// lane — 96 at 12 columns, 128 at 16 — so it does not fit in 64 alongside anything else. Forcing it
// with `__launch_bounds__(POW_THREADS, 2)` compiles and then spills **924 bytes of stores and 1060
// of loads** per thread, which is no trade at any occupancy. The rectangle has to shrink to fit, and
// every rectangle that fits measures slower (above).
//
// **The two ways past ~138 TH/s**, both of which are rewrites rather than schedule
// changes, and both of which were tried on this branch and measured a loss:
//
//   1. *Split a tile row's columns across warps.* Halves the accumulator per lane
//      (48 registers at 12 columns with a split of 2), which is exactly the register
//      relief the profile says occupancy needs. Measured on the real kernel:
//      ```text
//        16x12 split=1  137.9 TH/s   1 block/SM    <- shipped
//        16x12 split=2  115.6 TH/s   1 block/SM
//        16x12 split=2, min 2 blocks/SM   115.5 TH/s   (registers now fit; still slower)
//        16x8  split=2, min 2 blocks/SM   127.9 TH/s
//         8x8  split=2, min 2 blocks/SM    72.7 TH/s
//      ```
//      The reason is that the A operand depends on the tile row alone, so every column
//      slice of a row re-loads the same A fragment. The split halves the accumulator but
//      doubles the A `ldmatrix` count, and an issue-bound kernel pays for that more than
//      it gains from the second block. A split of 3 or more cannot even launch at 16 rows:
//      `POW_THREADS = 16 * 3 * 32 = 1536`, the thread limit per SM, so the launch is
//      refused before it can be timed.
//   2. *Fewer instructions per multiply.* The same `mma` count with no `ldmatrix`
//      operand loads runs at **228 TH/s** against 138 here — but the fragment loads are
//      not removable: `m16n8k32` reads its operands from registers that only `ldmatrix`
//      fills, and `ldmatrix.x4` already replaces four `LDS.32` per operand. m16n8k32
//      offers no wider fragment, so there is no instruction that does the same multiply
//      with fewer loads.
//
// What *did* move it, for the record: widening columns 8 -> 12 is +5.4% and `POW_SMEM_T_DYNAMIC`
// a further +0.5%. Neither is 200 TH/s, and neither was close.
#ifndef POW_BLOCK_ROWS
#define POW_BLOCK_ROWS 16
#endif
#ifndef POW_BLOCK_COLS
#define POW_BLOCK_COLS 12
#endif
#ifndef POW_STAGES
#define POW_STAGES 2
#endif
// Row strips walked before advancing the column strip. 1 is the identity (linear block order).
#ifndef POW_L2_GROUP
#define POW_L2_GROUP 1
#endif
// 1 = `cp.async.cg` (bypass L1, the default), 0 = `cp.async.ca` (cache in L1). Measured, not assumed.
#ifndef POW_CP_ASYNC_CA
#define POW_CP_ASYNC_CA 0
#endif
// 1 = put the transcript buffer in the dynamic shared allocation rather than `static __shared__`.
// See the declaration in `tokenminer_search_grid`. On by default: it measures 138.4 against 137.5
// for the static form at the shipped 16x12, and it makes the block's whole shared-memory footprint
// one number the launch requests instead of two the reader has to add.
#ifndef POW_SMEM_T_DYNAMIC
#define POW_SMEM_T_DYNAMIC 1
#endif
// 1 = hash the transcript in place out of `sT` instead of copying it into a local `message[64]`.
// The words are already contiguous little-endian bytes, so the copy is redundant; it is also the
// only local array in the kernel and the profile attributes all 7.04 MB of local-memory traffic to
// spilling, which a 64-byte buffer live across a call is exactly the shape of. See the hash site.
#ifndef POW_HASH_DIRECT
#define POW_HASH_DIRECT 0
#endif
#define POW_WARPS POW_BLOCK_ROWS
#define POW_THREADS (POW_WARPS * 32)
#define POW_TILES_PER_BLOCK (POW_BLOCK_ROWS * POW_BLOCK_COLS)

/// Marks the staging buffers as dynamic rather than static `__shared__`.
///
/// Every architecture stages through one dynamic allocation. A static `__shared__` array is capped at
/// 48 KiB on every card we ship for -- sm_75 included, whose 64 KiB budget does not raise that cap
/// for static arrays -- and `32 x 4` needs more than that before the transcript buffer is counted, so
/// static cannot express this geometry at all. The host raises the dynamic ceiling per function with
/// `cudaFuncSetAttribute` before the first launch; see `pow_opt_in_shared_memory`.
#define POW_DYNAMIC_SMEM 1
// `__align__(16)` because both consumers of this allocation require it and neither can supply it: the
// staging writes 16 bytes at a time with `cp.async`, and the `ldmatrix` fold addresses 16-byte row
// starts. The driver's default alignment for `extern __shared__` is only what the array's element type
// asks for, which for `signed char` is 1. `POW_SMEM_STRIDE` is 48 and every stage offset is a multiple
// of it, so one alignment on the base covers every row of both buffers.
extern __shared__ __align__(16) signed char pow_dynamic_smem[];

// Which fold this architecture gets.
//
// `mma.m16n8k32` arrived in Ampere. Turing (sm_75) has no int8 tensor cores at all, so it gets a
// DP4A fold instead -- a genuinely different instruction stream, several times the instruction count
// for the same multiply-accumulates, and the best that silicon offers. Turing is not a fallback out
// of convenience: a T4 is a common cloud mining card, and dropping it from `build.rs`'s arch table to
// avoid writing this would be trading real hardware support for code.
//
// `POW_FORCE_DP4A` selects it on any architecture. That is not decoration -- this kernel has two
// implementations of a consensus-critical fold, and without a way to run the Turing one on whatever
// card the developer happens to have, the sm_75 cubin would ship having only ever been compiled.
#if defined(POW_FORCE_DP4A) || (__CUDA_ARCH__ < 800)
#define POW_DP4A 1
#else
#define POW_DP4A 0
#endif

// Whether the `mma` path loads its fragments with `ldmatrix` instead of four `LDS.32` apiece.
//
// `ldmatrix` arrived in Turing, so every architecture this miner ships an `mma` path for has it and
// there is no architecture here for which the fallback would be the only option -- it exists to be
// *measurable*, not to be portable. `POW_FORCE_LDS32` selects it on any card, for the same reason
// `POW_FORCE_DP4A` exists: two implementations of a consensus-critical fold, one of which has to be
// runnable on whatever card the developer happens to have, or the other ships having only ever been
// compiled.
#if defined(POW_FORCE_LDS32) || POW_DP4A
#define POW_LDMATRIX 0
#else
#define POW_LDMATRIX 1
#endif

// k staged per tile. `mma.m16n8k32` takes k = 32, so a 64-wide stage is exactly two instructions'
// worth and the DP4A fold consumes the same 32 bytes as four four-byte dots per lane, so the
// staging, the step count and the fold schedule are shared by both paths and only the multiply
// differs.
//
// **The tensor path stages 64, the DP4A path 32, and the reason is measured, not a preference.**
// At 32 the step loop issues one `mma` k-block between a `cp.async` wait and a `BAR.SYNC`, and the
// profiler named that barrier as the single largest stall in the kernel -- 3.99 of 15.67 cycles per
// issued instruction, ahead of every other term. Widening the stage to 64 halves the barriers per
// unit of multiply-accumulates: the same stall falls to 0.92, the total to 10.56, and the top stall
// becomes `math_pipe_throttle`, which is the one that means the math pipe is the constraint.
// Five alternating runs at production geometry put this at **+77%**, 71.7 -> 127.1 TMAC/s, with no
// overlap between the two ranges. It also drops `long_scoreboard` from 2.34 to 0.52: a deeper step
// hides the `cp.async` latency on its own, which is why stage depth on its own never paid.
//
// Turing keeps 32. Its `aw[2][POW_BK / 4]` staging array doubles with the stage width, and at 64 the
// DP4A path spills 76 bytes against 20 at 32 -- both already capped at 128 registers by
// `__launch_bounds__`. A card with no tensor cores to begin with is the wrong place to spend
// registers on hiding a barrier, so the two paths stage differently and the host mirrors the split.
//
// `rank` must be a whole number of stages or `steps_per_rank` is zero and the fold's modulo divides
// by zero; the host checks that, and a stage width above `rank` is the easy way to hit it.
//
// Overridable so the stage width can be swept rather than assumed. The barrier count per unit of
// multiply-accumulates scales as `1/POW_BK`, which is why going 32 -> 64 was worth +77% historically
// and why 128 is worth measuring: at 128 the step loop issues four `mma` k-blocks between a
// `cp.async` wait and a `BAR.SYNC` instead of two. It costs shared memory (the staging buffers scale
// with `POW_BK`) and the host's `pow_smem_stride`/`pow_bk` have to mirror whatever is chosen here.
#if POW_DP4A
#ifndef POW_BK
#define POW_BK 32
#endif
#else
#ifndef POW_BK
#define POW_BK 64
#endif
#endif

// The shared row stride: `POW_BK + 16`, so every staged row is padded past its own width. An unpadded
// stride is a poor choice: 32 is 8 banks wide, so the eight row starts a lane group touches fold onto
// each other four deep and every fragment load costs four times what it should. Padding to `POW_BK + 16`
// is 12 banks at `POW_BK` 32 and 20 at 64, and both give lane `l` eight distinct row starts. Padding
// rather than XOR swizzling, because it keeps the copy and the
// fragment load as plain 16- and 32-bit accesses; a swizzle would have to be undone by the same code
// that wrote it, and there is nothing to buy for it.
//
// The rule that makes this work is `stride == 16 (mod 32)`, which is what keeps `ldmatrix`
// conflict-free: a matrix reads eight rows of 16 bytes, so their row starts must land on eight
// distinct bank groups. 48 and 80 both satisfy it; **64 does not** -- 64 is `0 (mod 32)`, so rows
// alternate between two starts and every `ldmatrix` costs four times what it should. That is worth
// knowing because 64 is the width a stage wants most, and it is why the pad is there.
#define POW_SMEM_STRIDE (POW_BK + 16)

// Stages in flight, defined with the rest of the geometry above so a benchmark can override it.
// Two is the fewest that overlap the next k-step's copy with the current one's multiply, and a
// barrier is needed between them either way.

// Bytes of shared memory a block's three buffers take, laid out contiguously in the dynamic
// allocation: A, then B, then the transcript. `POW_SMEM_STRIDE` pads each staged row, so these
// scale with the geometry rather than with the data.
#define POW_SMEM_A_BYTES (POW_STAGES * POW_BLOCK_ROWS * 16 * POW_SMEM_STRIDE)
#define POW_SMEM_B_BYTES (POW_STAGES * POW_BLOCK_COLS * 16 * POW_SMEM_STRIDE)
#define POW_SMEM_T_BYTES (POW_WARPS * POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS * 4)

// What a launch must request as its dynamic shared-memory size -- the staged operands only.
//
// `POW_SMEM_T_BYTES` is deliberately *not* in it: the transcript buffer is static `__shared__`, which
// the driver allocates on top of whatever the launch asks for. A launch that passes `POW_SMEM_BYTES`
// instead double-counts the transcript, and on the tensor path that is 61,440 + 8,192 = 69,632
// against a 48 KiB default -- so the launch fails with `cudaErrorInvalidValue` and names
// nothing about shared memory. The two totals differ by exactly the transcript, which is the whole
// reason both names exist. This is no longer hypothetical: at `POW_BK` 64 the tensor path asks for
// 61,440 bytes and genuinely needs the opt-in `pow_opt_in_shared_memory` performs.
#define POW_SMEM_DYNAMIC_BYTES (POW_SMEM_A_BYTES + POW_SMEM_B_BYTES)

// The whole per-block shared-memory footprint, for sizing a carveout or checking a budget against the
// device limit. This is the number that has to fit the device; `POW_SMEM_DYNAMIC_BYTES` is the number
// that has to be passed to the launch.
#define POW_SMEM_BYTES (POW_SMEM_DYNAMIC_BYTES + POW_SMEM_T_BYTES)

__device__ __forceinline__ unsigned int pow_ld32(const signed char* p) {
    return *reinterpret_cast<const unsigned int*>(p);
}

// Four signed bytes as the `char4` that `__dp4a`'s signed int8 overload takes.
//
// The overload is `__dp4a(char4, char4, int)` -- packed, not four scalars -- so a 32-bit staged load
// has to be taken apart by hand. Little-endian, so this is the same byte order the load produced and
// no shift is out of place.
//
// `char4`, not `uchar4`: CUDA has both, and the unsigned one is `__dp4a(uchar4, uchar4, unsigned
// int)`. Our operands are `signal + noise` in `[-127, 126]`, so the signed form is the correct one
// and picking the other gives a plausible-looking kernel that computes a different function of the
// same bytes.
__device__ __forceinline__ unsigned int pow_dp4a(unsigned int acc, unsigned int a,
                                                 unsigned int b) {
    char4 av;
    char4 bv;
    av.x = (signed char)(a);
    av.y = (signed char)(a >> 8);
    av.z = (signed char)(a >> 16);
    av.w = (signed char)(a >> 24);
    bv.x = (signed char)(b);
    bv.y = (signed char)(b >> 8);
    bv.z = (signed char)(b >> 16);
    bv.w = (signed char)(b >> 24);
    return (unsigned int)__dp4a(av, bv, (int)acc);
}

// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` -- the int8 tensor-core multiply, sm_80 and up.
//
// This replaces `wmma` for two reasons. `wmma` leaves the accumulator's cell-to-register mapping
// opaque, and this kernel has to read every cell out of it; and there is no `wmma` int8 form that
// reaches these operands anyway. The layout below is the m16n8k32 one, which the ISA fixes rather
// than the compiler choosing, and it was checked cell by cell against a hand-computed product rather
// than inferred from the PTX manual:
//
//   A (16 x 32): row = lane >> 2, k = (lane & 3) * 4 for {0..3}; a1 is that row + 8, a2 is k + 16.
//   B (32 x  8): k = (lane & 3) * 4 for {0..3}; b1 is k + 16. Column = lane >> 2.
//   C/D (16 x 8, s32): row = lane >> 2, column = (lane & 3) * 2 for {0,1}; c2/c3 are column + 8.
#if !POW_DP4A
__device__ __forceinline__ void pow_mma(unsigned int* d, unsigned int a0, unsigned int a1,
                                        unsigned int a2, unsigned int a3,
                                        unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// One `ldmatrix.x4` loads the four registers an `mma` fragment pair wants out of a 16-row, 32-byte
// shared tile -- the A operand of one `mma`, or the B operand of two.
//
// The mapping is not a coincidence and it is the only reason this works without a swizzle. Read the
// tile as four 8x8 matrices of 16-bit elements -- rows 0-7 / bytes 0-15, rows 8-15 / bytes 0-15,
// rows 0-7 / bytes 16-31, rows 8-15 / bytes 16-31 -- and `ldmatrix` hands lane `l` the two 16-bit
// elements at row `l >> 2`, columns `(l & 3) * 2` and `+1`, which is four bytes at
// `(l & 3) * 4 .. + 3`. Those four matrices are exactly `a0..a3` for A, and for B exactly the
// `b0`/`b1` of both eight-column halves at once. So one instruction replaces four `LDS.32`.
//
// Lane `l` supplies the address of row `(l & 7)` of matrix `l >> 3`, which is the row/offset pair
// below -- and note it is the same pair for A and for B, so there is one loader rather than two.
//
// **No swizzle is needed, and that is worth being sure about rather than assuming.** `ldmatrix`
// reads eight rows of 16 bytes per matrix, so a matrix is conflict-free exactly when its eight rows
// start on 32 distinct banks. `POW_SMEM_STRIDE` is `16 (mod 32)` at either stage width, so rows 0-7
// land on eight distinct starts and each row's four banks are disjoint from the rest: 48 bytes gives
// banks 0, 12, 24, 4, 16, 28, 8, 20, and 80 gives 0, 20, 8, 28, 16, 4, 24, 12. The pitch was chosen
// to make the `LDS.32` path conflict-free and it happens to satisfy `ldmatrix` too; the two choices
// are not in tension and there is nothing to buy from replacing it.
//
// `smem` must be 16-byte aligned, which the 48-byte stride preserves for every row and offset here,
// and which the `__shared__` declarations state explicitly because the `cp.async` above needs it too.
#if POW_LDMATRIX
__device__ __forceinline__ void pow_ldmatrix_x4(unsigned int* out, const signed char* smem,
                                                int lane) {
    const int row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int koff = (lane >> 4) * 16;
    const unsigned int addr =
        (unsigned int)__cvta_generic_to_shared(smem + row * POW_SMEM_STRIDE + koff);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(addr));
}
#endif

// Wait until at most `N` `cp.async` groups are still in flight.
//
// `cp.async.wait_group` takes an immediate, so the count has to be a template parameter rather than
// an argument. On sm_75 there is no `cp.async` and the copies are plain stores, which a
// `__syncthreads` is the correct -- and only -- way to wait on.
template <int N>
__device__ __forceinline__ void pow_cp_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    __syncthreads();
#endif
}

__device__ __forceinline__ void pow_cp_async_fence() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/// One 16-byte copy per thread, global to shared.
///
/// `chunk` walks the staged rows in `POW_BK / 16`-byte pieces, so the *caller's* loop bound is what
/// keeps the copy inside the rows the grid really has -- fewer than the block's own extent whenever
/// the grid is smaller than one block rectangle, which the self-tests deliberately arrange.
///
/// That bound is `rows * 16 * (POW_BK / 16)` and all three factors are load-bearing: a tile row is
/// sixteen matrix rows, and each of those holds `POW_BK` bytes to be copied as `POW_BK / 16` chunks.
/// Halving it still compiles, still runs, and silently stages a quarter of the operand.
__device__ __forceinline__ void pow_stage_copy(signed char* smem,
                                               const signed char* src,
                                               int ldm,
                                               int kk,
                                               int chunk) {
    const int row = chunk / (POW_BK / 16);
    const int offset = (chunk % (POW_BK / 16)) * 16;
    signed char* dst = smem + row * POW_SMEM_STRIDE + offset;
    const signed char* from = src + (long long)row * ldm + kk + offset;
#if __CUDA_ARCH__ >= 800
    // `POW_CP_ASYNC_CA` selects the L1-caching form of `cp.async`.
    //
    // `.cg` (the default) bypasses L1 and caches only in L2. `.ca` also caches in L1. Which is right
    // depends entirely on whether two blocks on this SM read the same bytes, and the original
    // comment here asserted they never do — "the staged bytes are read once by one block and never
    // reused by another". That is false. A 16x8 block reads 16 rows of A, and the grid has
    // `tiles_per_row / 8` = 1024 column strips per row strip, so ~1024 blocks read those same 16 rows
    // of A. With 84 SMs and linear block order those blocks are spread far apart in time, which is
    // why the measured effect of this switch was small -- but the reuse is real and is what the
    // bandwidth arithmetic in `POW_BLOCK_ROWS` is about.
    //
    // Overridable so the choice can be measured rather than assumed.
#if POW_CP_ASYNC_CA
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                     (unsigned int)__cvta_generic_to_shared(dst)), "l"(from));
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     (unsigned int)__cvta_generic_to_shared(dst)), "l"(from));
#endif
#else
    *(uint4*)dst = *(const uint4*)from;
#endif
}

/// Searches a whole batch of 16x16 tiles: fold the summed operands, hash each transcript, compare
/// against the bound, and claim the region if a tile wins.
///
/// The whole of steps 5 and 6 of the pipeline in one kernel, because a candidate that has to make a
/// round trip to the host to be rejected is a candidate that costs a PCIe synchronisation to
/// discover nothing -- and there are tens of millions of candidates per attempt, so one launch per
/// tile is not an option either.
///
/// `a_sum` is the noise-applied `A` as `m_tiles * 16` rows of `k` and `b_sum` the same for `Bt`, so a
/// tile's operands are the sixteen rows and sixteen columns at its position and the fragment pair
/// covers them without any pattern indexing -- the job's row and column patterns are resolved when
/// those matrices are gathered.
///
/// **Geometry.** The batch is `tiles` tiles of a grid `tiles_per_row` wide, laid out in tiles-per-row
/// order, so it covers `tiles / tiles_per_row` whole rows. Block `b` owns rectangle
/// `(b / cols_per_strip, b % cols_per_strip)`: `POW_BLOCK_ROWS` tile rows by `POW_BLOCK_COLS` tile
/// columns. Both divisions are ceilings, because a grid narrower than one strip still needs a strip
/// for the fraction of one it has, and the trailing strip is short rather than absent. The host has
/// to compute exactly this for `gridDim.x`.
///
/// **The fold.** Per rank block over the *running* totals, so the reduction cannot be hoisted out of
/// the loop: `acc` is never reset, and each fold XORs everything accumulated so far. Doing it once at
/// the end leaves the first transcript word right and every later word wrong.
///
/// The reduction is a warp shuffle rather than a shared-memory tree, and the reason is specific
/// rather than clever: what a rank block contributes is the XOR of all 256 cells, and **XOR does not
/// care which cell a value came from**. Each lane XORs its own four accumulator registers and the
/// five `__shfl_xor` steps reduce the warp -- no shared memory, no barrier, no second pass. The
/// reference miner does the same, and names the fold a direct in-register `shfl_xor` as the reason
/// its kernel beats the hand-`ldm` one.
///
/// **Regions.** The grid is cut into fixed-size **regions**; `found` and `coord` have one slot per
/// region, so a batch of regions in flight cannot have one region's win overwrite another's, and the
/// host can tell which region won without reading anything else back. A block can straddle a region
/// boundary whenever a region is smaller than a block rectangle, so the early-out below is guarded to
/// the single-region case and the claim computes its region per tile instead. Production regions are
/// whole tile rows, far wider than a block, so the guard is always true there.
///
/// The key is `dnsA`, not the job key: consensus hashes the transcript under the A noise seed, and
/// this is the one place the two differ.
///
/// `bound` is the 256-bit target as eight big-endian words, most significant group at index 0. The
/// digest is compared against it the way the verifier does -- as a little-endian integer -- so the two
/// ends of the digest meet in the middle of the walk; see the compare near the bottom.
///
/// `transcript_out`, when non-null, receives every tile's sixteen transcript words whether or not it
/// won: `POW_TILES_PER_BLOCK * 16` words per block. It is how the fold is pinned without a second
/// implementation of this grid's indexing to disagree about.
// `__launch_bounds__` is not advisory here. The DP4A path wants 158 registers, and 158 * 512 threads
// is 80,896 against a 65,536-register file, so without this the sm_75 cubin compiles cleanly and then
// fails every launch with `cudaErrorLaunchOutOfResources`. Capping at one block's worth of registers
// makes ptxas spill the excess instead, which costs the fallback path speed and is the right trade for
// a card that has no tensor cores to begin with. The `mma` paths sit at 94 to 128 and are untouched.
//
// The cap is the *tensor* path's, and it is the binding constraint on the block rectangle rather than
// a formality. The accumulator alone is `POW_BLOCK_COLS * 2 * 4` registers per lane, so a rectangle
// of N columns asks for 8N of them before anything else. At N = 16 that is 128, and ptxas responds by
// spilling: measured with `-Xptxas -v`, `POW_BLOCK_COLS` 12 spills 16 bytes (negligible) while 16
// spills 1600 stores / 1808 loads and the kernel falls off a cliff to 17 TH/s against 138 for 12
// columns. That collapse is a register spill, not shared memory and not traffic — 16x16 fits in the
// 101,376 B shared-memory budget either way.
//
// Overridable so the cap can be swept rather than assumed.
#ifndef POW_MIN_BLOCKS
#define POW_MIN_BLOCKS 1
#endif
extern "C" __global__ __launch_bounds__(POW_THREADS, POW_MIN_BLOCKS) void tokenminer_search_grid(
        const signed char* a_sum,          // m_tiles * 16 rows of k, row-major
        const signed char* b_sum,          // n_tiles * 16 rows of k, row-major
        const unsigned int* key_words,     // eight words: dnsA
        const unsigned int* bound,         // eight words, big-endian, most significant first
        unsigned int* found,               // one word per region, atomicCAS
        unsigned int* coord,               // two words per region: (row, col) of the winning tile
        unsigned int* transcript_out,      // POW_TILES_PER_BLOCK * 16 words per block, or null
        unsigned int first_tile,           // grid index of this batch's first tile
        unsigned int tiles,                // tiles in this batch, a whole number of tile rows
        unsigned int tiles_per_row,        // tiles in one row of the tile grid
        unsigned int tiles_per_region,
        int k,
        int rank) {
// The staged A and B buffers are one dynamic allocation, sliced here in the order `POW_SMEM_*_BYTES`
    // lays them out, which is what puts the geometry's shared-memory cost in one place the host can
    // be checked against. The transcript buffer is still `static __shared__`: it is small enough that
    // it fits the static cap on its own, and leaving it static keeps it out of the launch's
    // `shared_mem_bytes`.
    signed char* const smem = pow_dynamic_smem;

    // The transcript buffer, addressed as one flat array of `POW_WARPS * POW_BLOCK_COLS *
    // POW_TRANSCRIPT_WORDS` words under `POW_SMEM_T_DYNAMIC`, and as a `[warp][col][word]` array
    // otherwise. Both are the same bytes in the same order; only the indexing syntax differs, so
    // every use goes through `pow_sT_AT` below and never indexes `sT` directly.
    //
    // `POW_SMEM_T_DYNAMIC` exists for one reason: it is what stands between the shipped geometry and
    // a wider rectangle. At `POW_BLOCK_COLS` 16 the static form is 16 KB on top of 80 KB of staging,
    // which is 96 KB against a 101,376 B opt-in limit -- it fits, but it leaves the SM unable to hold
    // a second block, and 16x16 measured 17 TH/s against 138 for 16x12. Moving it to the dynamic
    // allocation does not change the total (the driver draws both from the same budget); it makes
    // the whole footprint one number the launch requests and one number the host can check.
    //
    // Off by default. The geometry that ships does not need it, and the layout change is not one to
    // make for its own sake.
#if POW_SMEM_T_DYNAMIC
    unsigned int* const sT = reinterpret_cast<unsigned int*>(smem + POW_SMEM_DYNAMIC_BYTES);
#define POW_ST_AT(w, c, d) sT[((w) * POW_BLOCK_COLS + (c)) * POW_TRANSCRIPT_WORDS + (d)]
#else
    __shared__ unsigned int sT[POW_WARPS][POW_BLOCK_COLS][POW_TRANSCRIPT_WORDS];
#define POW_ST_AT(w, c, d) sT[(w)][(c)][(d)]
#endif

    // Typed views of the staged operands, so the rest of the kernel keeps indexing them as
    // `[stage][row * POW_SMEM_STRIDE]` and cannot drift out of step with the allocation above.
    //
    // A and B need separate types: one stage of A is `POW_BLOCK_ROWS` rows and one stage of B is
    // `POW_BLOCK_COLS`, so a shared typedef makes `sB[1]` step by the wrong stride and the second
    // stage lands outside the allocation. That is not a subtle arithmetic slip either -- it is an
    // out-of-bounds write on every launch past stage zero, which compute-sanitizer reports as an
    // invalid `__shared__` write and the driver surfaces as an illegal address on the next call.
    typedef signed char pow_a_stage_t[POW_BLOCK_ROWS * 16 * POW_SMEM_STRIDE];
    typedef signed char pow_b_stage_t[POW_BLOCK_COLS * 16 * POW_SMEM_STRIDE];
    pow_a_stage_t* const sA = reinterpret_cast<pow_a_stage_t*>(smem);
    pow_b_stage_t* const sB = reinterpret_cast<pow_b_stage_t*>(smem + POW_SMEM_A_BYTES);

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
#if POW_DP4A || !POW_LDMATRIX
    const int lane_group = lane >> 2;      // which row of the 8 this lane serves
#endif
#if POW_DP4A || !POW_LDMATRIX
    const int lane_k = (lane & 3) * 4;     // and which four k values
#endif
#if !POW_DP4A && !POW_LDMATRIX
    const int lane_k_hi = lane_k + 16;
#endif

    // The grid is walked a batch at a time and `first_tile` is where this batch starts, so it has to
    // be applied here rather than added to the answer afterwards: `row0` is what indexes the operand
    // rows below, so a batch-local row would fold the same tile rows over and over while reporting
    // entirely plausible coordinates. What lands in `coord` is then the grid's own row and column, and
    // the host takes it as it comes.
    const unsigned int first_row = first_tile / tiles_per_row;
    const unsigned int rows_in_batch = tiles / tiles_per_row;
    const unsigned int cols_per_strip = (tiles_per_row + POW_BLOCK_COLS - 1) / POW_BLOCK_COLS;

    // Block -> rectangle mapping, with an optional L2-locality swizzle.
    //
    // Linear order gives each block a 16x8 rectangle whose row is `strip` and whose column strip is
    // `blockIdx.x % cols_per_strip`. Consecutive block IDs therefore walk *columns* while holding the
    // same rows, so the blocks resident at any instant read 16 rows of A (64 KiB, which fits L2
    // easily) but stride across all of B (512 MiB). Every one of those B reads misses L2.
    //
    // A "grouped" schedule -- walk a group of G row-strips before advancing the column -- makes the
    // concurrently-resident blocks span a square-ish patch of the tile grid instead of a line, so
    // both operands are shared and both fit in the 64 MB L2. This is the same swizzle CUTLASS and
    // Triton apply to GEMM tile schedules; here it is the difference between sharing one operand and
    // sharing both.
    //
    // `POW_L2_GROUP` of 1 is the identity (linear order) and is what an unset build gets. The group
    // count is in strips, not blocks, so a group of G covers G*POW_BLOCK_ROWS tile rows.
    unsigned int strip, col_block;
#if POW_L2_GROUP > 1
    const unsigned int strips_in_batch = rows_in_batch / POW_BLOCK_ROWS;   // full row strips
    const unsigned int groups_per_batch = (strips_in_batch + POW_L2_GROUP - 1) / POW_L2_GROUP;
    if (groups_per_batch > 0) {
        const unsigned int group = blockIdx.x / (POW_L2_GROUP * cols_per_strip);
        const unsigned int within = blockIdx.x - group * (POW_L2_GROUP * cols_per_strip);
        const unsigned int g_strip = within / cols_per_strip;
        col_block = within - g_strip * cols_per_strip;
        strip = group * POW_L2_GROUP + g_strip;
    } else {
        strip = blockIdx.x / cols_per_strip;
        col_block = blockIdx.x - strip * cols_per_strip;
    }
#else
    strip = blockIdx.x / cols_per_strip;
    col_block = blockIdx.x - strip * cols_per_strip;
#endif

    const unsigned int row0 = first_row + strip * POW_BLOCK_ROWS;
    const unsigned int col0 = col_block * POW_BLOCK_COLS;

    // Short on either axis when the batch has fewer tile rows than a block rectangle, or the grid is
    // narrower than one. Both are ordinary shapes for the self-tests, not only for degenerate grids.
    const int rows_valid = (int)min((unsigned int)POW_BLOCK_ROWS,
                                    rows_in_batch - strip * POW_BLOCK_ROWS);
    const int cols_valid = (int)min((unsigned int)POW_BLOCK_COLS, tiles_per_row - col0);

    // Another block in this region has already won. The rest of the region is wasted work, but it is
    // only wasted work -- the flag is read before the fold, so it is a plain load and not a barrier,
    // and blocks already past this point finish on their own.
    //
    // Only valid when the block's whole rectangle is in one region, which is checked rather than
    // assumed: a region narrower than a block would otherwise let a neighbouring region's win retire
    // this block before it has looked at any of its own tiles.
    //
    // Batch-local, like the claim below and for the same reason. `strip_first` is an absolute tile, so
    // dividing it by the region size without taking `first_tile` off reads `found` at the region's
    // index in the *whole grid* -- which for the production geometry is `64 * batch + 16 * strip`
    // against a 64-word buffer, so on every batch after the first this reads and atomically writes
    // another allocation. `strip_first >= first_tile` always, so the subtraction cannot wrap.
    const unsigned int strip_first = strip * POW_BLOCK_ROWS * tiles_per_row;
    const bool one_region = strip_first / tiles_per_region
                         == (strip_first + POW_BLOCK_ROWS * tiles_per_row - 1) / tiles_per_region;
    if (one_region && atomicAdd(found + strip_first / tiles_per_region, 0u) != 0u) {
        return;
    }

    const signed char* a_base = a_sum + (long long)row0 * 16 * k;
    const signed char* b_base = b_sum + (long long)col0 * 16 * k;

    // Zeroed, because the fold rotates and XORs into the running transcript: an uninitialised word
    // would give a different answer on every launch while the arithmetic stayed correct, which is the
    // worst failure mode this file has. `tokenminer_jackpot_fold` gets this from a zeroed device
    // allocation instead, since it has no shared transcript of its own.
    //
    // This loop is what makes `POW_SMEM_T_DYNAMIC` safe: a `static __shared__` array is zeroed by the
    // driver, but a slice of the dynamic allocation is not, so with the transcript buffer moved there
    // this loop is the only thing standing between a launch and an uninitialised transcript.
    //
    // The barrier is load-bearing rather than tidy. The zeroing is spread over every thread, so each
    // warp clears slices other warps own; without the barrier a warp that has already folded its own
    // slice has it zeroed underneath by one still in the loop, and every transcript comes back empty.
    for (int i = tid; i < POW_WARPS * POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS; i += POW_THREADS) {
        const int w = i / (POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS);
        const int rest = i - w * POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS;
        const int t = rest / POW_TRANSCRIPT_WORDS;
        POW_ST_AT(w, t, rest - t * POW_TRANSCRIPT_WORDS) = 0u;
    }
    __syncthreads();

    // Never reset between rank blocks: the fold reads the running totals.
    //
    // Both layouts spend the same 64 registers per lane on a warp's whole 16x16 tile, and both index
    // it as `acc[tile_column][cell]`. They differ only in what a "cell" is: four s32 holding a
    // 16x8 `mma` accumulator, or eight s32 each holding one cell of the 16x16 outright.
#if POW_DP4A
    unsigned int acc[POW_BLOCK_COLS][8];
    #pragma unroll
    for (int t = 0; t < POW_BLOCK_COLS; ++t) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc[t][r] = 0u;
        }
    }
#else
    unsigned int acc[POW_BLOCK_COLS][2][4];
    #pragma unroll
    for (int t = 0; t < POW_BLOCK_COLS; ++t) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[t][h][r] = 0u;
            }
        }
    }
#endif

    // `rank` has to be a whole number of staged k-steps or the fold below never fires and every
    // transcript comes back zero; the host checks it before the launch.
    const int steps_per_rank = rank / POW_BK;
    const int total_steps = k / POW_BK;

    for (int s = 0; s < POW_STAGES - 1; ++s) {
        for (int chunk = tid; chunk < rows_valid * 16 * (POW_BK / 16); chunk += POW_THREADS) {
            pow_stage_copy(sA[s], a_base, k, s * POW_BK, chunk);
        }
        for (int chunk = tid; chunk < cols_valid * 16 * (POW_BK / 16); chunk += POW_THREADS) {
            pow_stage_copy(sB[s], b_base, k, s * POW_BK, chunk);
        }
        pow_cp_async_fence();
    }

    for (int step = 0; step < total_steps; ++step) {
        // `cp.async.wait_group` takes an immediate count of groups still allowed to be in flight.
        // With S stages the newest commit is for the *next* step, so S-1 may remain outstanding and
        // the wait is for `S - 2`. At S == 2 that is 0 -- wait for everything -- which is correct but
        // serialises the copy against the multiply, because the group being waited on is the very one
        // just issued for the next step. At S == 1 it is -1, an illegal immediate that makes every
        // launch fail, so the count is clamped rather than left to underflow.
        pow_cp_async_wait<(POW_STAGES > 2) ? (POW_STAGES - 2) : 0>();
        __syncthreads();

        // Issued before the multiply so the copy overlaps it, and fenced after: the group has to be
        // committed even on the last step, or the wait above is waiting on the wrong count.
        const int next = step + POW_STAGES - 1;
        const int write_stage = (step + POW_STAGES - 1) % POW_STAGES;
        if (next < total_steps) {
            for (int chunk = tid; chunk < rows_valid * 16 * (POW_BK / 16); chunk += POW_THREADS) {
                pow_stage_copy(sA[write_stage], a_base, k, next * POW_BK, chunk);
            }
            for (int chunk = tid; chunk < cols_valid * 16 * (POW_BK / 16); chunk += POW_THREADS) {
                pow_stage_copy(sB[write_stage], b_base, k, next * POW_BK, chunk);
            }
        }
        pow_cp_async_fence();

        const int rd = step % POW_STAGES;

        // One warp per tile row, and each warp holds that row's sixteen columns. The row's A operand
        // is loaded once and reused across all `POW_BLOCK_COLS` of them, which is what the columns
        // axis of the block rectangle buys.
        //
        // A column the block rectangle does not cover is still multiplied, because the loop bound is
        // the block's extent and not `cols_valid`. That is deliberate and harmless: those
        // accumulators reach no transcript, because the hash below is gated on `cols_valid`, and the
        // reads stay inside the `sB` array -- shared memory has no bounds check, but it also cannot
        // fault, so this is wasted arithmetic rather than undefined behaviour.
        if (warp < rows_valid) {
#if POW_DP4A
            // Lane `l` owns rows `l >> 2` and `(l >> 2) + 8`, columns `(l & 3) * 4 + j` for
            // j in 0..3. Eight cells per lane per tile column, which is the whole 16x16 over 32
            // lanes -- the same register count the `mma` path spends on it.
            const signed char* a_lo = sA[rd] + (warp * 16 + lane_group) * POW_SMEM_STRIDE;
            const signed char* a_hi = a_lo + 8 * POW_SMEM_STRIDE;
            const int col_quarter = (lane & 3) * 4;

            int aw[2][POW_BK / 4];
            #pragma unroll
            for (int kw = 0; kw < POW_BK / 4; ++kw) {
                aw[0][kw] = (int)pow_ld32(a_lo + kw * 4);
                aw[1][kw] = (int)pow_ld32(a_hi + kw * 4);
            }

            #pragma unroll
            for (int t = 0; t < POW_BLOCK_COLS; ++t) {
                #pragma unroll
                for (int kw = 0; kw < POW_BK / 4; ++kw) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const unsigned int b =
                            pow_ld32(sB[rd] + (t * 16 + col_quarter + j) * POW_SMEM_STRIDE + kw * 4);
                        acc[t][j]     = pow_dp4a(acc[t][j],     aw[0][kw], b);
                        acc[t][4 + j] = pow_dp4a(acc[t][4 + j], aw[1][kw], b);
                    }
                }
            }
#else
            // Two `mma` tiles of eight columns, since m16n8k32 produces an 8-wide tile.
#if POW_LDMATRIX
            // Nine `LDSM` for the thirty-six `LDS.32` this used to take. One for A, loaded once and
            // reused across all eight tile columns -- which is what the rectangle's rows axis buys --
            // and one per tile column for B, each covering both eight-column halves at once.
            unsigned int a[4];
            // One `ldmatrix` per 32-byte k-block, so a 64-wide stage issues two per tile and the
            // barrier above is amortised over twice the multiply-accumulates. `ldmatrix` addresses
            // 16 bytes within the block it is given, so the block index is just a pointer offset --
            // which also keeps every fragment address 16-byte aligned, as the copy that wrote it was.
            #pragma unroll
            for (int kb = 0; kb < POW_BK / 32; ++kb) {
                pow_ldmatrix_x4(a, sA[rd] + warp * 16 * POW_SMEM_STRIDE + kb * 32, lane);

                #pragma unroll
                for (int t = 0; t < POW_BLOCK_COLS; ++t) {
                    unsigned int b[4];
                    pow_ldmatrix_x4(b, sB[rd] + t * 16 * POW_SMEM_STRIDE + kb * 32, lane);
                    pow_mma(acc[t][0], a[0], a[1], a[2], a[3], b[0], b[2]);
                    pow_mma(acc[t][1], a[0], a[1], a[2], a[3], b[1], b[3]);
                }
            }
#else
            const signed char* a_row = sA[rd] + (warp * 16 + lane_group) * POW_SMEM_STRIDE;
            const signed char* a_row_hi = a_row + 8 * POW_SMEM_STRIDE;

            // As above, one pass per 32-byte k-block within the stage; `lane_k` and `lane_k_hi`
            // address the four k values each side of an `mma` fragment, and `kb * 32` steps to the
            // next one along the staged row.
            #pragma unroll
            for (int kb = 0; kb < POW_BK / 32; ++kb) {
                const unsigned int a0 = pow_ld32(a_row + kb * 32 + lane_k);
                const unsigned int a1 = pow_ld32(a_row_hi + kb * 32 + lane_k);
                const unsigned int a2 = pow_ld32(a_row + kb * 32 + lane_k_hi);
                const unsigned int a3 = pow_ld32(a_row_hi + kb * 32 + lane_k_hi);

                #pragma unroll
                for (int t = 0; t < POW_BLOCK_COLS; ++t) {
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const signed char* b_row =
                            sB[rd] + (t * 16 + h * 8 + lane_group) * POW_SMEM_STRIDE + kb * 32;
                        pow_mma(acc[t][h], a0, a1, a2, a3,
                                pow_ld32(b_row + lane_k), pow_ld32(b_row + lane_k_hi));
                    }
                }
            }
#endif
#endif
        }

        if ((step + 1) % steps_per_rank == 0) {
            const int slot = (step / steps_per_rank) % POW_TRANSCRIPT_WORDS;
            unsigned int pv[POW_BLOCK_COLS];
            #pragma unroll
            for (int t = 0; t < POW_BLOCK_COLS; ++t) {
                unsigned int x = 0u;
                if (warp < rows_valid) {
#if POW_DP4A
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        x ^= acc[t][r];
                    }
#else
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        x ^= acc[t][h][0] ^ acc[t][h][1] ^ acc[t][h][2] ^ acc[t][h][3];
                    }
#endif
                }
                // Position-independent, which is the whole reason this is a shuffle and not a tree:
                // nothing downstream can tell which cell of the 16x16 a contribution came from.
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    x ^= __shfl_xor_sync(0xffffffffu, x, off);
                }
                pv[t] = x;
            }
            // `__syncwarp` rather than `__syncthreads`: both the read and the write of `sT[warp]` are
            // this warp's, so the only ordering that matters is the warp's own.
            __syncwarp();
            if (lane == 0) {
                #pragma unroll
                for (int t = 0; t < POW_BLOCK_COLS; ++t) {
const unsigned int prev = POW_ST_AT(warp, t, slot);
                    POW_ST_AT(warp, t, slot) = ((prev << POW_HASH_ROT) | (prev >> (32 - POW_HASH_ROT)))
                                           ^ pv[t];
                }
            }
            __syncwarp();
        }
    }

    __syncthreads();

    // Four lanes per tile: each reads all sixteen transcript words, so every lane hashes a whole
    // transcript and no lane sits idle while another waits on a reduction. That makes a warp's eight
    // tile columns `lane / 4`, and the gate below is the whole warp -- there is no lane of a warp
    // without a tile. `lane / 4` and not `tid / 4`: the tile row is already `warp`, so a block-wide
    // division hands warp 1 the tile columns eight to fifteen, and every warp past the first hashes
    // nothing that clears `cols_valid`.
    const int tile_in_warp = lane / 4;
    const bool hash_here = (lane < POW_BLOCK_COLS * 4) && (warp < rows_valid)
                        && (tile_in_warp < cols_valid);

    unsigned int words[16];
    if (hash_here) {
        // POW_SKIP_HASH is a benchmark-only ablation: it removes the per-tile BLAKE3 and the
        // bound compare so the harness can price the hash against the fold. Never defined in the
        // shipped build, so this is the production path.
#if defined(POW_SKIP_HASH)
        if (key_words[0] == 0xffffffffu) { words[0] = 1u; }
#else
        // The sixteen transcript words are already contiguous 64 bytes in `sT`, and BLAKE3 loads
        // them little-endian, so `b3_chunk_cv` can read them in place. `POW_HASH_DIRECT` skips
        // building `message[]` entirely -- which is not a micro-optimisation: the profile shows
        // 7.04 MB of *local* memory traffic per launch at 100% overhead, i.e. register spilling,
        // and an 80-byte stack frame with 16 bytes of spill stores. A 64-byte local buffer held
        // live across the call is exactly the shape of thing that spills.
#if POW_HASH_DIRECT
        b3_chunk_cv(reinterpret_cast<const unsigned char*>(&POW_ST_AT(warp, tile_in_warp, 0)), 64u,
                    0ull, key_words, B3_KEYED_HASH, B3_ROOT, words);
#else
        // Measured and kept: reading the transcript in place instead of copying it into a local
        // `message[64]` is *correct and marginally cheaper* but does not move the kernel — 137.9
        // against 138.8, inside run-to-run noise, and the register count stays at 128 either way.
        // The spill is not this buffer. It is the accumulator running into the `__launch_bounds__`
        // cap, which `-Xptxas -v` shows as an 80-byte stack frame and 16 bytes of spill stores
        // whether or not the buffer is there. Left on because it removes a redundant copy, not
        // because it is a speedup — the comment above says so, and the measurement that decided
        // it is recorded here rather than the flag being quietly defaulted.
        unsigned char message[64];
        #pragma unroll
        for (int i = 0; i < POW_TRANSCRIPT_WORDS; ++i) {
            const unsigned int word = POW_ST_AT(warp, tile_in_warp, i);
            message[i * 4]     = (unsigned char)(word);
            message[i * 4 + 1] = (unsigned char)(word >> 8);
            message[i * 4 + 2] = (unsigned char)(word >> 16);
            message[i * 4 + 3] = (unsigned char)(word >> 24);
        }
        b3_chunk_cv(message, 64u, 0ull, key_words, B3_KEYED_HASH, B3_ROOT, words);
#endif
#endif

        // The verifier's rule is `U256::from_little_endian(hash_jackpot) <= bound`
        // (`sanity_checks::check_jackpot_against_nbits`), so digest byte 0 is the *least* significant
        // one and `words[i]` -- already the little-endian read of digest bytes `[4i..4i+4)` -- is the
        // bound's group `7 - i`, most significant group last in the digest and first in the bound.
        //
        // So the walk pairs the bound from the top with the digest from the other end, and needs no
        // byte reversal at all: `words[7 - i]` already reads `d[4(7-i)] + d[4(7-i)+1]*2^8 + ...`,
        // which is the same expression as the bound's group `i`, `d[28-4i]*2^24 + ... + d[31-4i]`.
        //
        // Both halves of that are easy to get wrong in ways nothing downstream can see. The hash stays
        // a real BLAKE3 digest either way, and roughly half of all candidates clear *some* reading of
        // the bound, so a search with the wrong one reports wins the pool rejects and drops the ones
        // it should have kept.
        bool under = true;
        for (int i = 0; i < 8; ++i) {
            const unsigned int have = words[7 - i];
            const unsigned int want = bound[i];
            if (have > want) {
                under = false;
                break;
            }
            if (have < want) {
                break;
            }
        }

        if (under) {
            // Absolute tile row and column, but a **batch-local** region: `found` and `coord` are
            // sized for the regions of this batch, and the host reads `found[..regions]` back
            // knowing nothing else. So the region's linear index has the batch's own offset taken
            // off, not added on.
            //
            // `row0` already carries `first_tile / tiles_per_row`, so `tr * tiles_per_row + tc` is
            // the absolute tile index and subtracting `first_tile` lands on the batch-local one.
            // Getting this wrong is silent in one direction and destructive in the other: adding
            // `first_tile` again puts the index past the end of `found` on every batch after the
            // first, and dropping the offset entirely makes every batch claim region `tr` of the
            // *grid*, which for the production geometry is 8192 times past the end of a 64-word
            // buffer.
            const unsigned int tr = row0 + (unsigned int)warp;
            const unsigned int tc = col0 + (unsigned int)tile_in_warp;
            const unsigned int region = (tr * tiles_per_row + tc - first_tile) / tiles_per_region;
            if (atomicCAS(found + region, 0u, 1u) == 0u) {
                coord[region * 2]     = tr;
                coord[region * 2 + 1] = tc;
            }
        }
    }

    if (transcript_out != nullptr) {
        // Written unconditionally, including for the tiles the rectangle does not cover: a tile
        // outside the grid has to read as something specific, or a grid-shape bug hides behind the
        // tiles that happen to be covered.
        const unsigned int block_base = blockIdx.x * POW_TILES_PER_BLOCK;
        for (int i = tid; i < POW_TILES_PER_BLOCK * POW_TRANSCRIPT_WORDS; i += POW_THREADS) {
            const int w = i / (POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS);
            const int rest = i - w * POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS;
            const int t = rest / POW_TRANSCRIPT_WORDS;
            const int word = rest - t * POW_TRANSCRIPT_WORDS;
            unsigned int value = 0u;
            if (w < rows_valid && t < cols_valid) {
                value = POW_ST_AT(w, t, word);
            }
            transcript_out[block_base * POW_TRANSCRIPT_WORDS + i] = value;
        }
    }
}


/// `ApEA = A + EAL·EARᵀ` for one row window, as `signed char`.
///
/// The noise-apply, and the step that makes the fold a single GEMM: consensus multiplies `A + noise`
/// rather than expanding into four products, and this is where the sum is formed. The reference
/// splits it into a DP4A product into `i32` and then a quantise-and-clamp, because it wants the
/// `i32` side-product for its own tensor-core path; the operand ranges already put the sum in
/// `[-127, 126]`, so one pass with no intermediate is both smaller and exact.
///
/// Only `rows` rows of `out` are written. `dense` is *already* the matching window of EAL or EBR —
/// the search asks `tokenminer_noise_dense` for exactly those rows — so there is no absolute row
/// index to reconcile here, which is why this kernel takes no `first_row`.
///
/// `rank` must be a multiple of 32 and `k` a multiple of `rank`; the host checks both.
extern "C" __global__ void tokenminer_noise_apply(
        const signed char* signal,        // rows x k, row-major
        const signed char* dense,         // rows x rank, row-major (the EAL or EBR window)
        const unsigned int* pairs,        // k pairs, from tokenminer_noise_perm
        signed char* out,                 // rows x k, row-major
        unsigned int rows,
        unsigned int k,
        unsigned int rank) {
    const unsigned long long total = (unsigned long long)rows * k;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += stride) {
        const unsigned int u = (unsigned int)(i / k);
        const unsigned int l = (unsigned int)(i % k);

        // The sparse pair names two columns of this row's dense noise; the difference is the entry.
        const unsigned int first = pairs[l * 2];
        const unsigned int second = pairs[l * 2 + 1];

        const signed char* dense_row = dense + (unsigned long long)u * rank;
        const int sum = (int)signal[i] + (int)dense_row[first] - (int)dense_row[second];

        out[i] = (signed char)sum;
    }
}