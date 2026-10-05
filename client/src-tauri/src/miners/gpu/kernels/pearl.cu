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
// tile, and that is the entire reason this kernel is not memory-bound. One tile per block re-reads
// both of its operands whole: sixteen rows of `k` from each side, so `32 * k` bytes moved for
// `16 * 16 * k` multiply-accumulates. A rectangle amortises each side across the other axis --
// `16 * k * (1/cols + 1/rows)` bytes per tile -- which is 12 KiB against 128 KiB at the shipped
// dimensions. The per-tile form runs at about 66% of an RTX 5080's DRAM bandwidth and cannot go
// faster; this one needs under a third of it and is bound by instruction issue instead.
//
// These four are measured, not guessed, and the measurement agrees with the arithmetic once the
// arithmetic is done in the right order.
//
// Per staged k-step a block moves `POW_BLOCK_ROWS * 16 * POW_BK` bytes of A and
// `POW_BLOCK_COLS * 16 * POW_BK` of B through shared memory for `POW_BLOCK_ROWS * POW_BLOCK_COLS *
// 16 * 16 * POW_BK` multiply-accumulates, so per-tile traffic is `POW_BK * (1 / cols + 1 / rows)`
// bytes. That is minimised by splitting the rectangle as evenly as it can go, and it is the shipped
// `16 x 8` that wins it, not `32 x 4`: 6.0 bytes per tile against 9.0. `32 x 4` moves **half again**
// as much shared-memory traffic, because halving `cols` doubles the A term and only a quarter of the
// saving comes back from doubling `rows`.
//
// Measured on an RTX 5080 at production geometry, `32 x 4` is **16% slower** -- 61.9 against 73.5
// TMAC/s -- which is what the traffic term predicts and nothing else needs to explain. It is *not* an
// occupancy effect. The accumulator is 64 registers per lane either way, and the register file is
// what caps residency: this kernel compiles to 98 registers, and 98 * 512 = 50,176 against 65,536,
// so it sits at **one** block per SM. `32 x 4` measured 64 registers and *did* reach two blocks per
// SM -- 32 warps against 16 -- and still lost, because the traffic it added cost more than the
// doubled occupancy bought. Do not re-derive this as a barrier-overlap story: the doubled occupancy
// was real and it did not pay.
//
// So the rectangle stays `16 x 8`, and the lever that would actually help is the one this leaves
// open: `POW_SMEM_STRIDE` pads every staged row by 50%, so the block spends 45,056 bytes where
// 36,864 would do. That is not the occupancy lever either -- at one block per SM the register file,
// not shared memory, is the binding constraint -- but it is what stands between this geometry and a
// smaller stride, and a stride of 32 needs an XOR swizzle to keep the fragment loads off the same
// banks. That is the next thing to try, not a bigger rectangle.
#define POW_BLOCK_ROWS 16
#define POW_BLOCK_COLS 8
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

// k staged per tile. `mma.m16n8k32` takes k = 32, so this is exactly one instruction's worth and a
// staged buffer is one fragment deep. The DP4A fold consumes the same 32 bytes as four four-byte
// dots per lane, so the staging, the step count and the fold schedule are shared by both paths and
// only the multiply differs.
#define POW_BK 32

// The shared row stride: 48 bytes rather than 32. A 32-byte stride is 8 banks wide, so the eight row
// starts a lane group touches fold onto each other four deep and every fragment load costs four times
// what it should. 48 bytes is 12 banks, and lane `l` reads word `row * 12 + l % 4` -- 32 distinct
// banks. Padding rather than XOR swizzling, because it keeps the copy and the fragment load as plain
// 16- and 32-bit accesses; a swizzle would have to be undone by the same code that wrote it, and
// there is nothing to buy for it.
#define POW_SMEM_STRIDE (POW_BK + 16)

// Stages in flight. Two is the fewest that overlap the next k-step's copy with the current one's
// multiply, and a barrier is needed between them either way.
#define POW_STAGES 2

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
// instead double-counts the transcript, and at this geometry that is 45,056 + 9,216 = 54,272 against
// the 48 KiB a block gets with no opt-in -- so the launch fails with `cudaErrorInvalidValue` and names
// nothing about shared memory. The two totals differ by exactly the transcript, which is the whole
// reason both names exist.
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
// start on 32 distinct banks. `POW_SMEM_STRIDE` is 48 bytes = 12 banks, and twelve divides evenly
// into thirty-two's factors, so rows 0-7 start at banks 0, 12, 24, 4, 16, 28, 8, 20 -- eight
// distinct starts, and each row's four banks are disjoint from the rest. The pitch was chosen to make
// the `LDS.32` path conflict-free and it happens to satisfy `ldmatrix` too; the two choices are not
// in tension and there is nothing to buy from replacing it.
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
    // `.cg` bypasses L1: the staged bytes are read once by one block and never reused by another, so
    // caching them only evicts something that would have been.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     (unsigned int)__cvta_generic_to_shared(dst)), "l"(from));
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
extern "C" __global__ __launch_bounds__(POW_THREADS) void tokenminer_search_grid(
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
    __shared__ unsigned int sT[POW_WARPS][POW_BLOCK_COLS][POW_TRANSCRIPT_WORDS];

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
    const unsigned int strip = blockIdx.x / cols_per_strip;
    const unsigned int col_block = blockIdx.x - strip * cols_per_strip;

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
    // The barrier is load-bearing rather than tidy. The zeroing is spread over every thread, so each
    // warp clears slices other warps own; without the barrier a warp that has already folded its own
    // slice has it zeroed underneath by one still in the loop, and every transcript comes back empty.
    for (int i = tid; i < POW_WARPS * POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS; i += POW_THREADS) {
        const int w = i / (POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS);
        const int rest = i - w * POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS;
        const int t = rest / POW_TRANSCRIPT_WORDS;
        sT[w][t][rest - t * POW_TRANSCRIPT_WORDS] = 0u;
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
        pow_cp_async_wait<POW_STAGES - 2>();
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
            pow_ldmatrix_x4(a, sA[rd] + warp * 16 * POW_SMEM_STRIDE, lane);

            #pragma unroll
            for (int t = 0; t < POW_BLOCK_COLS; ++t) {
                unsigned int b[4];
                pow_ldmatrix_x4(b, sB[rd] + t * 16 * POW_SMEM_STRIDE, lane);
                pow_mma(acc[t][0], a[0], a[1], a[2], a[3], b[0], b[2]);
                pow_mma(acc[t][1], a[0], a[1], a[2], a[3], b[1], b[3]);
            }
#else
            const signed char* a_row = sA[rd] + (warp * 16 + lane_group) * POW_SMEM_STRIDE;
            const signed char* a_row_hi = a_row + 8 * POW_SMEM_STRIDE;

            const unsigned int a0 = pow_ld32(a_row + lane_k);
            const unsigned int a1 = pow_ld32(a_row_hi + lane_k);
            const unsigned int a2 = pow_ld32(a_row + lane_k_hi);
            const unsigned int a3 = pow_ld32(a_row_hi + lane_k_hi);

            #pragma unroll
            for (int t = 0; t < POW_BLOCK_COLS; ++t) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const signed char* b_row =
                        sB[rd] + (t * 16 + h * 8 + lane_group) * POW_SMEM_STRIDE;
                    pow_mma(acc[t][h], a0, a1, a2, a3,
                            pow_ld32(b_row + lane_k), pow_ld32(b_row + lane_k_hi));
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
                    const unsigned int prev = sT[warp][t][slot];
                    sT[warp][t][slot] = ((prev << POW_HASH_ROT) | (prev >> (32 - POW_HASH_ROT)))
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
        unsigned char message[64];
        #pragma unroll
        for (int i = 0; i < POW_TRANSCRIPT_WORDS; ++i) {
            const unsigned int word = sT[warp][tile_in_warp][i];
            message[i * 4]     = (unsigned char)(word);
            message[i * 4 + 1] = (unsigned char)(word >> 8);
            message[i * 4 + 2] = (unsigned char)(word >> 16);
            message[i * 4 + 3] = (unsigned char)(word >> 24);
        }
        b3_chunk_cv(message, 64u, 0ull, key_words, B3_KEYED_HASH, B3_ROOT, words);

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
                value = sT[w][t][word];
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