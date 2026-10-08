// fold_sm120.cuh
//
// Hand-written sm_120 int8 fold: warp tile `16 * FOLD_M_ATOMS` x 128, one `ldmatrix.x4` per 32-byte
// k-block, staged through shared memory with a `cp.async` pipeline.
//
// The n-extent is forced. A candidate is 2 rows x 128 columns and the fold's XOR is over all 256 of its
// cells, so a warp can finish a candidate's reduction without leaving the warp only if it owns all 128 of
// the candidate's columns. The m-extent is free, and it is the only shape knob:
//
//   m_atoms   mma per ldmatrix    accumulators/thread   fits a 128-row block?
//     1            1.78                  64                  yes
//     2            3.20                 128                  yes
//     3            4.36                 192                  no -- needs a 384-row block
//     4            5.33                 256                  no -- past the 255-register ceiling
//
// The blob sits at 4x8 (ratio 4.0) and pays a cross-warp reduction for it. Everything about this file is
// about reaching a comparable ratio without that reduction, at `m_atoms = 3`.
//
// Three things here were wrong for a long time and the harness could not see any of them, because it
// copied the blob's output into the buffer before running ours -- so an unwritten word read as the blob's
// own answer and every comparison passed. Each is worth naming because each produced a *plausible* result:
//
//   * `fold_mma` declared its accumulator write-only. An `mma` adds into `%0..%3`, so with `"=r"` and the
//     same array listed again as an input the instruction added into whatever register `%0` happened to
//     name. The fold read all zeros for every candidate, and a zero fold made every timing look good.
//   * the checkpoint index was not added to the store address, so all sixteen checkpoints wrote one word
//     and only the last survived.
//   * `fold_mma` was handed `&bf[0]`, whose second word is matrix 1 -- the *second* eight-column atom's
//     k 0-15 -- where the first atom's k 16-31 belongs. `ldmatrix.x4` puts the two k-halves of an atom in
//     matrices 0 and 2, not 0 and 1.
//
// The staging also only wrote one B tile-column per warp, which left the columns above the warp count
// uninitialised. At m=2 there are four warps and columns 4..7 were never loaded at all.

#pragma once

#ifndef FOLD_M_ATOMS
#define FOLD_M_ATOMS 2
#endif
#ifndef FOLD_STAGES
#define FOLD_STAGES 3
#endif
#ifndef FOLD_BLOCK_ROWS
#define FOLD_BLOCK_ROWS 128
#endif

// One `ldmatrix.x4` covers a 16-row, 32-byte k-block -- four 8x8 b16 matrices. `FOLD_BK` is the k extent
// of one stage, and it has to be a whole number of those blocks; 32 is the only one the addressing below
// implements, so it is not a knob. (At 64 the kernel compiled and ran and produced *nearly* right numbers,
// which is the worst kind of wrong.)
#define FOLD_BK 32

#define FOLD_N_ATOMS 16
// B fragments per k-block: 128 columns of 16 = eight, each feeding two n8 atoms of the accumulator
// (`FOLD_N_ATOMS` is the *accumulator* count, which is twice this -- conflating the two reads past
// the end of the accumulator array and is a wrong answer, not a fault).
#define FOLD_B_TILES 8
#define FOLD_WARP_ROWS (16 * FOLD_M_ATOMS)
#define FOLD_WARPS (FOLD_BLOCK_ROWS / FOLD_WARP_ROWS)
#define FOLD_THREADS (FOLD_WARPS * 32)
#define FOLD_TILES_PER_BLOCK (FOLD_BLOCK_ROWS / 128)

// `16 (mod 32)` is the rule that keeps `ldmatrix` conflict-free: a matrix is eight rows of 16 bytes, so
// its rows must start on eight distinct banks. A plain `FOLD_BK` is `0 (mod 32)`, which puts every row on
// the same bank and costs four times what it should.
#define FOLD_STRIDE (FOLD_BK + 16)

// The verifier's fold group is `r = 128` and the blob checkpoints every 128 columns, so the checkpoint
// period is 128 regardless of how the k-blocks are staged.
#define FOLD_CHECKPOINT_STAGES (128 / FOLD_BK)

// `FOLD_STAGES - 1` k-blocks are in flight. Stage count is a correctness parameter, not just a depth:
// two stages would have the issue for k-block `kb+1` land in the same buffer a lagging warp is still
// reading, and there is no barrier between the two to stop it.
#define FOLD_PREFETCH (FOLD_STAGES - 1)

#define FOLD_GROUP_SIZE_M 8
#define FOLD_CANDIDATES 64
#define FOLD_WORDS 16

// The two staging arrays are one allocation with different per-stage strides. Stepping both by the total
// would put `A`'s stage 1 in the middle of `B`.
#define FOLD_A_STAGE (FOLD_BLOCK_ROWS * FOLD_STRIDE)
// `FOLD_BPIPE` is `FOLD_BDIRECT` plus register double-buffering: B stays out of shared *and* its
// global load is issued a full k-block before it is consumed, so the mma never waits on it. A naive
// B-direct reads B back synchronously inside the same iteration and pays the whole latency; that gap
// is what this closes.
#ifdef FOLD_BPIPE
#define FOLD_BDIRECT
#endif
// `FOLD_BDIRECT` reads B straight from a fragment-ordered global buffer (one `LDG.128` a lane a
// k-block) instead of staging it through shared, so it needs no B shared at all. That is the point:
// B is 8 of every warp's 10 `ldmatrix`, and the shared read of B is the largest single term in the
// fold's energy (see the energy probe). `L1` serves the warps that share a k-block's 4 KB.
#ifdef FOLD_BDIRECT
#define FOLD_B_STAGE 0
#else
#define FOLD_B_STAGE (8 * 16 * FOLD_STRIDE)
#endif
#define FOLD_SMEM (FOLD_STAGES * (FOLD_A_STAGE + FOLD_B_STAGE))

// One `ldmatrix.x4` over a 16-row, 32-byte tile: the A operand of one `mma`, or the B operand of two.
//
// The four matrices are ordered by the addresses the lanes supply, and the addressing below makes them
// (rows 0-7, bytes 0-15), (rows 8-15, bytes 0-15), (rows 0-7, bytes 16-31), (rows 8-15, bytes 16-31).
// Lane `l` supplies the address of row `(l&7)` of matrix `l>>3`, and receives the four bytes at row
// `l>>2`, bytes `(l&3)*4 .. +3` of each matrix.
__device__ __forceinline__ void fold_ldmatrix_x4(unsigned int* out, const signed char* smem, int lane) {
    const int row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int koff = (lane >> 4) * 16;
    const unsigned int addr =
        (unsigned int)__cvta_generic_to_shared(smem + row * FOLD_STRIDE + koff);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(addr));
}

// `b0` and `b1` are values rather than a pointer because the two k-halves of one B atom are matrices 0
// and 2 of the `ldmatrix.x4` result, not 0 and 1 -- see `fold_ldmatrix_x4`.
__device__ __forceinline__ void fold_mma(unsigned int* d, const unsigned int* a,
                                         unsigned int b0, unsigned int b1) {
    // `"+r"`: the accumulator is `{%0,%1,%2,%3}` on both sides of the instruction. Declaring it `"=r"`
    // and passing `d[]` again as an input does not tie the two together -- the input operands start at
    // `%4`, so `%0` is not required to hold `d[]` on entry and the mma adds into whatever it names.
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b0), "r"(b1));
}

__device__ __forceinline__ void fold_cp_async(unsigned int saddr, const signed char* gptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr));
}

// One lane's B fragment for one 16-column block of one k-block, read from the fragment-ordered
// buffer the host builds. `ld.global.nc` so it goes through the read-only path, `v4` so it is one
// 128-bit transaction. Volatile because these loads are the only thing feeding the mma and ptxas
// must not hoist, sink or merge them across the barrier.
//
// The four words are exactly what `fold_ldmatrix_x4` would have returned for the same block --
// matrices (0,1) and (2,3) are b0/b1 and b2/b3 -- so `fold_mma` takes them unchanged.
__device__ __forceinline__ void fold_ldg_b(unsigned int* bf, const signed char* p) {
    asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(bf[0]), "=r"(bf[1]), "=r"(bf[2]), "=r"(bf[3])
                 : "l"(p)
                 : "memory");
}

// The fold. `a_sum` is `(m, k)` row-major, `b_sum` is `(n, k)` row-major -- the same bytes as the
// `(k, n)` column-major view the mma wants -- and `transcripts` is `(tiles, 64, 16)` u32.
//
// `dbg` may be null. When it is not, block 0's warp 0 writes the staged bytes and the loaded fragments of
// its first k-block, which is what separated "staging is empty" from "the fragment load is wrong" when
// this kernel was silent.
extern "C" __global__ __launch_bounds__(FOLD_THREADS) void fold_sm120(
        const signed char* __restrict__ a_sum,
        const signed char* __restrict__ b_sum,
        unsigned int* __restrict__ transcripts,
        int m, int n, int k, int stride_am, int stride_bn,
        unsigned int* __restrict__ dbg) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int num_pid_m = m / FOLD_BLOCK_ROWS;
    const int num_pid_n = n / 128;

    // The same swizzle the blob uses, so the L2 behaviour is comparable and not a second variable. The
    // transcript index is the *linear* tile index regardless of the swizzle.
    const int pid = blockIdx.x;
    const int num_pid_in_group = FOLD_GROUP_SIZE_M * num_pid_n;
    const int group_id = pid / num_pid_in_group;
    const int first_pid_m = group_id * FOLD_GROUP_SIZE_M;
    int gsz = num_pid_m - first_pid_m;
    if (gsz > FOLD_GROUP_SIZE_M) gsz = FOLD_GROUP_SIZE_M;
    const int local = pid - group_id * num_pid_in_group;
    const int pid_m = first_pid_m + (local % gsz);
    const int pid_n = local / gsz;

    extern __shared__ __align__(16) unsigned char smem[];
    signed char* sA = (signed char*)smem;
    signed char* sB = (signed char*)smem + FOLD_STAGES * FOLD_A_STAGE;

    // Warp `warp` owns rows `16*FOLD_M_ATOMS*warp .. +` of the block. Those fall in tile
    // `warp * FOLD_WARP_ROWS / 128` of the block, and the candidate base inside it is
    // `(warp * FOLD_WARP_ROWS % 128) / 2` -- a candidate is two rows, so the row offset halves.
    const int warp_rows_in_block = warp * FOLD_WARP_ROWS;
    const int tile_m = pid_m * FOLD_TILES_PER_BLOCK + warp_rows_in_block / 128;
    const int cand_base = (warp_rows_in_block % 128) / 2;

    const signed char* a_base = a_sum + (long long)(pid_m * FOLD_BLOCK_ROWS + warp_rows_in_block) * stride_am;

    const int k_blocks = k / FOLD_BK;
    const int a_chunks = FOLD_WARP_ROWS * FOLD_BK / (32 * 16);

    // B is 128 rows of `FOLD_BK` bytes -- eight tile-columns of sixteen rows -- and it is staged *once per
    // block* and split across the warps. Loading it per warp instead (each warp reading all 128 rows) is
    // correct but copies the whole tile `FOLD_WARPS` times: at m=2 that was four redundant copies, which
    // is where most of this kernel's staging traffic went.
    const int b_chunks_total = 128 * (FOLD_BK / 16);
    const int b_chunks_per_warp = b_chunks_total / FOLD_WARPS;
    const int b_per_lane = b_chunks_per_warp / 32;

#ifdef FOLD_BDIRECT
    // B's fragment-ordered buffer for this block's 128 columns. The host lays it out as
    // [n/128][k/FOLD_BK][t 8][lane 32][4 words], where `t` is a sixteen-column block and the four
    // words are the four the equivalent `fold_ldmatrix_x4` would have handed the lane (matrices 0/2
    // are the first atom's two k-halves, 1/3 the second's -- see `fold_ldmatrix_x4`). Fixed `t` and
    // 32 lanes touch one contiguous 512-byte run, so a warp's load is fully coalesced.
    const unsigned int* b_frag_base =
        (const unsigned int*)b_sum + (long long)pid_n * (k / FOLD_BK) * 8 * 32 * 4;
#endif

    // A is loaded by the warp that owns its rows, so it is already split; only B was redundant.
    auto issue = [&](int stage, int kb) {
        #pragma unroll
        for (int i = 0; i < a_chunks; ++i) {
            const int cid = lane + 32 * i;
            const int r = cid / (FOLD_BK / 16);
            const int c = (cid % (FOLD_BK / 16)) * 16;
            fold_cp_async((unsigned int)__cvta_generic_to_shared(
                              sA + stage * FOLD_A_STAGE + warp_rows_in_block * FOLD_STRIDE + r * FOLD_STRIDE + c),
                          a_base + (long long)r * stride_am + kb * FOLD_BK + c);
        }

#ifndef FOLD_BDIRECT
        #pragma unroll
        for (int i = 0; i < b_per_lane; ++i) {
            const int cid = warp * b_chunks_per_warp + lane + 32 * i;
            const int r = cid / (FOLD_BK / 16);
            const int c = (cid % (FOLD_BK / 16)) * 16;
            fold_cp_async((unsigned int)__cvta_generic_to_shared(
                              sB + stage * FOLD_B_STAGE + r * FOLD_STRIDE + c),
                          b_sum + (long long)(pid_n * 128 + r) * stride_bn + kb * FOLD_BK + c);
        }
#endif

        asm volatile("cp.async.commit_group;\n");
    };

    // Prologue: `FOLD_PREFETCH` k-blocks in flight, so the wait below never drains the queue dry.
    #pragma unroll
    for (int s = 0; s < FOLD_PREFETCH; ++s) {
        if (s < k_blocks) issue(s, s);
    }

    unsigned int acc[FOLD_M_ATOMS][FOLD_N_ATOMS][4];
    #pragma unroll
    for (int a = 0; a < FOLD_M_ATOMS; ++a) {
        #pragma unroll
        for (int i = 0; i < FOLD_N_ATOMS; ++i) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[a][i][r] = 0u;
        }
    }

    // The transcript slot of the warp's *first* candidate. A slot is `FOLD_WORDS` words, one per
    // checkpoint, so the word checkpoint `chk` writes is `slot + chk`.
    const long long out_base =
        ((long long)(tile_m * num_pid_n + pid_n) * FOLD_CANDIDATES + cand_base) * FOLD_WORDS;
    const int g = lane >> 3;

#ifdef FOLD_BPIPE
    // B double-buffered in registers. `bcur` holds the k-block the mma below consumes; `bnext` is
    // filled by the prefetch at the top of the iteration with the *following* k-block. At the end of
    // the iteration `bnext` is rotated into `bcur` -- an element-wise copy of static arrays, which
    // ptxas renames away when it unrolls the loop by two, so the register file holds two genuine
    // buffers and no move executes. What it buys: the `LDG.128` for k-block `kb+1` is in flight for
    // the whole of k-block `kb`'s mma, which is the latency the naive `FOLD_BDIRECT` pays on the
    // critical path. Same coalesced 512-byte run per (t, warp) as `FOLD_BDIRECT`.
    unsigned int bcur[FOLD_B_TILES][4], bnext[FOLD_B_TILES][4];
    #pragma unroll
    for (int t = 0; t < FOLD_B_TILES; ++t) {
        fold_ldg_b(bcur[t], (const signed char*)(b_frag_base + ((long long)t * 32 + lane) * 4));
    }
#endif

#ifdef FOLD_BPIPE
    #pragma unroll 2
#endif
    for (int kb = 0; kb < k_blocks; ++kb) {
#ifdef FOLD_BPIPE
        // Issue the next k-block's B *before* the barrier, into the other register buffer. The load
        // retires into the scoreboard and is in flight across the `cp.async` wait, the barrier and the
        // whole k-block of mma -- so the mma never waits on it. That coverage is the entire difference
        // between this and `FOLD_BDIRECT`, which reads B synchronously and pays the latency inline.
        if (kb + 1 < k_blocks) {
            #pragma unroll
            for (int t = 0; t < FOLD_B_TILES; ++t) {
                fold_ldg_b(bnext[t],
                           (const signed char*)(b_frag_base
                               + (((long long)(kb + 1) * FOLD_B_TILES + t) * 32 + lane) * 4));
            }
        }
#endif
        // At most `FOLD_PREFETCH - 1` groups pending, so the oldest -- stage `kb % FOLD_STAGES` -- is
        // complete. On the tail the queue is shorter than the prefetch depth and the request has to be
        // zero, or the wait returns before the stage it is about to read has landed.
        if (kb + FOLD_PREFETCH < k_blocks) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(FOLD_PREFETCH - 1));
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        const int stage = kb % FOLD_STAGES;
        const signed char* stageA = sA + stage * FOLD_A_STAGE;
#if defined(FOLD_BDIRECT) && !defined(FOLD_BPIPE)
        // All eight of this k-block's B fragments, one 16-byte load a lane a tile-column. Loaded
        // before the A fragments so the global latency overlaps the ldmatrix, and once per block
        // rather than once per m-atom.
        unsigned int bgl[FOLD_B_TILES][4];
        #pragma unroll
        for (int t = 0; t < FOLD_B_TILES; ++t) {
            fold_ldg_b(bgl[t], (const signed char*)(b_frag_base
                            + (((long long)kb * FOLD_B_TILES + t) * 32 + lane) * 4));
        }
#elif !defined(FOLD_BDIRECT)
        const signed char* stageB = sB + stage * FOLD_B_STAGE;
#endif

        if (dbg != nullptr && kb == 0 && blockIdx.x == 0 && warp == 0 && lane == 0) {
            dbg[0] = (unsigned int)(unsigned char)stageA[0];
            dbg[1] = (unsigned int)(unsigned char)stageA[FOLD_STRIDE];
            dbg[2] = (unsigned int)(unsigned char)stageA[16 * FOLD_STRIDE];
#ifdef FOLD_BPIPE
            dbg[3] = bcur[0][0]; dbg[4] = bcur[0][1]; dbg[5] = bcur[0][2];
#elif defined(FOLD_BDIRECT)
            dbg[3] = bgl[0][0]; dbg[4] = bgl[0][1]; dbg[5] = bgl[0][2];
#else
            dbg[3] = (unsigned int)(unsigned char)stageB[0];
            dbg[4] = (unsigned int)(unsigned char)stageB[FOLD_STRIDE];
            dbg[5] = (unsigned int)(unsigned char)stageB[16 * FOLD_STRIDE];
#endif
        }

        #pragma unroll
        for (int a = 0; a < FOLD_M_ATOMS; ++a) {
            unsigned int af[4];
            fold_ldmatrix_x4(af, stageA + warp_rows_in_block * FOLD_STRIDE + a * 16 * FOLD_STRIDE, lane);
            if (dbg != nullptr && kb == 0 && a == 0 && blockIdx.x == 0 && warp == 0 && lane == 0) {
                dbg[6] = af[0]; dbg[7] = af[1]; dbg[8] = af[2]; dbg[9] = af[3];
            }

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
#ifdef FOLD_BPIPE
                const unsigned int* bf = bcur[t];
#elif defined(FOLD_BDIRECT)
                const unsigned int* bf = bgl[t];
#else
                unsigned int bfre[4];
                fold_ldmatrix_x4(bfre, stageB + t * 16 * FOLD_STRIDE, lane);
                const unsigned int* bf = bfre;
                if (dbg != nullptr && kb == 0 && a == 0 && t == 0 && blockIdx.x == 0 && warp == 0 &&
                    lane == 0) {
                    dbg[10] = bfre[0]; dbg[11] = bfre[1]; dbg[12] = bfre[2]; dbg[13] = bfre[3];
                }
#endif
                // Matrices 0/2 are the two k-halves of this tile-column's first eight-column atom, 1/3
                // the two halves of the second.
                fold_mma(acc[a][2 * t], af, bf[0], bf[2]);
                fold_mma(acc[a][2 * t + 1], af, bf[1], bf[3]);
            }
        }

#ifdef FOLD_BPIPE
        // Rotate: the next k-block's B becomes the current one. Static element-wise copy of arrays of
        // constant size, so it is pure register renaming once the loop is unrolled.
        if (kb + 1 < k_blocks) {
            #pragma unroll
            for (int t = 0; t < FOLD_B_TILES; ++t) {
                #pragma unroll
                for (int w = 0; w < 4; ++w) bcur[t][w] = bnext[t][w];
            }
        }
#endif

        // Issue for a later k-block *after* the barrier, so the write cannot land in a buffer a lagging
        // warp is still reading. `kb + FOLD_PREFETCH` is `FOLD_STAGES - 1` ahead, which is the same
        // buffer as `kb - 1` -- read one iteration ago, and separated from this write by the barrier above.
        if (kb + FOLD_PREFETCH < k_blocks) {
            issue((kb + FOLD_PREFETCH) % FOLD_STAGES, kb + FOLD_PREFETCH);
        }

        if (kb % FOLD_CHECKPOINT_STAGES == FOLD_CHECKPOINT_STAGES - 1) {
            const int chk = kb / FOLD_CHECKPOINT_STAGES;

            unsigned int px[FOLD_M_ATOMS][2];
            #pragma unroll
            for (int a = 0; a < FOLD_M_ATOMS; ++a) {
                px[a][0] = 0u;   // row 16a + l/4     -> candidate cand_base + 8a + g
                px[a][1] = 0u;   // row 16a + l/4 + 8 -> candidate cand_base + 8a + g + 4
                #pragma unroll
                for (int i = 0; i < FOLD_N_ATOMS; ++i) {
                    px[a][0] ^= acc[a][i][0] ^ acc[a][i][1];
                    px[a][1] ^= acc[a][i][2] ^ acc[a][i][3];
                }

                // Butterfly over the eight lanes of group `g`: selectors 4, 2, 1 touch bits 2..0 and
                // leave `lane>>3` alone, so the reduction stays inside the group.
                #pragma unroll
                for (int s = 4; s >= 1; s >>= 1) {
                    px[a][0] ^= __shfl_xor_sync(0xffffffffu, px[a][0], s);
                    px[a][1] ^= __shfl_xor_sync(0xffffffffu, px[a][1], s);
                }
            }

            if ((lane & 7) == 0) {
                #pragma unroll
                for (int a = 0; a < FOLD_M_ATOMS; ++a) {
                    transcripts[out_base + (8 * a + g) * FOLD_WORDS + chk] = px[a][0];
                    transcripts[out_base + (8 * a + g + 4) * FOLD_WORDS + chk] = px[a][1];
                }
            }
        }
    }
}
