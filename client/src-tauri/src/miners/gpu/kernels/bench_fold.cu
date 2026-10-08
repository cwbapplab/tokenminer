// bench_fold.cu
//
// Times the hand-written 16x128 fold against the Triton blob on the SAME input, and byte-compares the
// transcripts before either number is believed.
//
// The comparison is not circular: the blob is already held against `zk_pow` by gate D, so agreeing with
// the blob on identical input means agreeing with the verifier. XOR is commutative and the int32
// accumulator cannot overflow here (max |sum| = 2048 * 127 * 127 = 33,022,976 < 2^31), so changing the
// reduction tree or the fragment order cannot change a transcript value. Any disagreement is a bug, not
// a rounding difference.
//
// Three things make the green check actually bite, and all are deliberate:
//   1. the transcripts are poisoned with 0xcccc_cccc before each run, so an unwritten slot is
//      distinguishable from a written zero;
//   2. the poison check runs on *both* outputs. It ran on the blob's alone for one round, and that made
//      the identity check vacuously true: the harness copied the blob's output into the buffer before
//      running ours, so a kernel that wrote nothing agreed with the blob on every word. Ours has to be
//      run into a poisoned buffer and be seen to have written every slot before its agreement means
//      anything.
//   3. one word of the blob's output is flipped and the harness must report a mismatch. A comparison
//      that cannot fail is not a comparison.
//
// Build:
//   nvcc -O3 -arch=sm_120a -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 \
//        -ccbin <msvc-bin> -o bench_fold.exe bench_fold.cu -lcuda
// Run:
//   bench_fold.exe <path-to-pearl_search_norotl_kernel.ptx>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "fold_sm120.cuh"

#define CHECK_CU(x)                                                                     \
    do {                                                                                \
        CUresult r_ = (x);                                                              \
        if (r_ != CUDA_SUCCESS) {                                                       \
            const char* msg = nullptr;                                                  \
            cuGetErrorString(r_, &msg);                                                 \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,           \
                         msg ? msg : "unknown driver error");                           \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

#define CHECK_RT(x)                                                                     \
    do {                                                                                \
        cudaError_t e_ = (x);                                                           \
        if (e_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,           \
                         cudaGetErrorString(e_));                                       \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

#ifndef FOLD_TEST_M
#define FOLD_TEST_M 131072
#endif
#ifndef FOLD_TEST_N
#define FOLD_TEST_N 131072
#endif

static const int M = FOLD_TEST_M;
static const int N = FOLD_TEST_N;
static const int K = 2048;
static const int TILES = (M / 128) * (N / 128);
static const int WORDS = TILES * FOLD_CANDIDATES * FOLD_WORDS;

// Our grid is in *blocks*, not tiles: a block can cover more than one 128-row tile, which is what makes
// `m_atoms = 3` expressible at all. A block that does not divide `M` is not a smaller grid -- the last
// partial block is never launched and the candidates inside it are never written, which would show up as
// a comparison failure for the wrong reason. So the shape is checked before anything runs.
static const int BLOCKS = (M / FOLD_BLOCK_ROWS) * (N / 128);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: bench_fold.exe <path-to-pearl_search_norotl_kernel.ptx>\n");
        return 1;
    }

    if (M % FOLD_BLOCK_ROWS != 0 || N % 128 != 0) {
        std::fprintf(stderr,
                    "M = %d is not a whole number of %d-row blocks, so the last partial block is never \
                     launched and the comparison would fail for the wrong reason.\n",
                    M, FOLD_BLOCK_ROWS);
        return 1;
    }

    FILE* f = std::fopen(argv[1], "rb");
    if (!f) {
        std::fprintf(stderr, "cannot read the blob: %s\n", argv[1]);
        return 1;
    }
    std::vector<char> ptx;
    for (int c; (c = std::fgetc(f)) != EOF;) ptx.push_back((char)c);
    std::fclose(f);
    // `cuModuleLoadData` takes a NUL-terminated string. A vector built by pushing bytes is not, and the
    // failure it produces is a bare "PTX JIT compilation failed" with nothing pointing at the cause.
    ptx.push_back('\0');

    CUcontext ctx;
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    // CUDA 13 renamed `cuCtxCreate` to `cuCtxCreate_v4` with an extra argument. The primary context is
    // also the one the runtime calls in this file see, which is what lets `cudaMemcpy` and the launch of
    // `fold_sm120` share the device with the driver-API launch of the blob.
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    // The blob is PTX, so the driver JIT is what turns it into SASS and `CU_JIT_MAX_REGISTERS` is a real
    // knob at load time. The recorded 211 registers cap it at `65536 / (128 * 211) = 2` blocks/SM while
    // smem would allow 3, so `maxregs` is the one lever that can move occupancy without touching the
    // Triton source. `maxregs = 0` means "no cap", which is the production path.
    auto load_blob = [&](int maxregs, CUmodule* out_mod, CUfunction* out_fn) {
        if (maxregs > 0) {
            CUjit_option opt[1] = {CU_JIT_MAX_REGISTERS};
            void* val[1] = {(void*)(intptr_t)maxregs};
            CHECK_CU(cuModuleLoadDataEx(out_mod, ptx.data(), 1, opt, val));
        } else {
            CHECK_CU(cuModuleLoadData(out_mod, ptx.data()));
        }
        CHECK_CU(cuModuleGetFunction(out_fn, *out_mod, "_pearl_search_norotl_kernel"));
        CHECK_CU(cuFuncSetAttribute(*out_fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                    (int)33792));
    };

    CUmodule mod;
    CUfunction blob;
    load_blob(0, &mod, &blob);

    const int blob_smem = 33792;

    int blob_regs = 0, blob_block = 0;
    CHECK_CU(cuFuncGetAttribute(&blob_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, blob));
    CHECK_CU(cuFuncGetAttribute(&blob_block, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, blob));

    int ours_regs = 0;
    CHECK_RT(cudaFuncSetAttribute(fold_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize, FOLD_SMEM));
    cudaFuncAttributes oa;
    CHECK_RT(cudaFuncGetAttributes(&oa, (const void*)fold_sm120));

    const int ours_smem = FOLD_SMEM;
    const int sms = 84;
    std::printf("blob   : %d regs/thread, block %d, smem %d -> %d blocks/SM\n", blob_regs, blob_block,
                blob_smem, 65536 / (128 * blob_regs));
    std::printf("ours   : %d regs/thread, block %d, smem %d -> %d blocks/SM\n", (int)oa.numRegs,
                FOLD_THREADS, ours_smem, 65536 / (FOLD_THREADS * (int)oa.numRegs));

    signed char* a_sum = nullptr;
    signed char* b_sum = nullptr;
    unsigned int* transcripts = nullptr;
    CHECK_RT(cudaMalloc(&a_sum, (size_t)M * K));
    CHECK_RT(cudaMalloc(&b_sum, (size_t)N * K));
    CHECK_RT(cudaMalloc(&transcripts, (size_t)WORDS * 4));

    // Input: a fixed pattern, not random, so a disagreement is reproducible. Values in [-64, 63] so the
    // `satfinite` path is never exercised -- a test that saturates measures saturation, not layout.
    //
    // `FOLD_BENCH_ALLONES=1` fills both operands with 1 instead, which makes the fold's answer known: the
    // k-block sums are nonzero by construction, so a side that reports all zeros is not folding the data
    // rather than folding it to zero.
    const bool all_ones = std::getenv("FOLD_BENCH_ALLONES") != nullptr;
    std::vector<signed char> ha((size_t)M * K), hb((size_t)N * K);
    // A full-period generator, not a modular one. Anything of the form `(a*i + b) % 128` has period 128,
    // and at `K = 2048` that is sixteen periods: every row comes out *identical*, the fold's window pairs
    // equal cells, and the answer is zero for a reason that has nothing to do with the kernel. Two earlier
    // versions of this harness measured that zero and called it agreement. A value that is a function of
    // the whole index has no such period.
    auto pattern = [](size_t i, unsigned long long salt) {
        unsigned long long x = i * 0x9E3779B97F4A7C15ull + salt;
        x ^= x >> 29;
        x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 32;
        return (signed char)((int)(x % 128) - 64);
    };
    for (size_t i = 0; i < ha.size(); ++i) ha[i] = all_ones ? 1 : pattern(i, 0x1111ull);
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = all_ones ? 1 : pattern(i, 0x9999ull);

    // `FOLD_BENCH_IMPULSE=1`: one non-zero cell in each operand, so the answer is known by hand. A[0][0]
    // and B[0][0] are both 1 and nothing else is, so candidate 0's window (rows {0,1}, columns 0..127) has
    // exactly one non-zero product -- at column 0 -- and its fold is 1, while every other candidate folds
    // to 0. Every other mode makes the expected value a computation; this one makes it a fact.
    const bool impulse = std::getenv("FOLD_BENCH_IMPULSE") != nullptr;
    if (impulse) {
        std::fill(ha.begin(), ha.end(), (signed char)0);
        std::fill(hb.begin(), hb.end(), (signed char)0);
        ha[0] = 1;
        hb[0] = 1;
    }

    CHECK_RT(cudaMemcpy(a_sum, ha.data(), ha.size(), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(b_sum, hb.data(), hb.size(), cudaMemcpyHostToDevice));

#ifdef FOLD_BDIRECT
    // B' in the fragment order the B-direct fold reads, built on the host once:
    //   [n/128][k/FOLD_BK][t 8][lane 32][4 u32]
    // The four words are exactly what `fold_ldmatrix_x4` would hand lane `l` for tile-column `t`
    // of one k-block, so the fold's mma take them unchanged. Lane `l` *receives* matrix m's row
    // `l>>2`, bytes `(l&3)*4 .. +3` -- not the row it supplied an address for -- and the four
    // matrices are (rows 0-7, k 0-15), (rows 8-15, k 0-15), (rows 0-7, k 16-31), (rows 8-15,
    // k 16-31); see `fold_ldmatrix_x4`. Getting this mapping wrong is exactly the class of bug the
    // correctness check below exists to catch, so it is derived here, not eyeballed.
    signed char* bfrag = nullptr;
    CHECK_RT(cudaMalloc(&bfrag, (size_t)N * K));
    {
        const int nblk = N / 128, kblk = K / FOLD_BK;
        std::vector<unsigned char> hf((size_t)N * K);
        for (int n0 = 0; n0 < nblk; ++n0) {
            for (int kb = 0; kb < kblk; ++kb) {
                for (int t = 0; t < 8; ++t) {
                    const int rowbase = n0 * 128 + t * 16;
                    for (int l = 0; l < 32; ++l) {
                        unsigned char* dst = hf.data()
                            + (((size_t)(n0 * kblk + kb) * 8 + t) * 32 + l) * 16;
                        const int r0 = rowbase + (l >> 2);
                        const int r1 = rowbase + 8 + (l >> 2);
                        const int k0 = kb * FOLD_BK + (l & 3) * 4;
                        const int k2 = k0 + 16;
                        for (int b = 0; b < 4; ++b) {
                            dst[b] = (unsigned char)hb[(size_t)r0 * K + k0 + b];
                            dst[4 + b] = (unsigned char)hb[(size_t)r1 * K + k0 + b];
                            dst[8 + b] = (unsigned char)hb[(size_t)r0 * K + k2 + b];
                            dst[12 + b] = (unsigned char)hb[(size_t)r1 * K + k2 + b];
                        }
                    }
                }
            }
        }
        CHECK_RT(cudaMemcpy(bfrag, hf.data(), hf.size(), cudaMemcpyHostToDevice));
        std::printf("B' fragment order built (%d column blocks x %d k-blocks x 8 x 32 x 16 B)\n",
                    nblk, kblk);
    }
#endif
    std::printf("operands: %s\n",
                impulse ? "impulse (A[0][0] = B[0][0] = 1, expect candidate 0 word 0 = 1)"
                        : all_ones ? "all ones" : "spread pattern");

    std::vector<unsigned int> ref(WORDS), got(WORDS);

    auto run_blob = [&]() {
        unsigned int m = M, n = N, k = K;
        void* params[10] = {&a_sum, &b_sum, &transcripts, &m, &n, &k, &k, &k, nullptr, nullptr};
        unsigned long long null0 = 0, null1 = 0;
        params[8] = &null0;
        params[9] = &null1;
        CHECK_CU(cuLaunchKernel(blob, TILES, 1, 1, 128, 1, 1, blob_smem, 0, params, nullptr));
    };

    auto run_blob_fn = [&](CUfunction fn) {
        unsigned int m = M, n = N, k = K;
        void* params[10] = {&a_sum, &b_sum, &transcripts, &m, &n, &k, &k, &k, nullptr, nullptr};
        unsigned long long null0 = 0, null1 = 0;
        params[8] = &null0;
        params[9] = &null1;
        CHECK_CU(cuLaunchKernel(fn, TILES, 1, 1, 128, 1, 1, blob_smem, 0, params, nullptr));
    };

    unsigned int* dbg = nullptr;
    CHECK_RT(cudaMalloc(&dbg, 16 * sizeof(unsigned int)));

    auto run_ours = [&]() {
#ifdef FOLD_BDIRECT
        // The B argument is the fragment-ordered buffer: same bytes, different order.
        fold_sm120<<<BLOCKS, FOLD_THREADS, FOLD_SMEM>>>(a_sum, bfrag, transcripts, M, N, K, K, K, dbg);
#else
        fold_sm120<<<BLOCKS, FOLD_THREADS, FOLD_SMEM>>>(a_sum, b_sum, transcripts, M, N, K, K, K, dbg);
#endif
    };

    // Poison, so an unwritten slot is not mistaken for a written zero. The poison is its own buffer and
    // it is uploaded before *each* run -- uploading the blob's output into `transcripts` before running
    // ours is the trap this code fell into once: an unwritten slot then inherits the blob's value, the
    // comparison agrees, and the harness reports a pass for a kernel that wrote nothing.
    const std::vector<unsigned int> poison(WORDS, 0xccccccccu);

    CHECK_RT(cudaMemcpy(transcripts, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice));
    run_blob();
    CHECK_RT(cudaDeviceSynchronize());
    CHECK_RT(cudaMemcpy(ref.data(), transcripts, ref.size() * 4, cudaMemcpyDeviceToHost));

    int poison_left = 0, blob_nonzero = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] == 0xccccccccu) ++poison_left;
        if (ref[i] != 0u) ++blob_nonzero;
    }
    std::printf("blob wrote every slot: %s (%d poisoned words left, %d non-zero words)\n",
                poison_left == 0 ? "yes" : "NO", poison_left, blob_nonzero);

    CHECK_RT(cudaMemcpy(transcripts, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice));
    run_ours();
    CHECK_RT(cudaDeviceSynchronize());
    CHECK_RT(cudaMemcpy(got.data(), transcripts, got.size() * 4, cudaMemcpyDeviceToHost));

    {
        unsigned int d[16] = {};
        CHECK_RT(cudaMemcpy(d, dbg, sizeof(d), cudaMemcpyDeviceToHost));
        std::printf("tap: sA[0]=%u sA[1row]=%u sA[16row]=%u sB[0]=%u sB[1row]=%u sB[16row]=%u\n",
                    d[0], d[1], d[2], d[3], d[4], d[5]);
        std::printf("     af=%08x %08x %08x %08x   bf=%08x %08x %08x %08x\n",
                    d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13]);
    }

    int ours_poison_left = 0, ours_nonzero = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == 0xccccccccu) ++ours_poison_left;
        if (got[i] != 0u) ++ours_nonzero;
    }
    std::printf("ours wrote every slot: %s (%d poisoned words left, %d non-zero words)\n",
                ours_poison_left == 0 ? "yes" : "NO", ours_poison_left, ours_nonzero);
    if (ours_poison_left) {
        std::printf("  the identity check below is vacuous: %d slots were never written\n",
                    ours_poison_left);
    }

    int first_diff = -1, diffs = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) {
            ++diffs;
            if (first_diff < 0) first_diff = (int)i;
        }
    }
    // Which words each side actually wrote, for the first two candidate slots. A slot that is all poison
    // on our side but not on the blob's is the shape a missing checkpoint write produces, and the printed
    // positions say which checkpoint index the write went to.
    for (int slot = 0; slot < 2; ++slot) {
        std::printf("tile 0 candidate %d\n  blob:", slot);
        for (int i = 0; i < FOLD_WORDS; ++i) std::printf(" %08x", ref[slot * FOLD_WORDS + i]);
        std::printf("\n  ours:");
        for (int i = 0; i < FOLD_WORDS; ++i) std::printf(" %08x", got[slot * FOLD_WORDS + i]);
        std::printf("\n");
    }

    std::printf("transcripts identical: %s", diffs == 0 ? "yes" : "NO");
    if (diffs) {
        const int idx = first_diff;
        std::printf(" (%d differ, first at candidate %d word %d: blob %08x ours %08x)", diffs,
                    idx / FOLD_WORDS, idx % FOLD_WORDS, ref[idx], got[idx]);
    }
    std::printf("\n");

    // The sabotage: flip one word and the harness must see it. Without this the comparison could be
    // vacuously true.
    {
        const int probe = 12345;
        unsigned int saved = ref[probe];
        ref[probe] ^= 0x1u;
        int seen = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (ref[i] != got[i]) ++seen;
        }
        std::printf("sabotage: flipped word %d, harness reports %d mismatches -> %s\n", probe, seen,
                    seen == 1 ? "the check bites" : "THE CHECK IS BLIND");
        ref[probe] = saved;
    }

    // Timing, paired. The blob's number moved 17% between two runs of the same binary, so a blob figure
    // from one run cannot be compared with an ours figure from another. Each iteration launches the two
    // back to back in the same stream, so both see the same clock and thermal state, and the events are
    // read after the queue drains so no sync cost is charged to either.
    const int iters = 30;
    std::vector<cudaEvent_t> bs(iters), be(iters), os(iters), oe(iters);
    for (int i = 0; i < iters; ++i) {
        CHECK_RT(cudaEventCreate(&bs[i]));
        CHECK_RT(cudaEventCreate(&be[i]));
        CHECK_RT(cudaEventCreate(&os[i]));
        CHECK_RT(cudaEventCreate(&oe[i]));
    }

    run_blob();
    run_ours();
    CHECK_RT(cudaDeviceSynchronize());

    for (int i = 0; i < iters; ++i) {
        CHECK_RT(cudaEventRecord(bs[i]));
        run_blob();
        CHECK_RT(cudaEventRecord(be[i]));
        CHECK_RT(cudaEventRecord(os[i]));
        run_ours();
        CHECK_RT(cudaEventRecord(oe[i]));
    }
    CHECK_RT(cudaDeviceSynchronize());

    double bsum = 0.0, osum = 0.0, bmin = 1e9, omin = 1e9;
    for (int i = 0; i < iters; ++i) {
        float b = 0.0f, o = 0.0f;
        CHECK_RT(cudaEventElapsedTime(&b, bs[i], be[i]));
        CHECK_RT(cudaEventElapsedTime(&o, os[i], oe[i]));
        bsum += b;
        osum += o;
        if (b < bmin) bmin = b;
        if (o < omin) omin = o;
    }

    const double macs = (double)TILES * 128 * 128 * K;
    if (impulse || all_ones) {
        // The operands are almost all zeros here, and the memory system compresses zero blocks: the blob
        // measured 184 TMAC/s on the impulse fill against 133 on the spread pattern, same instructions.
        // The comparisons above are still valid -- they are per-value -- but these times are not.
        std::printf("NOTE: impulse/all-ones operands compress in memory; the timings below are NOT\n"
                    "      comparable to a spread-pattern run. Correctness only.\n");
    }

    std::printf("blob mean %7.1f ms -> %6.1f TMAC/s   best %7.1f -> %6.1f\n", bsum / iters,
                macs / (bsum / iters) / 1e9, bmin, macs / bmin / 1e9);
    std::printf("ours mean %7.1f ms -> %6.1f TMAC/s   best %7.1f -> %6.1f\n", osum / iters,
                macs / (osum / iters) / 1e9, omin, macs / omin / 1e9);

    // Optional: a pre-compiled cubin to time against the JIT'd PTX. `ptxas --maxrregcount` is ignored on
    // a `.reqntid` entry point, so the only way to reach 168 registers and a third block/SM is to bake
    // `.maxnreg` in and hand the driver the cubin. Comparing the two is what says whether the occupancy
    // the cap buys is worth the spill it costs.
    if (argc >= 3) {
        FILE* cf = std::fopen(argv[2], "rb");
        if (!cf) {
            std::fprintf(stderr, "cannot read the cubin: %s\n", argv[2]);
            return 1;
        }
        std::vector<char> cubin;
        for (int c; (c = std::fgetc(cf)) != EOF;) cubin.push_back((char)c);
        std::fclose(cf);

        CUmodule cm;
        CUfunction cf2;
        CHECK_CU(cuModuleLoadData(&cm, cubin.data()));
        CHECK_CU(cuModuleGetFunction(&cf2, cm, "_pearl_search_norotl_kernel"));
        CHECK_CU(cuFuncSetAttribute(cf2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                    (int)33792));
        int cr = 0, cb = 0;
        CHECK_CU(cuFuncGetAttribute(&cr, CU_FUNC_ATTRIBUTE_NUM_REGS, cf2));
        CHECK_CU(cuFuncGetAttribute(&cb, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cf2));

        CHECK_RT(cudaMemcpy(transcripts, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice));
        run_blob_fn(cf2);
        CHECK_RT(cudaDeviceSynchronize());
        CHECK_RT(cudaMemcpy(got.data(), transcripts, got.size() * 4, cudaMemcpyDeviceToHost));
        int cmism = 0, cleft = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (got[i] != ref[i]) ++cmism;
            if (got[i] == 0xccccccccu) ++cleft;
        }

        for (int w = 0; w < 5; ++w) run_blob_fn(cf2);
        CHECK_RT(cudaDeviceSynchronize());
        float cs = 0.0f, cmin = 1e9f;
        cudaEvent_t c0, c1;
        CHECK_RT(cudaEventCreate(&c0));
        CHECK_RT(cudaEventCreate(&c1));
        for (int i = 0; i < iters; ++i) {
            CHECK_RT(cudaEventRecord(c0));
            run_blob_fn(cf2);
            CHECK_RT(cudaEventRecord(c1));
            CHECK_RT(cudaEventSynchronize(c1));
            float el = 0.0f;
            CHECK_RT(cudaEventElapsedTime(&el, c0, c1));
            cs += el;
            if (el < cmin) cmin = el;
        }
        cudaEventDestroy(c0);
        cudaEventDestroy(c1);

        std::printf("cubin  : %d regs/thread, block %d, smem %d -> %d blocks/SM\n", cr, cb, blob_smem,
                    65536 / (128 * cr));
        std::printf("cubin  mean %7.1f ms -> %6.1f TMAC/s   best %7.1f -> %6.1f   (identity: %s, %d left)\n",
                    cs / iters, macs / (cs / iters) / 1e9, cmin, macs / cmin / 1e9,
                    cmism == 0 ? "ok" : "MISMATCH", cleft);
        cuModuleUnload(cm);
    }

    for (int i = 0; i < iters; ++i) {
        cudaEventDestroy(bs[i]);
        cudaEventDestroy(be[i]);
        cudaEventDestroy(os[i]);
        cudaEventDestroy(oe[i]);
    }

    // The occupancy sweep. `65536 / (128 * regs)` is the block count the register file allows; smem
    // allows 3 and threads 12, so a cap that lands under 171 is what buys the third block. Each capped
    // module is also re-run through the identity check, because a spill-heavy JIT is exactly the thing
    // that could come out fast and wrong.
    for (int cap : {0, 168, 160, 128}) {
        CUmodule m2;
        CUfunction f2;
        load_blob(cap, &m2, &f2);
        int r2 = 0;
        CHECK_CU(cuFuncGetAttribute(&r2, CU_FUNC_ATTRIBUTE_NUM_REGS, f2));

        CHECK_RT(cudaMemcpy(transcripts, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice));
        run_blob_fn(f2);
        CHECK_RT(cudaDeviceSynchronize());
        CHECK_RT(cudaMemcpy(got.data(), transcripts, got.size() * 4, cudaMemcpyDeviceToHost));
        int mism = 0, left = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (got[i] != ref[i]) ++mism;
            if (got[i] == 0xccccccccu) ++left;
        }

        for (int w = 0; w < 5; ++w) run_blob_fn(f2);
        CHECK_RT(cudaDeviceSynchronize());
        float s = 0.0f;
        cudaEvent_t e0, e1;
        CHECK_RT(cudaEventCreate(&e0));
        CHECK_RT(cudaEventCreate(&e1));
        for (int i = 0; i < iters; ++i) {
            CHECK_RT(cudaEventRecord(e0));
            run_blob_fn(f2);
            CHECK_RT(cudaEventRecord(e1));
            CHECK_RT(cudaEventSynchronize(e1));
            float el = 0.0f;
            CHECK_RT(cudaEventElapsedTime(&el, e0, e1));
            s += el;
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);

        std::printf("cap %4d -> %3d regs, %d blocks/SM, %6.1f ms -> %6.1f TMAC/s  (identity: %s, %d left)\n",
                    cap, r2, 65536 / (128 * r2), s / iters, macs / (s / iters) / 1e9,
                    mism == 0 ? "ok" : "MISMATCH", left);
        cuModuleUnload(m2);
    }

    cudaFree(a_sum);
    cudaFree(b_sum);
    cudaFree(transcripts);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
