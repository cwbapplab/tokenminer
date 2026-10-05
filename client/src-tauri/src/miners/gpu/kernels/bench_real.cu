// Times the REAL `tokenminer_search_grid` from pearl.cu, unmodified, at production geometry.
//
// Every earlier probe was a reconstruction, and a reconstruction that disagrees with the thing it
// reconstructs tells you about the reconstruction. This compiles the shipped kernel itself (via
// kernels.cu) and drives it with the same grid and launch parameters pearl_gpu.rs uses:
// REGIONS_PER_BATCH=64 tile rows, 16x8 block rectangles, k=4096, rank=256, no transcript buffer.
//
// A bound nothing can clear is used, so the retire path is not what is being timed — this is the
// pure fold+hash+compare throughput of the shipped binary.
//
// Build:
//   nvcc -O3 -arch=sm_120a -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 \
//        -ccbin <msvc-bin> -o bench_real.exe bench_real.cu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// `kernels.cu` includes probe.cu, which stamps `__CUDA_ARCH__` into a kernel body. `__CUDA_ARCH__`
// is only defined during nvcc's device pass, and this file is also compiled by the host pass for
// main(), so the include has to be split across the two. `pearl.cu` is what we are timing; probe.cu
// only holds the self-test anchors, so its absence here does not affect the fold.
#include "blake3.cuh"
#include "pearl.cu"

#define CHECK(x)                                                                                  \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,                      \
                         cudaGetErrorString(e_));                                                  \
            std::exit(1);                                                                          \
        }                                                                                         \
    } while (0)

// Mirrors the host's constants in cuda.rs.
//
// `POW_BK` cannot be read from pearl.cu's macros here: `POW_DP4A` is `__CUDA_ARCH__ < 800`, and
// `__CUDA_ARCH__` is undefined in nvcc's HOST pass (where main() lives), so the host pass sees
// 0 and would derive BK=32 while the device pass — the code that actually runs — sees 1200 and
// derives BK=64. Reading the macro therefore under-allocates shared memory and the launch faults.
// So the tensor path is spelled out here, exactly as `CudaBackend::pow_bk` spells it out for sm_120.
static const int K = 4096;
static const int RANK = 256;
static const int TILE = 16;
static const int REGIONS_PER_BATCH = 64;
// These three come from pearl.cu's macros, which the harness can override with -DPOW_BLOCK_ROWS=...
// so a geometry can be A/B'd against the real kernel rather than a reconstruction of it.
static const int DEV_BK = 64;
static const int DEV_STRIDE = DEV_BK + 16;
// The transcript buffer `sT` is `POW_WARPS * POW_BLOCK_COLS * POW_TRANSCRIPT_WORDS` words. Under
// POW_SMEM_T_DYNAMIC it lives in the dynamic allocation and has to be *requested*; otherwise it is
// `static __shared__` and the driver allocates it on top, out of the same 100 KB budget. Either way
// the total is the same -- what changes is how much of it the launch has to ask for, which is why
// the host's `pow_smem_dynamic_bytes` and this have to agree exactly.
static const int DEV_SMEM_T = POW_BLOCK_ROWS * POW_BLOCK_COLS * 16 * 4;
#if POW_SMEM_T_DYNAMIC
static const int DEV_SMEM =
    POW_STAGES * (POW_BLOCK_ROWS + POW_BLOCK_COLS) * 16 * DEV_STRIDE + DEV_SMEM_T;
static const int DEV_SMEM_STATIC = 0;
#else
static const int DEV_SMEM = POW_STAGES * (POW_BLOCK_ROWS + POW_BLOCK_COLS) * 16 * DEV_STRIDE;
static const int DEV_SMEM_STATIC = DEV_SMEM_T;
#endif

// `pearl.cu` hard-codes the geometry. Rather than editing the shipped kernel for each experiment,
// the harness re-includes it under a -D override that the file's own #defines cannot see, because
// every geometry constant in pearl.cu is written as an unconditional `#define`. The only lever
// available from the command line is POW_FORCE_LDS32 / POW_FORCE_DP4A, which select the fold. So
// the geometry experiments are done by patching pearl.cu, rebuilding this harness, and comparing
// the single-launch number below -- which is exactly how the 8x8 result was obtained.
#ifndef HARNESS_TILE_ROWS
#define HARNESS_TILE_ROWS POW_BLOCK_ROWS
#endif

int main(int argc, char** argv) {
    // Production: m = n = 131072 -> 8192 tile rows and columns -> 8192 tiles per row.
    const int tiles_per_row = argc > 1 ? atoi(argv[1]) : 8192;

    const size_t abytes = (size_t)tiles_per_row * TILE * K;
    const size_t bbytes = abytes;
    signed char *a = nullptr, *b = nullptr;
    CHECK(cudaMalloc(&a, abytes));
    CHECK(cudaMalloc(&b, bbytes));
    CHECK(cudaMemset(a, 1, abytes));
    CHECK(cudaMemset(b, 2, bbytes));

    unsigned *key = nullptr, *bound = nullptr, *found = nullptr, *coord = nullptr;
    CHECK(cudaMalloc(&key, 32));
    CHECK(cudaMalloc(&bound, 32));
    const int regions = REGIONS_PER_BATCH;
    CHECK(cudaMalloc(&found, regions * 4));
    CHECK(cudaMalloc(&coord, regions * 2 * 4));

    unsigned hk[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    // A bound nothing clears: 0x0000...00ffff..ff, so no tile ever claims and the retire path
    // never fires. We are timing the fold, not the winner search.
    unsigned hb[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0xffffu};
    CHECK(cudaMemcpy(key, hk, 32, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(bound, hb, 32, cudaMemcpyHostToDevice));

    const int cols_per_strip = (tiles_per_row + POW_BLOCK_COLS - 1) / POW_BLOCK_COLS;
    const int rows_in_batch = REGIONS_PER_BATCH;
    const int blocks = ((rows_in_batch + POW_BLOCK_ROWS - 1) / POW_BLOCK_ROWS) * cols_per_strip;
    const int tiles = rows_in_batch * tiles_per_row;
    const int threads = POW_BLOCK_ROWS * 32;

    const size_t smem = DEV_SMEM;

    std::printf("shipped tokenminer_search_grid\n");
    std::printf("  block rect   %dx%d   threads=%d   stages=%d   BK=%d (device pass)\n",
                POW_BLOCK_ROWS, POW_BLOCK_COLS, threads, POW_STAGES, DEV_BK);
    std::printf("  tiles/row=%d  batch tiles=%d  blocks=%d  smem=%zu B\n", tiles_per_row, tiles,
                blocks, smem);

    cudaFuncAttributes fa;
    CHECK(cudaFuncGetAttributes(&fa, (const void*)tokenminer_search_grid));
    int regsPerBlock = fa.numRegs * threads;
    int regsPerSm = 0;
    CHECK(cudaDeviceGetAttribute(&regsPerSm, cudaDevAttrMaxRegistersPerMultiprocessor, 0));
    int smemPerSm = 0;
    CHECK(cudaDeviceGetAttribute(&smemPerSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    std::printf("  regs=%d  -> %d blocks/SM by regs, %d by smem\n", fa.numRegs,
                regsPerSm / (regsPerBlock ? regsPerBlock : 1), smemPerSm / (int)smem);

    int smemPerSmOptin = 0;
    CHECK(cudaDeviceGetAttribute(&smemPerSmOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    std::printf("  smem: dynamic %zu B + static sT %d B = %zu B total (limit %d B)\n", smem,
                DEV_SMEM_STATIC, smem + DEV_SMEM_STATIC, smemPerSmOptin);
    if (smem + DEV_SMEM_STATIC > (size_t)smemPerSmOptin) {
        std::printf("  DOES NOT FIT: dynamic + static exceeds the opt-in limit; not measurable.\n");
        return 1;
    }
    if (smem > 48 * 1024) {
        CHECK(cudaFuncSetAttribute(tokenminer_search_grid,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    }

    // A NULL transcript buffer: production passes one (`search_grid` sends a null device
    // pointer), and a non-null here would make the kernel write transcripts back every tile.
    auto launch = [&] {
        tokenminer_search_grid<<<blocks, threads, smem>>>(
            a, b, key, bound, found, coord, (unsigned*)nullptr, 0u, (unsigned)tiles,
            (unsigned)tiles_per_row, (unsigned)tiles_per_row, K, RANK);
    };

    // A null transcript buffer, so the kernel skips the write-back that production skips.
    launch();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("  LAUNCH FAILED: %s\n", cudaGetErrorString(e));
        return 1;
    }

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int r = 0; r < 7; ++r) {
        CHECK(cudaEventRecord(e0));
        launch();
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = ms < best ? ms : best;
    }
    const float batch_ms = best;

    // One tile is TILE*TILE*K = 2^20 MACs at k=4096, and a TH is 10^12 of them, so tiles/s times
    // 2^20 / 1e12 is TH/s.
    const double tiles_per_sec = (double)tiles / (best / 1000.0);
    const double th = tiles_per_sec * (TILE * (double)TILE * K / 1e12);
    std::printf("\n  %d tiles in %.3f ms  ->  %.2f Mtiles/s  ->  %.2f TH/s\n", tiles, best,
                tiles_per_sec / 1e6, th);
    // Machine-readable line for sweep_real.bat: no leading whitespace, so findstr /B matches.
    std::printf("single: %.2f\n", th);

    // The hash ablation. POW_SKIP_HASH removes only the BLAKE3 and the bound compare, leaving the
    // fold, the staging, the barriers and the transcript build untouched. The difference between
    // this and the number above is what the per-tile hash actually costs -- the shipped kernel
    // spends four lanes on each tile's hash, computing the same digest four times.
    {
        int* fs = nullptr;
        int* cs = nullptr;
        CHECK(cudaMalloc(&fs, (size_t)REGIONS_PER_BATCH * 4));
        CHECK(cudaMalloc(&cs, (size_t)REGIONS_PER_BATCH * 2 * 4));
        auto lb = [&] {
            tokenminer_search_grid<<<blocks, threads, smem>>>(
                a, b, key, bound, (unsigned*)fs, (unsigned*)cs, (unsigned*)nullptr, 0u,
                (unsigned)tiles, (unsigned)tiles_per_row, (unsigned)tiles_per_row, K, RANK);
        };
        lb();
        cudaError_t e2 = cudaDeviceSynchronize();
        if (e2 != cudaSuccess) {
            std::printf("  ablation launch failed: %s\n", cudaGetErrorString(e2));
            CHECK(cudaGetLastError());
        } else {
            float bh = 1e30f;
            for (int r = 0; r < 7; ++r) {
                CHECK(cudaEventRecord(e0));
                lb();
                CHECK(cudaEventRecord(e1));
                CHECK(cudaEventSynchronize(e1));
                float ms;
                CHECK(cudaEventElapsedTime(&ms, e0, e1));
                bh = ms < bh ? ms : bh;
            }
            double thn = (double)tiles / (bh / 1000.0) * (TILE * (double)TILE * K / 1e12);
            std::printf("  no-hash ablation: %.3f ms -> %.2f TH/s  (hash costs %.1f%%)\n", bh, thn,
                        100.0 * (1.0 - (double)best / bh));
            std::printf("single_nohash: %.2f\n", thn);
        }
        CHECK(cudaFree(fs));
        CHECK(cudaFree(cs));
    }
    const double total_tiles = (double)tiles_per_row * tiles_per_row;
    std::printf("  full %dx%d grid (1 attempt) = %.0f tiles -> %.1f ms -> %.2f TH/s\n", tiles_per_row,
                tiles_per_row, total_tiles, best * (tiles_per_row / (double)REGIONS_PER_BATCH),
                th);

    // What the miner's host loop actually pays: the same grid driven as REGIONS batches, each
    // followed by a blocking read of the flags, exactly as pearl_gpu.rs does. If this is far
    // below the single-launch number, the loss is the per-batch sync, not the kernel.
    //
    // Swept over the batch size, because REGIONS_PER_BATCH is a host constant and the sync is
    // per batch: a bigger batch means fewer syncs but more wasted work once a region wins.
    for (int regions : {16, 32, 64, 128, 256, 512}) {
        int* f2 = nullptr;
        int* c2 = nullptr;
        CHECK(cudaMalloc(&f2, regions * 4));
        CHECK(cudaMalloc(&c2, regions * 2 * 4));
        int tpr = tiles_per_row;
        int rows_in = regions;
        int blk = ((rows_in + POW_BLOCK_ROWS - 1) / POW_BLOCK_ROWS) * cols_per_strip;
        int tl = rows_in * tpr;
        int nb = tpr / regions;
        auto lb = [&](int which) {
            tokenminer_search_grid<<<blk, threads, smem>>>(
                a, b, key, bound, (unsigned*)f2, (unsigned*)c2, (unsigned*)nullptr,
                (unsigned)(which * tl), (unsigned)tl, (unsigned)tpr, (unsigned)tpr, K, RANK);
        };
        lb(0);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("  regions=%4d  LAUNCH FAILED\n", regions);
            CHECK(cudaGetLastError());
            continue;
        }
        float lm = 1e30f;
        for (int r = 0; r < 5; ++r) {
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < nb; ++i) {
                lb(i);
                unsigned flags[512] = {0};
                CHECK(cudaMemcpy(flags, f2, (size_t)regions * 4, cudaMemcpyDeviceToHost));
            }
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            lm = ms < lm ? ms : lm;
        }
        double all = (double)tl * nb;
        std::printf("  regions=%4d  batches=%4d  %7.1f ms  ->  %6.2f TH/s\n", regions, nb, lm,
                    all / (lm / 1000.0) * (TILE * (double)TILE * K / 1e12));
        CHECK(cudaFree(f2));
        CHECK(cudaFree(c2));
    }

    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    CHECK(cudaGetLastError());
    return 0;
}