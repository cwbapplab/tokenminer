// Times the wide entry point against the shipped one on the SAME input, and checks that the two agree
// tile by tile before either number is believed.
//
// The hoist changes the instruction schedule, not the arithmetic: the same fragments reach the same
// `mma` in the same order, so the transcripts must be identical. If they are not, a throughput number
// is measuring a different function, which is the failure this harness exists to catch.
//
// Build:
//   nvcc -O3 -arch=sm_120a -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 \
//        -ccbin <msvc-bin> -o bench_wide.exe bench_wide.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "blake3.cuh"
// The shipped side of the A/B is the production configuration: two stages. A sweep
// of the wide side's stage count must not take the shipped kernel along with it,
// because the shipped rectangle at three stages (119,808 B) is past the opt-in
// limit and its launch would fault -- with the correctness pass still reporting
// `identical`, which is the misleading failure this pin exists to prevent. The
// wide kernel reads POW_W_STAGES, which the sweep sets; pearl.cu's shipped
// kernel reads POW_STAGES, pinned here at its production value.
#ifdef POW_W_STAGES_OVERRIDE
#define POW_W_STAGES POW_W_STAGES_OVERRIDE
#else
#define POW_W_STAGES 2
#endif
#define POW_STAGES 2
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

static const int K = 4096;
static const int RANK = 256;
static const int TILE = 16;
static const int REGIONS_PER_BATCH = 64;
static const int DEV_BK = 64;
static const int DEV_STRIDE = DEV_BK + 16;

// Shipped geometry, as `bench_real.cu` computes it. The shipped kernel always runs
// at two stages -- that is its production configuration -- so its side of the A/B is
// pinned here and only the wide side answers the stage sweep. Without the pin, a
// three-stage sweep takes the shipped baseline past the opt-in limit with it and the
// baseline launch fails, leaving a wide number with nothing to compare against.
static const int SH_TILES = POW_BLOCK_ROWS * POW_BLOCK_COLS;
static const int SH_STAGES = 2;
static const int SH_SMEM_T = POW_BLOCK_ROWS * POW_BLOCK_COLS * 16 * 4;
static const int SH_SMEM = SH_STAGES * (POW_BLOCK_ROWS + POW_BLOCK_COLS) * 16 * DEV_STRIDE + SH_SMEM_T;

// Wide geometry: staged A per row slot, staged B per column slot, transcript per warp tile.
static const int W_TILES = POW_W_BLOCK_ROWS * POW_W_BLOCK_COLS;
static const int W_SMEM_T = POW_W_WARPS * POW_W_TILES_PER_WARP * 16 * 4;
static const int W_SMEM = POW_W_STAGES * (POW_W_ROW_SLOTS * (POW_W_ROW_TILES * 16) +
                                          POW_W_COL_SLOTS * (POW_W_COL_TILES * 16)) * DEV_STRIDE + W_SMEM_T;

static const unsigned int HOST_KEY[8] = {1, 2, 3, 4, 5, 6, 7, 8};
// A bound nothing clears, so no tile claims and the retire path never fires: this times the fold.
static const unsigned int HOST_BOUND[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0xffffu};
static unsigned int *key, *bound;

static float time_launch(int blocks, int threads, int smem, void (*launch)(int, int, int),
                         const void* func) {
    int optin = 0;
    CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem > optin) {
        std::printf("  smem %d B exceeds the %d B opt-in limit: not measurable\n", smem, optin);
        return -1.0f;
    }
    CHECK(cudaFuncSetAttribute((const void*)func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    launch(blocks, threads, smem);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("  LAUNCH FAILED: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1.0f;
    }
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int r = 0; r < 7; ++r) {
        CHECK(cudaEventRecord(e0));
        launch(blocks, threads, smem);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = ms < best ? ms : best;
    }
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    return best;
}

static signed char *g_a, *g_b;
static unsigned int *g_found, *g_coord, *g_transcript;
static int g_tiles, g_tiles_per_row;
// Production passes a null transcript buffer, so the throughput number must exclude the write-back;
// the correctness pass needs the buffer. This flips which one a launch sees.
static int g_null_transcript;

static void launch_shipped(int blocks, int threads, int smem) {
    int cols_per_strip = (g_tiles_per_row + POW_BLOCK_COLS - 1) / POW_BLOCK_COLS;
    (void)blocks;
    // blocks is recomputed inside so the mapping matches the kernel's own derivation.
    int rows_in_batch = REGIONS_PER_BATCH;
    int blk = ((rows_in_batch + POW_BLOCK_ROWS - 1) / POW_BLOCK_ROWS) * cols_per_strip;
    tokenminer_search_grid<<<blk, threads, smem>>>(g_a, g_b, key, bound, g_found, g_coord,
                                                   g_null_transcript ? nullptr : g_transcript, 0u,
                                                   (unsigned)g_tiles, (unsigned)g_tiles_per_row,
                                                   (unsigned)g_tiles_per_row, K, RANK);
}

static void launch_wide(int blocks, int threads, int smem) {
    int cols_per_strip = (g_tiles_per_row + POW_W_BLOCK_COLS - 1) / POW_W_BLOCK_COLS;
    int rows_in_batch = REGIONS_PER_BATCH;
    int blk = ((rows_in_batch + POW_W_BLOCK_ROWS - 1) / POW_W_BLOCK_ROWS) * cols_per_strip;
    tokenminer_search_grid_wide<<<blk, threads, smem>>>(g_a, g_b, key, bound, g_found, g_coord,
                                                        g_null_transcript ? nullptr : g_transcript, 0u,
                                                        (unsigned)g_tiles, (unsigned)g_tiles_per_row,
                                                        (unsigned)g_tiles_per_row, K, RANK);
}

int main(int argc, char** argv) {
    const int tiles_per_row = argc > 1 ? atoi(argv[1]) : 256;
    g_tiles_per_row = tiles_per_row;
    g_tiles = REGIONS_PER_BATCH * tiles_per_row;

    const size_t bytes = (size_t)tiles_per_row * TILE * K;
    CHECK(cudaMalloc(&g_a, bytes));
    CHECK(cudaMalloc(&g_b, bytes));
    CHECK(cudaMemset(g_a, 1, bytes));
    CHECK(cudaMemset(g_b, 2, bytes));

    CHECK(cudaMalloc(&key, 32));
    CHECK(cudaMalloc(&bound, 32));
    CHECK(cudaMalloc(&g_found, REGIONS_PER_BATCH * 4));
    CHECK(cudaMalloc(&g_coord, REGIONS_PER_BATCH * 2 * 4));
    const size_t tbytes = (size_t)g_tiles * 16 * 4;
    CHECK(cudaMalloc(&g_transcript, tbytes));

    CHECK(cudaMemcpy(key, HOST_KEY, 32, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(bound, HOST_BOUND, 32, cudaMemcpyHostToDevice));

    // ---- Bit-exactness: same input, both entry points, tile by tile. ----
    std::vector<unsigned int> ref((size_t)g_tiles * 16, 0u);
    std::vector<unsigned int> got((size_t)g_tiles * 16, 0u);

    int sh_blocks = ((REGIONS_PER_BATCH + POW_BLOCK_ROWS - 1) / POW_BLOCK_ROWS) *
                    ((tiles_per_row + POW_BLOCK_COLS - 1) / POW_BLOCK_COLS);
    int w_blocks = ((REGIONS_PER_BATCH + POW_W_BLOCK_ROWS - 1) / POW_W_BLOCK_ROWS) *
                   ((tiles_per_row + POW_W_BLOCK_COLS - 1) / POW_W_BLOCK_COLS);

    CHECK(cudaMemset(g_transcript, 0, tbytes));
    launch_shipped(sh_blocks, POW_BLOCK_ROWS * 32, SH_SMEM);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(ref.data(), g_transcript, tbytes, cudaMemcpyDeviceToHost));

    CHECK(cudaMemset(g_transcript, 0, tbytes));
    launch_wide(w_blocks, POW_W_THREADS, W_SMEM);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got.data(), g_transcript, tbytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int b = 0; b < sh_blocks; ++b) {
        int strip = b / ((tiles_per_row + POW_BLOCK_COLS - 1) / POW_BLOCK_COLS);
        int col_block = b - strip * ((tiles_per_row + POW_BLOCK_COLS - 1) / POW_BLOCK_COLS);
        int row0 = strip * POW_BLOCK_ROWS;
        int col0 = col_block * POW_BLOCK_COLS;
        for (int r = 0; r < POW_BLOCK_ROWS; ++r) {
            for (int c = 0; c < POW_BLOCK_COLS; ++c) {
                int gr = row0 + r, gc = col0 + c;
                if (gr >= REGIONS_PER_BATCH || gc >= tiles_per_row) continue;
                size_t tile = (size_t)gr * tiles_per_row + gc;
                for (int w = 0; w < 16; ++w) {
                    if (ref[tile * 16 + w] != got[tile * 16 + w]) {
                        if (mismatches < 8) {
                            std::printf("  MISMATCH tile (%d,%d) word %d: shipped %08x wide %08x\n",
                                        gr, gc, w, ref[tile * 16 + w], got[tile * 16 + w]);
                        }
                        ++mismatches;
                    }
                }
            }
        }
    }
    std::printf("bit-exactness over %d tiles: %s\n", g_tiles,
                mismatches == 0 ? "identical" : "DIFFERENT");

    // ---- Throughput, both entry points, same buffers, null transcript (as production). ----
    g_null_transcript = 1;
    const double th_units = TILE * (double)TILE * K / 1e12;
    float sh_ms = time_launch(sh_blocks, POW_BLOCK_ROWS * 32, SH_SMEM, launch_shipped,
                              (const void*)tokenminer_search_grid);
    float w_ms = time_launch(w_blocks, POW_W_THREADS, W_SMEM, launch_wide,
                             (const void*)tokenminer_search_grid_wide);
    if (sh_ms > 0) std::printf("shipped %dx%d: %.3f ms -> %.2f TH/s\n", POW_BLOCK_ROWS,
                               POW_BLOCK_COLS, sh_ms, (double)g_tiles / (sh_ms / 1000.0) * th_units);
    if (w_ms > 0) std::printf("wide    %dx%d:  %.3f ms -> %.2f TH/s\n", POW_W_BLOCK_ROWS,
                              POW_W_BLOCK_COLS, w_ms, (double)g_tiles / (w_ms / 1000.0) * th_units);

    // Occupancy, because the shape trade is exactly a trade between the load ratio and the warps the
    // scheduler can see.
    int regs_per_sm = 0, smem_per_sm = 0;
    CHECK(cudaDeviceGetAttribute(&regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0));
    CHECK(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    cudaFuncAttributes fa;
    CHECK(cudaFuncGetAttributes(&fa, (const void*)tokenminer_search_grid_wide));
    std::printf("  wide: regs=%d  smem=%d B  blocks/SM by regs=%d by smem=%d  warps/SM=%d\n",
                fa.numRegs, W_SMEM, regs_per_sm / (fa.numRegs * POW_W_THREADS),
                smem_per_sm / W_SMEM,
                (regs_per_sm / (fa.numRegs * POW_W_THREADS)) * POW_W_WARPS);
    std::printf("  shipped smem=%d B\n", SH_SMEM);
    if (sh_ms > 0 && w_ms > 0) {
        std::printf("delta: %+.1f%%\n", 100.0 * (sh_ms / w_ms - 1.0));
    }

    CHECK(cudaFree(g_a));
    CHECK(cudaFree(g_b));
    CHECK(cudaFree(g_found));
    CHECK(cudaFree(g_coord));
    CHECK(cudaFree(g_transcript));
    return mismatches == 0 ? 0 : 1;
}
