// bench_energy.cu
//
// TMAC/s per watt, for the search blob and for the register-resident tensor peak.
//
// The box's RTX 5080 is power-limited to 252 W (70% of its 360 W default), so at the cap the SM
// clock is whatever fits the wattage and hashrate = clock x MACs/clock. That makes ENERGY per MAC
// the quantity that sets hashrate: a change that does the same MACs for fewer joules raises the
// clock 1:1. This measures where those joules go.
//
// The comparison is the point. `peak` is the same `mma` with both operands in registers -- no
// shared memory, no global traffic, no staging loop. Its joules/MAC is the arithmetic floor.
// Everything the blob spends above that is the operand path, and that surplus is the budget any
// energy optimization has to work against.
//
// Each mode loops for a wall-clock budget so the card reaches its steady power state; the harness
// prints MACs and seconds, and the caller samples `nvidia-smi --query-gpu=power.draw` during the
// run. Sampling is deliberately outside this process: NVML inside would add a second reader to a
// meter that already averages over a window.
//
// Build:
//   nvcc -O3 -std=c++17 -Xcompiler /Zc:preprocessor -arch=sm_120a -ccbin <msvc-bin>
//        -o bench_energy.exe bench_energy.cu -lcuda

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "fold_sm120.cuh"

#ifdef FOLD_BDIRECT
#define FOLD_BDIRECT_STR "ours B-direct"
#else
#define FOLD_BDIRECT_STR "ours staged-B"
#endif

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

// The same shape the fold's own harness uses, so the two numbers are comparable. `BLOCK_M` /
// `BLOCK_N` narrow the grid for the L2-resident control: at 128 MiB an operand is far past this
// chip's 64 MiB L2 (so it is DRAM-fed), and a grid whose operands fit L2 isolates the on-chip
// energy from the DRAM energy.
static int M = 131072;
static int N = 131072;
static const int K = 2048;
static int TILES = (M / 128) * (N / 128);
static const int BLOB_SMEM = 33792;

#define NACC 8

__global__ __launch_bounds__(128) void peak_kernel(unsigned int* sink) {
    unsigned int a[4], b[2];
    unsigned int acc[NACC][4];
    const unsigned int t = threadIdx.x * 2654435761u + 1u;
    #pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = t + i;
    b[0] = t ^ 0x9e3779b9u;
    b[1] = t ^ 0x85ebca6bu;
    #pragma unroll
    for (int j = 0; j < NACC; ++j) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = t + j + i;
    }
    // A plain outer loop with no bound the compiler can hoist out: REPS is large enough that the
    // launch is long, and `sink`'s guard keeps the accumulators live.
    for (int r = 0; r < 4096; ++r) {
        #pragma unroll
        for (int j = 0; j < NACC; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
    }
    unsigned int s = 0;
    #pragma unroll
    for (int j = 0; j < NACC; ++j) s ^= acc[j][0] ^ acc[j][1] ^ acc[j][2] ^ acc[j][3];
    if (s == 0xdeadbeefu) sink[threadIdx.x] = s;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: bench_energy.exe peak|blob <ptx> [seconds]\n");
        return 1;
    }
    const std::string mode = argv[1];
    const bool peak = mode == "peak";
    const bool ours = mode == "ours";
    if (ours) {
        // Our hand-written fold, at the m=2 shape that tiles a 128-row block. FOLD_BDIRECT is a
        // compile-time switch: this binary is built twice, once with it, and the two are compared.
        int d0 = 0;
        CHECK_RT(cudaGetDevice(&d0));
    }
    double budget = argc >= 4 ? std::atof(argv[3]) : 8.0;
    if (const char* bm = std::getenv("BENCH_BLOCK_M")) M = std::atoi(bm);
    if (const char* bn = std::getenv("BENCH_BLOCK_N")) N = std::atoi(bn);
    TILES = (M / 128) * (N / 128);

    int dev = 0;
    CHECK_RT(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CHECK_RT(cudaGetDeviceProperties(&prop, dev));
    // The cap is read from the environment, not NVML: the meter belongs to the caller, which is
    // already sampling it, and the harness is the wrong place for a second reader.
    double cap_w = 252.0;
    if (const char* e = std::getenv("PEAK_WATT")) cap_w = std::atof(e);

    unsigned int* sink = nullptr;
    CHECK_RT(cudaMalloc(&sink, 128 * sizeof(unsigned int)));

    double macs_per_launch = 0.0;
    CUmodule mod = nullptr;
    CUfunction blob = nullptr;
    signed char *a_sum = nullptr, *b_sum = nullptr;
    unsigned int* transcripts = nullptr;

    signed char* b_use = nullptr;
    if (ours) {
        CHECK_RT(cudaMalloc(&a_sum, (size_t)M * K));
        CHECK_RT(cudaMalloc(&b_sum, (size_t)N * K));
        CHECK_RT(cudaMalloc(&transcripts, (size_t)TILES * 64 * 16 * 4));
        std::vector<signed char> ha((size_t)M * K), hb((size_t)N * K);
        auto pattern = [](size_t i, unsigned long long salt) {
            unsigned long long x = i * 0x9E3779B97F4A7C15ull + salt;
            x ^= x >> 29;
            x *= 0xBF58476D1CE4E5B9ull;
            x ^= x >> 32;
            return (signed char)((int)(x % 128) - 64);
        };
        for (size_t i = 0; i < ha.size(); ++i) ha[i] = pattern(i, 0x1111ull);
        for (size_t i = 0; i < hb.size(); ++i) hb[i] = pattern(i, 0x9999ull);
        CHECK_RT(cudaMemcpy(a_sum, ha.data(), ha.size(), cudaMemcpyHostToDevice));
        CHECK_RT(cudaMemcpy(b_sum, hb.data(), hb.size(), cudaMemcpyHostToDevice));
#ifdef FOLD_BDIRECT
        signed char* bfrag = nullptr;
        CHECK_RT(cudaMalloc(&bfrag, (size_t)N * K));
        {
            const int nblk = N / 128, kblk = K / FOLD_BK;
            std::vector<unsigned char> hf((size_t)N * K);
            for (int n0 = 0; n0 < nblk; ++n0)
              for (int kb = 0; kb < kblk; ++kb)
                for (int t = 0; t < 8; ++t) {
                    const int rowbase = n0 * 128 + t * 16;
                    for (int l = 0; l < 32; ++l) {
                        unsigned char* dst =
                            hf.data() + (((size_t)(n0 * kblk + kb) * 8 + t) * 32 + l) * 16;
                        const int r0 = rowbase + (l >> 2), r1 = rowbase + 8 + (l >> 2);
                        const int k0 = kb * FOLD_BK + (l & 3) * 4, k2 = k0 + 16;
                        for (int b = 0; b < 4; ++b) {
                            dst[b] = (unsigned char)hb[(size_t)r0 * K + k0 + b];
                            dst[4 + b] = (unsigned char)hb[(size_t)r1 * K + k0 + b];
                            dst[8 + b] = (unsigned char)hb[(size_t)r0 * K + k2 + b];
                            dst[12 + b] = (unsigned char)hb[(size_t)r1 * K + k2 + b];
                        }
                    }
                }
            CHECK_RT(cudaMemcpy(bfrag, hf.data(), hf.size(), cudaMemcpyHostToDevice));
        }
        b_use = bfrag;
#else
        b_use = b_sum;
#endif
        CHECK_RT(cudaFuncSetAttribute(fold_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      FOLD_SMEM));
        macs_per_launch = (double)TILES * 128 * 128 * K;
    } else if (!peak) {
        if (argc < 3) { std::fprintf(stderr, "blob mode needs the ptx path\n"); return 1; }
        FILE* f = std::fopen(argv[2], "rb");
        if (!f) { std::fprintf(stderr, "cannot read the blob: %s\n", argv[2]); return 1; }
        std::vector<char> ptx;
        for (int c; (c = std::fgetc(f)) != EOF;) ptx.push_back((char)c);
        std::fclose(f);
        ptx.push_back('\0');
        CHECK_CU(cuModuleLoadData(&mod, ptx.data()));
        CHECK_CU(cuModuleGetFunction(&blob, mod, "_pearl_search_norotl_kernel"));
        CHECK_CU(cuFuncSetAttribute(blob, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                    BLOB_SMEM));

        CHECK_RT(cudaMalloc(&a_sum, (size_t)M * K));
        CHECK_RT(cudaMalloc(&b_sum, (size_t)N * K));
        CHECK_RT(cudaMalloc(&transcripts, (size_t)TILES * 64 * 16 * 4));
        // A spread pattern, so the memory system cannot compress it. A CONSTANT fill compresses
        // (this box reads ~40% faster on one), and a zero fill more so -- either measures the
        // memory path's best case, not the one production runs.
        std::vector<signed char> ha((size_t)M * K), hb((size_t)N * K);
        auto pattern = [](size_t i, unsigned long long salt) {
            unsigned long long x = i * 0x9E3779B97F4A7C15ull + salt;
            x ^= x >> 29;
            x *= 0xBF58476D1CE4E5B9ull;
            x ^= x >> 32;
            return (signed char)((int)(x % 128) - 64);
        };
        for (size_t i = 0; i < ha.size(); ++i) ha[i] = pattern(i, 0x1111ull);
        for (size_t i = 0; i < hb.size(); ++i) hb[i] = pattern(i, 0x9999ull);
        CHECK_RT(cudaMemcpy(a_sum, ha.data(), ha.size(), cudaMemcpyHostToDevice));
        CHECK_RT(cudaMemcpy(b_sum, hb.data(), hb.size(), cudaMemcpyHostToDevice));
        macs_per_launch = (double)TILES * 128 * 128 * K;
    } else {
        macs_per_launch = (double)(prop.multiProcessorCount * 8) * 4.0 * NACC * 4096.0 * 4096.0;
    }

    auto launch = [&]() {
        if (peak) {
            peak_kernel<<<prop.multiProcessorCount * 8, 128>>>(sink);
        } else if (ours) {
            const int blocks = (M / FOLD_BLOCK_ROWS) * (N / 128);
            fold_sm120<<<blocks, FOLD_THREADS, FOLD_SMEM>>>(a_sum, b_use, transcripts, M, N, K, K, K,
                                                            nullptr);
        } else {
            unsigned int m = M, n = N, k = K;
            void* params[10] = {&a_sum, &b_sum, &transcripts, &m, &n, &k, &k, &k, nullptr, nullptr};
            unsigned long long z0 = 0, z1 = 0;
            params[8] = &z0;
            params[9] = &z1;
            CHECK_CU(cuLaunchKernel(blob, TILES, 1, 1, 128, 1, 1, BLOB_SMEM, 0, params, nullptr));
        }
    };

    for (int i = 0; i < 3; ++i) launch();
    CHECK_RT(cudaDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    double secs = 0.0;
    long launches = 0;
    while (secs < budget) {
        launch();
        CHECK_RT(cudaDeviceSynchronize());
        ++launches;
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    const double total_macs = macs_per_launch * launches;
    const double tmacs = total_macs / secs / 1e12;
    std::printf("mode     : %s\n",
                peak ? "peak (register-resident mma)"
                     : ours ? (FOLD_BDIRECT_STR " (fold_sm120)")
                            : "blob (pearl_search_norotl)");
    std::printf("power cap: %.1f W\n", cap_w);
    std::printf("ran      : %ld launches in %.2f s -> %.1f TMAC/s\n", launches, secs, tmacs);
    std::printf("RESULT   : %.1f TMAC/s ; at the %.0f W cap that is %.3f TMAC/s per watt\n",
                tmacs, cap_w, tmacs / (cap_w));

    if (!peak) {
        cudaFree(a_sum); cudaFree(b_sum); cudaFree(transcripts);
        if (ours) cudaFree(b_use); else cuModuleUnload(mod);
    }
    cudaFree(sink);
    return 0;
}
