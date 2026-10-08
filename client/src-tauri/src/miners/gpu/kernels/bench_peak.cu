// bench_peak.cu
//
// What the tensor pipe can actually do on this chip, with no shared memory and no staging in the way.
//
// The fold measures 133 TMAC/s and every attempt to move it has landed on the same wall: the operand
// path, not the math. Whether that wall is worth attacking depends entirely on where the ceiling is --
// if the chip can only issue ~140 TMAC/s of int8 then the kernel is done, and if it can issue 400 then
// the staging path is worth a rewrite. This measures the ceiling directly: `ldmatrix` results are held
// in registers and the loop is nothing but `mma`, so the number here is a pure issue-rate bound.
//
// The mma is `m16n8k32`: 16*8*32 = 4096 MACs per instruction. Eight independent accumulators keep the
// pipe fed across the mma's latency; with fewer the loop stalls on the accumulation chain and measures
// latency instead of throughput.
//
// Build:
//   nvcc -O3 -arch=sm_120a -Xcompiler /wd4477 -ccbin <msvc-bin> -o bench_peak.exe bench_peak.cu -lcuda

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_RT(x)                                                                     \
    do {                                                                                \
        cudaError_t e_ = (x);                                                           \
        if (e_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,           \
                         cudaGetErrorString(e_));                                       \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

#ifndef NACC
#define NACC 8
#endif
#define REPS 4096


__global__ __launch_bounds__(128) void peak_kernel(unsigned int* sink) {
    // Seed the fragments from the thread id so nothing is constant-folded and no operand is zero.
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

    #pragma unroll 1
    for (int r = 0; r < REPS; ++r) {
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

int main() {
    int dev = 0, sms = 0, clock_khz = 0;
    CHECK_RT(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CHECK_RT(cudaGetDeviceProperties(&prop, dev));
    sms = prop.multiProcessorCount;
    // CUDA 13 dropped `cudaDeviceProp::clockRate`; the attribute is the replacement.
    CHECK_RT(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
    std::printf("device: %s, %d SMs, %.2f GHz max\n", prop.name, sms, clock_khz / 1e6);

    unsigned int* sink = nullptr;
    CHECK_RT(cudaMalloc(&sink, 128 * sizeof(unsigned int)));

    const int blocks = sms * 8;
    cudaFuncAttributes pa{};
    CHECK_RT(cudaFuncGetAttributes(&pa, (const void*)peak_kernel));
    std::printf("peak kernel: %d regs/thread, %d blocks/SM by registers\n", pa.numRegs,
                65536 / (128 * pa.numRegs));
    for (int warm = 0; warm < 3; ++warm) peak_kernel<<<blocks, 128>>>(sink);
    CHECK_RT(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CHECK_RT(cudaEventCreate(&e0));
    CHECK_RT(cudaEventCreate(&e1));
    CHECK_RT(cudaEventRecord(e0));
    peak_kernel<<<blocks, 128>>>(sink);
    CHECK_RT(cudaEventRecord(e1));
    CHECK_RT(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&ms, e0, e1));

    // Each warp issues NACC*REPS mma per iteration of the outer loop; a block is 4 warps.
    const double mma_count = (double)blocks * 4.0 * NACC * REPS;
    const double macs = mma_count * 4096.0;
    std::printf("%d blocks x 128 threads, %d mma/warp-rep\n", blocks, NACC);
    std::printf("%.3f ms -> %.1f TMAC/s  (%.1f TOPS int8)\n", ms, macs / (ms / 1e3) / 1e12,
                macs * 2.0 / (ms / 1e3) / 1e12);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(sink);
    return 0;
}
