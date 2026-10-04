// Self-test kernels: the checks a backend must pass before it is allowed to mine.
//
// These are small on purpose. The mining kernels are far larger, but they rest on the same two
// things verified here: the image the driver loads is the one we built, and the INT8 tensor-core
// math is bit-exact against a CPU reference.

#ifndef __CUDACC__
#error "probe.cu must be compiled by nvcc"
#endif

#include <mma.h>

// Stamps the architecture this image was compiled for, so the host can confirm which cubin the
// driver actually loaded, plus a thread index so the launch geometry can be sanity-checked.
extern "C" __global__ void tokenminer_arch_probe(unsigned int* out) {
    out[0] = (unsigned int)__CUDA_ARCH__;
    out[1] = blockIdx.x * blockDim.x + threadIdx.x;
}

// A single-thread INT8 dot product over `k` signed bytes.
//
// The scalar fallback shape, kept because it is trivially checkable: if this disagrees with the
// host, the problem is argument passing or the data, not the tensor-core path.
extern "C" __global__ void tokenminer_i8_dot(const signed char* a, const signed char* b, int* out, int k) {
    int acc = 0;
    for (int i = 0; i < k; ++i) {
        acc += (int)a[i] * (int)b[i];
    }

    *out = acc;
}

// C[M x N] = A[M x K] * B[K x N], all signed 8-bit in, 32-bit accumulators out.
//
// This is the operation the mining kernel is built on, and the whole reason the GPU path exists:
// tensor cores do this orders of magnitude faster than the CPU search. One block per 16x16 output
// tile, using `wmma` so the fragment layouts are the compiler's problem rather than a hand-rolled
// `ldmatrix`/`mma.sync` — an unoptimised first cut that is easy to prove correct. Pipelining,
// persistent CTAs and TMA come with the real kernel.
//
// M, N and K must all be multiples of 16.
extern "C" __global__ void tokenminer_i8_gemm(
    const signed char* a,
    const signed char* b,
    int* c,
    int m,
    int n,
    int k) {
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * 16;

    if (tile_row >= m || tile_col >= n) {
        return;
    }

    using namespace nvcuda;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;

    wmma::fill_fragment(acc, 0);

    for (int kk = 0; kk < k; kk += 16) {
        // A is row-major M x K.
        wmma::load_matrix_sync(a_frag, a + tile_row * k + kk, k);
        // B is row-major K x N, so the leading dimension is N for both layouts of interest here.
        wmma::load_matrix_sync(b_frag, b + kk * n + tile_col, n);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    wmma::store_matrix_sync(c + tile_row * n + tile_col, acc, n, wmma::mem_row_major);
}

// Mirrors `zk_pow::circuit::pearl_program`; the consensus constants, not tunables.
#define JACKPOT_WORDS 16
#define LROT_PER_TILE 13

// The Pearl jackpot for one tile pair, matching `zk_pow::circuit::chip::compute_jackpot` exactly.
//
// `secret_a` is h x k and `secret_b` is w x k, both row-major i8, with `noise_a` / `noise_b` shaped
// to match. The operands are *summed before multiplying*, so each factor is in [-128, 128] and
// cannot be fed to INT8 tensor cores directly — the host has to expand
// (a + na)(b + nb) = ab + a*nb + na*b + na*nb into four INT8 products. (That expansion is the next
// step; this kernel is the plain-loop version that pins the semantics down.)
//
// One block handles one tile pair. The cell sums live in shared memory because they *carry across*
// rank blocks: the canonical fold XORs the running totals after each block, not per-block sums.
// Getting that wrong leaves the first jackpot word correct and every later word wrong.
extern "C" __global__ void tokenminer_jackpot(
    const signed char* secret_a,
    const signed char* noise_a,
    const signed char* secret_b,
    const signed char* noise_b,
    unsigned int* jackpot,
    int k,
    int rank,
    int h,
    int w) {
    extern __shared__ int shared[];

    int* cells = shared;                                  // h * w running sums
    unsigned int* partials = (unsigned int*)(cells + h * w);

    const int tid = threadIdx.x;
    const int cell_count = h * w;
    const int rank_blocks = k / rank;

    for (int i = tid; i < cell_count; i += blockDim.x) {
        cells[i] = 0;
    }
    __syncthreads();

    for (int block = 0; block < rank_blocks; ++block) {
        const int lo = block * rank;
        const int hi = lo + rank;

        // Each (u, v) cell is owned by exactly one thread, so the read-modify-write is race-free.
        for (int v = tid; v < w; v += blockDim.x) {
            const signed char* b_row = secret_b + (long)v * k;
            const signed char* nb_row = noise_b + (long)v * k;

            for (int u = 0; u < h; ++u) {
                const signed char* a_row = secret_a + (long)u * k;
                const signed char* na_row = noise_a + (long)u * k;

                int sum = cells[u * w + v];
                for (int l = lo; l < hi; ++l) {
                    const int a = a_row[l];
                    const int na = na_row[l];
                    const int b = b_row[l];
                    const int nb = nb_row[l];

                    // The four-way expansion, kept explicit because it is what the tensor-core
                    // version rests on: `(a + na)` is in [-128, 128] and so cannot be an INT8
                    // operand, but each of a, na, b, nb can be. Summing the four products over the
                    // rank block gives the same value as multiplying the sums.
                    sum += a * b + a * nb + na * b + na * nb;
                }
                cells[u * w + v] = sum;
            }
        }
        __syncthreads();

        unsigned int partial = 0;
        for (int i = tid; i < cell_count; i += blockDim.x) {
            partial ^= (unsigned int)cells[i];
        }

        partials[tid] = partial;
        __syncthreads();

        for (int span = blockDim.x / 2; span > 0; span >>= 1) {
            if (tid < span) {
                partials[tid] ^= partials[tid + span];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const unsigned int xored_tile = partials[0];
            const int slot = block % JACKPOT_WORDS;
            const unsigned int rotated = (jackpot[slot] << LROT_PER_TILE) | (jackpot[slot] >> (32 - LROT_PER_TILE));
            jackpot[slot] = rotated ^ xored_tile;
        }

        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Keyed BLAKE3 over the jackpot message.
// ---------------------------------------------------------------------------

__device__ __constant__ unsigned int B3_IV[4] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
};

__device__ __constant__ unsigned char B3_SCHEDULE[7][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    { 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8},
    { 3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1},
    {10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6},
    {12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4},
    { 9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7},
    {11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13},
};

#define B3_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define B3_G(a, b, c, d, mx, my)        \
    do {                                 \
        v[a] = v[a] + v[b] + (mx);       \
        v[d] = B3_ROTR(v[d] ^ v[a], 16); \
        v[c] = v[c] + v[d];              \
        v[b] = B3_ROTR(v[b] ^ v[c], 12); \
        v[a] = v[a] + v[b] + (my);       \
        v[d] = B3_ROTR(v[d] ^ v[a], 8);  \
        v[c] = v[c] + v[d];              \
        v[b] = B3_ROTR(v[b] ^ v[c], 7);  \
    } while (0)

#define B3_ROUND(r)                                                                  \
    do {                                                                             \
        B3_G(0, 4,  8, 12, block[B3_SCHEDULE[r][ 0]], block[B3_SCHEDULE[r][ 1]]);     \
        B3_G(1, 5,  9, 13, block[B3_SCHEDULE[r][ 2]], block[B3_SCHEDULE[r][ 3]]);     \
        B3_G(2, 6, 10, 14, block[B3_SCHEDULE[r][ 4]], block[B3_SCHEDULE[r][ 5]]);     \
        B3_G(3, 7, 11, 15, block[B3_SCHEDULE[r][ 6]], block[B3_SCHEDULE[r][ 7]]);     \
        B3_G(0, 5, 10, 15, block[B3_SCHEDULE[r][ 8]], block[B3_SCHEDULE[r][ 9]]);     \
        B3_G(1, 6, 11, 12, block[B3_SCHEDULE[r][10]], block[B3_SCHEDULE[r][11]]);     \
        B3_G(2, 7,  8, 13, block[B3_SCHEDULE[r][12]], block[B3_SCHEDULE[r][13]]);     \
        B3_G(3, 4,  9, 14, block[B3_SCHEDULE[r][14]], block[B3_SCHEDULE[r][15]]);     \
    } while (0)

// Keyed BLAKE3 of a 64-byte message: exactly one chunk, so exactly one compression with the ROOT
// flag. Mirrors `pearl_blake3::blake3_digest(message, Some(key))`, which is what
// `compute_jackpot_hash` applies to the sixteen little-endian jackpot words.
extern "C" __global__ void tokenminer_jackpot_hash(
    const unsigned char* message,
    const unsigned char* key,
    unsigned char* digest) {
    unsigned int block[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        block[i] = (unsigned int)message[i * 4]
                 | ((unsigned int)message[i * 4 + 1] << 8)
                 | ((unsigned int)message[i * 4 + 2] << 16)
                 | ((unsigned int)message[i * 4 + 3] << 24);
    }

    unsigned int v[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = (unsigned int)key[i * 4]
             | ((unsigned int)key[i * 4 + 1] << 8)
             | ((unsigned int)key[i * 4 + 2] << 16)
             | ((unsigned int)key[i * 4 + 3] << 24);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[8 + i] = B3_IV[i];
    }
    v[12] = 0;   // chunk counter low: the whole message is one chunk
    v[13] = 0;   // chunk counter high
    v[14] = 64;  // block length
    v[15] = 27;  // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH

    B3_ROUND(0);
    B3_ROUND(1);
    B3_ROUND(2);
    B3_ROUND(3);
    B3_ROUND(4);
    B3_ROUND(5);
    B3_ROUND(6);

    // A 32-byte root output is the first eight words folded with the last eight.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int word = v[i] ^ v[i + 8];
        digest[i * 4]     = (unsigned char)(word);
        digest[i * 4 + 1] = (unsigned char)(word >> 8);
        digest[i * 4 + 2] = (unsigned char)(word >> 16);
        digest[i * 4 + 3] = (unsigned char)(word >> 24);
    }
}
