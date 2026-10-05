// Self-test kernels: the checks a backend must pass before it is allowed to mine.
//
// These are small on purpose. The mining kernels are far larger, but they rest on the same two
// things verified here: the image the driver loads is the one we built, and the INT8 tensor-core
// math is bit-exact against a CPU reference.

#ifndef __CUDACC__
#error "probe.cu must be compiled by nvcc"
#endif

#include <mma.h>

#include "blake3.cuh"

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
// to match. Consensus multiplies the operands *summed*, so this expands
// (a + na)(b + nb) = ab + a*nb + na*b + na*nb into four INT8 products rather than summing first.
//
// That expansion is deliberate and superseded: the sums are in [-127, 126] and do fit an INT8
// operand, so `tokenminer_jackpot_fold` below gets there in one GEMM. This kernel stays for tile
// shapes `wmma` cannot hold — any h or w that is not a multiple of 16 — and because it is the
// literal expansion of the reference expression, it pins the semantics the fast kernel has to match.
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

// The summed-operand fold: one INT8 tensor-core GEMM over `a_sum` and `b_sum`, folded into the
// sixteen-word transcript `compute_jackpot` returns.
//
// This is the operation the whole miner exists to run, and it is one GEMM rather than four because
// consensus multiplies the *sums*. `signal` is in [-64, 63] and the noise is a difference of two
// entries of a [-32, 31] dense row, so `signal + noise` is in [-127, 126] and still fits in an
// INT8 tensor-core operand. `tokenminer_jackpot` above does the same job the long way round, with
// the four-term expansion, and both are held against `compute_jackpot`; this one is the one the
// search will use, and the other covers tile shapes whose h or w is not a whole 16.
//
// `h` and `w` must both be multiples of 16, the shape `wmma` can hold in one fragment pair. The
// reference miner's production tile is 16x16 (`HT = 16` in `luckypool_miner.py`), but the pattern
// arrives in the job, so this is a dispatch decision and not an assumption.
//
// B is read as `col_major` against a `k`-strided pointer: the fragment wants element (l, v) at
// `ptr[l + v * ldm]`, and `b_sum[v * k + l]` is exactly that. Loading it `row_major` — as
// `tokenminer_i8_gemm` does, because that kernel takes a genuine K x N matrix — would transpose the
// tile and quietly hash a different one.
extern "C" __global__ void tokenminer_jackpot_fold(
    const signed char* a_sum,          // h x k, row-major
    const signed char* b_sum,          // w x k, row-major
    unsigned int* jackpot,             // JACKPOT_WORDS words, zeroed on entry
    int k,
    int rank,
    int h,
    int w) {
    using namespace nvcuda;

    __shared__ int cells[16 * 16];
    __shared__ unsigned int partials[32];

    const int tid = threadIdx.x;
    const int rank_blocks = k / rank;

    // Never reset between rank blocks: the transcript folds the running totals, so resetting here
    // would leave the first word right and every later word wrong.
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;
    wmma::fill_fragment(acc, 0);

    for (int block = 0; block < rank_blocks; ++block) {
        const int lo = block * rank;
        const int hi = lo + rank;

        for (int kk = lo; kk < hi; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, a_sum + kk, k);
            wmma::load_matrix_sync(b_frag, b_sum + kk, k);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        // The fragment layout is the compiler's problem, but reading arbitrary cells back out of it
        // is not, so it goes through shared memory once per rank block rather than once per element.
        __syncthreads();
        wmma::store_matrix_sync(cells, acc, 16, wmma::mem_row_major);
        __syncthreads();

        unsigned int partial = 0;
        for (int i = tid; i < 16 * 16; i += blockDim.x) {
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
            const unsigned int slot = (unsigned int)(block % JACKPOT_WORDS);
            const unsigned int rotated = (jackpot[slot] << LROT_PER_TILE)
                                       | (jackpot[slot] >> (32 - LROT_PER_TILE));
            jackpot[slot] = rotated ^ partials[0];
        }
    }
}

// Keyed BLAKE3 over the jackpot message has no kernel of its own: a 64-byte message is exactly one
// chunk, so it is `tokenminer_blake3_root_chunk` in `pearl.cu`. One implementation, and a bug in it
// fails the jackpot check and the tree check together rather than hiding behind a second copy.
