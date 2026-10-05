// BLAKE3 on the device: the compression function, and the three node shapes the mining path needs
// — a chunk, a parent, and a root.
//
// This is deliberately the same tree `pearl_blake3` builds on the host, not a shortcut. A Pearl
// proof commits to its matrices with a Merkle root over 1024-byte chunk leaves, and the noise
// seeds chain off those roots; the pool re-derives both with `pearl_blake3`. So the device needs
// keyed and unkeyed hashing over *arbitrarily many* chunks, with the chunk-CV tree walked exactly
// the way the reference walks it — including the carry-up of an odd node and the promotion of the
// last pair to the root. `check_blake3_tree` holds the two to each other.

#ifndef TOKENMINER_BLAKE3_CUH
#define TOKENMINER_BLAKE3_CUH

// Domain separation. The reference defines the same five in `pearl-blake3/src/hasher.rs`.
#define B3_CHUNK_START 1u
#define B3_CHUNK_END 2u
#define B3_PARENT 4u
#define B3_ROOT 8u
#define B3_KEYED_HASH 16u

#define B3_BLOCK_LEN 64u
#define B3_CHUNK_LEN 1024u
#define B3_WORDS 8u  // a chaining value is eight words, which is half of the 16-word state

__device__ __constant__ unsigned int B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};

// The message schedule for all seven rounds: round r reads the block permuted r times. Expanded
// here so the rounds themselves unroll over constant indices and the state stays in registers.
__device__ __constant__ unsigned char B3_SCHEDULE[7][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    { 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8},
    { 3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1},
    {10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6},
    {12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4},
    { 9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7},
    {11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13},
};

__device__ __forceinline__ unsigned int b3_rotr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}

#define B3_G(v, a, b, c, d, mx, my)       \
    do {                                  \
        v[a] = v[a] + v[b] + (mx);        \
        v[d] = b3_rotr(v[d] ^ v[a], 16);  \
        v[c] = v[c] + v[d];               \
        v[b] = b3_rotr(v[b] ^ v[c], 12);  \
        v[a] = v[a] + v[b] + (my);        \
        v[d] = b3_rotr(v[d] ^ v[a], 8);   \
        v[c] = v[c] + v[d];               \
        v[b] = b3_rotr(v[b] ^ v[c], 7);   \
    } while (0)

#define B3_ROUND(v, r)                                                                      \
    do {                                                                                    \
        B3_G(v, 0, 4,  8, 12, block[B3_SCHEDULE[r][ 0]], block[B3_SCHEDULE[r][ 1]]);        \
        B3_G(v, 1, 5,  9, 13, block[B3_SCHEDULE[r][ 2]], block[B3_SCHEDULE[r][ 3]]);        \
        B3_G(v, 2, 6, 10, 14, block[B3_SCHEDULE[r][ 4]], block[B3_SCHEDULE[r][ 5]]);        \
        B3_G(v, 3, 7, 11, 15, block[B3_SCHEDULE[r][ 6]], block[B3_SCHEDULE[r][ 7]]);        \
        B3_G(v, 0, 5, 10, 15, block[B3_SCHEDULE[r][ 8]], block[B3_SCHEDULE[r][ 9]]);        \
        B3_G(v, 1, 6, 11, 12, block[B3_SCHEDULE[r][10]], block[B3_SCHEDULE[r][11]]);        \
        B3_G(v, 2, 7,  8, 13, block[B3_SCHEDULE[r][12]], block[B3_SCHEDULE[r][13]]);        \
        B3_G(v, 3, 4,  9, 14, block[B3_SCHEDULE[r][14]], block[B3_SCHEDULE[r][15]]);        \
    } while (0)

// One BLAKE3 compression.
//
// `cv` is the incoming chaining value, `block` the sixteen message words, `counter` the *chunk*
// index, and `flags` the domain separation. The counter is the chunk counter, not a block counter:
// every block of a chunk compresses with the same counter, which is what the reference does
// (`ChunkState::output` passes `self.chunk_counter`; `c/blake3.c` likewise). The full 16-word output
// is written because the root of a tree is the first 32 bytes of it and the second Output block is
// what makes the final squeeze cheap — keeping one routine avoids two subtly different ones.
__device__ __forceinline__ void b3_compress(
        const unsigned int cv[8],
        const unsigned int block[16],
        unsigned long long counter,
        unsigned int block_len,
        unsigned int flags,
        unsigned int out[16]) {
    unsigned int v[16];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = cv[i];
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[8 + i] = B3_IV[i];
    }
    v[12] = (unsigned int)(counter & 0xffffffffull);
    v[13] = (unsigned int)(counter >> 32);
    v[14] = block_len;
    v[15] = flags;

    B3_ROUND(v, 0);
    B3_ROUND(v, 1);
    B3_ROUND(v, 2);
    B3_ROUND(v, 3);
    B3_ROUND(v, 4);
    B3_ROUND(v, 5);
    B3_ROUND(v, 6);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = v[i] ^ v[i + 8];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i + 8] = v[i + 8] ^ cv[i];
    }
}

// Little-endian 32-bit load of `nbytes` (<= 64) into a zeroed block.
//
// The zeroing is load-bearing: BLAKE3 pads the final short block with zeroes rather than leaving
// whatever was in the buffer.
__device__ __forceinline__ void b3_load_block(
        unsigned int block[16],
        const unsigned char* data,
        unsigned int nbytes) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        block[i] = 0;
    }

    const unsigned int words = nbytes / 4;
    for (unsigned int i = 0; i < words; ++i) {
        block[i] = (unsigned int)data[i * 4]
                 | ((unsigned int)data[i * 4 + 1] << 8)
                 | ((unsigned int)data[i * 4 + 2] << 16)
                 | ((unsigned int)data[i * 4 + 3] << 24);
    }

    const unsigned int tail = nbytes % 4;
    if (tail != 0) {
        unsigned int last = 0;
        for (unsigned int i = 0; i < tail; ++i) {
            last |= (unsigned int)data[words * 4 + i] << (8 * i);
        }
        block[words] = last;
    }
}

// The chaining value of one chunk, or of the whole input when `final_flags` carries ROOT.
//
// `chunk_index` is the compression counter for every block of the chunk. It is what makes a chunk's
// CV depend on where it sits in the input rather than only on its contents.
__device__ __forceinline__ void b3_chunk_cv(
        const unsigned char* data,
        unsigned int len,
        unsigned long long chunk_index,
        const unsigned int key_words[8],
        unsigned int base_flags,
        unsigned int final_flags,
        unsigned int out[16]) {
    unsigned int cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cv[i] = key_words[i];
    }

    unsigned int pos = 0;

    // An empty chunk is one zero-length block. `len` is never 0 for the mining matrices, but the
    // reference allows it and a self-test that pins the general case is worth the branch.
    if (len == 0) {
        unsigned int block[16];
        b3_load_block(block, data, 0);
        b3_compress(cv, block, chunk_index, 0,
                    base_flags | B3_CHUNK_START | B3_CHUNK_END | final_flags, out);
        return;
    }

    while (pos < len) {
        const unsigned int rest = (len - pos < B3_BLOCK_LEN) ? (len - pos) : B3_BLOCK_LEN;

        unsigned int flags = base_flags;
        if (pos == 0) {
            flags |= B3_CHUNK_START;
        }
        if (pos + rest == len) {
            flags |= B3_CHUNK_END | final_flags;
        }

        unsigned int block[16];
        b3_load_block(block, data + pos, rest);

        unsigned int next[16];
        b3_compress(cv, block, chunk_index, rest, flags, next);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            cv[i] = next[i];
        }
        pos += rest;
    }

    // The final compression already produced the chaining value; re-derive it from the state so
    // `out` is filled for both the last block and the empty-chunk path above.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = cv[i];
    }
    #pragma unroll
    for (int i = 8; i < 16; ++i) {
        out[i] = 0;
    }
}

// A parent node. `left` and `right` are eight-word chaining values.
__device__ __forceinline__ void b3_parent_cv(
        const unsigned int left[8],
        const unsigned int right[8],
        const unsigned int key_words[8],
        unsigned int base_flags,
        unsigned int final_flags,
        unsigned int out[16]) {
    unsigned int block[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        block[i] = left[i];
        block[i + 8] = right[i];
    }

    b3_compress(key_words, block, 0ull, B3_BLOCK_LEN,
                base_flags | B3_PARENT | final_flags, out);
}

// Writes a chaining value out as the little-endian bytes the rest of the code compares.
__device__ __forceinline__ void b3_store_words(unsigned char* out, const unsigned int words[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i * 4]     = (unsigned char)(words[i]);
        out[i * 4 + 1] = (unsigned char)(words[i] >> 8);
        out[i * 4 + 2] = (unsigned char)(words[i] >> 16);
        out[i * 4 + 3] = (unsigned char)(words[i] >> 24);
    }
}

// The same word read big-endian, i.e. what `b3_store_words` would call the bytes at
// `out + (7 - i) * 4` read as a big-endian integer.
//
// Needed wherever a digest is compared against a value the host sent as an integer. Comparing a
// little-endian word against a big-endian one orders that group by its bytes rather than by its
// value, which reverses it: `0x00ca867a` (digest bytes `7a 86 ca 00`) looks *smaller* than
// `0xffca867a` (bound bytes `7a 86 ca ff`), so a group that is actually larger reads as smaller and
// the bound inverts. Both halves of a comparison have to agree on the byte order *and* on the index
// order, or the verdict is a coin flip on the boundary rather than an arithmetic comparison.
__device__ __forceinline__ unsigned int b3_big_endian_word(unsigned int le_word) {
    return __byte_perm(le_word, 0u, 0x0123);
}

// The 64-byte message the noise tensors are built from.
//
// `get_random_hash` hashes `pad32(i32) || seed` under a key, which is one BLAKE3 block: the first
// four bytes carry `1 + index` at `prepend * 4` and the last thirty-two are the seed, with zeroes
// between. Reproducing the message rather than the hash lets every index share one code path, and
// keeps the `1 +` (rather than `index`) visible here where it is easy to see.
__device__ __forceinline__ void b3_noise_message(
        unsigned char message[64],
        const unsigned int seed_words[8],
        unsigned int index,
        unsigned int prepend) {
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        message[i] = 0;
    }

    const unsigned int value = 1u + index;
    message[prepend * 4]     = (unsigned char)(value);
    message[prepend * 4 + 1] = (unsigned char)(value >> 8);
    message[prepend * 4 + 2] = (unsigned char)(value >> 16);
    message[prepend * 4 + 3] = (unsigned char)(value >> 24);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        message[32 + i * 4]     = (unsigned char)(seed_words[i]);
        message[32 + i * 4 + 1] = (unsigned char)(seed_words[i] >> 8);
        message[32 + i * 4 + 2] = (unsigned char)(seed_words[i] >> 16);
        message[32 + i * 4 + 3] = (unsigned char)(seed_words[i] >> 24);
    }
}

// The high 32 bits of a 64-bit product of two 32-bit words, which is how the sparse permutation
// picks its second index (`mul_hi_u32`).
__device__ __forceinline__ unsigned int b3_mul_hi_u32(unsigned int a, unsigned int b) {
    return (unsigned int)(((unsigned long long)a * (unsigned long long)b) >> 32);
}

#endif  // TOKENMINER_BLAKE3_CUH