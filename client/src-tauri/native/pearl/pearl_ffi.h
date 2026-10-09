// Flat C ABI over llmjob's Pearl CUDA core.
//
// The vendored core (vendor/llmjob/earn/native/src/pearl_host.cu) exposes an
// extern "C" API, but two of its structs — PearlSearchResult and PearlProofSide
// — carry std::vector, so they cannot cross into Rust as-is. This shim keeps the
// vectors on the C++ side and hands Rust borrowed pointer+length slices instead.
//
// This header is the contract. It is included by pearl_shim.cpp and mirrored by
// hand in ../src/miners/pearl/abi.rs; both sides compile, and pearl_abi_check.cpp
// static_asserts the layout against the vendored pearl_config.h, so a submodule
// bump that changes the geometry fails the build instead of mining nothing.
//
// Rules the Rust side must hold to:
//   * Every pointer in PearlHitFlat / PearlSideFlat borrows storage owned by the
//     PearlFfiHandle, and is valid only until the next call on that handle.
//     Copy out of them before calling again.
//   * No call throws. Errors come back as -1/0/1 plus a message in `err`.

#ifndef PEARL_FFI_H
#define PEARL_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PEARL_FFI_HEADER_BYTES 76
#define PEARL_FFI_HASH_BYTES 32
#define PEARL_FFI_CHUNK_BYTES 1024   // one Merkle leaf; leaves_len == leaf count * this
#define PEARL_FFI_DIGEST_BYTES 32    // one sibling digest

// Mirror of the vendored PearlProfile. Field order, widths and padding are
// asserted equal by pearl_abi_check.cpp; passing this by pointer lets the shim
// hand the real struct straight to pearl_host_create without unpacking fields.
typedef struct PearlProfileFlat {
  uint32_t k;               // common dimension; must be 2048 (PEARL_FOLD_K)
  uint16_t rank;            // must be 128 (PEARL_FOLD_RANK)
  uint16_t mma_type;        // 0 = Int7xInt7ToInt32
  uint32_t m;               // miner's own workload height (not hashed)
  uint32_t n;               // miner's own workload width (not hashed)
  uint32_t seed_derivation; // 0 = cert-v3 salted, 1 = legacy
  uint32_t col_batch;       // column offsets one launch covers
  uint32_t hash_big_endian; // 0 = little-endian jackpot compare (the reference)
  uint32_t operand_fill;    // 0 = hashed, 1 = constant
} PearlProfileFlat;

// One side of a share proof. `leaves` is `leaf_indices_len * PEARL_FFI_CHUNK_BYTES`
// long; `siblings` is a whole number of PEARL_FFI_DIGEST_BYTES.
typedef struct PearlSideFlat {
  const uint32_t *leaf_indices;
  uint64_t leaf_indices_len;
  const uint8_t *leaves;
  uint64_t leaves_len;
  const uint8_t *siblings;
  uint64_t siblings_len;
  uint8_t root[PEARL_FFI_HASH_BYTES];
  uint64_t total_leaves;
} PearlSideFlat;

// One hit. `proof` is the core's 64-byte transcript, not the wire proof: the
// wire proof is assembled on the Rust side from proof_a/proof_bt.
typedef struct PearlHitFlat {
  uint8_t jackpot_hash[PEARL_FFI_HASH_BYTES];
  uint8_t a_seed[PEARL_FFI_HASH_BYTES];
  uint8_t b_seed[PEARL_FFI_HASH_BYTES];
  uint64_t nonce;
  uint64_t salt;
  uint64_t attempts; // this batch's region count (what `collect` reports)
  const uint8_t *proof;
  uint64_t proof_len;
  PearlSideFlat proof_a;
  PearlSideFlat proof_bt;
  int found;
} PearlHitFlat;

typedef struct PearlFfiHandle PearlFfiHandle;

// Choose the GPU and make it current on this thread. Must be called before
// pearl_ffi_create. Writes the card name into `name`; returns the chosen index,
// or -1 with a message in `err`.
int pearl_ffi_select_device(const PearlProfileFlat *p, int requested, char *name,
                            size_t name_len, char *err, size_t err_len);

// Allocate the device state. `device` is the index pearl_ffi_select_device
// returned; it is remembered so other threads can make that device current.
// Null on failure, with a message in `err`.
PearlFfiHandle *pearl_ffi_create(const PearlProfileFlat *p, int device, char *err,
                                 size_t err_len);
void pearl_ffi_destroy(PearlFfiHandle *h);

// Point this thread at the card the context lives on. The search thread must
// call this once before driving the handle.
void pearl_ffi_bind_thread(PearlFfiHandle *h);

// Make the device the context lives on current on a DIFFERENT thread, before
// that thread touches any of the handle's state.
//
// The core's every cudaFree and every cudaMalloc happens inside a DeviceScope
// that reads ctx->device, so a thread that never selected the device can still
// destroy a context safely — but only if the runtime can see the device at all.
// On a Windows cuInit(0)-less driver (or a machine whose only GPU is
// cudaComputeModeProhibited) any device query without a prior cuInit fails with
// cudaErrorNoDevice, and cudaSetDevice on a device the process has never
// enumerated fails the same way — which would make the destroy path (and only
// the destroy path, since the miner thread selects properly) fail. Call this
// once on whichever thread will drop the handle when that is not the thread
// that created it.
int pearl_ffi_reselect_device(PearlFfiHandle *h, char *err, size_t err_len);

// Load a job. header is 76 bytes, target is 32 bytes (big-endian), salt is this
// card's slice of the search space.
void pearl_ffi_set_job(PearlFfiHandle *h, const uint8_t *header, const uint8_t *target,
                       uint64_t salt);
void pearl_ffi_reseed(PearlFfiHandle *h, uint64_t salt);

// Search one batch synchronously. 1 = hit (out filled), 0 = no hit, -1 = error.
int pearl_ffi_search(PearlFfiHandle *h, uint64_t nonce_base, uint32_t batch,
                     PearlHitFlat *out, char *err, size_t err_len);

// Pipelined search: submit queues a batch without waiting (at most two in
// flight) and reports its region count; collect waits for the oldest and reports
// it as pearl_ffi_search would. 1 = hit, 0 = none, -1 = error.
int pearl_ffi_submit(PearlFfiHandle *h, uint64_t nonce_base, uint32_t batch,
                     uint64_t *regions_out, char *err, size_t err_len);
int pearl_ffi_collect(PearlFfiHandle *h, PearlHitFlat *out, char *err, size_t err_len);
// The region count of the most recent pearl_ffi_collect, whether or not it hit.
//
// The core only hands `attempts` back through its hit result, but a batch that
// found nothing is exactly as much work as one that did, and it is the common
// case at a pool's share difficulty — so a hashrate counted from hits alone
// reads near zero while the card is fully busy. The shim records the count on
// the collect itself (it is valid even when collect returned 0) so the caller can
// count *finished* batches. Negative or unset before the first collect.
int64_t pearl_ffi_last_regions(PearlFfiHandle *h);
int pearl_ffi_pending(PearlFfiHandle *h);
// The collected batch's further hits, one per call, after the one collect
// returned. 1 = hit, 0 = no more, -1 = error.
int pearl_ffi_next_hit(PearlFfiHandle *h, PearlHitFlat *out);

// Which fold this card runs, for the startup log. Never null.
const char *pearl_ffi_fold_name(PearlFfiHandle *h);

#ifdef __cplusplus
}
#endif

#endif // PEARL_FFI_H
