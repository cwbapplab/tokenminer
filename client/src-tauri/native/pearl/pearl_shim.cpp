// Thin C++ shim: llmjob's extern "C" Pearl core -> the flat ABI in pearl_ffi.h.
//
// The vendored core already exposes extern "C" functions, but their result
// structs carry std::vector, which Rust cannot read. This file mirrors those
// structs (the same way pearl_core.cc does, so it can call the same functions),
// calls the core, and republishes each result as borrowed pointer+length slices.
//
// The vectors live in the handle, so the pointers a PearlHitFlat hands out stay
// valid until the next call on that handle. That is the entire reason the handle
// is an opaque struct rather than just the Ctx*.

#include "pearl_ffi.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

// PearlProfile is declared in pearl_config.h, which the shim does not include (it
// is a device-shared header). Forward-declare it as the tagged struct the core
// uses; pearl_abi_check.cpp includes the real one and static_asserts that our
// PearlProfileFlat is laid out identically, so a mismatch fails the build.
typedef struct PearlProfile PearlProfile;

// --- Mirror of the vendored structs and entry points ------------------------
// Declared identically in pearl_host.cu. Kept in this one form deliberately:
// pulling a shared header in would drag the CUDA sources' definitions into the
// host compiler. pearl_abi_check.cpp asserts the layout matches.

struct PearlProofSide {
  std::vector<uint32_t> leaf_indices;
  std::vector<uint8_t> leaves;    // leaf_indices.size() * 1024
  std::vector<uint8_t> siblings;  // n * 32
  uint8_t root[PEARL_FFI_HASH_BYTES];
  uint64_t total_leaves;
};

struct PearlSearchResult {
  uint8_t jackpot_hash[PEARL_FFI_HASH_BYTES];
  uint8_t a_seed[PEARL_FFI_HASH_BYTES];
  uint8_t b_seed[PEARL_FFI_HASH_BYTES];
  uint64_t nonce;
  uint64_t salt;
  std::vector<uint8_t> proof;
  PearlProofSide proof_a;
  PearlProofSide proof_bt;
  bool found;
};

extern "C" {
int pearl_host_select_device(const PearlProfile *profile, int requested, char *name,
                             size_t name_len, char *err, size_t err_len);
void *pearl_host_create(const PearlProfile *profile, char *err, size_t err_len);
void pearl_host_destroy(void *ctx);
void pearl_host_bind_thread(void *ctx);
void pearl_host_set_job_salted(void *ctx, const uint8_t *header, const uint8_t *target,
                               uint64_t salt);
void pearl_host_reseed(void *ctx, uint64_t salt);
bool pearl_host_search(void *ctx, uint64_t nonce_base, uint32_t batch,
                       PearlSearchResult *out, uint64_t *attempts, char *err,
                       size_t err_len);
bool pearl_host_submit(void *ctx, uint64_t nonce_base, uint32_t batch, uint64_t *regions,
                       char *err, size_t err_len);
bool pearl_host_collect(void *ctx, PearlSearchResult *out, uint64_t *attempts, char *err,
                        size_t err_len);
int pearl_host_pending(void *ctx);
bool pearl_host_next_hit(void *ctx, PearlSearchResult *out);
const char *pearl_host_fold_name(void *ctx);
}

// --- Handle -----------------------------------------------------------------

struct PearlFfiHandle {
  void *ctx = nullptr;
  // The profile and the chosen device index, kept so a thread that did not
  // create the context can make that device current before dropping it.
  PearlProfileFlat profile = {};
  int device = 0;
  char name[256] = {0};
  // The most recent result, kept alive so PearlHitFlat's pointers remain valid
  // until the next call. `ok` is what the last call returned.
  PearlSearchResult last;
  int ok = 0;
};

namespace {

// Fill a flat side from the shim's own copy of a result side.
void fill_side(PearlSideFlat *out, const PearlProofSide &s) {
  out->leaf_indices = s.leaf_indices.empty() ? nullptr : s.leaf_indices.data();
  out->leaf_indices_len = s.leaf_indices.size();
  out->leaves = s.leaves.empty() ? nullptr : s.leaves.data();
  out->leaves_len = s.leaves.size();
  out->siblings = s.siblings.empty() ? nullptr : s.siblings.data();
  out->siblings_len = s.siblings.size();
  memcpy(out->root, s.root, PEARL_FFI_HASH_BYTES);
  out->total_leaves = s.total_leaves;
}

// Republish the handle's stored result. `attempts` is only meaningful on the
// collect path; search records it too.
void publish(PearlFfiHandle *h, PearlHitFlat *out, uint64_t attempts) {
  const PearlSearchResult &r = h->last;
  memcpy(out->jackpot_hash, r.jackpot_hash, PEARL_FFI_HASH_BYTES);
  memcpy(out->a_seed, r.a_seed, PEARL_FFI_HASH_BYTES);
  memcpy(out->b_seed, r.b_seed, PEARL_FFI_HASH_BYTES);
  out->nonce = r.nonce;
  out->salt = r.salt;
  out->attempts = attempts;
  out->proof = r.proof.empty() ? nullptr : r.proof.data();
  out->proof_len = r.proof.size();
  fill_side(&out->proof_a, r.proof_a);
  fill_side(&out->proof_bt, r.proof_bt);
  out->found = r.found ? 1 : 0;
}

void clear_hit(PearlHitFlat *out) {
  memset(out, 0, sizeof(*out));
  out->found = 0;
}

// The profile is passed by pointer and laid out identically to the core's
// PearlProfile (asserted at build time), so no per-field unpacking is needed.
const PearlProfile *as_profile(const PearlProfileFlat *p) {
  return reinterpret_cast<const PearlProfile *>(p);
}

} // namespace

// --- ABI --------------------------------------------------------------------

int pearl_ffi_select_device(const PearlProfileFlat *p, int requested, char *name,
                            size_t name_len, char *err, size_t err_len) {
  if (!p) {
    if (err && err_len) snprintf(err, err_len, "no profile supplied");
    return -1;
  }
  return pearl_host_select_device(as_profile(p), requested, name, name_len, err, err_len);
}

PearlFfiHandle *pearl_ffi_create(const PearlProfileFlat *p, int device, char *err,
                                size_t err_len) {
  if (!p) {
    if (err && err_len) snprintf(err, err_len, "no profile supplied");
    return nullptr;
  }
  void *ctx = pearl_host_create(as_profile(p), err, err_len);
  if (!ctx) return nullptr;
  PearlFfiHandle *h = new (std::nothrow) PearlFfiHandle();
  if (!h) {
    pearl_host_destroy(ctx);
    if (err && err_len) snprintf(err, err_len, "out of host memory");
    return nullptr;
  }
  h->ctx = ctx;
  h->profile = *p;
  h->device = device;
  return h;
}

void pearl_ffi_destroy(PearlFfiHandle *h) {
  if (!h) return;
  if (h->ctx) pearl_host_destroy(h->ctx);
  delete h;
}

void pearl_ffi_bind_thread(PearlFfiHandle *h) {
  if (h && h->ctx) pearl_host_bind_thread(h->ctx);
}

int pearl_ffi_reselect_device(PearlFfiHandle *h, char *err, size_t err_len) {
  if (!h || !h->ctx) {
    if (err && err_len) snprintf(err, err_len, "reselect called with no context");
    return -1;
  }
  // pearl_host_select_device sets the device for the calling thread, so handing
  // it the context's own index makes that index current here. It is the only
  // entry point that enumerates devices first, which is what a thread dropping
  // a context has to do before any per-device call will work at all.
  int index = pearl_host_select_device(as_profile(&h->profile), h->device, nullptr, 0, err,
                                       err_len);
  if (index < 0) return -1;
  if (index != h->device) {
    if (err && err_len)
      snprintf(err, err_len, "device %d was selected but this thread landed on %d", h->device,
               index);
    return -1;
  }
  return index;
}

void pearl_ffi_set_job(PearlFfiHandle *h, const uint8_t *header, const uint8_t *target,
                       uint64_t salt) {
  if (h && h->ctx) pearl_host_set_job_salted(h->ctx, header, target, salt);
}

void pearl_ffi_reseed(PearlFfiHandle *h, uint64_t salt) {
  if (h && h->ctx) pearl_host_reseed(h->ctx, salt);
}

int pearl_ffi_search(PearlFfiHandle *h, uint64_t nonce_base, uint32_t batch,
                     PearlHitFlat *out, char *err, size_t err_len) {
  if (!h || !h->ctx || !out) {
    if (out) clear_hit(out);
    if (err && err_len) snprintf(err, err_len, "search called with no context");
    return -1;
  }
  uint64_t attempts = 0;
  clear_hit(out);
  bool ok = pearl_host_search(h->ctx, nonce_base, batch, &h->last, &attempts, err, err_len);
  // A false return is "no hit", not an error, unless a message was written — the
  // core writes `err` only on a real CUDA fault.
  if (!ok && err && err_len && err[0]) return -1;
  h->ok = ok ? 1 : 0;
  if (ok) publish(h, out, attempts);
  return h->ok;
}

int pearl_ffi_submit(PearlFfiHandle *h, uint64_t nonce_base, uint32_t batch,
                     uint64_t *regions_out, char *err, size_t err_len) {
  if (regions_out) *regions_out = 0;
  if (!h || !h->ctx) {
    if (err && err_len) snprintf(err, err_len, "submit called with no context");
    return -1;
  }
  if (!pearl_host_submit(h->ctx, nonce_base, batch, regions_out, err, err_len)) {
    if (err && err_len && err[0]) return -1;
    return 0; // pipeline full or no job — advisably a retry, not an error
  }
  return 1;
}

int pearl_ffi_collect(PearlFfiHandle *h, PearlHitFlat *out, char *err, size_t err_len) {
  if (!h || !h->ctx || !out) {
    if (out) clear_hit(out);
    if (err && err_len) snprintf(err, err_len, "collect called with no context");
    return -1;
  }
  uint64_t attempts = 0;
  clear_hit(out);
  bool ok = pearl_host_collect(h->ctx, &h->last, &attempts, err, err_len);
  if (!ok && err && err_len && err[0]) return -1;
  h->ok = ok ? 1 : 0;
  if (ok) publish(h, out, attempts);
  return h->ok;
}

int pearl_ffi_pending(PearlFfiHandle *h) {
  if (!h || !h->ctx) return 0;
  return pearl_host_pending(h->ctx);
}

int pearl_ffi_next_hit(PearlFfiHandle *h, PearlHitFlat *out) {
  if (!h || !h->ctx || !out) return 0;
  clear_hit(out);
  if (!pearl_host_next_hit(h->ctx, &h->last)) return 0;
  publish(h, out, 0);
  return 1;
}

const char *pearl_ffi_fold_name(PearlFfiHandle *h) {
  if (!h || !h->ctx) return "no context";
  const char *n = pearl_host_fold_name(h->ctx);
  return n ? n : "unresolved";
}
