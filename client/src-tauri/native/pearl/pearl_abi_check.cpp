// Compile-time and run-time check that our flat ABI still matches the vendored
// Pearl core. build.rs compiles AND runs this; a mismatch fails the build.
//
// Why this exists: PearlProfileFlat is reinterpret_cast to the core's
// PearlProfile (pearl_shim.cpp), and the result structs are mirrored by hand. A
// submodule bump that changes either would otherwise compile cleanly and mine
// garbage — the constructors and geometry still "work", the hashes are just
// wrong. The static_asserts make that a build error.

#include "pearl_config.h"   // the vendored header
#include "pearl_ffi.h"      // our contract

#include <cstddef>
#include <cstdint>
#include <cstdio>

// The scalar fields the shim hands to pearl_host_* must line up field for field.
// sizeof catches added/removed fields; the offsetof set catches reordering and
// any change to a field's width.
static_assert(sizeof(PearlProfileFlat) == sizeof(PearlProfile),
              "PearlProfileFlat no longer matches PearlProfile in size");
static_assert(offsetof(PearlProfileFlat, k) == offsetof(PearlProfile, k), "k moved");
static_assert(offsetof(PearlProfileFlat, rank) == offsetof(PearlProfile, rank), "rank moved");
static_assert(offsetof(PearlProfileFlat, mma_type) == offsetof(PearlProfile, mma_type),
              "mma_type moved");
static_assert(offsetof(PearlProfileFlat, m) == offsetof(PearlProfile, m), "m moved");
static_assert(offsetof(PearlProfileFlat, n) == offsetof(PearlProfile, n), "n moved");
static_assert(offsetof(PearlProfileFlat, seed_derivation) ==
                  offsetof(PearlProfile, seed_derivation),
              "seed_derivation moved");
static_assert(offsetof(PearlProfileFlat, col_batch) == offsetof(PearlProfile, col_batch),
              "col_batch moved");
static_assert(offsetof(PearlProfileFlat, hash_big_endian) ==
                  offsetof(PearlProfile, hash_big_endian),
              "hash_big_endian moved");
static_assert(offsetof(PearlProfileFlat, operand_fill) == offsetof(PearlProfile, operand_fill),
              "operand_fill moved");

// The geometry the Rust side hardcodes in config.rs. If a submodule bump moves
// any of these, the Rust maths (config52, the fold) would silently disagree with
// the kernel.
static_assert(PEARL_CONFIG_BYTES == 52, "config52 width changed");
static_assert(PEARL_HEADER_BYTES == 76, "header width changed");
static_assert(PEARL_HASH_BYTES == 32, "hash width changed");
static_assert(PEARL_ROWS_COUNT == 16, "rows pattern count changed");
static_assert(PEARL_COLS_COUNT == 16, "cols pattern count changed");
static_assert(PEARL_FOLD_RANK == 128u, "fold rank changed");
static_assert(PEARL_FOLD_K == 2048u, "fold k changed");
static_assert(PEARL_FFI_CHUNK_BYTES == 1024, "Merkle chunk width changed");

int main() {
  // A fingerprint the build log can show, so a reader can see what was checked.
  std::printf(
      "pearl ABI ok: profile=%zu config52=%d header=%d rank=%u k=%u rows=%d cols=%d "
      "chunk=%d\n",
      sizeof(PearlProfile), PEARL_CONFIG_BYTES, PEARL_HEADER_BYTES, PEARL_FOLD_RANK,
      PEARL_FOLD_K, PEARL_ROWS_COUNT, PEARL_COLS_COUNT, PEARL_FFI_CHUNK_BYTES);
  return 0;
}
