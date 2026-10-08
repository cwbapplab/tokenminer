// kernels_only.cu
//
// The translation unit compiled into `pearl_gemm.fatbin`, embedded via `include_bytes!` and loaded
// with `cuModuleLoadData`. Built by `build.rs::build_fatbin`, which is the reference's
// `csrc/build_fatbin.sh`: SASS for sm_86 / sm_89 / sm_120 plus `compute_86` PTX as the JIT fallback.
//
// Contents: the four headers the split search path needs (algorithm only — `__device__` functions),
// then the extern-C entry points the Rust launch layer looks up by name in `fatbin.rs::symbols`. The
// entry points are the only kernels in this TU, so the image contains exactly the symbols the launch
// layer can look up and nothing else.
//
// Four of the reference's nine headers are in scope. `merkle_sm80.cuh`, `merkle_combine_sm80.cuh`,
// `noising_sm80.cuh`, `noising_smem_sm80.cuh` and
// `pearl_gemm_search_perthread_smem_pipelined_sm80.cuh` are deliberately absent: tokenminer already
// has a verified commitment path (`check_commitment`, `check_job_key`), noising is the Triton blob,
// and the search is the Triton blob — not a ported CUDA kernel. The reference's own search kernel is
// 1x16 atoms per warp, so porting it would be a step down from the blob's 4x8.
//
// No template instantiations. The reference needs them because its entry points are templated kernels
// at file scope, and nvcc emits no code for a template that is never instantiated. Here every entry
// point is a plain `extern "C"` kernel, so the symbols exist without help.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "pearl_gemm/blake3_sm80.cuh"
#include "pearl_gemm/noise_generation_sm80.cuh"
#include "pearl_gemm/pearl_blake3_compare_sm80.cuh"
#include "pearl_gemm/pow_scan_emit_sm80.cuh"

#include "extern_c_shims.inc"
