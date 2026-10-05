// The single translation unit `build.rs` compiles.
//
// The driver API loads one cubin as one module, and every entry point the backend looks up has to
// be in it — so the kernels are split across files for reading, not for linking, and this file
// pulls them into one compilation. `nvcc -cubin` takes a single input, which is the constraint
// behind that arrangement.

#ifndef __CUDACC__
#error "kernels.cu must be compiled by nvcc"
#endif

#include "probe.cu"
#include "pearl.cu"