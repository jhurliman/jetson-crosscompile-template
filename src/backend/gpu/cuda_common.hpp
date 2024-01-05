#include "cuda/types.hpp"

// When compiling against CUDA and libstdc++ 13+ with clang <= 17, we run into
// the issue described at <https://github.com/llvm/llvm-project/issues/62939>
// where a conflict between the `__noinline__` macro defined in the CUDA headers
// and the `__noinline__` macro defined in the libstdc++ headers causes
// compilation to fail. The `nvcc` compiler and clang 18+ include a workaround
// for this issue, which we manually apply here.

// Save the original definitions of the macros, if they exist
#pragma push_macro("__noinline__")
#pragma push_macro("__noclone__")
#pragma push_macro("__cold__")

// Undefine the macros
#undef __noinline__
#undef __noclone__
#undef __cold__

// Now include the standard library headers
#include <string>

// Restore the original macro definitions
#pragma pop_macro("__noinline__")
#pragma pop_macro("__noclone__")
#pragma pop_macro("__cold__")

// Now include the CUDA headers
#include <cuda_runtime.h>

#define HAS_CUDA_11_2 (CUDA_MAJOR == 11 && CUDA_MINOR >= 2) || (CUDA_MAJOR > 11)

inline std::string CudaErrorMessage(const cudaError_t error) {
  return std::string(cudaGetErrorName(error)) + ": " + cudaGetErrorString(error);
}
