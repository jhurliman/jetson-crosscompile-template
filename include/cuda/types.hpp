#pragma once

#ifdef USE_GPU
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
#include <memory>
#include <string>

// Restore the original macro definitions
#pragma pop_macro("__noinline__")
#pragma pop_macro("__noclone__")
#pragma pop_macro("__cold__")

// Now include the CUDA headers
#include <cuda_runtime_api.h>

#define HAS_CUDA_11_2 (CUDA_MAJOR == 11 && CUDA_MINOR >= 2) || (CUDA_MAJOR > 11)
#else
#include <string>

constexpr uint cudaHostAllocMapped = 2;
typedef struct CUstream_st* cudaStream_t;
using cudaError_t = int;

constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
#endif

#include "../errors.hpp"

enum class CudaMemAttachFlag {
  Global = 1,
  Host = 2,
  Single = 4,
};

enum class StreamPriority {
  High = 0,
  Normal = 1,
  Low = 2,
};

struct StreamError {
  cudaError_t errorCode;
  std::string errorMessage;

  StreamError(cudaError_t code, const std::string& message)
    : errorCode(code),
      errorMessage(message) {}
};

inline std::string cudaErrorMessage(const cudaError_t error) {
#ifdef USE_GPU
  return std::string(cudaGetErrorName(error)) + ": " + cudaGetErrorString(error);
#else
  return "CUDA error: " + std::to_string(error);
#endif
}
