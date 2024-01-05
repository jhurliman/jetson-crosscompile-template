#include "cuda/types.hpp"

#include <cuda_runtime.h>

#include <string>

#define HAS_CUDA_11_2 (CUDA_MAJOR == 11 && CUDA_MINOR >= 2) || (CUDA_MAJOR > 11)

inline std::string CudaErrorMessage(const cudaError_t error) {
  return std::string(cudaGetErrorName(error)) + ": " + cudaGetErrorString(error);
}
