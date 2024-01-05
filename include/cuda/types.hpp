#pragma once

#include <string>

#define cudaHostAllocMapped 0x02

#ifdef USE_GPU
#include <cuda_runtime_api.h>
using CudaErrorType = cudaError_t;
#else
using CudaErrorType = int;
constexpr CudaErrorType cudaSuccess = 0;
constexpr CudaErrorType cudaErrorInvalidValue = 1;
constexpr CudaErrorType cudaErrorMemoryAllocation = 2;
#endif

// Forward declaration of cudaStream_t
typedef struct CUstream_st* cudaStream_t;

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
  CudaErrorType errorCode;
  std::string errorMessage;

  StreamError(CudaErrorType code, const std::string& message)
    : errorCode(code),
      errorMessage(message) {}
};
