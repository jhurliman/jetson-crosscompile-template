#pragma once

#include "cuda/stream.hpp" // IWYU pragma: export

#include <cuda_runtime_api.h>
#include <tl/expected.hpp>

#define CUDA_EXPECTED(call)                                                                        \
  do {                                                                                             \
    const cudaError_t err = (call);                                                                \
    if (err != cudaSuccess) {                                                                      \
      return tl::make_unexpected(StreamError{err, cudaGetErrorString(err), __FILE__, __LINE__});   \
    }                                                                                              \
  } while (false)

#define CUDA_OPTIONAL(call)                                                                        \
  do {                                                                                             \
    const cudaError_t err = (call);                                                                \
    if (err != cudaSuccess) {                                                                      \
      return StreamError{err, cudaGetErrorString(err), __FILE__, __LINE__};                        \
    }                                                                                              \
  } while (false)

#define CUDA_EXPECTED_INIT()                                                                       \
  do {                                                                                             \
    const auto maybeErr = cuda::ensureInitialized();                                               \
    if (maybeErr) {                                                                                \
      return tl::make_unexpected(                                                                  \
        StreamError{maybeErr->errorCode, maybeErr->errorMessage, __FILE__, __LINE__});             \
    }                                                                                              \
  } while (false)

#define CUDA_CHECK_PREVIOUS_ERROR()                                                                \
  do {                                                                                             \
    const cudaError_t prevError = cudaGetLastError();                                              \
    if (prevError != cudaSuccess) {                                                                \
      return StreamError{prevError,                                                                \
        std::string{"Previous error: "} + cudaGetErrorString(prevError),                           \
        __FILE__,                                                                                  \
        __LINE__};                                                                                 \
    }                                                                                              \
  } while (false)

#ifdef __INTELLISENSE__

#define CUDA_KERNEL(kernel, numBlocks, blockSize, sharedMemSize, stream, ...)                      \
  do {                                                                                             \
    CUDA_CHECK_PREVIOUS_ERROR();                                                                   \
    kernel(__VA_ARGS__);                                                                           \
    CUDA_OPTIONAL(cudaGetLastError());                                                             \
  } while (false)

#else

#define CUDA_KERNEL(kernel, numBlocks, blockSize, sharedMemSize, stream, ...)                      \
  do {                                                                                             \
    CUDA_CHECK_PREVIOUS_ERROR();                                                                   \
    kernel<<<numBlocks, blockSize, sharedMemSize, stream>>>(__VA_ARGS__);                          \
    CUDA_OPTIONAL(cudaGetLastError());                                                             \
  } while (false)

#endif
