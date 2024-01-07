#include "cuda/stream.hpp"

#include "cuda_expected.hpp"

#include <nvToolsExtCudaRt.h>

namespace cuda {

std::optional<StreamError> ensureInitialized() {
  static bool initialized = false;
  if (initialized) { return {}; }
  initialized = true;

  // Set the device flags
  CUDA_OPTIONAL(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));

  // Initialize the CUDA runtime
  CUDA_OPTIONAL(cudaFree(nullptr));

  // Get the current device
  int device;
  CUDA_OPTIONAL(cudaGetDevice(&device));

  // Ensure mapped pinned allocations are supported
  int canMapHostMemory;
  CUDA_OPTIONAL(cudaDeviceGetAttribute(&canMapHostMemory, cudaDevAttrCanMapHostMemory, device));

  int concurrentManagedAccess;
  CUDA_OPTIONAL(
    cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, device));

  if (!canMapHostMemory) {
    return StreamError{cudaErrorUnknown, "Device does not support mapped pinned allocations"};
  }

  if (!concurrentManagedAccess) {
    return StreamError{cudaErrorUnknown, "Device does not support concurrent managed access"};
  }

  return {};
}

tl::expected<cudaStream_t, StreamError> createStream(
  const std::string_view name, const StreamPriority priority) {
  CUDA_EXPECTED_INIT();

  // Get the min/max priorities for the current device
  int minPriority, maxPriority;
  CUDA_EXPECTED(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));

  // Calculate the priority for the new stream
  int priorityValue;
  switch (priority) {
  case StreamPriority::High:
    priorityValue = maxPriority;
    break;
  case StreamPriority::Normal:
    priorityValue = (maxPriority + minPriority) / 2;
    if (priorityValue == maxPriority) { priorityValue = minPriority; }
    break;
  case StreamPriority::Low:
    priorityValue = minPriority;
    break;
  }

  // Create the stream
  cudaStream_t stream;
  CUDA_EXPECTED(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priorityValue));

  // Set the stream name
  nvtxNameCudaStreamA(stream, name.data());

  return stream;
}

std::optional<StreamError> destroyStream(cudaStream_t stream) {
  CUDA_OPTIONAL(cudaStreamDestroy(stream));
  return {};
}

std::optional<StreamError> synchronizeStream(cudaStream_t stream) {
  CUDA_OPTIONAL(cudaStreamSynchronize(stream));
  return {};
}

} // namespace cuda
