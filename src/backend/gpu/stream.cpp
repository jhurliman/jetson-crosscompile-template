#include "cuda/stream.hpp"

#include "cuda_common.hpp"

#include <nvToolsExtCudaRt.h>

tl::expected<cudaStream_t, StreamError> createStream(
  const std::string_view name, const StreamPriority priority) {
  // Get the min/max priorities for the current device
  int minPriority, maxPriority;
  cudaError_t error = cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);
  if (error != cudaSuccess) {
    return tl::make_unexpected(StreamError{error, CudaErrorMessage(error)});
  }

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
  error = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priorityValue);
  if (error != cudaSuccess) {
    return tl::make_unexpected(StreamError{error, CudaErrorMessage(error)});
  }

  // Set the stream name
  nvtxNameCudaStreamA(stream, name.data());

  return stream;
}

void destroyStream(cudaStream_t stream) {
  cudaStreamDestroy(stream);
}
