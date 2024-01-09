#pragma once

#include "cuda_wrappers/CudaBuffer.hpp"
#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * @brief NvmmBuffer is an abstract base class that provides a RAII wrapper for NVIDIA Jetson
 * Multimedia API NvBuffer allocations.
 */
class NvmmBuffer {
public:
  NvmmBuffer() = default;

  virtual ~NvmmBuffer() = default;

  NvmmBuffer(const NvmmBuffer&) = delete;
  NvmmBuffer& operator=(const NvmmBuffer&) = delete;

  NvmmBuffer(NvmmBuffer&&) = default;
  NvmmBuffer& operator=(NvmmBuffer&&) = default;

  virtual size_t size() const = 0;

  virtual int fd() = 0;
  virtual int fd() const = 0;

  virtual std::optional<NvmmError> copyFrom(const CudaBuffer& src,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    cudaStream_t stream) = 0;

  virtual std::optional<StreamError> copyFromHost(
    const void* src, size_t dstOffset, size_t count, cudaStream_t stream) = 0;

  virtual std::optional<StreamError> copyTo(CudaBuffer& dst,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    cudaStream_t stream) const = 0;

  virtual std::optional<StreamError> copyToHost(void* dst,
    size_t srcOffset,
    size_t count,
    cudaStream_t stream,
    bool synchronize = true) const = 0;

  virtual std::optional<StreamError> memset(std::byte value, size_t count, cudaStream_t stream) = 0;
};
