#pragma once

#include "types.hpp"

#include <tl/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * @brief CudaBuffer is an abstract base class that provides a common interface for various CUDA
 * allocation types (e.g. host-pinned, device, unified, etc.).
 */
class CudaBuffer {
public:
  virtual ~CudaBuffer() = default;

  virtual size_t size() const = 0;

  virtual void* cudaData() = 0;
  virtual const void* cudaData() const = 0;

  virtual bool isDevice() const = 0;

  virtual std::optional<StreamError> copyFrom(const CudaBuffer& src,
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
