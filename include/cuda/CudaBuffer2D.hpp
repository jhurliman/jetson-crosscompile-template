#pragma once

#include "CudaBuffer.hpp"
#include "types.hpp"

#include <tl/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * @brief CudaBuffer is an abstract base class that provides a common interface for various CUDA
 * allocation types (e.g. host-pinned, device, device pitched, unified, etc.) of two-dimensional
 * data buffers (e.g. images, matrices, etc.).
 */
class CudaBuffer2D : public CudaBuffer {
public:
  virtual ~CudaBuffer2D() = default;

  virtual size_t width() const = 0;
  virtual size_t height() const = 0;
  virtual size_t pitch() const = 0;

  virtual std::optional<StreamError> copyFrom2D(const CudaBuffer2D& src,
    size_t srcX,
    size_t srcY,
    size_t dstX,
    size_t dstY,
    size_t width,
    size_t height,
    cudaStream_t stream) = 0;

  virtual std::optional<StreamError> copyFromHost2D(const void* src,
    size_t dstX,
    size_t dstY,
    size_t width,
    size_t height,
    cudaStream_t stream) = 0;

  virtual std::optional<StreamError> copyTo2D(CudaBuffer2D& dst,
    size_t srcX,
    size_t srcY,
    size_t dstX,
    size_t dstY,
    size_t width,
    size_t height,
    cudaStream_t stream) const = 0;

  virtual std::optional<StreamError> copyToHost2D(void* dst,
    size_t srcX,
    size_t srcY,
    size_t width,
    size_t height,
    cudaStream_t stream,
    bool synchronize = true) const = 0;
};
