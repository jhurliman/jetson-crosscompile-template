#pragma once

#include "CudaBuffer.hpp"
#include "types.hpp"

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
  CudaBuffer2D() = default;

  ~CudaBuffer2D() override = default;

  CudaBuffer2D(const CudaBuffer2D&) = delete;
  CudaBuffer2D& operator=(const CudaBuffer2D&) = delete;
  CudaBuffer2D(CudaBuffer2D&&) = delete;
  CudaBuffer2D& operator=(CudaBuffer2D&&) = delete;

  virtual size_t capacity() const = 0;
  virtual size_t widthBytes() const = 0;
  virtual size_t height() const = 0;
  virtual size_t pitch() const = 0;

  virtual std::optional<StreamError> copyFrom2D(const CudaBuffer2D& src,
    size_t srcX,
    size_t srcY,
    size_t dstX,
    size_t dstY,
    size_t widthBytes,
    size_t height,
    cudaStream_t stream) = 0;

  virtual std::optional<StreamError> copyFromHost2D(const void* src,
    size_t srcPitch,
    size_t dstX,
    size_t dstY,
    size_t widthBytes,
    size_t height,
    cudaStream_t stream) = 0;

  virtual std::optional<StreamError> copyTo2D(CudaBuffer2D& dst,
    size_t srcX,
    size_t srcY,
    size_t dstX,
    size_t dstY,
    size_t widthBytes,
    size_t height,
    cudaStream_t stream) const = 0;

  virtual std::optional<StreamError> copyToHost2D(void* dst,
    size_t dstPitch,
    size_t srcX,
    size_t srcY,
    size_t widthBytes,
    size_t height,
    cudaStream_t stream,
    bool synchronize = true) const = 0;
};
