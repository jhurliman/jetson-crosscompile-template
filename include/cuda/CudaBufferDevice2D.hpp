#pragma once

#include "CudaBuffer2D.hpp"
#include "CudaBufferDevice.hpp"

#include <tl/expected.hpp>

class CudaBufferDevice2D : public CudaBuffer2D {
public:
  static tl::expected<std::unique_ptr<CudaBufferDevice2D>, StreamError> create(
    size_t widthBytes, size_t height, cudaStream_t stream);

  ~CudaBufferDevice2D() override = default;

  size_t size() const override;
  size_t capacity() const override;
  size_t widthBytes() const override;
  size_t height() const override;
  size_t pitch() const override;

  void* cudaData() override;
  const void* cudaData() const override;

  bool isDevice() const override { return true; }

  std::optional<StreamError> copyFrom(const CudaBuffer& src,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    cudaStream_t stream) override;

  std::optional<StreamError> copyFromHost(
    const void* src, size_t dstOffset, size_t count, cudaStream_t stream) override;

  std::optional<StreamError> copyTo(CudaBuffer& dst,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    cudaStream_t stream) const override;

  std::optional<StreamError> copyToHost(void* dst,
    size_t srcOffset,
    size_t count,
    cudaStream_t stream,
    bool synchronize = true) const override;

  std::optional<StreamError> memset(std::byte value, size_t count, cudaStream_t stream) override;

  std::optional<StreamError> copyFrom2D(const CudaBuffer2D& src,
    size_t srcX,
    size_t srcY,
    size_t dstX,
    size_t dstY,
    size_t widthBytes,
    size_t height,
    cudaStream_t stream) override;

  std::optional<StreamError> copyFromHost2D(const void* src,
    size_t srcPitch,
    size_t dstX,
    size_t dstY,
    size_t widthBytes,
    size_t height,
    cudaStream_t stream) override;

  std::optional<StreamError> copyTo2D(CudaBuffer2D& dst,
    size_t srcX,
    size_t srcY,
    size_t dstX,
    size_t dstY,
    size_t widthBytes,
    size_t height,
    cudaStream_t stream) const override;

  std::optional<StreamError> copyToHost2D(void* dst,
    size_t dstPitch,
    size_t srcX,
    size_t srcY,
    size_t widthBytes,
    size_t height,
    cudaStream_t stream,
    bool synchronize = true) const override;

private:
  CudaBufferDevice2D(std::unique_ptr<CudaBufferDevice> buffer, size_t widthBytes, size_t height);

  CudaBufferDevice2D(const CudaBufferDevice2D&) = delete;
  CudaBufferDevice2D& operator=(const CudaBufferDevice2D&) = delete;

  CudaBufferDevice2D(CudaBufferDevice2D&&) = delete;
  CudaBufferDevice2D& operator=(CudaBufferDevice2D&&) = delete;

  std::unique_ptr<CudaBufferDevice> buffer_;
  size_t widthBytes_;
  size_t height_;
};
