#pragma once

#include "CudaBuffer2D.hpp"

class CudaBufferPitched : public CudaBuffer2D {
public:
  static tl::expected<std::unique_ptr<CudaBufferPitched>, StreamError> create(
    size_t width, size_t height);

  ~CudaBufferPitched() override;

  size_t size() const override;
  size_t width() const override;
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
    size_t width,
    size_t height,
    cudaStream_t stream) override;

  std::optional<StreamError> copyFromHost2D(const void* src,
    size_t dstX,
    size_t dstY,
    size_t width,
    size_t height,
    cudaStream_t stream) override;

  std::optional<StreamError> copyTo2D(CudaBuffer2D& dst,
    size_t srcX,
    size_t srcY,
    size_t dstX,
    size_t dstY,
    size_t width,
    size_t height,
    cudaStream_t stream) const override;

  std::optional<StreamError> copyToHost2D(void* dst,
    size_t srcX,
    size_t srcY,
    size_t width,
    size_t height,
    cudaStream_t stream,
    bool synchronize = true) const override;

private:
  CudaBufferPitched(void* data, size_t width, size_t height, size_t pitch);

  CudaBufferPitched(const CudaBufferPitched&) = delete;
  CudaBufferPitched& operator=(const CudaBufferPitched&) = delete;

  CudaBufferPitched(CudaBufferPitched&&) = delete;
  CudaBufferPitched& operator=(CudaBufferPitched&&) = delete;

  size_t width_ = 0;
  size_t height_ = 0;
  size_t pitch_ = 0;
  void* data_ = nullptr;
};
