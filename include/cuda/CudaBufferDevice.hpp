#pragma once

#include "CudaBuffer.hpp"

#include <memory>

class CudaBufferDevice : public CudaBuffer {
public:
  static tl::expected<std::unique_ptr<CudaBufferDevice>, StreamError> create(
    size_t byteSize, cudaStream_t stream);

  ~CudaBufferDevice() override;

  size_t size() const override;

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

private:
  CudaBufferDevice(void* data, size_t byteSize, cudaStream_t stream);

  CudaBufferDevice(const CudaBufferDevice&) = delete;
  CudaBufferDevice& operator=(const CudaBufferDevice&) = delete;

  CudaBufferDevice(CudaBufferDevice&&) = delete;
  CudaBufferDevice& operator=(CudaBufferDevice&&) = delete;

  size_t size_;
  void* data_;
  cudaStream_t stream_;
};
