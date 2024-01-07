#pragma once

#include "CudaBuffer.hpp"

#include <memory>

class CudaBufferHostPinned : public CudaBuffer {
public:
  static tl::expected<std::unique_ptr<CudaBufferHostPinned>, StreamError> create(
    size_t byteSize, CudaHostPinnedFlags flags = CudaHostPinnedFlags::Mapped);

  ~CudaBufferHostPinned() override;

  size_t size() const override;

  void* cudaData() override;
  const void* cudaData() const override;

  bool isDevice() const override { return false; }

  const std::byte* hostData() const;
  std::byte* hostData();

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
  CudaBufferHostPinned(std::byte* data, size_t byteSize);

  CudaBufferHostPinned(const CudaBufferHostPinned&) = delete;
  CudaBufferHostPinned& operator=(const CudaBufferHostPinned&) = delete;

  CudaBufferHostPinned(CudaBufferHostPinned&&) = delete;
  CudaBufferHostPinned& operator=(CudaBufferHostPinned&&) = delete;

  size_t size_ = 0;
  std::byte* data_ = nullptr;
};
