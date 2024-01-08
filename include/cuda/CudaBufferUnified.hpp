#pragma once

#include "CudaBuffer.hpp"

#include <tl/expected.hpp>

#include <memory>

class CudaBufferUnified : public CudaBuffer {
public:
  static tl::expected<std::unique_ptr<CudaBufferUnified>, StreamError> create(
    size_t byteSize, CudaMemAttachFlag flag = CudaMemAttachFlag::Global);

  static tl::expected<std::unique_ptr<CudaBufferUnified>, StreamError> createFromHostData(
    const void* data,
    size_t byteSize,
    cudaStream_t stream,
    CudaMemAttachFlag flag = CudaMemAttachFlag::Global);

  ~CudaBufferUnified() override;

  size_t size() const override;

  void* cudaData() override;
  const void* cudaData() const override;

  bool isDevice() const override;

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

  std::optional<StreamError> prefetch(CudaMemAttachFlag flag, cudaStream_t stream);

private:
  CudaBufferUnified(void* data, size_t byteSize, bool isDevice);

  CudaBufferUnified(const CudaBufferUnified&) = delete;
  CudaBufferUnified& operator=(const CudaBufferUnified&) = delete;

  CudaBufferUnified(CudaBufferUnified&&) = delete;
  CudaBufferUnified& operator=(CudaBufferUnified&&) = delete;

  size_t size_;
  std::byte* data_;
  bool isDevice_;
};
