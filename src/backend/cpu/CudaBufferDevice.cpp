#include "cuda/CudaBufferDevice.hpp"

#include <cstring>

tl::expected<std::unique_ptr<CudaBufferDevice>, StreamError> CudaBufferDevice::create(
  size_t byteSize, cudaStream_t stream) {
  (void)stream;
  void* data = malloc(byteSize);
  if (data == nullptr) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "malloc failed"});
  }
  return std::unique_ptr<CudaBufferDevice>(new CudaBufferDevice(data, byteSize, stream));
}

CudaBufferDevice::CudaBufferDevice(void* data, size_t byteSize, cudaStream_t stream)
  : size_(byteSize),
    data_(data),
    stream_(stream) {}

CudaBufferDevice::~CudaBufferDevice() {
  (void)stream_;
  free(data_);
}

size_t CudaBufferDevice::size() const {
  return size_;
}

void* CudaBufferDevice::cudaData() {
  return data_;
}

const void* CudaBufferDevice::cudaData() const {
  return data_;
}

std::optional<StreamError> CudaBufferDevice::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferDevice::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  ::memcpy(dstPtr, src, count);
  return {};
}

std::optional<StreamError> CudaBufferDevice::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferDevice::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  (void)stream;
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferDevice::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  (void)stream;
  std::byte* dstPtr = static_cast<std::byte*>(data_);
  ::memset(dstPtr, int(value), count);
  return {};
}
