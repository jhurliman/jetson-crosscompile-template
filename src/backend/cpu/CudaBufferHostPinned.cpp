#include "cuda/CudaBufferHostPinned.hpp"

#include <cstring>

tl::expected<std::unique_ptr<CudaBufferHostPinned>, StreamError> CudaBufferHostPinned::create(
  size_t byteSize, uint flags) {
  (void)flags;
  std::byte* data = static_cast<std::byte*>(malloc(byteSize));
  if (data == nullptr) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "malloc failed"});
  }
  return std::unique_ptr<CudaBufferHostPinned>(new CudaBufferHostPinned(data, byteSize));
}

CudaBufferHostPinned::CudaBufferHostPinned(std::byte* data, size_t byteSize)
  : size_(byteSize),
    data_(data) {}

CudaBufferHostPinned::~CudaBufferHostPinned() {
  free(data_);
}

size_t CudaBufferHostPinned::size() const {
  return size_;
}

void* CudaBufferHostPinned::cudaData() {
  return data_;
}

const void* CudaBufferHostPinned::cudaData() const {
  return data_;
}

const std::byte* CudaBufferHostPinned::hostData() const {
  return data_;
}

std::byte* CudaBufferHostPinned::hostData() {
  return data_;
}

std::optional<StreamError> CudaBufferHostPinned::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferHostPinned::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  ::memcpy(dstPtr, src, count);
  return {};
}

std::optional<StreamError> CudaBufferHostPinned::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferHostPinned::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  (void)stream;
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferHostPinned::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_);
  ::memset(dstPtr, int(value), count);
  return {};
}
