#include "cuda/CudaBufferUnified.hpp"

#include <cstring>

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)

tl::expected<std::unique_ptr<CudaBufferUnified>, StreamError> CudaBufferUnified::create(
  size_t byteSize, CudaMemAttachFlag flag) {
  (void)flag;
  if (byteSize == 0) {
    return std::unique_ptr<CudaBufferUnified>(new CudaBufferUnified(nullptr, 0, true));
  }
  void* data = ::malloc(byteSize);
  if (data == nullptr) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "malloc failed"});
  }
  return std::unique_ptr<CudaBufferUnified>(new CudaBufferUnified(data, byteSize, true));
}

tl::expected<std::unique_ptr<CudaBufferUnified>, StreamError> CudaBufferUnified::createFromHostData(
  const void* data, size_t byteSize, cudaStream_t stream, CudaMemAttachFlag flag) {
  auto res = CudaBufferUnified::create(byteSize, flag);
  if (!res) { return tl::make_unexpected(res.error()); }
  auto bufferPtr = std::move(*res);

  // Copy host data to this buffer (memcpy)
  std::optional<StreamError> err = bufferPtr->copyFromHost(data, 0, byteSize, stream);
  if (err) { return tl::make_unexpected(err.value()); }
  return std::move(bufferPtr);
}

CudaBufferUnified::CudaBufferUnified(void* data, size_t byteSize, bool isDevice)
  : size_(byteSize),
    data_(static_cast<std::byte*>(data)),
    isDevice_(isDevice) {}

CudaBufferUnified::~CudaBufferUnified() {
  ::free(data_);
}

size_t CudaBufferUnified::size() const {
  return size_;
}

void* CudaBufferUnified::cudaData() {
  return data_;
}

const void* CudaBufferUnified::cudaData() const {
  return data_;
}

const std::byte* CudaBufferUnified::hostData() const {
  return data_;
}

std::byte* CudaBufferUnified::hostData() {
  return data_;
}

bool CudaBufferUnified::isDevice() const {
  return isDevice_;
}

std::optional<StreamError> CudaBufferUnified::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferUnified::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  ::memcpy(dstPtr, src, count);
  return {};
}

std::optional<StreamError> CudaBufferUnified::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferUnified::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  (void)stream;
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferUnified::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  (void)stream;
  ::memset(data_, int(value), count);
  return {};
}

std::optional<StreamError> CudaBufferUnified::prefetch(
  CudaMemAttachFlag flag, cudaStream_t stream) {
  (void)flag;
  (void)stream;
  return {};
}

// NOLINTEND(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
