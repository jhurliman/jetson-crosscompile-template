#include "nvmm/NvmmBuffer.hpp"

#include "cuda/CudaBuffer.hpp"
#include "cuda_expected.hpp"
#include "nvmm/types.hpp"

#include <cstring>
#include <limits>

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)

tl::expected<std::unique_ptr<NvmmBuffer>, StreamError> NvmmBuffer::create(size_t byteSize) {
  if (byteSize > std::numeric_limits<int32_t>::max()) {
    return tl::make_unexpected(StreamError{cudaErrorInvalidValue, "byteSize too large"});
  }
  if (byteSize == 0) { return std::unique_ptr<NvmmBuffer>(new NvmmBuffer(nullptr, -1, 0, 0)); }

  void* data = std::malloc(byteSize);
  if (data == nullptr) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "malloc failed"});
  }

  return std::unique_ptr<NvmmBuffer>(
    new NvmmBuffer(static_cast<std::byte*>(data), 0, byteSize, NV_BUFFER_SIZE));
}

NvmmBuffer::NvmmBuffer(std::byte* pVirtAddr, int fd, size_t byteSize, size_t nvBufferSize)
  : data_(pVirtAddr),
    size_(byteSize),
    nvBufferSize_(nvBufferSize),
    fd_(fd) {}

NvmmBuffer::~NvmmBuffer() {
  if (!data_) { return; }

  std::free(data_);
}

size_t NvmmBuffer::size() const {
  return size_;
}

size_t NvmmBuffer::nvBufferSize() const {
  return nvBufferSize_;
}

int NvmmBuffer::fd() {
  return fd_;
}

int NvmmBuffer::fd() const {
  return fd_;
}

std::byte* NvmmBuffer::data() {
  return data_;
}

const std::byte* NvmmBuffer::data() const {
  return data_;
}

std::optional<StreamError> NvmmBuffer::copyFrom(const NvmmBuffer& src,
  size_t srcOffset,
  size_t dstOffset,
  size_t count,
  NvBufferSession session) {
  (void)session;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.data()) + srcOffset;
  std::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> NvmmBuffer::copyFromCuda(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  // void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  // const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  // const auto copyType = src.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  // CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream));

  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  const auto copyType = src.isDevice() ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream));

  return {};
}

std::optional<StreamError> NvmmBuffer::copyFromHost(
  const void* src, size_t dstOffset, size_t count) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  std::memcpy(dstPtr, src, count);
  return {};
}

std::optional<StreamError> NvmmBuffer::copyTo(NvmmBuffer& dst,
  size_t srcOffset,
  size_t dstOffset,
  size_t count,
  NvBufferSession session) const {
  return dst.copyFrom(*this, srcOffset, dstOffset, count, session);
}

std::optional<StreamError> NvmmBuffer::copyToCuda(CudaBuffer& dst,
  size_t srcOffset,
  size_t dstOffset,
  size_t count,
  cudaStream_t stream,
  bool synchronize) const {
  (void)stream;
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  const auto copyType = dst.isDevice() ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream));

  return {};
}

std::optional<StreamError> NvmmBuffer::copyToHost(
  void* dst, size_t srcOffset, size_t count, bool synchronize) const {
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  std::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> NvmmBuffer::memset(std::byte value, size_t count) {
  std::byte* dstPtr = static_cast<std::byte*>(data_);
  std::memset(dstPtr, int(value), count);
  return {};
}

// NOLINTEND(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
