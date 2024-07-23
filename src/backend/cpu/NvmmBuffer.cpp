#include "nvmm/NvmmBuffer.hpp"

#include "cuda/types.hpp"

#include <cstring>
#include <limits>

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)

tl::expected<std::unique_ptr<NvmmBuffer>, NvmmError> NvmmBuffer::create(size_t byteSize) {
  if (byteSize > std::numeric_limits<int32_t>::max()) {
    return tl::make_unexpected(NvmmError{cudaErrorInvalidValue, "byteSize too large"});
  }
  if (byteSize == 0) { return std::unique_ptr<NvmmBuffer>(new NvmmBuffer(nullptr, 0, 0, 0)); }

  void* data = ::malloc(byteSize);
  if (data == nullptr) {
    return tl::make_unexpected(NvmmError{cudaErrorMemoryAllocation, "malloc failed"});
  }

  return std::unique_ptr<NvmmBuffer>(new NvmmBuffer(data, -1, byteSize, byteSize));
}

NvmmBuffer::NvmmBuffer(void* pVirtAddr, int fd, size_t byteSize, size_t nvBufferSize)
  : data_(pVirtAddr),
    size_(byteSize),
    nvBufferSize_(nvBufferSize),
    fd_(fd) {}

NvmmBuffer::~NvmmBuffer() {
  if (!data_) { return; }

  ::free(data_);
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

void* NvmmBuffer::data() {
  return data_;
}

const void* NvmmBuffer::data() const {
  return data_;
}

std::optional<NvmmError> NvmmBuffer::copyFrom(const NvmmBuffer& src,
  size_t srcOffset,
  size_t dstOffset,
  size_t count,
  NvBufferSession session) {
  (void)session;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.data()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<NvmmError> NvmmBuffer::copyFromCuda(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<NvmmError> NvmmBuffer::copyFromHost(const void* src, size_t dstOffset, size_t count) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  ::memcpy(dstPtr, src, count);
  return {};
}

std::optional<NvmmError> NvmmBuffer::copyTo(NvmmBuffer& dst,
  size_t srcOffset,
  size_t dstOffset,
  size_t count,
  NvBufferSession session) const {
  return dst.copyFrom(*this, srcOffset, dstOffset, count, session);
}

std::optional<NvmmError> NvmmBuffer::copyToCuda(CudaBuffer& dst,
  size_t srcOffset,
  size_t dstOffset,
  size_t count,
  cudaStream_t stream,
  bool synchronize) const {
  (void)stream;
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<NvmmError> NvmmBuffer::copyToHost(
  void* dst, size_t srcOffset, size_t count, bool synchronize) const {
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<NvmmError> NvmmBuffer::memset(std::byte value, size_t count) {
  std::byte* dstPtr = static_cast<std::byte*>(data_);
  std::memset(dstPtr, int(value), count);
  return {};
}

// NOLINTEND(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
