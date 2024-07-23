#include "nvmm/NvmmBuffer.hpp"

#include "cuda/CudaBuffer.hpp"
#include "cuda/types.hpp"
#include "nvmm/NvmmMemMap.hpp"

#include <nvbuf_utils.h>

#include <cstring>
#include <iostream>
#include <limits>

tl::expected<std::unique_ptr<NvmmBuffer>, StreamError> NvmmBuffer::create(size_t byteSize) {
  if (byteSize > std::numeric_limits<int32_t>::max()) {
    return tl::make_unexpected(StreamError{cudaErrorInvalidValue, "byteSize too large"});
  }
  if (byteSize == 0) { return std::unique_ptr<NvmmBuffer>(new NvmmBuffer(nullptr, -1, 0, 0)); }

  NvBufferCreateParams params{};
  params.width = 0;
  params.height = 0;
  params.payloadType = NvBufferPayload_MemHandle;
  params.memsize = int32_t(byteSize);
  params.layout = NvBufferLayout_Pitch;
  params.colorFormat = NvBufferColorFormat_GRAY8;
  params.nvbuf_tag = NvBufferTag_NONE;

  int dmabuf_fd = -1;
  const int res = NvBufferCreateEx(&dmabuf_fd, &params);
  if (res != 0 || dmabuf_fd < 0) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "NvBufferCreateEx failed"});
  }

  // Read back the actual buffer size
  NvBufferParams bufferParams{};
  const int res2 = NvBufferGetParams(dmabuf_fd, &bufferParams);
  if (res2 != 0) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "NvBufferGetParams failed"});
  }

  void* data = bufferParams.nv_buffer;
  const size_t nvBufferSize = size_t(bufferParams.nv_buffer_size);
  return std::unique_ptr<NvmmBuffer>(
    new NvmmBuffer(static_cast<std::byte*>(data), dmabuf_fd, byteSize, nvBufferSize));
}

NvmmBuffer::NvmmBuffer(std::byte* pVirtAddr, int fd, size_t byteSize, size_t nvBufferSize)
  : data_(pVirtAddr),
    size_(byteSize),
    nvBufferSize_(nvBufferSize),
    fd_(fd) {}

NvmmBuffer::~NvmmBuffer() {
  if (fd_ < 0) { return; }

  const int res = NvBufferDestroy(fd_);
  if (res != 0) {
    // TODO: Logging callback for NvmmBuffer?
    std::cerr << "NvBufferDestroy failed: " << res << "\n";
  }
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
  if (srcOffset == 0 && dstOffset == 0 && count == src.size() && src.size() == size()) {
    // Fast path: the entire buffer is being copied. Perform a DMA buffer-to-buffer copy
    NvBufferTransformParams params{};
    params.session = session;
    const int res = NvBufferTransform(src.fd(), fd_, &params);
    if (res != 0) {
      return StreamError{cudaErrorMapBufferObjectFailed, "NvBufferTransform failed"};
    }
    return {};
  }

  // Slow path: NvBufferTransform doesn't support partial copies of 1D buffers. Memory-map both
  // buffers and copy the data manually

  // Sanity check the validity of the copy operation
  if (srcOffset + count > src.size() || dstOffset + count > size()) {
    return StreamError{cudaErrorInvalidValue, "Out-of-bounds copy operation"};
  }

  // Get a memory-mapped read pointer to the source buffer
  auto maybeSrcMap = NvmmMemMap::create(src.fd(), NvmmBufferMemAccess::Read);
  if (!maybeSrcMap) {
    return StreamError{cudaErrorMapBufferObjectFailed, "NvmmMemMap::create failed"};
  }
  NvmmMemMap srcMap = std::move(*maybeSrcMap.value());

  // Get a memory-mapped write pointer to the destination buffer
  auto maybeDstMap = NvmmMemMap::create(fd_, NvmmBufferMemAccess::Write);
  if (!maybeDstMap) {
    return StreamError{cudaErrorMapBufferObjectFailed, "NvmmMemMap::create failed"};
  }
  NvmmMemMap dstMap = std::move(*maybeDstMap.value());

  std::memcpy(dstMap.data() + dstOffset, srcMap.data() + srcOffset, count);
  return dstMap.syncForDevice();
}

std::optional<StreamError> NvmmBuffer::copyFromCuda(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  // Get a memory-mapped write pointer to the buffer
  auto maybeMap = NvmmMemMap::create(fd_, NvmmBufferMemAccess::Write);
  if (!maybeMap) {
    return StreamError{cudaErrorMapBufferObjectFailed, "NvmmMemMap::create failed"};
  }
  NvmmMemMap map = std::move(*maybeMap.value());

  // Copy the data from the CudaBuffer to the buffer as a device-to-host transfer
  const auto err = src.copyToHost(map.data() + dstOffset, srcOffset, count, stream);
  if (err) { return StreamError{err->errorCode, err->errorMessage}; }

  return map.syncForDevice();
}

std::optional<StreamError> NvmmBuffer::copyFromHost(
  const void* src, size_t dstOffset, size_t count) {
  // Get a memory-mapped write pointer to the buffer
  auto maybeMap = NvmmMemMap::create(fd_, NvmmBufferMemAccess::Write);
  if (!maybeMap) {
    return StreamError{cudaErrorMapBufferObjectFailed, "NvmmMemMap::create failed"};
  }
  NvmmMemMap map = std::move(*maybeMap.value());

  std::memcpy(map.data() + dstOffset, src, count);
  return map.syncForDevice();
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
  // Get a memory-mapped read pointer to the buffer
  auto maybeMap = NvmmMemMap::create(fd_, NvmmBufferMemAccess::Read);
  if (!maybeMap) {
    return StreamError{cudaErrorMapBufferObjectFailed, "NvmmMemMap::create failed"};
  }
  NvmmMemMap map = std::move(*maybeMap.value());

  if (synchronize) {
    // Synchronize to host memory
    const auto err = map.syncForCpu();
    if (err) { return StreamError{err->errorCode, err->errorMessage}; }
  }

  // Copy the data from the buffer to the CudaBuffer as a host-to-device transfer
  const auto err = dst.copyFromHost(map.data() + srcOffset, dstOffset, count, stream);
  if (err) { return StreamError{err->errorCode, err->errorMessage}; }

  return {};
}

std::optional<StreamError> NvmmBuffer::copyToHost(
  void* dst, size_t srcOffset, size_t count, bool synchronize) const {
  // Get a memory-mapped read pointer to the buffer
  auto maybeMap = NvmmMemMap::create(fd_, NvmmBufferMemAccess::Read);
  if (!maybeMap) {
    return StreamError{cudaErrorMapBufferObjectFailed, "NvmmMemMap::create failed"};
  }
  NvmmMemMap map = std::move(*maybeMap.value());

  if (synchronize) {
    // Synchronize to host memory
    const auto err = map.syncForCpu();
    if (err) { return StreamError{err->errorCode, err->errorMessage}; }
  }

  // Copy the data from the buffer to the host memory
  std::memcpy(dst, map.data() + srcOffset, count);
  return {};
}

std::optional<StreamError> NvmmBuffer::memset(std::byte value, size_t count) {
  // Get a memory-mapped write pointer to the buffer
  auto maybeMap = NvmmMemMap::create(fd_, NvmmBufferMemAccess::Write);
  if (!maybeMap) {
    return StreamError{cudaErrorMapBufferObjectFailed, "NvmmMemMap::create failed"};
  }
  NvmmMemMap map = std::move(*maybeMap.value());

  std::memset(map.data(), int(value), count);
  return map.syncForDevice();
}
