#pragma once

#include "errors.hpp"
#include "types.hpp"

#include <tl/expected.hpp>

#include <cstddef>
#include <memory>
#include <optional>

class CudaBuffer;
typedef struct CUstream_st* cudaStream_t;

/**
 * NvmmBuffer provides a RAII wrapper for NVIDIA Jetson Multimedia API NvBuffer allocations of
 * linear memory buffers. For 2D image buffers, use `NvmmImageBuffer` instead.
 */
class NvmmBuffer {
public:
  static tl::expected<std::unique_ptr<NvmmBuffer>, StreamError> create(size_t byteSize);

  NvmmBuffer() = default;

  virtual ~NvmmBuffer();

  NvmmBuffer(const NvmmBuffer&) = delete;
  NvmmBuffer& operator=(const NvmmBuffer&) = delete;

  NvmmBuffer(NvmmBuffer&&) = default;
  NvmmBuffer& operator=(NvmmBuffer&&) = default;

  size_t size() const;
  size_t nvBufferSize() const;

  int fd();
  int fd() const;

  std::byte* data();
  const std::byte* data() const;

  std::optional<StreamError> copyFrom(const NvmmBuffer& src,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    NvBufferSession session);

  std::optional<StreamError> copyFromCuda(
    const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream);

  std::optional<StreamError> copyFromHost(const void* src, size_t dstOffset, size_t count);

  std::optional<StreamError> copyTo(NvmmBuffer& dst,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    NvBufferSession session) const;

  std::optional<StreamError> copyToCuda(CudaBuffer& dst,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    cudaStream_t stream,
    bool synchronize = true) const;

  std::optional<StreamError> copyToHost(
    void* dst, size_t srcOffset, size_t count, bool synchronize = true) const;

  std::optional<StreamError> memset(std::byte value, size_t count);

protected:
  NvmmBuffer(std::byte* pVirtAddr, int fd, size_t byteSize, size_t nvBufferSize);

private:
  std::byte* data_{};
  size_t size_{};
  size_t nvBufferSize_{};
  int fd_ = -1;
};
