#pragma once

#include "types.hpp"

#include <tl/expected.hpp>

#include <cstddef>
#include <memory>
#include <optional>

class CudaBuffer;
typedef struct CUstream_st* cudaStream_t;

/**
 * NvmmBuffer is an abstract base class that provides a RAII wrapper for NVIDIA Jetson Multimedia
 * API NvBuffer allocations.
 */
class NvmmBuffer {
public:
  static tl::expected<std::unique_ptr<NvmmBuffer>, NvmmError> create(size_t byteSize);

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

  void* data();
  const void* data() const;

  std::optional<NvmmError> copyFrom(const NvmmBuffer& src,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    NvBufferSession session);

  std::optional<NvmmError> copyFromCuda(
    const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream);

  std::optional<NvmmError> copyFromHost(const void* src, size_t dstOffset, size_t count);

  std::optional<NvmmError> copyTo(NvmmBuffer& dst,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    NvBufferSession session) const;

  std::optional<NvmmError> copyToCuda(CudaBuffer& dst,
    size_t srcOffset,
    size_t dstOffset,
    size_t count,
    cudaStream_t stream,
    bool synchronize = true) const;

  std::optional<NvmmError> copyToHost(
    void* dst, size_t srcOffset, size_t count, bool synchronize = true) const;

  std::optional<NvmmError> memset(std::byte value, size_t count);

protected:
  NvmmBuffer(void* pVirtAddr, int fd, size_t byteSize, size_t nvBufferSize);

private:
  void* data_{};
  size_t size_{};
  size_t nvBufferSize_{};
  int fd_ = -1;
};
