#pragma once

#include "NvmmBuffer.hpp"

class NvmmImageBuffer : public NvmmBuffer {
public:
  static tl::expected<std::unique_ptr<NvmmImageBuffer>, NvmmError> create(
    size_t width, size_t height, NvmmColorFormat format, NvmmBufferLayout layout);

  ~NvmmImageBuffer();

  NvmmImageBuffer(const NvmmImageBuffer&) = delete;
  NvmmImageBuffer& operator=(const NvmmImageBuffer&) = delete;

  NvmmImageBuffer(NvmmImageBuffer&&) = default;
  NvmmImageBuffer& operator=(NvmmImageBuffer&&) = default;

  size_t width() const;
  size_t height() const;
  NvmmColorFormat format() const;
  NvmmBufferLayout layout() const;
  size_t pitch() const;

private:
  NvmmImageBuffer(
    int fd, size_t width, size_t height, NvmmColorFormat format, NvmmBufferLayout layout);

  size_t width_;
  size_t height_;
  NvmmColorFormat format_;
  NvmmBufferLayout layout_;
};
