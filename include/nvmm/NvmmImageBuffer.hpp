#pragma once

#include "NvmmBuffer.hpp"
#include "NvmmImageBufferInfo.hpp"
#include "types.hpp"

class NvmmImageBuffer : public NvmmBuffer {
public:
  static tl::expected<std::unique_ptr<NvmmImageBuffer>, StreamError> create(size_t width,
    size_t height,
    NvmmColorFormat format = NvmmColorFormat::YUV420,
    NvmmBufferLayout layout = NvmmBufferLayout::BlockLinear);

  ~NvmmImageBuffer() override = default;

  NvmmImageBuffer(const NvmmImageBuffer&) = delete;
  NvmmImageBuffer& operator=(const NvmmImageBuffer&) = delete;

  NvmmImageBuffer(NvmmImageBuffer&&) = default;
  NvmmImageBuffer& operator=(NvmmImageBuffer&&) = default;

  // Color format of this image buffer
  NvmmColorFormat format() const;
  // Number of planes of hardware buffer
  size_t numPlanes() const;

  // Pixel width of the first plane
  size_t width() const;
  // Pixel height of the first plane
  size_t height() const;

  const NvmmImageBufferInfo& info() const;

  std::optional<StreamError> copyFrom2D(const NvmmImageBuffer& src,
    const Rect& srcRect,
    const Rect& dstRect,
    NvBufferSession session,
    NvmmTransformFilter filter = NvmmTransformFilter::Smart);

  std::optional<StreamError> copyFromHost2D(const std::vector<std::byte*>& srcPlanes);

  std::optional<StreamError> copyTo2D(NvmmImageBuffer& dst,
    const Rect& srcRect,
    const Rect& dstRect,
    NvBufferSession session,
    NvmmTransformFilter filter = NvmmTransformFilter::Smart) const;

  std::optional<StreamError> copyToHost2D(const std::vector<std::byte*>& dstPlanes) const;

private:
  NvmmImageBuffer(std::byte* pVirtAddr,
    int fd,
    size_t byteSize,
    size_t nvBufferSize,
    const NvmmImageBufferInfo& info);

  NvmmImageBufferInfo info_;
};
