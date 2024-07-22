#pragma once

#include "NvmmBuffer.hpp"
#include "types.hpp"

#include <array>

class NvmmImageBuffer : public NvmmBuffer {
public:
  static constexpr size_t MAX_NUM_PLANES = 4;
  using PlaneArray = std::array<size_t, MAX_NUM_PLANES>;
  using PlaneLayoutArray = std::array<NvmmBufferLayout, MAX_NUM_PLANES>;

  static tl::expected<std::unique_ptr<NvmmImageBuffer>, NvmmError> create(
    size_t width, size_t height, NvmmColorFormat format, NvmmBufferLayout layout);

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

  // Pixel width of each plane
  const PlaneArray& widths() const;
  // Pixel height of each plane
  const PlaneArray& heights() const;
  // Pitch of each plane
  const PlaneArray& pitches() const;
  // Memory offset values for each plane
  const PlaneArray& offsets() const;
  // Byte size of each plane
  const PlaneArray& sizes() const;
  // Layout type of each plane
  const PlaneLayoutArray& layouts() const;

  std::optional<NvmmError> copyFrom2D(const NvmmImageBuffer& src,
    const Rect& srcRect,
    const Rect& dstRect,
    NvBufferSession session,
    NvmmTransformFilter filter = NvmmTransformFilter::Smart);

  std::optional<NvmmError> copyFromHost2D(const std::vector<uint8_t*>& srcPlanes);

  std::optional<NvmmError> copyTo2D(NvmmImageBuffer& dst,
    const Rect& srcRect,
    const Rect& dstRect,
    NvBufferSession session,
    NvmmTransformFilter filter = NvmmTransformFilter::Smart) const;

  std::optional<NvmmError> copyToHost2D(const std::vector<uint8_t*>& dstPlanes) const;

private:
  NvmmImageBuffer(void* pVirtAddr,
    int fd,
    size_t byteSize,
    size_t nvBufferSize,
    NvmmColorFormat format,
    size_t numPlanes,
    const PlaneArray& widths,
    const PlaneArray& heights,
    const PlaneArray& pitches,
    const PlaneArray& offsets,
    const PlaneArray& sizes,
    const PlaneLayoutArray& layouts);

  NvmmColorFormat format_;
  size_t numPlanes_;
  PlaneArray widths_;
  PlaneArray heights_;
  PlaneArray pitches_;
  PlaneArray offsets_;
  PlaneArray sizes_;
  PlaneLayoutArray layouts_;
};
