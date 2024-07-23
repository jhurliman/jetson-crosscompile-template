#include "nvmm/NvmmImageBuffer.hpp"

#include "cuda/types.hpp"

#include <cstring>
#include <limits>

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory,
// cppcoreguidelines-pro-bounds-constant-array-index)

NvmmImageBuffer::NvmmImageBuffer(std::byte* pVirtAddr,
  int fd,
  size_t byteSize,
  size_t nvBufferSize,
  const NvmmImageBufferInfo& info)
  : NvmmBuffer(pVirtAddr, fd, byteSize, nvBufferSize),
    info_(info) {}

tl::expected<std::unique_ptr<NvmmImageBuffer>, StreamError> NvmmImageBuffer::create(
  size_t width, size_t height, NvmmColorFormat format, NvmmBufferLayout layout) {
  (void)layout;
  if (width > std::numeric_limits<int32_t>::max() || height > std::numeric_limits<int32_t>::max()) {
    return tl::make_unexpected(StreamError{cudaErrorInvalidValue, "width or height too large"});
  }
  if (width == 0 || height == 0) {
    return std::unique_ptr<NvmmImageBuffer>(new NvmmImageBuffer(nullptr, -1, 0, 0, {}));
  }

  const NvmmImageBufferInfo info = GetImageBufferInfo(width, height, format, layout);

  // Create the underlying buffer
  const size_t byteSize = info.size();
  void* data = std::malloc(byteSize);
  if (data == nullptr) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "malloc failed"});
  }

  return std::unique_ptr<NvmmImageBuffer>(
    new NvmmImageBuffer(static_cast<std::byte*>(data), 0, byteSize, NV_BUFFER_SIZE, info));
}

NvmmColorFormat NvmmImageBuffer::format() const {
  return info_.format;
}

size_t NvmmImageBuffer::numPlanes() const {
  return info_.numPlanes;
}

size_t NvmmImageBuffer::width() const {
  return info_.widths[0];
}

size_t NvmmImageBuffer::height() const {
  return info_.heights[0];
}

const NvmmImageBufferInfo& NvmmImageBuffer::info() const {
  return info_;
}

std::optional<StreamError> NvmmImageBuffer::copyFrom2D(const NvmmImageBuffer& src,
  const Rect& srcRect,
  const Rect& dstRect,
  NvBufferSession session,
  NvmmTransformFilter filter) {
  // ::memcpy-based emulation of DMA buffer to buffer 2D copy
  (void)session;
  (void)filter;

  const auto& srcInfo = src.info();
  const auto& dstInfo = info();

  if (dstInfo.format != srcInfo.format) {
    return StreamError{cudaErrorInvalidValue, "Incompatible format"};
  }
  if (srcRect.left + srcRect.width > src.width() || srcRect.top + srcRect.height > src.height() ||
    dstRect.left + dstRect.width > width() || dstRect.top + dstRect.height > height()) {
    return StreamError{cudaErrorInvalidValue, "Invalid srcRect or dstRect"};
  }

  for (size_t i = 0; i < srcInfo.numPlanes; ++i) {
    // Calculate adjusted rectangles for each plane
    const size_t scaleX = srcInfo.widths[0] / srcInfo.widths[i];
    const size_t scaleY = srcInfo.heights[0] / srcInfo.heights[i];

    Rect adjustedSrcRect = {uint32_t(srcRect.left / scaleX),
      uint32_t(srcRect.top / scaleY),
      uint32_t(srcRect.width / scaleX),
      uint32_t(srcRect.height / scaleY)};

    Rect adjustedDstRect = {uint32_t(dstRect.left / scaleX),
      uint32_t(dstRect.top / scaleY),
      uint32_t(dstRect.width / scaleX),
      uint32_t(dstRect.height / scaleY)};

    // Copy each row
    for (uint32_t row = 0; row < adjustedSrcRect.height; ++row) {
      const std::byte* srcPlaneData = src.data() + srcInfo.offsets[i];
      std::byte* dstPlaneData = data() + dstInfo.offsets[i];

      const std::byte* srcRow =
        srcPlaneData + (adjustedSrcRect.top + row) * srcInfo.pitches[i] + adjustedSrcRect.left;
      std::byte* dstRow =
        dstPlaneData + (adjustedDstRect.top + row) * dstInfo.pitches[i] + adjustedDstRect.left;

      std::memcpy(dstRow, srcRow, adjustedSrcRect.width);
    }
  }

  return {};
}

std::optional<StreamError> NvmmImageBuffer::copyFromHost2D(
  const std::vector<std::byte*>& srcPlanes) {
  const unsigned int planes = static_cast<unsigned int>(numPlanes());
  if (srcPlanes.size() != planes) {
    return StreamError{cudaErrorInvalidValue, "srcPlanes.size() != numPlanes()"};
  }

  const auto& info = this->info();
  for (unsigned int i = 0; i < planes; ++i) {
    const unsigned int width = static_cast<unsigned int>(info.widths[i]);
    const unsigned int height = static_cast<unsigned int>(info.heights[i]);
    const unsigned int pitch = static_cast<unsigned int>(info.pitches[i]);

    const std::byte* srcPlaneData = srcPlanes[i];
    std::byte* dstPlaneData = data() + info.offsets[i];

    for (unsigned int row = 0; row < height; ++row) {
      const std::byte* srcRow = srcPlaneData + row * pitch;
      std::byte* dstRow = dstPlaneData + row * pitch;

      std::memcpy(dstRow, srcRow, width);
    }
  }

  return {};
}

std::optional<StreamError> NvmmImageBuffer::copyTo2D(NvmmImageBuffer& dst,
  const Rect& srcRect,
  const Rect& dstRect,
  NvBufferSession session,
  NvmmTransformFilter filter) const {
  return dst.copyFrom2D(*this, srcRect, dstRect, session, filter);
}

std::optional<StreamError> NvmmImageBuffer::copyToHost2D(
  const std::vector<std::byte*>& dstPlanes) const {
  const unsigned int planes = static_cast<unsigned int>(numPlanes());
  if (dstPlanes.size() != planes) {
    return StreamError{cudaErrorInvalidValue, "dstPlanes.size() != numPlanes()"};
  }

  const auto& info = this->info();
  for (unsigned int i = 0; i < planes; ++i) {
    const unsigned int width = static_cast<unsigned int>(info.widths[i]);
    const unsigned int height = static_cast<unsigned int>(info.heights[i]);
    const unsigned int pitch = static_cast<unsigned int>(info.pitches[i]);

    const std::byte* srcPlaneData = data() + info.offsets[i];
    std::byte* dstPlaneData = dstPlanes[i];

    for (unsigned int row = 0; row < height; ++row) {
      const std::byte* srcRow = srcPlaneData + row * pitch;
      std::byte* dstRow = dstPlaneData + row * pitch;

      std::memcpy(dstRow, srcRow, width);
    }
  }

  return {};
}

// NOLINTEND(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory,
// cppcoreguidelines-pro-bounds-constant-array-index)
