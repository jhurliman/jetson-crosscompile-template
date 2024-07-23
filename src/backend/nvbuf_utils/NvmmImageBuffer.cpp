#include "nvmm/NvmmImageBuffer.hpp"

#include "cuda/types.hpp"
#include "formats.hpp"

#include <dlfcn.h>
#include <nvbuf_utils.h>

#include <limits>
#include <optional>

// Jetpack r32.7+ introduced additional color formats, changing the ordering of the color format
// enum. We detect if we're running on 32.7+ by checking if the NvBufferCreateCompressed() function
// is available
static bool HasExtendedColorFormats() {
  static std::optional<bool> hasExtendedColorFormats;
  if (hasExtendedColorFormats.has_value()) { return hasExtendedColorFormats.value(); }

  void* handle = dlopen("libnvbuf_utils.so", RTLD_LAZY);
  if (handle == nullptr) { return false; }

  void* sym = dlsym(handle, "NvBufferCreateCompressed");
  dlclose(handle);

  hasExtendedColorFormats = sym != nullptr;
  return hasExtendedColorFormats.value();
}

// Check which version of Jetpack is running (specifically, which version of libnvbuf_utils.so is
// available) and return the corresponding NvBufferColorFormat value. If Jetpack 32.6 or older is
// running, some color formats such as YUV422 are unavailable and std::nullopt is returned
static std::optional<NvBufferColorFormat> GetNvColorFormat(NvmmColorFormat format) {
  if (HasExtendedColorFormats()) { return NvBufferColorFormat(NvBufferColorFormat32_7(format)); }

  const auto result = NvBufferColorFormat32_6(format);
  if (!result) { return {}; }
  return NvBufferColorFormat(result.value());
}

constexpr NvBufferRect MakeNvBufferRect(const Rect& rect) {
  return NvBufferRect{rect.top, rect.left, rect.width, rect.height};
}

NvmmImageBuffer::NvmmImageBuffer(std::byte* pVirtAddr,
  int fd,
  size_t byteSize,
  size_t nvBufferSize,
  const NvmmImageBufferInfo& info)
  : NvmmBuffer(pVirtAddr, fd, byteSize, nvBufferSize),
    info_(info) {}

tl::expected<std::unique_ptr<NvmmImageBuffer>, StreamError> NvmmImageBuffer::create(
  size_t width, size_t height, NvmmColorFormat format, NvmmBufferLayout layout) {
  if (width > std::numeric_limits<int32_t>::max() || height > std::numeric_limits<int32_t>::max()) {
    return tl::make_unexpected(StreamError{cudaErrorInvalidValue, "width or height too large"});
  }
  if (width == 0 || height == 0) {
    return std::unique_ptr<NvmmImageBuffer>(new NvmmImageBuffer(nullptr, -1, 0, 0, {}));
  }

  const auto nvFormat = GetNvColorFormat(format);
  if (!nvFormat) {
    return tl::make_unexpected(StreamError{cudaErrorInvalidValue, "Color format unsupported"});
  }

  NvBufferCreateParams params{};
  params.width = int32_t(width);
  params.height = int32_t(height);
  params.payloadType = NvBufferPayload_SurfArray;
  params.memsize = 0;
  params.layout = NvBufferLayout(layout);
  params.colorFormat = nvFormat.value();
  params.nvbuf_tag = NvBufferTag_NONE;

  int dmabuf_fd = -1;
  const int res = NvBufferCreateEx(&dmabuf_fd, &params);
  if (res != 0 || dmabuf_fd < 0) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "NvBufferCreateEx failed"});
  }

  // Read back the actual buffer parameters
  NvBufferParams out{};
  const int res2 = NvBufferGetParams(dmabuf_fd, &out);
  if (res2 != 0) {
    NvBufferDestroy(dmabuf_fd);
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "NvBufferGetParams failed"});
  }

  void* data = out.nv_buffer;
  const size_t byteSize =
    size_t(out.psize[0]) + size_t(out.psize[1]) + size_t(out.psize[2]) + size_t(out.psize[3]);
  const size_t nvBufferSize = out.nv_buffer_size;
  const size_t numPlanes = out.num_planes;
  const PlaneArray widths = {out.width[0], out.width[1], out.width[2], out.width[3]};
  const PlaneArray heights = {out.height[0], out.height[1], out.height[2], out.height[3]};
  const PlaneArray pitches = {out.pitch[0], out.pitch[1], out.pitch[2], out.pitch[3]};
  const PlaneArray offsets = {out.offset[0], out.offset[1], out.offset[2], out.offset[3]};
  const PlaneArray sizes = {out.psize[0], out.psize[1], out.psize[2], out.psize[3]};
  const PlaneLayoutArray layouts = {NvmmBufferLayout(out.layout[0]),
    NvmmBufferLayout(out.layout[1]),
    NvmmBufferLayout(out.layout[2]),
    NvmmBufferLayout(out.layout[3])};

  const NvmmImageBufferInfo info{
    format, numPlanes, widths, heights, pitches, offsets, sizes, layouts};
  return std::unique_ptr<NvmmImageBuffer>(
    new NvmmImageBuffer(static_cast<std::byte*>(data), dmabuf_fd, byteSize, nvBufferSize, info));
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
  // DMA buffer to buffer copy via NvBufferTransform

  if (info().format != src.info().format) {
    return StreamError{cudaErrorInvalidValue, "Incompatible format"};
  }
  if (srcRect.left + srcRect.width > src.width() || srcRect.top + srcRect.height > src.height() ||
    dstRect.left + dstRect.width > width() || dstRect.top + dstRect.height > height()) {
    return StreamError{cudaErrorInvalidValue, "Invalid srcRect or dstRect"};
  }

  uint32_t flags = 0;
  if (srcRect.top != 0 || srcRect.left != 0 || srcRect.width != src.width() ||
    srcRect.height != src.height()) {
    flags |= NVBUFFER_TRANSFORM_CROP_SRC;
  }
  if (dstRect.top != 0 || dstRect.left != 0 || dstRect.width != width() ||
    dstRect.height != height()) {
    flags |= NVBUFFER_TRANSFORM_CROP_DST;
  }
  if (srcRect != dstRect) { flags |= NVBUFFER_TRANSFORM_FILTER; }

  NvBufferTransformParams params{};
  params.session = session;
  params.transform_flag = flags;
  params.transform_filter = NvBufferTransform_Filter(filter);
  params.src_rect = MakeNvBufferRect(srcRect);
  params.dst_rect = MakeNvBufferRect(dstRect);
  const int res = NvBufferTransform(src.fd(), fd(), &params);
  if (res != 0) { return StreamError{cudaErrorMapBufferObjectFailed, "NvBufferTransform failed"}; }

  return {};
}

std::optional<StreamError> NvmmImageBuffer::copyFromHost2D(
  const std::vector<std::byte*>& srcPlanes) {
  const unsigned int planes = static_cast<unsigned int>(numPlanes());
  if (srcPlanes.size() != planes) {
    return StreamError{cudaErrorInvalidValue, "srcPlanes.size() != numPlanes()"};
  }

  // Copy each plane from host to the hardware buffer
  const auto& info = this->info();
  for (unsigned int i = 0; i < planes; ++i) {
    const unsigned int width = static_cast<unsigned int>(info.widths.at(i));
    const unsigned int height = static_cast<unsigned int>(info.heights.at(i));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    unsigned char* srcPtr = reinterpret_cast<unsigned char*>(srcPlanes[i]);
    const int res = Raw2NvBuffer(srcPtr, i, width, height, fd());
    if (res != 0) {
      return StreamError{
        cudaErrorInvalidValue, "Raw2NvBuffer failed for plane " + std::to_string(i)};
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

  // Copy each plane from the hardware buffer to host
  const auto& info = this->info();
  for (unsigned int i = 0; i < planes; ++i) {
    const unsigned int width = static_cast<unsigned int>(info.widths.at(i));
    const unsigned int height = static_cast<unsigned int>(info.heights.at(i));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    unsigned char* dstPtr = reinterpret_cast<unsigned char*>(dstPlanes[i]);
    const int res = NvBuffer2Raw(fd(), i, width, height, dstPtr);
    if (res != 0) {
      return StreamError{
        cudaErrorInvalidValue, "NvBuffer2Raw failed for plane " + std::to_string(i)};
    }
  }

  return {};
}
