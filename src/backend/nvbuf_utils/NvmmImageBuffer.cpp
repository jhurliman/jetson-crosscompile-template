#include "nvmm/NvmmImageBuffer.hpp"

#include "cuda/types.hpp"

#include <nvbuf_utils.h>

#include <iostream>
#include <limits>
#include <sstream>

static std::string ToString(const NvBufferParams& params) {
  std::ostringstream oss;
  oss << "NvBufferParams{";
  oss << "dmabuf_fd=" << params.dmabuf_fd << ", ";
  oss << "nv_buffer=" << params.nv_buffer << ", ";
  oss << "payloadType=" << params.payloadType << ", ";
  oss << "memsize=" << params.memsize << ", ";
  oss << "nv_buffer_size=" << params.nv_buffer_size << ", ";
  oss << "pixel_format=" << params.pixel_format << ", ";
  oss << "num_planes=" << params.num_planes << ", ";
  oss << "width=[";
  const size_t numPlanes = std::min(size_t(params.num_planes), size_t(MAX_NUM_PLANES));
  for (size_t i = 0; i < numPlanes; ++i) {
    oss << params.width[i];
    if (i + 1 < numPlanes) { oss << ", "; }
  }
  oss << "], height=[";
  for (size_t i = 0; i < numPlanes; ++i) {
    oss << params.height[i];
    if (i + 1 < numPlanes) { oss << ", "; }
  }
  oss << "], pitch=[";
  for (size_t i = 0; i < numPlanes; ++i) {
    oss << params.pitch[i];
    if (i + 1 < numPlanes) { oss << ", "; }
  }
  oss << "], offset=[";
  for (size_t i = 0; i < numPlanes; ++i) {
    oss << params.offset[i];
    if (i + 1 < numPlanes) { oss << ", "; }
  }
  oss << "], psize=[";
  for (size_t i = 0; i < numPlanes; ++i) {
    oss << params.psize[i];
    if (i + 1 < numPlanes) { oss << ", "; }
  }
  oss << "], layout=[";
  for (size_t i = 0; i < numPlanes; ++i) {
    oss << params.layout[i];
    if (i + 1 < numPlanes) { oss << ", "; }
  }
  oss << "]}";
  return oss.str();
}

constexpr NvBufferRect MakeNvBufferRect(const Rect& rect) {
  return NvBufferRect{rect.top, rect.left, rect.width, rect.height};
}

NvmmImageBuffer::NvmmImageBuffer(void* pVirtAddr,
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
  const PlaneLayoutArray& layouts)
  : NvmmBuffer(pVirtAddr, fd, byteSize, nvBufferSize),
    format_(format),
    numPlanes_(numPlanes),
    widths_(widths),
    heights_(heights),
    pitches_(pitches),
    offsets_(offsets),
    sizes_(sizes),
    layouts_(layouts) {}

tl::expected<std::unique_ptr<NvmmImageBuffer>, NvmmError> NvmmImageBuffer::create(
  size_t width, size_t height, NvmmColorFormat format, NvmmBufferLayout layout) {
  if (width > std::numeric_limits<int32_t>::max() || height > std::numeric_limits<int32_t>::max()) {
    return tl::make_unexpected(NvmmError{cudaErrorInvalidValue, "width or height too large"});
  }
  if (width == 0 || height == 0) {
    return std::unique_ptr<NvmmImageBuffer>(
      new NvmmImageBuffer(nullptr, -1, 0, 0, format, 0, {}, {}, {}, {}, {}, {}));
  }

  NvBufferCreateParams params{};
  params.width = int32_t(width);
  params.height = int32_t(height);
  params.payloadType = NvBufferPayload_SurfArray;
  params.memsize = 0;
  params.layout = NvBufferLayout(layout);
  params.colorFormat = NvBufferColorFormat(format);
  params.nvbuf_tag = NvBufferTag_NONE;

  int dmabuf_fd = -1;
  const int res = NvBufferCreateEx(&dmabuf_fd, &params);
  if (res != 0 || dmabuf_fd < 0) {
    return tl::make_unexpected(NvmmError{cudaErrorMemoryAllocation, "NvBufferCreateEx failed"});
  }

  // Read back the actual buffer parameters
  NvBufferParams bufferParams{};
  const int res2 = NvBufferGetParams(dmabuf_fd, &bufferParams);
  if (res2 != 0) {
    return tl::make_unexpected(NvmmError{cudaErrorMemoryAllocation, "NvBufferGetParams failed"});
  }

  std::cout << "NvBufferCreateEx: " << res << ", dmabuf_fd: " << dmabuf_fd
            << ", params: " << ToString(bufferParams) << "\n";

  void* pVirtAddr = bufferParams.nv_buffer;
  const size_t byteSize = size_t(bufferParams.psize[0]) + size_t(bufferParams.psize[1]) +
    size_t(bufferParams.psize[2]) + size_t(bufferParams.psize[3]);
  const size_t nvBufferSize = bufferParams.nv_buffer_size;
  const size_t numPlanes = bufferParams.num_planes;
  const PlaneArray widths = {
    bufferParams.width[0], bufferParams.width[1], bufferParams.width[2], bufferParams.width[3]};
  const PlaneArray heights = {
    bufferParams.height[0], bufferParams.height[1], bufferParams.height[2], bufferParams.height[3]};
  const PlaneArray pitches = {
    bufferParams.pitch[0], bufferParams.pitch[1], bufferParams.pitch[2], bufferParams.pitch[3]};
  const PlaneArray offsets = {
    bufferParams.offset[0], bufferParams.offset[1], bufferParams.offset[2], bufferParams.offset[3]};
  const PlaneArray sizes = {
    bufferParams.psize[0], bufferParams.psize[1], bufferParams.psize[2], bufferParams.psize[3]};
  const PlaneLayoutArray layouts = {NvmmBufferLayout(bufferParams.layout[0]),
    NvmmBufferLayout(bufferParams.layout[1]),
    NvmmBufferLayout(bufferParams.layout[2]),
    NvmmBufferLayout(bufferParams.layout[3])};

  return std::unique_ptr<NvmmImageBuffer>(new NvmmImageBuffer(pVirtAddr,
    dmabuf_fd,
    byteSize,
    nvBufferSize,
    format,
    numPlanes,
    widths,
    heights,
    pitches,
    offsets,
    sizes,
    layouts));
}

NvmmColorFormat NvmmImageBuffer::format() const {
  return format_;
}

size_t NvmmImageBuffer::numPlanes() const {
  return numPlanes_;
}

size_t NvmmImageBuffer::width() const {
  return widths_[0];
}

size_t NvmmImageBuffer::height() const {
  return heights_[0];
}

const NvmmImageBuffer::PlaneArray& NvmmImageBuffer::widths() const {
  return widths_;
}

const NvmmImageBuffer::PlaneArray& NvmmImageBuffer::heights() const {
  return heights_;
}

const NvmmImageBuffer::PlaneArray& NvmmImageBuffer::pitches() const {
  return pitches_;
}

const NvmmImageBuffer::PlaneArray& NvmmImageBuffer::offsets() const {
  return offsets_;
}

const NvmmImageBuffer::PlaneArray& NvmmImageBuffer::sizes() const {
  return sizes_;
}

const NvmmImageBuffer::PlaneLayoutArray& NvmmImageBuffer::layouts() const {
  return layouts_;
}

std::optional<NvmmError> NvmmImageBuffer::copyFrom2D(const NvmmImageBuffer& src,
  const Rect& srcRect,
  const Rect& dstRect,
  NvBufferSession session,
  NvmmTransformFilter filter) {
  // DMA buffer to buffer copy via NvBufferTransform

  uint32_t flags = 0;
  if (srcRect.top != 0 || srcRect.left != 0 || srcRect.width != src.widths()[0] ||
    srcRect.height != src.heights()[0]) {
    flags |= NVBUFFER_TRANSFORM_CROP_SRC;
  }
  if (dstRect.top != 0 || dstRect.left != 0 || dstRect.width != widths()[0] ||
    dstRect.height != heights()[0]) {
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
  if (res != 0) { return NvmmError{cudaErrorMapBufferObjectFailed, "NvBufferTransform failed"}; }

  return {};
}

std::optional<NvmmError> NvmmImageBuffer::copyFromHost2D(const std::vector<uint8_t*>& srcPlanes) {
  const unsigned int planes = static_cast<unsigned int>(numPlanes());
  if (srcPlanes.size() != planes) {
    return NvmmError{cudaErrorInvalidValue, "srcPlanes.size() != numPlanes()"};
  }

  // Copy each plane from host to the hardware buffer
  for (unsigned int i = 0; i < planes; ++i) {
    const unsigned int width = static_cast<unsigned int>(widths().at(i));
    const unsigned int height = static_cast<unsigned int>(heights().at(i));
    const int res = Raw2NvBuffer(srcPlanes[i], i, width, height, fd());
    if (res != 0) {
      return NvmmError{cudaErrorInvalidValue, "Raw2NvBuffer failed for plane " + std::to_string(i)};
    }
  }

  return {};
}

std::optional<NvmmError> NvmmImageBuffer::copyTo2D(NvmmImageBuffer& dst,
  const Rect& srcRect,
  const Rect& dstRect,
  NvBufferSession session,
  NvmmTransformFilter filter) const {
  return dst.copyFrom2D(*this, srcRect, dstRect, session, filter);
}

std::optional<NvmmError> NvmmImageBuffer::copyToHost2D(
  const std::vector<uint8_t*>& dstPlanes) const {
  const unsigned int planes = static_cast<unsigned int>(numPlanes());
  if (dstPlanes.size() != planes) {
    return NvmmError{cudaErrorInvalidValue, "dstPlanes.size() != numPlanes()"};
  }

  // Copy each plane from the hardware buffer to host
  for (unsigned int i = 0; i < planes; ++i) {
    const unsigned int width = static_cast<unsigned int>(widths().at(i));
    const unsigned int height = static_cast<unsigned int>(heights().at(i));
    const int res = NvBuffer2Raw(fd(), i, width, height, dstPlanes[i]);
    if (res != 0) {
      return NvmmError{cudaErrorInvalidValue, "NvBuffer2Raw failed for plane " + std::to_string(i)};
    }
  }

  return {};
}
