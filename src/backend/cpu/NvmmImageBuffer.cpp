#include "nvmm/NvmmImageBuffer.hpp"

#include "cuda/types.hpp"

#include <limits>

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)

// YUV420,               // BT.601 colorspace - YUV420 multi-planar.
// YVU420,               // BT.601 colorspace - YUV420 multi-planar.
// YUV422,               // BT.601 colorspace - YUV422 multi-planar.
// YUV420_ER,            // BT.601 colorspace - YUV420 ER multi-planar.
// YVU420_ER,            // BT.601 colorspace - YVU420 ER multi-planar.
// NV12,                 // BT.601 colorspace - Y/CbCr 4:2:0 multi-planar.
// NV12_ER,              // BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar.
// NV21,                 // BT.601 colorspace - Y/CbCr 4:2:0 multi-planar.
// NV21_ER,              // BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar.
// UYVY,                 // BT.601 colorspace - YUV 4:2:2 planar.
// UYVY_ER,              // BT.601 colorspace - YUV ER 4:2:2 planar.
// VYUY,                 // BT.601 colorspace - YUV 4:2:2 planar.
// VYUY_ER,              // BT.601 colorspace - YUV ER 4:2:2 planar.
// YUYV,                 // BT.601 colorspace - YUV 4:2:2 planar.
// YUYV_ER,              // BT.601 colorspace - YUV ER 4:2:2 planar.
// YVYU,                 // BT.601 colorspace - YUV 4:2:2 planar.
// YVYU_ER,              // BT.601 colorspace - YUV ER 4:2:2 planar.
// ABGR32,               // LegacyRGBA colorspace - BGRA-8-8-8-8 planar.
// XRGB32,               // LegacyRGBA colorspace - XRGB-8-8-8-8 planar.
// ARGB32,               // LegacyRGBA colorspace - ARGB-8-8-8-8 planar.
// NV12_10LE,            // BT.601 colorspace - Y/CbCr 4:2:0 10-bit multi-planar.
// NV12_10LE_709,        // BT.709 colorspace - Y/CbCr 4:2:0 10-bit multi-planar.
// NV12_10LE_709_ER,     // BT.709_ER colorspace - Y/CbCr 4:2:0 10-bit multi-planar.
// NV12_10LE_2020,       // BT.2020 colorspace - Y/CbCr 4:2:0 10-bit multi-planar.
// NV21_10LE,            // BT.601 colorspace - Y/CrCb 4:2:0 10-bit multi-planar.
// NV12_12LE,            // BT.601 colorspace - Y/CbCr 4:2:0 12-bit multi-planar.
// NV12_12LE_2020,       // BT.2020 colorspace - Y/CbCr 4:2:0 12-bit multi-planar.
// NV21_12LE,            // BT.601 colorspace - Y/CrCb 4:2:0 12-bit multi-planar.
// YUV420_709,           // BT.709 colorspace - YUV420 multi-planar.
// YUV420_709_ER,        // BT.709 colorspace - YUV420 ER multi-planar.
// NV12_709,             // BT.709 colorspace - Y/CbCr 4:2:0 multi-planar.
// NV12_709_ER,          // BT.709 colorspace - Y/CbCr ER 4:2:0 multi-planar.
// YUV420_2020,          // BT.2020 colorspace - YUV420 multi-planar.
// NV12_2020,            // BT.2020 colorspace - Y/CbCr 4:2:0 multi-planar.
// SignedR16G16,         // Optical flow
// A32,                  // Optical flow SAD calculation Buffer format
// YUV444,               // BT.601 colorspace - YUV444 multi-planar.
// GRAY8,                // 8-bit grayscale.
// NV16,                 // BT.601 colorspace - Y/CbCr 4:2:2 multi-planar.
// NV16_10LE,            // BT.601 colorspace - Y/CbCr 4:2:2 10-bit semi-planar.
// NV24,                 // BT.601 colorspace - Y/CbCr 4:4:4 multi-planar.
// NV24_10LE,            // BT.601 colorspace - Y/CrCb 4:4:4 10-bit multi-planar.
// NV16_ER,              // BT.601_ER colorspace - Y/CbCr 4:2:2 multi-planar.
// NV24_ER,              // BT.601_ER colorspace - Y/CbCr 4:4:4 multi-planar.
// NV16_709,             // BT.709 colorspace - Y/CbCr 4:2:2 multi-planar.
// NV24_709,             // BT.709 colorspace - Y/CbCr 4:4:4 multi-planar.
// NV16_709_ER,          // BT.709_ER colorspace - Y/CbCr 4:2:2 multi-planar.
// NV24_709_ER,          // BT.709_ER colorspace - Y/CbCr 4:4:4 multi-planar.
// NV24_10LE_709,        // BT.709 colorspace - Y/CbCr 10 bit 4:4:4 multi-planar.
// NV24_10LE_709_ER,     // BT.709 ER colorspace - Y/CbCr 10 bit 4:4:4 multi-planar.
// NV24_10LE_2020,       // BT.2020 colorspace - Y/CbCr 10 bit 4:4:4 multi-planar.
// NV24_12LE_2020,       // BT.2020 colorspace - Y/CbCr 12 bit 4:4:4 multi-planar.
// RGBA_10_10_10_2_709,  // Non-linear RGB BT.709 colorspace - RGBA-10-10-10-2 planar.
// RGBA_10_10_10_2_2020, // Non-linear RGB BT.2020 colorspace - RGBA-10-10-10-2 planar.
// BGRA_10_10_10_2_709,  // Non-linear RGB BT.709 colorspace - BGRA-10-10-10-2 planar.
// BGRA_10_10_10_2_2020, // Non-linear RGB BT.2020 colorspace - BGRA-10-10-10-2 planar.
// Invalid               // Invalid color format.

// static size_t PlaneCountForColorFormat(NvmmColorFormat format) {
//   switch (format) {
//     case NvmmColorFormat::YUV420:
//     case NvmmColorFormat::YVU420:
//     case NvmmColorFormat::YUV422:
//     case NvmmColorFormat::YUV420_ER:
//     case NvmmColorFormat::YVU420_ER:
//       return 3;
//     case NvmmColorFormat::NV12:
//     case NvmmColorFormat::NV12_ER:
//     case NvmmColorFormat::NV21:
//     case NvmmColorFormat::NV21_ER:
//     case NvmmColorFormat::UYVY:
//     case NvmmColorFormat::UYVY_ER:
//     case NvmmColorFormat::VYUY:
//     default:
//       return 0;
//   }
// }

// NvmmImageBuffer::NvmmImageBuffer(void* pVirtAddr,
//   int fd,
//   size_t byteSize,
//   size_t nvBufferSize,
//   NvmmColorFormat format,
//   size_t numPlanes,
//   const PlaneArray& widths,
//   const PlaneArray& heights,
//   const PlaneArray& pitches,
//   const PlaneArray& offsets,
//   const PlaneArray& sizes,
//   const PlaneLayoutArray& layouts)
//   : NvmmBuffer(pVirtAddr, fd, byteSize, nvBufferSize),
//     format_(format),
//     numPlanes_(numPlanes),
//     widths_(widths),
//     heights_(heights),
//     pitches_(pitches),
//     offsets_(offsets),
//     sizes_(sizes),
//     layouts_(layouts) {}

// tl::expected<std::unique_ptr<NvmmImageBuffer>, NvmmError> NvmmImageBuffer::create(
//   size_t width, size_t height, NvmmColorFormat format, NvmmBufferLayout layout) {
//   if (width > std::numeric_limits<int32_t>::max() || height >
//   std::numeric_limits<int32_t>::max()) {
//     return tl::make_unexpected(NvmmError{cudaErrorInvalidValue, "width or height too large"});
//   }
//   if (width == 0 || height == 0) {
//     return std::unique_ptr<NvmmImageBuffer>(
//       new NvmmImageBuffer(nullptr, -1, 0, 0, format, 0, {}, {}, {}, {}, {}, {}));
//   }

//   void* pVirtAddr = bufferParams.nv_buffer;
//   const size_t byteSize = size_t(bufferParams.psize[0]) + size_t(bufferParams.psize[1]) +
//     size_t(bufferParams.psize[2]) + size_t(bufferParams.psize[3]);
//   const size_t nvBufferSize = bufferParams.nv_buffer_size;
//   const size_t numPlanes = bufferParams.num_planes;
//   const PlaneArray widths = {
//     bufferParams.width[0], bufferParams.width[1], bufferParams.width[2], bufferParams.width[3]};
//   const PlaneArray heights = {
//     bufferParams.height[0], bufferParams.height[1], bufferParams.height[2],
//     bufferParams.height[3]};
//   const PlaneArray pitches = {
//     bufferParams.pitch[0], bufferParams.pitch[1], bufferParams.pitch[2], bufferParams.pitch[3]};
//   const PlaneArray offsets = {
//     bufferParams.offset[0], bufferParams.offset[1], bufferParams.offset[2],
//     bufferParams.offset[3]};
//   const PlaneArray sizes = {
//     bufferParams.psize[0], bufferParams.psize[1], bufferParams.psize[2], bufferParams.psize[3]};
//   const PlaneLayoutArray layouts = {NvmmBufferLayout(bufferParams.layout[0]),
//     NvmmBufferLayout(bufferParams.layout[1]),
//     NvmmBufferLayout(bufferParams.layout[2]),
//     NvmmBufferLayout(bufferParams.layout[3])};

//   return std::unique_ptr<NvmmImageBuffer>(new NvmmImageBuffer(pVirtAddr,
//     dmabuf_fd,
//     byteSize,
//     nvBufferSize,
//     format,
//     numPlanes,
//     widths,
//     heights,
//     pitches,
//     offsets,
//     sizes,
//     layouts));
// }

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

// std::optional<NvmmError> NvmmImageBuffer::copyFrom2D(const NvmmImageBuffer& src,
//   const Rect& srcRect,
//   const Rect& dstRect,
//   NvBufferSession session,
//   NvmmTransformFilter filter) {
//   // DMA buffer to buffer copy via NvBufferTransform

//   uint32_t flags = 0;
//   if (srcRect.top != 0 || srcRect.left != 0 || srcRect.width != src.widths()[0] ||
//     srcRect.height != src.heights()[0]) {
//     flags |= NVBUFFER_TRANSFORM_CROP_SRC;
//   }
//   if (dstRect.top != 0 || dstRect.left != 0 || dstRect.width != widths()[0] ||
//     dstRect.height != heights()[0]) {
//     flags |= NVBUFFER_TRANSFORM_CROP_DST;
//   }
//   if (srcRect != dstRect) { flags |= NVBUFFER_TRANSFORM_FILTER; }

//   NvBufferTransformParams params{};
//   params.session = session;
//   params.transform_flag = flags;
//   params.transform_filter = NvBufferTransform_Filter(filter);
//   params.src_rect = MakeNvBufferRect(srcRect);
//   params.dst_rect = MakeNvBufferRect(dstRect);
//   const int res = NvBufferTransform(src.fd(), fd(), &params);
//   if (res != 0) { return NvmmError{cudaErrorMapBufferObjectFailed, "NvBufferTransform failed"}; }

//   return {};
// }

// std::optional<NvmmError> NvmmImageBuffer::copyFromHost2D(const std::vector<uint8_t*>& srcPlanes)
// {
//   const unsigned int planes = static_cast<unsigned int>(numPlanes());
//   if (srcPlanes.size() != planes) {
//     return NvmmError{cudaErrorInvalidValue, "srcPlanes.size() != numPlanes()"};
//   }

//   // Copy each plane from host to the hardware buffer
//   for (unsigned int i = 0; i < planes; ++i) {
//     const unsigned int width = static_cast<unsigned int>(widths().at(i));
//     const unsigned int height = static_cast<unsigned int>(heights().at(i));
//     const int res = Raw2NvBuffer(srcPlanes[i], i, width, height, fd());
//     if (res != 0) {
//       return NvmmError{cudaErrorInvalidValue, "Raw2NvBuffer failed for plane " +
//       std::to_string(i)};
//     }
//   }

//   return {};
// }

// std::optional<NvmmError> NvmmImageBuffer::copyTo2D(NvmmImageBuffer& dst,
//   const Rect& srcRect,
//   const Rect& dstRect,
//   NvBufferSession session,
//   NvmmTransformFilter filter) const {
//   return dst.copyFrom2D(*this, srcRect, dstRect, session, filter);
// }

// std::optional<NvmmError> NvmmImageBuffer::copyToHost2D(
//   const std::vector<uint8_t*>& dstPlanes) const {
//   const unsigned int planes = static_cast<unsigned int>(numPlanes());
//   if (dstPlanes.size() != planes) {
//     return NvmmError{cudaErrorInvalidValue, "dstPlanes.size() != numPlanes()"};
//   }

//   // Copy each plane from the hardware buffer to host
//   for (unsigned int i = 0; i < planes; ++i) {
//     const unsigned int width = static_cast<unsigned int>(widths().at(i));
//     const unsigned int height = static_cast<unsigned int>(heights().at(i));
//     const int res = NvBuffer2Raw(fd(), i, width, height, dstPlanes[i]);
//     if (res != 0) {
//       return NvmmError{cudaErrorInvalidValue, "NvBuffer2Raw failed for plane " +
//       std::to_string(i)};
//     }
//   }

//   return {};
// }

// NOLINTEND(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
