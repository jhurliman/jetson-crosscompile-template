#pragma once

#include <string> // IWYU pragma: export

static constexpr size_t NV_BUFFER_SIZE = 1008;

// Forward declare opaque type NvBufferSession
typedef struct _NvBufferSession* NvBufferSession;

// clang-format off

enum class NvmmColorFormat {
  YUV420,               // BT.601 colorspace - YUV420 multi-planar.
  YVU420,               // BT.601 colorspace - YUV420 multi-planar.
  YUV422,               // BT.601 colorspace - YUV422 multi-planar.
  YUV420_ER,            // BT.601 colorspace - YUV420 ER multi-planar.
  YVU420_ER,            // BT.601 colorspace - YVU420 ER multi-planar.
  NV12,                 // BT.601 colorspace - Y/CbCr 4:2:0 multi-planar.
  NV12_ER,              // BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar.
  NV21,                 // BT.601 colorspace - Y/CbCr 4:2:0 multi-planar.
  NV21_ER,              // BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar.
  UYVY,                 // BT.601 colorspace - YUV 4:2:2 planar.
  UYVY_ER,              // BT.601 colorspace - YUV ER 4:2:2 planar.
  VYUY,                 // BT.601 colorspace - YUV 4:2:2 planar.
  VYUY_ER,              // BT.601 colorspace - YUV ER 4:2:2 planar.
  YUYV,                 // BT.601 colorspace - YUV 4:2:2 planar.
  YUYV_ER,              // BT.601 colorspace - YUV ER 4:2:2 planar.
  YVYU,                 // BT.601 colorspace - YUV 4:2:2 planar.
  YVYU_ER,              // BT.601 colorspace - YUV ER 4:2:2 planar.
  ABGR32,               // LegacyRGBA colorspace - BGRA-8-8-8-8 planar.
  XRGB32,               // LegacyRGBA colorspace - XRGB-8-8-8-8 planar.
  ARGB32,               // LegacyRGBA colorspace - ARGB-8-8-8-8 planar.
  NV12_10LE,            // BT.601 colorspace - Y/CbCr 4:2:0 10-bit multi-planar.
  NV12_10LE_709,        // BT.709 colorspace - Y/CbCr 4:2:0 10-bit multi-planar.
  NV12_10LE_709_ER,     // BT.709_ER colorspace - Y/CbCr 4:2:0 10-bit multi-planar.
  NV12_10LE_2020,       // BT.2020 colorspace - Y/CbCr 4:2:0 10-bit multi-planar.
  NV21_10LE,            // BT.601 colorspace - Y/CrCb 4:2:0 10-bit multi-planar.
  NV12_12LE,            // BT.601 colorspace - Y/CbCr 4:2:0 12-bit multi-planar.
  NV12_12LE_2020,       // BT.2020 colorspace - Y/CbCr 4:2:0 12-bit multi-planar.
  NV21_12LE,            // BT.601 colorspace - Y/CrCb 4:2:0 12-bit multi-planar.
  YUV420_709,           // BT.709 colorspace - YUV420 multi-planar.
  YUV420_709_ER,        // BT.709 colorspace - YUV420 ER multi-planar.
  NV12_709,             // BT.709 colorspace - Y/CbCr 4:2:0 multi-planar.
  NV12_709_ER,          // BT.709 colorspace - Y/CbCr ER 4:2:0 multi-planar.
  YUV420_2020,          // BT.2020 colorspace - YUV420 multi-planar.
  NV12_2020,            // BT.2020 colorspace - Y/CbCr 4:2:0 multi-planar.
  SignedR16G16,         // Optical flow
  A32,                  // Optical flow SAD calculation Buffer format
  YUV444,               // BT.601 colorspace - YUV444 multi-planar.
  GRAY8,                // 8-bit grayscale.
  NV16,                 // BT.601 colorspace - Y/CbCr 4:2:2 multi-planar.
  NV16_10LE,            // BT.601 colorspace - Y/CbCr 4:2:2 10-bit semi-planar.
  NV24,                 // BT.601 colorspace - Y/CbCr 4:4:4 multi-planar.
  NV24_10LE,            // BT.601 colorspace - Y/CrCb 4:4:4 10-bit multi-planar.
  NV16_ER,              // BT.601_ER colorspace - Y/CbCr 4:2:2 multi-planar.
  NV24_ER,              // BT.601_ER colorspace - Y/CbCr 4:4:4 multi-planar.
  NV16_709,             // BT.709 colorspace - Y/CbCr 4:2:2 multi-planar.
  NV24_709,             // BT.709 colorspace - Y/CbCr 4:4:4 multi-planar.
  NV16_709_ER,          // BT.709_ER colorspace - Y/CbCr 4:2:2 multi-planar.
  NV24_709_ER,          // BT.709_ER colorspace - Y/CbCr 4:4:4 multi-planar.
  NV24_10LE_709,        // BT.709 colorspace - Y/CbCr 10 bit 4:4:4 multi-planar.
  NV24_10LE_709_ER,     // BT.709 ER colorspace - Y/CbCr 10 bit 4:4:4 multi-planar.
  NV24_10LE_2020,       // BT.2020 colorspace - Y/CbCr 10 bit 4:4:4 multi-planar.
  NV24_12LE_2020,       // BT.2020 colorspace - Y/CbCr 12 bit 4:4:4 multi-planar.
  RGBA_10_10_10_2_709,  // Non-linear RGB BT.709 colorspace - RGBA-10-10-10-2 planar.
  RGBA_10_10_10_2_2020, // Non-linear RGB BT.2020 colorspace - RGBA-10-10-10-2 planar.
  BGRA_10_10_10_2_709,  // Non-linear RGB BT.709 colorspace - BGRA-10-10-10-2 planar.
  BGRA_10_10_10_2_2020, // Non-linear RGB BT.2020 colorspace - BGRA-10-10-10-2 planar.
  Invalid               // Invalid color format.
};

enum class NvmmBufferLayout {
  Pitch,       // Pitch layout.
  BlockLinear, // Block linear layout.
};

enum class NvmmBufferMemAccess {
  Read,      // Memory read.
  Write,     // Memory write.
  ReadWrite, // Memory read & write.
};

enum class NvmmTransformFilter {
  Nearest,
  Bilinear,
  FiveTap,
  TenTap,
  Smart,
  Nicest,
};

// clang-format on

struct Rect {
  uint32_t top;
  uint32_t left;
  uint32_t width;
  uint32_t height;
};

inline bool operator==(const Rect& lhs, const Rect& rhs) {
  return lhs.top == rhs.top && lhs.left == rhs.left && lhs.width == rhs.width &&
    lhs.height == rhs.height;
}

inline bool operator!=(const Rect& lhs, const Rect& rhs) {
  return !(lhs == rhs);
}
