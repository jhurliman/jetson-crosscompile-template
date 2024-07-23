/**
 * NVIDIA Jetpack 32.7 introduced new NvBufferColorFormat enum values intermixed with the existing
 * values. This file includes helper methods for converting NvmmColorFormat to the corresponding
 * NvBufferColorFormat enum value for both Jetpack 32.6 and 32.7, assuming the caller knows the
 * Jetpack version.
 */

#pragma once

#include "nvmm/types.hpp"

enum class NvBufferColorFormat32_6 {
  /** BT.601 colorspace - YUV420 multi-planar. */
  NvBufferColorFormat_YUV420,
  /** BT.601 colorspace - YUV420 multi-planar. */
  NvBufferColorFormat_YVU420,
  /** BT.601 colorspace - YUV420 ER multi-planar. */
  NvBufferColorFormat_YUV420_ER,
  /** BT.601 colorspace - YVU420 ER multi-planar. */
  NvBufferColorFormat_YVU420_ER,
  /** BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12,
  /** BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12_ER,
  /** BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. */
  NvBufferColorFormat_NV21,
  /** BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar. */
  NvBufferColorFormat_NV21_ER,
  /** BT.601 colorspace - YUV 4:2:2 planar. */
  NvBufferColorFormat_UYVY,
  /** BT.601 colorspace - YUV ER 4:2:2 planar. */
  NvBufferColorFormat_UYVY_ER,
  /** BT.601 colorspace - YUV 4:2:2 planar. */
  NvBufferColorFormat_VYUY,
  /** BT.601 colorspace - YUV ER 4:2:2 planar. */
  NvBufferColorFormat_VYUY_ER,
  /** BT.601 colorspace - YUV 4:2:2 planar. */
  NvBufferColorFormat_YUYV,
  /** BT.601 colorspace - YUV ER 4:2:2 planar. */
  NvBufferColorFormat_YUYV_ER,
  /** BT.601 colorspace - YUV 4:2:2 planar. */
  NvBufferColorFormat_YVYU,
  /** BT.601 colorspace - YUV ER 4:2:2 planar. */
  NvBufferColorFormat_YVYU_ER,
  /** LegacyRGBA colorspace - BGRA-8-8-8-8 planar. */
  NvBufferColorFormat_ABGR32,
  /** LegacyRGBA colorspace - XRGB-8-8-8-8 planar. */
  NvBufferColorFormat_XRGB32,
  /** LegacyRGBA colorspace - ARGB-8-8-8-8 planar. */
  NvBufferColorFormat_ARGB32,
  /** BT.601 colorspace - Y/CbCr 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV12_10LE,
  /** BT.709 colorspace - Y/CbCr 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV12_10LE_709,
  /** BT.709_ER colorspace - Y/CbCr 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV12_10LE_709_ER,
  /** BT.2020 colorspace - Y/CbCr 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV12_10LE_2020,
  /** BT.601 colorspace - Y/CrCb 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV21_10LE,
  /** BT.601 colorspace - Y/CbCr 4:2:0 12-bit multi-planar. */
  NvBufferColorFormat_NV12_12LE,
  /** BT.2020 colorspace - Y/CbCr 4:2:0 12-bit multi-planar. */
  NvBufferColorFormat_NV12_12LE_2020,
  /** BT.601 colorspace - Y/CrCb 4:2:0 12-bit multi-planar. */
  NvBufferColorFormat_NV21_12LE,
  /** BT.709 colorspace - YUV420 multi-planar. */
  NvBufferColorFormat_YUV420_709,
  /** BT.709 colorspace - YUV420 ER multi-planar. */
  NvBufferColorFormat_YUV420_709_ER,
  /** BT.709 colorspace - Y/CbCr 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12_709,
  /** BT.709 colorspace - Y/CbCr ER 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12_709_ER,
  /** BT.2020 colorspace - YUV420 multi-planar. */
  NvBufferColorFormat_YUV420_2020,
  /** BT.2020 colorspace - Y/CbCr 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12_2020,
  /** Optical flow */
  NvBufferColorFormat_SignedR16G16,
  /** Optical flow SAD calculation Buffer format */
  NvBufferColorFormat_A32,
  /** BT.601 colorspace - YUV444 multi-planar. */
  NvBufferColorFormat_YUV444,
  /** 8-bit grayscale. */
  NvBufferColorFormat_GRAY8,
  /** BT.601 colorspace - Y/CbCr 4:2:2 multi-planar. */
  NvBufferColorFormat_NV16,
  /** BT.601 colorspace - Y/CbCr 4:2:2 10-bit semi-planar. */
  NvBufferColorFormat_NV16_10LE,
  /** BT.601 colorspace - Y/CbCr 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24,
  /** BT.601_ER colorspace - Y/CbCr 4:2:2 multi-planar. */
  NvBufferColorFormat_NV16_ER,
  /** BT.601_ER colorspace - Y/CbCr 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_ER,
  /** BT.709 colorspace - Y/CbCr 4:2:2 multi-planar. */
  NvBufferColorFormat_NV16_709,
  /** BT.709 colorspace - Y/CbCr 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_709,
  /** BT.709_ER colorspace - Y/CbCr 4:2:2 multi-planar. */
  NvBufferColorFormat_NV16_709_ER,
  /** BT.709_ER colorspace - Y/CbCr 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_709_ER,
  /** BT.709 colorspace - Y/CbCr 10 bit 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_10LE_709,
  /** BT.709 ER colorspace - Y/CbCr 10 bit 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_10LE_709_ER,
  /** BT.2020 colorspace - Y/CbCr 10 bit 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_10LE_2020,
  /** BT.2020 colorspace - Y/CbCr 12 bit 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_12LE_2020,
  /** Non-linear RGB BT.709 colorspace - RGBA-10-10-10-2 planar. */
  NvBufferColorFormat_RGBA_10_10_10_2_709,
  /** Non-linear RGB BT.2020 colorspace - RGBA-10-10-10-2 planar. */
  NvBufferColorFormat_RGBA_10_10_10_2_2020,
  /** Invalid color format. */
  NvBufferColorFormat_Invalid,
};

enum class NvBufferColorFormat32_7 {
  /** BT.601 colorspace - YUV420 multi-planar. */
  NvBufferColorFormat_YUV420,
  /** BT.601 colorspace - YUV420 multi-planar. */
  NvBufferColorFormat_YVU420,
  /** BT.601 colorspace - YUV422 multi-planar. */
  NvBufferColorFormat_YUV422,
  /** BT.601 colorspace - YUV420 ER multi-planar. */
  NvBufferColorFormat_YUV420_ER,
  /** BT.601 colorspace - YVU420 ER multi-planar. */
  NvBufferColorFormat_YVU420_ER,
  /** BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12,
  /** BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12_ER,
  /** BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. */
  NvBufferColorFormat_NV21,
  /** BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar. */
  NvBufferColorFormat_NV21_ER,
  /** BT.601 colorspace - YUV 4:2:2 planar. */
  NvBufferColorFormat_UYVY,
  /** BT.601 colorspace - YUV ER 4:2:2 planar. */
  NvBufferColorFormat_UYVY_ER,
  /** BT.601 colorspace - YUV 4:2:2 planar. */
  NvBufferColorFormat_VYUY,
  /** BT.601 colorspace - YUV ER 4:2:2 planar. */
  NvBufferColorFormat_VYUY_ER,
  /** BT.601 colorspace - YUV 4:2:2 planar. */
  NvBufferColorFormat_YUYV,
  /** BT.601 colorspace - YUV ER 4:2:2 planar. */
  NvBufferColorFormat_YUYV_ER,
  /** BT.601 colorspace - YUV 4:2:2 planar. */
  NvBufferColorFormat_YVYU,
  /** BT.601 colorspace - YUV ER 4:2:2 planar. */
  NvBufferColorFormat_YVYU_ER,
  /** LegacyRGBA colorspace - BGRA-8-8-8-8 planar. */
  NvBufferColorFormat_ABGR32,
  /** LegacyRGBA colorspace - XRGB-8-8-8-8 planar. */
  NvBufferColorFormat_XRGB32,
  /** LegacyRGBA colorspace - ARGB-8-8-8-8 planar. */
  NvBufferColorFormat_ARGB32,
  /** BT.601 colorspace - Y/CbCr 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV12_10LE,
  /** BT.709 colorspace - Y/CbCr 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV12_10LE_709,
  /** BT.709_ER colorspace - Y/CbCr 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV12_10LE_709_ER,
  /** BT.2020 colorspace - Y/CbCr 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV12_10LE_2020,
  /** BT.601 colorspace - Y/CrCb 4:2:0 10-bit multi-planar. */
  NvBufferColorFormat_NV21_10LE,
  /** BT.601 colorspace - Y/CbCr 4:2:0 12-bit multi-planar. */
  NvBufferColorFormat_NV12_12LE,
  /** BT.2020 colorspace - Y/CbCr 4:2:0 12-bit multi-planar. */
  NvBufferColorFormat_NV12_12LE_2020,
  /** BT.601 colorspace - Y/CrCb 4:2:0 12-bit multi-planar. */
  NvBufferColorFormat_NV21_12LE,
  /** BT.709 colorspace - YUV420 multi-planar. */
  NvBufferColorFormat_YUV420_709,
  /** BT.709 colorspace - YUV420 ER multi-planar. */
  NvBufferColorFormat_YUV420_709_ER,
  /** BT.709 colorspace - Y/CbCr 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12_709,
  /** BT.709 colorspace - Y/CbCr ER 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12_709_ER,
  /** BT.2020 colorspace - YUV420 multi-planar. */
  NvBufferColorFormat_YUV420_2020,
  /** BT.2020 colorspace - Y/CbCr 4:2:0 multi-planar. */
  NvBufferColorFormat_NV12_2020,
  /** Optical flow */
  NvBufferColorFormat_SignedR16G16,
  /** Optical flow SAD calculation Buffer format */
  NvBufferColorFormat_A32,
  /** BT.601 colorspace - YUV444 multi-planar. */
  NvBufferColorFormat_YUV444,
  /** 8-bit grayscale. */
  NvBufferColorFormat_GRAY8,
  /** BT.601 colorspace - Y/CbCr 4:2:2 multi-planar. */
  NvBufferColorFormat_NV16,
  /** BT.601 colorspace - Y/CbCr 4:2:2 10-bit semi-planar. */
  NvBufferColorFormat_NV16_10LE,
  /** BT.601 colorspace - Y/CbCr 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24,
  /** BT.601 colorspace - Y/CrCb 4:4:4 10-bit multi-planar. */
  NvBufferColorFormat_NV24_10LE,
  /** BT.601_ER colorspace - Y/CbCr 4:2:2 multi-planar. */
  NvBufferColorFormat_NV16_ER,
  /** BT.601_ER colorspace - Y/CbCr 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_ER,
  /** BT.709 colorspace - Y/CbCr 4:2:2 multi-planar. */
  NvBufferColorFormat_NV16_709,
  /** BT.709 colorspace - Y/CbCr 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_709,
  /** BT.709_ER colorspace - Y/CbCr 4:2:2 multi-planar. */
  NvBufferColorFormat_NV16_709_ER,
  /** BT.709_ER colorspace - Y/CbCr 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_709_ER,
  /** BT.709 colorspace - Y/CbCr 10 bit 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_10LE_709,
  /** BT.709 ER colorspace - Y/CbCr 10 bit 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_10LE_709_ER,
  /** BT.2020 colorspace - Y/CbCr 10 bit 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_10LE_2020,
  /** BT.2020 colorspace - Y/CbCr 12 bit 4:4:4 multi-planar. */
  NvBufferColorFormat_NV24_12LE_2020,
  /** Non-linear RGB BT.709 colorspace - RGBA-10-10-10-2 planar. */
  NvBufferColorFormat_RGBA_10_10_10_2_709,
  /** Non-linear RGB BT.2020 colorspace - RGBA-10-10-10-2 planar. */
  NvBufferColorFormat_RGBA_10_10_10_2_2020,
  /** Non-linear RGB BT.709 colorspace - BGRA-10-10-10-2 planar. */
  NvBufferColorFormat_BGRA_10_10_10_2_709,
  /** Non-linear RGB BT.2020 colorspace - BGRA-10-10-10-2 planar. */
  NvBufferColorFormat_BGRA_10_10_10_2_2020,
  /** Invalid color format. */
  NvBufferColorFormat_Invalid,
};

// clang-format off

constexpr std::optional<uint32_t> NvBufferColorFormat32_6(NvmmColorFormat format) {
  switch (format) {
    case NvmmColorFormat::YUV420: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YUV420);
    case NvmmColorFormat::YVU420: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YVU420);
    case NvmmColorFormat::YUV422: return {};
    case NvmmColorFormat::YUV420_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YUV420_ER);
    case NvmmColorFormat::YVU420_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YVU420_ER);
    case NvmmColorFormat::NV12: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12);
    case NvmmColorFormat::NV12_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_ER);
    case NvmmColorFormat::NV21: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV21);
    case NvmmColorFormat::NV21_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV21_ER);
    case NvmmColorFormat::UYVY: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_UYVY);
    case NvmmColorFormat::UYVY_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_UYVY_ER);
    case NvmmColorFormat::VYUY: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_VYUY);
    case NvmmColorFormat::VYUY_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_VYUY_ER);
    case NvmmColorFormat::YUYV: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YUYV);
    case NvmmColorFormat::YUYV_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YUYV_ER);
    case NvmmColorFormat::YVYU: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YVYU);
    case NvmmColorFormat::YVYU_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YVYU_ER);
    case NvmmColorFormat::ABGR32: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_ABGR32);
    case NvmmColorFormat::XRGB32: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_XRGB32);
    case NvmmColorFormat::ARGB32: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_ARGB32);
    case NvmmColorFormat::NV12_10LE: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_10LE);
    case NvmmColorFormat::NV12_10LE_709: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_10LE_709);
    case NvmmColorFormat::NV12_10LE_709_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_10LE_709_ER);
    case NvmmColorFormat::NV12_10LE_2020: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_10LE_2020);
    case NvmmColorFormat::NV21_10LE: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV21_10LE);
    case NvmmColorFormat::NV12_12LE: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_12LE);
    case NvmmColorFormat::NV12_12LE_2020: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_12LE_2020);
    case NvmmColorFormat::NV21_12LE: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV21_12LE);
    case NvmmColorFormat::YUV420_709: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YUV420_709);
    case NvmmColorFormat::YUV420_709_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YUV420_709_ER);
    case NvmmColorFormat::NV12_709: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_709);
    case NvmmColorFormat::NV12_709_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_709_ER);
    case NvmmColorFormat::YUV420_2020: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YUV420_2020);
    case NvmmColorFormat::NV12_2020: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV12_2020);
    case NvmmColorFormat::SignedR16G16: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_SignedR16G16);
    case NvmmColorFormat::A32: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_A32);
    case NvmmColorFormat::YUV444: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_YUV444);
    case NvmmColorFormat::GRAY8: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_GRAY8);
    case NvmmColorFormat::NV16: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV16);
    case NvmmColorFormat::NV16_10LE: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV16_10LE);
    case NvmmColorFormat::NV24: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV24);
    case NvmmColorFormat::NV24_10LE: return {};
    case NvmmColorFormat::NV16_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV16_ER);
    case NvmmColorFormat::NV24_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV24_ER);
    case NvmmColorFormat::NV16_709: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV16_709);
    case NvmmColorFormat::NV24_709: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV24_709);
    case NvmmColorFormat::NV16_709_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV16_709_ER);
    case NvmmColorFormat::NV24_709_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV24_709_ER);
    case NvmmColorFormat::NV24_10LE_709: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV24_10LE_709);
    case NvmmColorFormat::NV24_10LE_709_ER: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV24_10LE_709_ER);
    case NvmmColorFormat::NV24_10LE_2020: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV24_10LE_2020);
    case NvmmColorFormat::NV24_12LE_2020: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_NV24_12LE_2020);
    case NvmmColorFormat::RGBA_10_10_10_2_709: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_RGBA_10_10_10_2_709);
    case NvmmColorFormat::RGBA_10_10_10_2_2020: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_RGBA_10_10_10_2_2020);
    case NvmmColorFormat::BGRA_10_10_10_2_709: return {};
    case NvmmColorFormat::BGRA_10_10_10_2_2020: return {};
    case NvmmColorFormat::Invalid: return uint32_t(NvBufferColorFormat32_6::NvBufferColorFormat_Invalid);
  }
}

// clang-format on

constexpr uint32_t NvBufferColorFormat32_7(NvmmColorFormat format) {
  return static_cast<uint32_t>(format);
}
