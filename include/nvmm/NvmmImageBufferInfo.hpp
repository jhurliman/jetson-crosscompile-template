#pragma once

#include "types.hpp"

#include <array>

static constexpr size_t MAX_NUM_PLANES = 4;

using PlaneArray = std::array<size_t, MAX_NUM_PLANES>;
using PlaneLayoutArray = std::array<NvmmBufferLayout, MAX_NUM_PLANES>;

/**
 * Describes the layout of an NvBuffer planar image buffer.
 */
struct NvmmImageBufferInfo {
  NvmmColorFormat format;
  size_t numPlanes;
  PlaneArray widths;
  PlaneArray heights;
  PlaneArray pitches;
  PlaneArray offsets;
  PlaneArray sizes;
  PlaneLayoutArray layouts;

  size_t size() const { return sizes[0] + sizes[1] + sizes[2] + sizes[3]; }
};

constexpr size_t BlockSizeBytes(size_t byteSize) {
  constexpr size_t BLOCK_SIZE = 1024 * 128;
  return (byteSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
}

constexpr size_t PitchSizeBytes(size_t widthBytes) {
  constexpr size_t PITCH_SIZE = 64;
  return (widthBytes + PITCH_SIZE - 1) / PITCH_SIZE * PITCH_SIZE;
}

constexpr NvmmImageBufferInfo GetImageBufferInfo(
  size_t width, size_t height, NvmmColorFormat format, NvmmBufferLayout layout) {
  constexpr NvmmBufferLayout Pitch = NvmmBufferLayout::Pitch;

  const size_t lumaWidth = size_t((ssize_t(width) + 1) & ~1);
  const size_t lumaHeight = size_t((ssize_t(height) + 1) & ~1);
  const size_t lumaWidthHalf = lumaWidth / 2;
  const size_t lumaHeightHalf = lumaHeight / 2;
  const size_t pitchFull = PitchSizeBytes(lumaWidth);
  const size_t pitchHalf = PitchSizeBytes(lumaWidthHalf);

  switch (format) {
  case NvmmColorFormat::YUV420:
  case NvmmColorFormat::YVU420:
  case NvmmColorFormat::YUV420_ER:
  case NvmmColorFormat::YVU420_ER:
  case NvmmColorFormat::YUV420_709:
  case NvmmColorFormat::YUV420_709_ER:
  case NvmmColorFormat::YUV420_2020: {
    const size_t size = BlockSizeBytes(pitchFull * lumaHeight);
    const size_t chromaSize = BlockSizeBytes(pitchHalf * lumaHeightHalf);
    return {
      format,
      3,
      { lumaWidth,  lumaWidthHalf,     lumaWidthHalf,     0},
      {lumaHeight, lumaHeightHalf,    lumaHeightHalf,     0},
      { pitchFull,      pitchHalf,         pitchHalf,     0},
      {         0,           size, size + chromaSize,     0},
      {      size,     chromaSize,        chromaSize,     0},
      {    layout,         layout,            layout, Pitch}
    };
  }
  case NvmmColorFormat::YUV422: {
    const size_t size = BlockSizeBytes(pitchFull * lumaHeight);
    const size_t chromaSize = BlockSizeBytes(pitchHalf * lumaHeight);
    return {
      format,
      3,
      { lumaWidth, lumaWidthHalf,     lumaWidthHalf,     0},
      {lumaHeight,    lumaHeight,        lumaHeight,     0},
      { pitchFull,     pitchHalf,         pitchHalf,     0},
      {         0,          size, size + chromaSize,     0},
      {      size,    chromaSize,        chromaSize,     0},
      {    layout,        layout,            layout, Pitch}
    };
  }
  case NvmmColorFormat::NV12:
  case NvmmColorFormat::NV12_ER:
  case NvmmColorFormat::NV21:
  case NvmmColorFormat::NV21_ER:
  case NvmmColorFormat::NV12_709:
  case NvmmColorFormat::NV12_709_ER:
  case NvmmColorFormat::NV12_2020: {
    const size_t size = BlockSizeBytes(pitchFull * lumaHeight);
    const size_t chromaSize = BlockSizeBytes(pitchFull * lumaHeight / 2);
    return {
      format,
      2,
      { lumaWidth,  lumaWidthHalf,     0,     0},
      {lumaHeight, lumaHeightHalf,     0,     0},
      { pitchFull,      pitchHalf,     0,     0},
      {         0,           size,     0,     0},
      {      size,     chromaSize,     0,     0},
      {    layout,         layout, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::UYVY:
  case NvmmColorFormat::UYVY_ER:
  case NvmmColorFormat::VYUY:
  case NvmmColorFormat::VYUY_ER:
  case NvmmColorFormat::YUYV:
  case NvmmColorFormat::YUYV_ER:
  case NvmmColorFormat::YVYU:
  case NvmmColorFormat::YVYU_ER: {
    const size_t bytesPerPixel = 2;
    const size_t pitch = PitchSizeBytes(lumaWidth * bytesPerPixel);
    const size_t size = BlockSizeBytes(pitch * lumaHeight);
    return {
      format,
      1,
      { lumaWidth,     0,     0,     0},
      {lumaHeight,     0,     0,     0},
      {     pitch,     0,     0,     0},
      {         0,     0,     0,     0},
      {      size,     0,     0,     0},
      {    layout, Pitch, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::ABGR32:
  case NvmmColorFormat::XRGB32:
  case NvmmColorFormat::ARGB32:
  case NvmmColorFormat::SignedR16G16:
  case NvmmColorFormat::A32: {
    const size_t bytesPerPixel = 4;
    const size_t pitch = PitchSizeBytes(lumaWidth * bytesPerPixel);
    const size_t size = BlockSizeBytes(pitch * lumaHeight);
    return {
      format,
      1,
      { lumaWidth,     0,     0,     0},
      {lumaHeight,     0,     0,     0},
      {     pitch,     0,     0,     0},
      {         0,     0,     0,     0},
      {      size,     0,     0,     0},
      {    layout, Pitch, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::NV12_10LE:
  case NvmmColorFormat::NV12_10LE_709:
  case NvmmColorFormat::NV12_10LE_709_ER:
  case NvmmColorFormat::NV12_10LE_2020:
  case NvmmColorFormat::NV21_10LE:
  case NvmmColorFormat::NV12_12LE:
  case NvmmColorFormat::NV12_12LE_2020:
  case NvmmColorFormat::NV21_12LE: {
    const size_t bytesPerPixel = 2;
    const size_t pitch = PitchSizeBytes(lumaWidth * bytesPerPixel);
    const size_t size = BlockSizeBytes(pitch * lumaHeight);
    const size_t chromaPitch = PitchSizeBytes(lumaWidthHalf * bytesPerPixel);
    const size_t chromaSize = BlockSizeBytes(chromaPitch * lumaHeightHalf);
    return {
      format,
      2,
      { lumaWidth,  lumaWidthHalf,     0,     0},
      {lumaHeight, lumaHeightHalf,     0,     0},
      {     pitch,    chromaPitch,     0,     0},
      {         0,           size,     0,     0},
      {      size,     chromaSize,     0,     0},
      {    layout,         layout, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::YUV444: {
    const size_t pitch = PitchSizeBytes(lumaWidth);
    const size_t size = BlockSizeBytes(pitch * lumaHeight);
    return {
      format,
      3,
      { lumaWidth,  lumaWidth,  lumaWidth,     0},
      {lumaHeight, lumaHeight, lumaHeight,     0},
      {     pitch,      pitch,      pitch,     0},
      {         0,       size,   size * 2,     0},
      {      size,       size,       size,     0},
      {    layout,     layout,     layout, Pitch}
    };
  }
  case NvmmColorFormat::GRAY8: {
    const size_t pitch = PitchSizeBytes(lumaWidth);
    const size_t size = BlockSizeBytes(pitch * lumaHeight);
    return {
      format,
      1,
      { lumaWidth,     0,     0,     0},
      {lumaHeight,     0,     0,     0},
      {     pitch,     0,     0,     0},
      {         0,     0,     0,     0},
      {      size,     0,     0,     0},
      {    layout, Pitch, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::NV16:
  case NvmmColorFormat::NV16_ER:
  case NvmmColorFormat::NV16_709:
  case NvmmColorFormat::NV16_709_ER: {
    const size_t pitch = PitchSizeBytes(lumaWidth);
    const size_t size = BlockSizeBytes(pitch * lumaHeight);
    const size_t chromaPitch = PitchSizeBytes(lumaWidthHalf * 2); // chroma 2bpp
    const size_t chromaSize = BlockSizeBytes(chromaPitch * lumaHeight);
    return {
      format,
      2,
      { lumaWidth, lumaWidthHalf,     0,     0},
      {lumaHeight,    lumaHeight,     0,     0},
      {     pitch,   chromaPitch,     0,     0},
      {         0,          size,     0,     0},
      {      size,    chromaSize,     0,     0},
      {    layout,        layout, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::NV16_10LE: {
    // 10 bits per pixel, aligned to 16 bits
    const size_t bytesPerLumaPixel = 2;
    const size_t lumaPitch = PitchSizeBytes(lumaWidth * bytesPerLumaPixel);
    const size_t lumaSize = BlockSizeBytes(lumaPitch * lumaHeight);
    const size_t chromaWidth = lumaWidth / 2;
    // Two 10-bit components per pixel (UV), aligned to 16 bits each
    const size_t bytesPerChromaPixel = 4;
    const size_t chromaPitch = PitchSizeBytes(chromaWidth * bytesPerChromaPixel);
    const size_t chromaSize = BlockSizeBytes(chromaPitch * lumaHeight);

    return {
      format,
      2, // 2 planes: Y, UV
      { lumaWidth, chromaWidth,     0,     0},
      {lumaHeight,  lumaHeight,     0,     0},
      { lumaPitch, chromaPitch,     0,     0},
      {         0,    lumaSize,     0,     0},
      {  lumaSize,  chromaSize,     0,     0},
      {    layout,      layout, Pitch, Pitch}  // Assuming the same layout for both planes
    };
  }
  case NvmmColorFormat::NV24:
  case NvmmColorFormat::NV24_ER:
  case NvmmColorFormat::NV24_709:
  case NvmmColorFormat::NV24_709_ER: {
    const size_t lumaPitch = PitchSizeBytes(lumaWidth);
    const size_t lumaSize = BlockSizeBytes(lumaPitch * lumaHeight);
    // UV pairs, each pair is 2 bytes and covers the entire width
    const size_t chromaPitch = PitchSizeBytes(lumaWidth * 2);
    const size_t chromaSize = BlockSizeBytes(chromaPitch * lumaHeight); // Full height, same as luma

    return {
      format,
      2,
      { lumaWidth,   lumaWidth,     0,     0},
      {lumaHeight,  lumaHeight,     0,     0},
      { lumaPitch, chromaPitch,     0,     0},
      {         0,    lumaSize,     0,     0},
      {  lumaSize,  chromaSize,     0,     0},
      {    layout,      layout, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::NV24_10LE:
  case NvmmColorFormat::NV24_10LE_709:
  case NvmmColorFormat::NV24_10LE_709_ER:
  case NvmmColorFormat::NV24_10LE_2020:
  case NvmmColorFormat::NV24_12LE_2020: {
    const size_t bytesPerLumaPixel = 2; // 10 bits per pixel, aligned to 16 bits
    const size_t lumaPitch = PitchSizeBytes(lumaWidth * bytesPerLumaPixel);
    const size_t lumaSize = BlockSizeBytes(lumaPitch * lumaHeight);
    const size_t bytesPerChromaPixel = 4; // Each UV pair is 20 bits, aligned to 32 bits (4 bytes)
    const size_t chromaPitch = PitchSizeBytes(lumaWidth * bytesPerChromaPixel);
    const size_t chromaSize = BlockSizeBytes(chromaPitch * lumaHeight);

    return {
      format,
      2, // 2 planes: Y, UV
      { lumaWidth,   lumaWidth,     0,     0},
      {lumaHeight,  lumaHeight,     0,     0},
      { lumaPitch, chromaPitch,     0,     0},
      {         0,    lumaSize,     0,     0},
      {  lumaSize,  chromaSize,     0,     0},
      {    layout,      layout, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::RGBA_10_10_10_2_709:
  case NvmmColorFormat::RGBA_10_10_10_2_2020:
  case NvmmColorFormat::BGRA_10_10_10_2_709:
  case NvmmColorFormat::BGRA_10_10_10_2_2020: {
    const size_t bytesPerPixel = 4; // Each pixel is 32 bits, packed into 4 bytes (10, 10, 10, 2)
    const size_t pitch = PitchSizeBytes(lumaWidth * bytesPerPixel);
    const size_t size = BlockSizeBytes(pitch * lumaHeight);

    return {
      format,
      1,
      { lumaWidth,     0,     0,     0},
      {lumaHeight,     0,     0,     0},
      {     pitch,     0,     0,     0},
      {         0,     0,     0,     0},
      {      size,     0,     0,     0},
      {    layout, Pitch, Pitch, Pitch}
    };
  }
  case NvmmColorFormat::Invalid:
  default: {
    return {
      format,
      0,
      {    0,     0,     0,     0},
      {    0,     0,     0,     0},
      {    0,     0,     0,     0},
      {    0,     0,     0,     0},
      {    0,     0,     0,     0},
      {Pitch, Pitch, Pitch, Pitch}
    };
  }
  }
}
