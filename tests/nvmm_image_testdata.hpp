#pragma once

#include "nvmm/NvmmImageBuffer.hpp"
#include "nvmm/types.hpp"

using PlaneArray = NvmmImageBuffer::PlaneArray;

constexpr NvmmBufferLayout BlockLinear = NvmmBufferLayout::BlockLinear;
constexpr NvmmBufferLayout Pitch = NvmmBufferLayout::Pitch;

struct AllocationTest {
  size_t width;
  size_t height;
  NvmmColorFormat format;
  size_t numPlanes;
  PlaneArray widths;
  PlaneArray heights;
  PlaneArray pitches;
  PlaneArray offsets;
  PlaneArray sizes;
  NvmmImageBuffer::PlaneLayoutArray layouts;
};

constexpr AllocationTest OnePlane(NvmmColorFormat format) {
  return {
    16,
    16,
    format,
    1,
    {         16,     0,     0,     0},
    {         16,     0,     0,     0},
    {         64,     0,     0,     0},
    {          0,     0,     0,     0},
    {     131072,     0,     0,     0},
    {BlockLinear, Pitch, Pitch, Pitch}
  };
}

constexpr AllocationTest TwoPlanes(NvmmColorFormat format) {
  return {
    16,
    16,
    format,
    2,
    {         16,           8,     0,     0},
    {         16,           8,     0,     0},
    {         64,          64,     0,     0},
    {          0,      131072,     0,     0},
    {     131072,      131072,     0,     0},
    {BlockLinear, BlockLinear, Pitch, Pitch}
  };
}

constexpr AllocationTest TwoPlanesFull(NvmmColorFormat format) {
  return {
    16,
    16,
    format,
    2,
    {         16,          16,     0,     0},
    {         16,          16,     0,     0},
    {         64,          64,     0,     0},
    {          0,      131072,     0,     0},
    {     131072,      131072,     0,     0},
    {BlockLinear, BlockLinear, Pitch, Pitch}
  };
}

constexpr AllocationTest ThreePlanes(NvmmColorFormat format) {
  return {
    16,
    16,
    format,
    3,
    {         16,           8,           8,     0},
    {         16,           8,           8,     0},
    {         64,          64,          64,     0},
    {          0,      131072,      262144,     0},
    {     131072,      131072,      131072,     0},
    {BlockLinear, BlockLinear, BlockLinear, Pitch}
  };
}

constexpr AllocationTest ThreePlanesFull(NvmmColorFormat format) {
  return {
    16,
    16,
    format,
    3,
    {         16,          16,          16,     0},
    {         16,          16,          16,     0},
    {         64,          64,          64,     0},
    {          0,      131072,      262144,     0},
    {     131072,      131072,      131072,     0},
    {BlockLinear, BlockLinear, BlockLinear, Pitch}
  };
}

static const std::vector<AllocationTest> NVMM_IMAGE_ALLOCATION_TESTS = {
  ThreePlanes(NvmmColorFormat::YUV420),
  ThreePlanes(NvmmColorFormat::YVU420),
 // ThreePlanes(NvmmColorFormat::YUV422), // Unsupported on Jetpack 32.6
  ThreePlanes(NvmmColorFormat::YUV420_ER),
  ThreePlanes(NvmmColorFormat::YVU420_ER),
  TwoPlanes(NvmmColorFormat::NV12),
  TwoPlanes(NvmmColorFormat::NV12_ER),
  TwoPlanes(NvmmColorFormat::NV21),
  TwoPlanes(NvmmColorFormat::NV21_ER),
  OnePlane(NvmmColorFormat::UYVY),
  OnePlane(NvmmColorFormat::UYVY_ER),
  OnePlane(NvmmColorFormat::VYUY),
  OnePlane(NvmmColorFormat::VYUY_ER),
  OnePlane(NvmmColorFormat::YUYV),
  OnePlane(NvmmColorFormat::YUYV_ER),
  OnePlane(NvmmColorFormat::YVYU),
  OnePlane(NvmmColorFormat::YVYU_ER),
  OnePlane(NvmmColorFormat::ABGR32),
  OnePlane(NvmmColorFormat::XRGB32),
  OnePlane(NvmmColorFormat::ARGB32),
  TwoPlanes(NvmmColorFormat::NV12_10LE),
  TwoPlanes(NvmmColorFormat::NV12_10LE_709),
  TwoPlanes(NvmmColorFormat::NV12_10LE_709_ER),
  TwoPlanes(NvmmColorFormat::NV12_10LE_2020),
  TwoPlanes(NvmmColorFormat::NV21_10LE),
  TwoPlanes(NvmmColorFormat::NV12_12LE),
  TwoPlanes(NvmmColorFormat::NV12_12LE_2020),
  TwoPlanes(NvmmColorFormat::NV21_12LE),
  ThreePlanes(NvmmColorFormat::YUV420_709),
  ThreePlanes(NvmmColorFormat::YUV420_709_ER),
  TwoPlanes(NvmmColorFormat::NV12_709),
  TwoPlanes(NvmmColorFormat::NV12_709_ER),
  ThreePlanes(NvmmColorFormat::YUV420_2020),
  TwoPlanes(NvmmColorFormat::NV12_2020),
  OnePlane(NvmmColorFormat::SignedR16G16),
  OnePlane(NvmmColorFormat::A32),
  ThreePlanesFull(NvmmColorFormat::YUV444),
  OnePlane(NvmmColorFormat::GRAY8),
 // Chroma plane is half width and full height
  {16,
   16,        NvmmColorFormat::NV16,
   2, {16, 8, 0, 0},
   {16, 16, 0, 0},
   {64, 64, 0, 0},
   {0, 131072, 0, 0},
   {131072, 131072, 0, 0},
   {BlockLinear, BlockLinear, Pitch, Pitch}},
 // Chroma plane is half width and full height
  {16,
   16,   NvmmColorFormat::NV16_10LE,
   2, {16, 8, 0, 0},
   {16, 16, 0, 0},
   {64, 64, 0, 0},
   {0, 131072, 0, 0},
   {131072, 131072, 0, 0},
   {BlockLinear, BlockLinear, Pitch, Pitch}},
  TwoPlanesFull(NvmmColorFormat::NV24),
 // TwoPlanes(NvmmColorFormat::NV24_10LE), // Unsupported on Jetpack 32.6
  // Chroma plane is half width and full height
  {16,
   16,     NvmmColorFormat::NV16_ER,
   2, {16, 8, 0, 0},
   {16, 16, 0, 0},
   {64, 64, 0, 0},
   {0, 131072, 0, 0},
   {131072, 131072, 0, 0},
   {BlockLinear, BlockLinear, Pitch, Pitch}},
  TwoPlanesFull(NvmmColorFormat::NV24_ER),
 // Chroma plane is half width and full height
  {16,
   16,    NvmmColorFormat::NV16_709,
   2, {16, 8, 0, 0},
   {16, 16, 0, 0},
   {64, 64, 0, 0},
   {0, 131072, 0, 0},
   {131072, 131072, 0, 0},
   {BlockLinear, BlockLinear, Pitch, Pitch}},
  TwoPlanesFull(NvmmColorFormat::NV24_709),
 // Chroma plane is half width and full height
  {16,
   16, NvmmColorFormat::NV16_709_ER,
   2, {16, 8, 0, 0},
   {16, 16, 0, 0},
   {64, 64, 0, 0},
   {0, 131072, 0, 0},
   {131072, 131072, 0, 0},
   {BlockLinear, BlockLinear, Pitch, Pitch}},
  TwoPlanesFull(NvmmColorFormat::NV24_709_ER),
  TwoPlanesFull(NvmmColorFormat::NV24_10LE_709),
  TwoPlanesFull(NvmmColorFormat::NV24_10LE_709_ER),
  TwoPlanesFull(NvmmColorFormat::NV24_10LE_2020),
  TwoPlanesFull(NvmmColorFormat::NV24_12LE_2020),
  TwoPlanesFull(NvmmColorFormat::NV24_12LE_2020),
  OnePlane(NvmmColorFormat::RGBA_10_10_10_2_709),
  OnePlane(NvmmColorFormat::RGBA_10_10_10_2_2020),
 // OnePlane(NvmmColorFormat::BGRA_10_10_10_2_709), // Unsupported on Jetpack 32.6
  // OnePlane(NvmmColorFormat::BGRA_10_10_10_2_2020), // Unsupported on Jetpack 32.6
};
