#pragma once

#include "CudaBuffer2D.hpp"

#include <tl/expected.hpp>

struct ArgumentError;

/**
 * @brief A lightweight wrapper around a CudaBuffer2D that provides a view of the buffer as a 2D
 * 8-bit unsigned integer array.
 */
class CudaImageView {
public:
  static tl::expected<CudaImageView, ArgumentError> fromBuffer(
    CudaBuffer2D& buffer, size_t x, size_t y, size_t widthBytes, size_t height);

  size_t widthBytes() const;

  size_t height() const;

  size_t pitch() const;

  size_t byteLength() const;

  size_t byteOffset() const;

  std::byte* data();

  const std::byte* data() const;

  bool isDevice() const;

  tl::expected<CudaImageView, ArgumentError> subview(
    size_t x, size_t y, size_t widthBytes, size_t height) const;

private:
  size_t x_;
  size_t y_;
  size_t widthBytes_;
  size_t height_;
  size_t pitch_;
  void* data_;
  bool isDevice_;

  CudaImageView(
    void* data, size_t x, size_t y, size_t widthBytes, size_t height, size_t pitch, bool isDevice);
};
