#pragma once

#include "CudaBuffer2D.hpp"

#include <tl/expected.hpp>

/**
 * @brief A lightweight wrapper around a CudaBuffer2D that provides a view of the buffer as a 2D
 * 8-bit unsigned integer array.
 */
class CudaImageView {
public:
  static size_t BYTES_PER_ELEMENT = 1;

  static tl::expected<CudaImageView, ArgumentError> fromBuffer(
    CudaBuffer2D& buffer, size_t x, size_t y, size_t widthBytes, size_t height) {
    if (x >= buffer.width() || y >= buffer.height()) {
      return tl::unexpected(ArgumentError("x or y out of bounds");
    }
    if (x + widthBytes > buffer.width() || y + height > buffer.height()) {
      return tl::unexpected(ArgumentError("widthBytes or height out of bounds");
    }
    return CudaImageView(buffer.cudaData(), x, y, widthBytes, height, buffer.isDevice());
  }

  size_t widthBytes() const { return widthBytes_; }

  size_t height() const { return height_; }

  size_t pitch() const { return pitch_; }

  size_t byteLength() const { return width_ * height_ * BYTES_PER_ELEMENT; }

  size_t byteOffset() const { return y_ * pitch_ + x_ * BYTES_PER_ELEMENT; }

  std::byte* data() { return static_cast<std::byte*>(data_) + byteOffset(); }

  const std::byte* data() const { return static_cast<const std::byte*>(data_) + byteOffset(); }

  bool isDevice() const { return isDevice_; }

  tl::expected < CudaImageView,
    subview(size_t x, size_t y, size_t widthBytes, size_t height) const {
    if (x >= widthBytes_ || y >= height_) { return CudaImageView(data_, 0, 0, 0, 0); }
    return CudaImageView(
      data_, x_ + x, y_ + y, std::min(widthBytes_ - x, widthBytes), std::min(height_ - y, height));
  }

private:
  size_t x_;
  size_t y_;
  size_t widthBytes_;
  size_t height_;
  size_t pitch_;
  void* data_;
  bool isDevice_;

  CudaImageView(
    void* data, size_t x, size_t y, size_t widthBytes, size_t height, size_t pitch, bool isDevice)
    : x_(x),
      y_(y),
      widthBytes_(widthBytes),
      height_(height),
      pitch_(pitch),
      data_(data),
      isDevice_(isDevice) {}
};
