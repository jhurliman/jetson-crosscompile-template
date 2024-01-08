#include "cuda/CudaImageView.hpp"

tl::expected<CudaImageView, ArgumentError> CudaImageView::fromBuffer(
  CudaBuffer2D& buffer, size_t x, size_t y, size_t widthBytes, size_t height) {
  if (x >= buffer.widthBytes() || y >= buffer.height()) {
    return tl::unexpected(ArgumentError("x or y out of bounds"));
  }
  if (x + widthBytes > buffer.widthBytes() || y + height > buffer.height()) {
    return tl::unexpected(ArgumentError("widthBytes or height out of bounds"));
  }
  return CudaImageView(
    buffer.cudaData(), x, y, widthBytes, height, buffer.pitch(), buffer.isDevice());
}

size_t CudaImageView::widthBytes() const {
  return widthBytes_;
}

size_t CudaImageView::height() const {
  return height_;
}

size_t CudaImageView::pitch() const {
  return pitch_;
}

size_t CudaImageView::byteLength() const {
  return widthBytes_ * height_;
}

size_t CudaImageView::byteOffset() const {
  return y_ * pitch_ + x_;
}

std::byte* CudaImageView::data() {
  return static_cast<std::byte*>(data_) + byteOffset();
}

const std::byte* CudaImageView::data() const {
  return static_cast<const std::byte*>(data_) + byteOffset();
}

bool CudaImageView::isDevice() const {
  return isDevice_;
}

tl::expected<CudaImageView, ArgumentError> CudaImageView::subview(
  size_t x, size_t y, size_t widthBytes, size_t height) const {
  if (x >= widthBytes_ || y >= height_) {
    return tl::unexpected(ArgumentError("x or y out of bounds"));
  }
  return CudaImageView(data_, x_ + x, y_ + y, widthBytes, height, pitch_, isDevice_);
}

CudaImageView::CudaImageView(
  void* data, size_t x, size_t y, size_t widthBytes, size_t height, size_t pitch, bool isDevice)
  : x_(x),
    y_(y),
    widthBytes_(widthBytes),
    height_(height),
    pitch_(pitch),
    data_(data),
    isDevice_(isDevice) {}
