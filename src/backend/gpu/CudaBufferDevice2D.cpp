#include "cuda/CudaBufferDevice2D.hpp"

tl::expected<std::unique_ptr<CudaBufferDevice2D>, StreamError> CudaBufferDevice2D::create(
  size_t width, size_t height, cudaStream_t stream) {
  return CudaBufferDevice::create(width * height, stream).map([&](auto&& bufferPtr) { // NOLINT
    return std::unique_ptr<CudaBufferDevice2D>(
      new CudaBufferDevice2D(std::move(bufferPtr), width, height));
  });
}

CudaBufferDevice2D::CudaBufferDevice2D(
  std::unique_ptr<CudaBufferDevice> buffer, size_t width, size_t height)
  : buffer_(std::move(buffer)),
    width_(width),
    height_(height) {}

size_t CudaBufferDevice2D::size() const {
  return buffer_->size();
}

size_t CudaBufferDevice2D::width() const {
  return width_;
}

size_t CudaBufferDevice2D::height() const {
  return height_;
}

size_t CudaBufferDevice2D::pitch() const {
  return width_;
}

void* CudaBufferDevice2D::cudaData() {
  return buffer_->cudaData();
}

const void* CudaBufferDevice2D::cudaData() const {
  return buffer_->cudaData();
}

std::optional<StreamError> CudaBufferDevice2D::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  return buffer_->copyFrom(src, srcOffset, dstOffset, count, stream);
}

std::optional<StreamError> CudaBufferDevice2D::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  return buffer_->copyFromHost(src, dstOffset, count, stream);
}

std::optional<StreamError> CudaBufferDevice2D::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  return buffer_->copyTo(dst, srcOffset, dstOffset, count, stream);
}

std::optional<StreamError> CudaBufferDevice2D::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  return buffer_->copyToHost(dst, srcOffset, count, stream, synchronize);
}

std::optional<StreamError> CudaBufferDevice2D::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  return buffer_->memset(value, count, stream);
}

std::optional<StreamError> CudaBufferDevice2D::copyFrom2D(const CudaBuffer2D& src,
  size_t srcX,
  size_t srcY,
  size_t dstX,
  size_t dstY,
  size_t width,
  size_t height,
  cudaStream_t stream) {
  // Check if this can be done as a single copy
  if (src.pitch() == pitch() && src.pitch() == width) {
    return copyFrom(src, srcY * src.pitch() + srcX, dstY * pitch() + dstX, width * height, stream);
  }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; ++y) {
    const auto err =
      copyFrom(src, (srcY + y) * src.pitch() + srcX, (dstY + y) * pitch() + dstX, width, stream);
    if (err) { return err; }
  }
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyFromHost2D(
  const void* src, size_t dstX, size_t dstY, size_t width, size_t height, cudaStream_t stream) {
  // Check if this can be done as a single copy
  if (pitch() == width) { return copyFromHost(src, dstY * pitch() + dstX, width * height, stream); }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; ++y) {
    const auto err = copyFromHost(src, (dstY + y) * pitch() + dstX, width, stream);
    if (err) { return err; }
  }
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyTo2D(CudaBuffer2D& dst,
  size_t srcX,
  size_t srcY,
  size_t dstX,
  size_t dstY,
  size_t width,
  size_t height,
  cudaStream_t stream) const {
  // Check if this can be done as a single copy
  if (pitch() == dst.pitch() && pitch() == width) {
    return copyTo(dst, srcY * pitch() + srcX, dstY * dst.pitch() + dstX, width * height, stream);
  }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; ++y) {
    const auto err =
      copyTo(dst, (srcY + y) * pitch() + srcX, (dstY + y) * dst.pitch() + dstX, width, stream);
    if (err) { return err; }
  }
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyToHost2D(void* dst,
  size_t srcX,
  size_t srcY,
  size_t width,
  size_t height,
  cudaStream_t stream,
  bool synchronize) const {
  // Check if this can be done as a single copy
  if (pitch() == width) {
    return copyToHost(dst, srcY * pitch() + srcX, width * height, stream, false);
  }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; ++y) {
    const auto err = copyToHost(dst, (srcY + y) * pitch() + srcX, width, stream, false);
    if (err) { return err; }
  }

  if (synchronize) { cudaStreamSynchronize(stream); }
  return {};
}
