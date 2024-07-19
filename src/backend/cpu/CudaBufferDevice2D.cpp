#include "cuda/CudaBufferDevice2D.hpp"

#include <cstring>

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)

tl::expected<std::unique_ptr<CudaBufferDevice2D>, StreamError> CudaBufferDevice2D::create(
  size_t widthBytes, size_t height, cudaStream_t stream) {
  (void)stream;

  // Create the underlying buffer
  const size_t byteSize = widthBytes * height;
  tl::expected<std::unique_ptr<CudaBufferDevice>, StreamError> bufferRes =
    CudaBufferDevice::create(byteSize, stream);
  if (!bufferRes) { return tl::make_unexpected(bufferRes.error()); }

  return std::unique_ptr<CudaBufferDevice2D>(
    new CudaBufferDevice2D(std::move(bufferRes.value()), widthBytes, height));
}

size_t CudaBufferDevice2D::size() const {
  return widthBytes_ * height_;
}

size_t CudaBufferDevice2D::capacity() const {
  return widthBytes_ * height_;
}

size_t CudaBufferDevice2D::widthBytes() const {
  return widthBytes_;
}

size_t CudaBufferDevice2D::height() const {
  return height_;
}

size_t CudaBufferDevice2D::pitch() const {
  return widthBytes_;
}

void* CudaBufferDevice2D::cudaData() {
  return buffer_->cudaData();
}

const void* CudaBufferDevice2D::cudaData() const {
  return buffer_->cudaData();
}

std::optional<StreamError> CudaBufferDevice2D::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(buffer_->cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(buffer_->cudaData()) + dstOffset;
  ::memcpy(dstPtr, src, count);
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(buffer_->cudaData()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  (void)stream;
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(buffer_->cudaData()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  (void)stream;
  std::byte* dstPtr = static_cast<std::byte*>(buffer_->cudaData());
  ::memset(dstPtr, int(value), count);
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyFrom2D(const CudaBuffer2D& src,
  size_t srcX,
  size_t srcY,
  size_t dstX,
  size_t dstY,
  size_t widthBytes,
  size_t height,
  cudaStream_t stream) {
  // Check if this can be done as a single copy
  if (src.pitch() == pitch() && src.pitch() == widthBytes) {
    return copyFrom(
      src, srcY * src.pitch() + srcX, dstY * pitch() + dstX, widthBytes * height, stream);
  }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; ++y) {
    const auto err = copyFrom(
      src, (srcY + y) * src.pitch() + srcX, (dstY + y) * pitch() + dstX, widthBytes, stream);
    if (err) { return err; }
  }
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyFromHost2D(const void* src,
  size_t srcPitch,
  size_t dstX,
  size_t dstY,
  size_t widthBytes,
  size_t height,
  cudaStream_t stream) {
  // Check if this can be done as a single copy
  if (widthBytes == srcPitch && pitch() == widthBytes) {
    return copyFromHost(src, dstY * pitch() + dstX, widthBytes * height, stream);
  }

  if (srcPitch < widthBytes) { return StreamError{cudaErrorInvalidValue, "srcPitch < widthBytes"}; }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; ++y) {
    const void* srcPtr = static_cast<const std::byte*>(src) + y * srcPitch;
    const auto err = copyFromHost(srcPtr, (dstY + y) * pitch() + dstX, widthBytes, stream);
    if (err) { return err; }
  }
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyTo2D(CudaBuffer2D& dst,
  size_t srcX,
  size_t srcY,
  size_t dstX,
  size_t dstY,
  size_t widthBytes,
  size_t height,
  cudaStream_t stream) const {
  // Check if this can be done as a single copy
  if (pitch() == dst.pitch() && pitch() == widthBytes) {
    return copyTo(
      dst, srcY * pitch() + srcX, dstY * dst.pitch() + dstX, widthBytes * height, stream);
  }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; ++y) {
    const auto err =
      copyTo(dst, (srcY + y) * pitch() + srcX, (dstY + y) * dst.pitch() + dstX, widthBytes, stream);
    if (err) { return err; }
  }
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyToHost2D(void* dst,
  size_t dstPitch,
  size_t srcX,
  size_t srcY,
  size_t widthBytes,
  size_t height,
  cudaStream_t stream,
  bool synchronize) const {
  (void)synchronize;

  // Check if this can be done as a single copy
  if (dstPitch == widthBytes && pitch() == dstPitch) {
    return copyToHost(dst, srcY * pitch() + srcX, widthBytes * height, stream, false);
  }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; y++) {
    void* dstPtr = static_cast<std::byte*>(dst) + y * dstPitch;
    const auto err = copyToHost(dstPtr, (srcY + y) * pitch() + srcX, widthBytes, stream, false);
    if (err) { return err; }
  }

  return {};
}

CudaBufferDevice2D::CudaBufferDevice2D(
  std::unique_ptr<CudaBufferDevice> buffer, size_t widthBytes, size_t height)
  : buffer_(std::move(buffer)),
    widthBytes_(widthBytes),
    height_(height) {}
