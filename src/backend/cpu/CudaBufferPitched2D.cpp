#include "cuda/CudaBufferPitched2D.hpp"

#include <cstring>

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)

tl::expected<std::unique_ptr<CudaBufferPitched2D>, StreamError> CudaBufferPitched2D::create(
  size_t widthBytes, size_t height) {
  if (widthBytes == 0 || height == 0) {
    return std::unique_ptr<CudaBufferPitched2D>(
      new CudaBufferPitched2D(nullptr, widthBytes, height, widthBytes));
  }
  const size_t byteSize = widthBytes * height;
  void* data = ::malloc(byteSize);
  if (data == nullptr) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "malloc failed"});
  }
  return std::unique_ptr<CudaBufferPitched2D>(
    new CudaBufferPitched2D(data, widthBytes, height, widthBytes));
}

CudaBufferPitched2D::~CudaBufferPitched2D() {
  ::free(data_);
}

size_t CudaBufferPitched2D::size() const {
  return widthBytes_ * height_;
}

size_t CudaBufferPitched2D::widthBytes() const {
  return widthBytes_;
}

size_t CudaBufferPitched2D::height() const {
  return height_;
}

size_t CudaBufferPitched2D::pitch() const {
  return pitch_;
}

void* CudaBufferPitched2D::cudaData() {
  return data_;
}

const void* CudaBufferPitched2D::cudaData() const {
  return data_;
}

std::optional<StreamError> CudaBufferPitched2D::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  ::memcpy(dstPtr, src, count);
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  (void)stream;
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  (void)stream;
  (void)synchronize;
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  ::memcpy(dstPtr, srcPtr, count);
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  (void)stream;
  std::byte* dstPtr = static_cast<std::byte*>(data_);
  ::memset(dstPtr, int(value), count);
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyFrom2D(const CudaBuffer2D& src,
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

std::optional<StreamError> CudaBufferPitched2D::copyFromHost2D(const void* src,
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
    const auto err = copyFromHost(src, (dstY + y) * pitch() + dstX, widthBytes, stream);
    if (err) { return err; }
  }
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyTo2D(CudaBuffer2D& dst,
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

std::optional<StreamError> CudaBufferPitched2D::copyToHost2D(void* dst,
  size_t srcX,
  size_t srcY,
  size_t widthBytes,
  size_t height,
  cudaStream_t stream,
  bool synchronize) const {
  (void)synchronize;

  // Check if this can be done as a single copy
  if (pitch() == widthBytes) {
    return copyToHost(dst, srcY * pitch() + srcX, widthBytes * height, stream, false);
  }

  // Otherwise, copy row by row
  for (size_t y = 0; y < height; ++y) {
    const auto err = copyToHost(dst, (srcY + y) * pitch() + srcX, widthBytes, stream, false);
    if (err) { return err; }
  }

  return {};
}

CudaBufferPitched2D::CudaBufferPitched2D(void* data, size_t widthBytes, size_t height, size_t pitch)
  : data_(data),
    widthBytes_(widthBytes),
    height_(height),
    pitch_(pitch) {}

// NOLINTEND(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
