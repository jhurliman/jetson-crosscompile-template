#include "cuda/CudaBufferDevice2D.hpp"

#include "cuda_expected.hpp"

tl::expected<std::unique_ptr<CudaBufferDevice2D>, StreamError> CudaBufferDevice2D::create(
  size_t widthBytes, size_t height, cudaStream_t stream) {
  return CudaBufferDevice::create(widthBytes * height, stream).map([&](auto&& bufferPtr) { // NOLINT
    return std::unique_ptr<CudaBufferDevice2D>(
      new CudaBufferDevice2D(std::move(bufferPtr), widthBytes, height));
  });
}

CudaBufferDevice2D::CudaBufferDevice2D(
  std::unique_ptr<CudaBufferDevice> buffer, size_t widthBytes, size_t height)
  : buffer_(std::move(buffer)),
    widthBytes_(widthBytes),
    height_(height) {}

size_t CudaBufferDevice2D::size() const {
  return buffer_->size();
}

size_t CudaBufferDevice2D::capacity() const {
  return buffer_->size();
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
  size_t widthBytes,
  size_t height,
  cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(buffer_->cudaData()) + dstY * pitch() + dstX;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcY * src.pitch() + srcX;
  const auto copyType = src.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  CUDA_OPTIONAL(
    cudaMemcpy2DAsync(dstPtr, pitch(), srcPtr, src.pitch(), widthBytes, height, copyType, stream));
  return {};
}

std::optional<StreamError> CudaBufferDevice2D::copyFromHost2D(const void* src,
  size_t srcPitch,
  size_t dstX,
  size_t dstY,
  size_t widthBytes,
  size_t height,
  cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(buffer_->cudaData()) + dstY * pitch() + dstX;
  CUDA_OPTIONAL(cudaMemcpy2DAsync(
    dstPtr, pitch(), src, srcPitch, widthBytes, height, cudaMemcpyHostToDevice, stream));
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
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstY * dst.pitch() + dstX;
  const void* srcPtr = static_cast<const std::byte*>(buffer_->cudaData()) + srcY * pitch() + srcX;
  const auto copyType = dst.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
  CUDA_OPTIONAL(
    cudaMemcpy2DAsync(dstPtr, dst.pitch(), srcPtr, pitch(), widthBytes, height, copyType, stream));
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
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(buffer_->cudaData()) + srcY * pitch() + srcX;
  CUDA_OPTIONAL(cudaMemcpy2DAsync(
    dstPtr, dstPitch, srcPtr, pitch(), widthBytes, height, cudaMemcpyDeviceToHost, stream));
  if (synchronize) { CUDA_OPTIONAL(cudaStreamSynchronize(stream)); }
  return {};
}
