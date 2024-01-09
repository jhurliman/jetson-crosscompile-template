#include "cuda/CudaBufferPitched2D.hpp"

#include "cuda_expected.hpp"

tl::expected<std::unique_ptr<CudaBufferPitched2D>, StreamError> CudaBufferPitched2D::create(
  size_t widthBytes, size_t height) {
  CUDA_EXPECTED_INIT();
  void* data;
  size_t pitch;
  CUDA_EXPECTED(cudaMallocPitch(&data, &pitch, widthBytes, height));
  return std::unique_ptr<CudaBufferPitched2D>(
    new CudaBufferPitched2D(data, widthBytes, height, pitch));
}

CudaBufferPitched2D::~CudaBufferPitched2D() {
  cudaFree(data_);
}

size_t CudaBufferPitched2D::size() const {
  return widthBytes_ * height_;
}

size_t CudaBufferPitched2D::capacity() const {
  return pitch_ * height_;
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
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  const auto copyType = src.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream));
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, src, count, cudaMemcpyHostToDevice, stream));
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  const auto copyType = dst.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream));
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, cudaMemcpyDeviceToHost, stream));
  if (synchronize) { CUDA_OPTIONAL(cudaStreamSynchronize(stream)); }
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  CUDA_OPTIONAL(cudaMemsetAsync(data_, int(value), count, stream));
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
  void* dstPtr = static_cast<std::byte*>(data_) + dstY * pitch() + dstX;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcY * src.pitch() + srcX;
  const auto copyType = src.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  CUDA_OPTIONAL(
    cudaMemcpy2DAsync(dstPtr, pitch(), srcPtr, src.pitch(), widthBytes, height, copyType, stream));
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyFromHost2D(const void* src,
  size_t srcPitch,
  size_t dstX,
  size_t dstY,
  size_t widthBytes,
  size_t height,
  cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstY * pitch() + dstX;
  CUDA_OPTIONAL(cudaMemcpy2DAsync(
    dstPtr, pitch(), src, srcPitch, widthBytes, height, cudaMemcpyHostToDevice, stream));
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
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstY * dst.pitch() + dstX;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcY * pitch() + srcX;
  const auto copyType = dst.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
  CUDA_OPTIONAL(
    cudaMemcpy2DAsync(dstPtr, dst.pitch(), srcPtr, pitch(), widthBytes, height, copyType, stream));
  return {};
}

std::optional<StreamError> CudaBufferPitched2D::copyToHost2D(void* dst,
  size_t dstPitch,
  size_t srcX,
  size_t srcY,
  size_t widthBytes,
  size_t height,
  cudaStream_t stream,
  bool synchronize) const {
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcY * pitch() + srcX;
  CUDA_OPTIONAL(cudaMemcpy2DAsync(
    dstPtr, dstPitch, srcPtr, pitch(), widthBytes, height, cudaMemcpyDeviceToHost, stream));
  if (synchronize) { CUDA_OPTIONAL(cudaStreamSynchronize(stream)); }
  return {};
}

CudaBufferPitched2D::CudaBufferPitched2D(void* data, size_t widthBytes, size_t height, size_t pitch)
  : data_(data),
    widthBytes_(widthBytes),
    height_(height),
    pitch_(pitch) {}
