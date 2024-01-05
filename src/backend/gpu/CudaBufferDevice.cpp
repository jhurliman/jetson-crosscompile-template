#include "cuda/CudaBufferDevice.hpp"

#include "cuda_common.hpp"

tl::expected<std::unique_ptr<CudaBufferDevice>, StreamError> CudaBufferDevice::create(
  size_t byteSize, cudaStream_t stream) {
  void* data;
#if HAS_CUDA_11_2
  const cudaError_t err = cudaMallocAsync(&data, byteSize, stream);
  if (err != cudaSuccess) { return tl::make_unexpected(StreamError{err, CudaErrorMessage(err)}); }
#else
  (void)stream; // unused parameter (for CUDA < 11.2)
  const cudaError_t err = cudaMalloc(&data, byteSize);
  if (err != cudaSuccess) { return tl::make_unexpected(StreamError{err, CudaErrorMessage(err)}); }
#endif
  return std::unique_ptr<CudaBufferDevice>(new CudaBufferDevice(data, byteSize, stream));
}

CudaBufferDevice::CudaBufferDevice(void* data, size_t byteSize, cudaStream_t stream)
  : size_(byteSize),
    data_(data),
    stream_(stream) {}

CudaBufferDevice::~CudaBufferDevice() {
#if HAS_CUDA_11_2
  cudaFreeAsync(data_, stream_);
#else
  (void)stream_; // unused parameter (for CUDA < 11.2)
  cudaFree(data_);
#endif
}

size_t CudaBufferDevice::size() const {
  return size_;
}

void* CudaBufferDevice::cudaData() {
  return data_;
}

const void* CudaBufferDevice::cudaData() const {
  return data_;
}

std::optional<StreamError> CudaBufferDevice::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  const auto copyType = src.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  const cudaError_t err = cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream);
  if (err != cudaSuccess) { return StreamError{err, CudaErrorMessage(err)}; }
  return {};
}

std::optional<StreamError> CudaBufferDevice::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const cudaError_t err = cudaMemcpyAsync(dstPtr, src, count, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) { return StreamError{err, CudaErrorMessage(err)}; }
  return {};
}

std::optional<StreamError> CudaBufferDevice::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  const auto copyType = dst.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
  const cudaError_t err = cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream);
  if (err != cudaSuccess) { return StreamError{err, CudaErrorMessage(err)}; }
  return {};
}

std::optional<StreamError> CudaBufferDevice::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  cudaError_t err = cudaMemcpyAsync(dstPtr, srcPtr, count, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) { return StreamError{err, CudaErrorMessage(err)}; }
  if (synchronize) {
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { return StreamError{err, CudaErrorMessage(err)}; }
  }
  return {};
}

std::optional<StreamError> CudaBufferDevice::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  const cudaError_t err = cudaMemsetAsync(data_, int(value), count, stream);
  if (err != cudaSuccess) { return StreamError{err, CudaErrorMessage(err)}; }
  return {};
}
