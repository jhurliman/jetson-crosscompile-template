#include "cuda/CudaBufferHostPinned.hpp"

#include <cuda_runtime.h>

tl::expected<std::unique_ptr<CudaBufferHostPinned>, StreamError> CudaBufferHostPinned::create(
  size_t byteSize, uint flags) {
  void* data;
  const cudaError_t err = cudaMallocHost(&data, byteSize, flags);
  if (err != cudaSuccess) { return tl::make_unexpected(StreamError{err, cudaErrorMessage(err)}); }
  return std::unique_ptr<CudaBufferHostPinned>(
    new CudaBufferHostPinned(static_cast<std::byte*>(data), byteSize));
}

CudaBufferHostPinned::CudaBufferHostPinned(std::byte* data, size_t byteSize)
  : size_(byteSize),
    data_(data) {}

CudaBufferHostPinned::~CudaBufferHostPinned() {
  cudaFreeHost(data_);
}

size_t CudaBufferHostPinned::size() const {
  return size_;
}

void* CudaBufferHostPinned::cudaData() {
  return data_;
}

const void* CudaBufferHostPinned::cudaData() const {
  return data_;
}

const std::byte* CudaBufferHostPinned::hostData() const {
  return data_;
}

std::byte* CudaBufferHostPinned::hostData() {
  return data_;
}

std::optional<StreamError> CudaBufferHostPinned::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  const auto copyType = src.isDevice() ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
  const cudaError_t err = cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream);
  if (err != cudaSuccess) { return StreamError{err, cudaErrorMessage(err)}; }
  return {};
}

std::optional<StreamError> CudaBufferHostPinned::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const cudaError_t err = cudaMemcpyAsync(dstPtr, src, count, cudaMemcpyHostToHost, stream);
  if (err != cudaSuccess) { return StreamError{err, cudaErrorMessage(err)}; }
  return {};
}

std::optional<StreamError> CudaBufferHostPinned::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  const auto copyType = dst.isDevice() ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
  const cudaError_t err = cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream);
  if (err != cudaSuccess) { return StreamError{err, cudaErrorMessage(err)}; }
  return {};
}

std::optional<StreamError> CudaBufferHostPinned::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  cudaError_t err = cudaMemcpyAsync(dstPtr, srcPtr, count, cudaMemcpyHostToHost, stream);
  if (err != cudaSuccess) { return StreamError{err, cudaErrorMessage(err)}; }
  if (synchronize) {
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { return StreamError{err, cudaErrorMessage(err)}; }
  }
  return {};
}

std::optional<StreamError> CudaBufferHostPinned::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  const cudaError_t err = cudaMemsetAsync(data_, int(value), count, stream);
  if (err != cudaSuccess) { return StreamError{err, cudaErrorMessage(err)}; }
  return {};
}
