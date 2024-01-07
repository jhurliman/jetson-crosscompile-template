#include "cuda/CudaBufferUnified.hpp"

#include "cuda/stream.hpp"
#include "cuda_expected.hpp"

tl::expected<std::unique_ptr<CudaBufferUnified>, StreamError> CudaBufferUnified::create(
  size_t byteSize, CudaMemAttachFlag flag) {
  CUDA_EXPECTED_INIT();
  void* data;
  CUDA_EXPECTED(cudaMallocManaged(&data, byteSize, uint(flag)));
  return std::unique_ptr<CudaBufferUnified>(new CudaBufferUnified(data, byteSize));
}

tl::expected<std::unique_ptr<CudaBufferUnified>, StreamError> CudaBufferUnified::createFromHostData(
  const void* data, size_t byteSize, cudaStream_t stream, CudaMemAttachFlag flag) {
  auto res = CudaBufferUnified::create(byteSize, flag);
  if (!res) { return tl::make_unexpected(res.error()); }
  auto bufferPtr = std::move(res.value());

  // Copy host data to this buffer
  const auto err = bufferPtr->copyFromHost(data, 0, byteSize, stream);
  if (err) { return tl::make_unexpected(err.value()); }
  return bufferPtr;
}

CudaBufferUnified::CudaBufferUnified(void* data, size_t byteSize)
  : size_(byteSize),
    data_(static_cast<std::byte*>(data)) {}

CudaBufferUnified::~CudaBufferUnified() {
  cudaFree(data_);
}

size_t CudaBufferUnified::size() const {
  return size_;
}

void* CudaBufferUnified::cudaData() {
  return data_;
}

const void* CudaBufferUnified::cudaData() const {
  return data_;
}

const std::byte* CudaBufferUnified::hostData() const {
  return data_;
}

std::byte* CudaBufferUnified::hostData() {
  return data_;
}

std::optional<StreamError> CudaBufferUnified::copyFrom(
  const CudaBuffer& src, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(src.cudaData()) + srcOffset;
  const auto copyType = src.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream));
  return {};
}

std::optional<StreamError> CudaBufferUnified::copyFromHost(
  const void* src, size_t dstOffset, size_t count, cudaStream_t stream) {
  void* dstPtr = static_cast<std::byte*>(data_) + dstOffset;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, src, count, cudaMemcpyHostToDevice, stream));
  return {};
}

std::optional<StreamError> CudaBufferUnified::copyTo(
  CudaBuffer& dst, size_t srcOffset, size_t dstOffset, size_t count, cudaStream_t stream) const {
  void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstOffset;
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  const auto copyType = dst.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, copyType, stream));
  return {};
}

std::optional<StreamError> CudaBufferUnified::copyToHost(
  void* dst, size_t srcOffset, size_t count, cudaStream_t stream, bool synchronize) const {
  void* dstPtr = static_cast<std::byte*>(dst);
  const void* srcPtr = static_cast<const std::byte*>(data_) + srcOffset;
  CUDA_OPTIONAL(cudaMemcpyAsync(dstPtr, srcPtr, count, cudaMemcpyDeviceToHost, stream));
  if (synchronize) { CUDA_OPTIONAL(cudaStreamSynchronize(stream)); }
  return {};
}

std::optional<StreamError> CudaBufferUnified::memset(
  std::byte value, size_t count, cudaStream_t stream) {
  CUDA_OPTIONAL(cudaMemsetAsync(data_, int(value), count, stream));
  return {};
}

std::optional<StreamError> CudaBufferUnified::prefetch(
  CudaMemAttachFlag flag, cudaStream_t stream) {
  // On Tegra platforms, `cudaStreamAttachMemAsync()` is used to prefetch
  // Unified Memory to CPU or GPU by transitioning the cudaMemAttach flag from
  // cudaMemAttachHost to cudaMemAttachGlobal/cudaMemAttachSingle or vice versa.
  // See:
  // <https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#effective-usage-of-unified-memory-on-tegra>
  //
  // On other platforms, `cudaMemPrefetchAsync()` is used to prefetch Unified
  // Memory to CPU or GPU by passing in the destination device as the dstDevice
  // parameter.
  //
  // Note that `cudaStreamAttachMemAsync()`-based prefetching is not implemented
  // in QNX and will have no effect.
#ifdef USE_T210
  CUDA_OPTIONAL(cudaStreamAttachMemAsync(stream, data_, size_, uint(flag)));
#else
  int dstDevice = cudaCpuDeviceId;
  if (flag != CudaMemAttachFlag::Host) { CUDA_OPTIONAL(cudaGetDevice(&dstDevice)); }
  CUDA_OPTIONAL(cudaMemPrefetchAsync(data_, size_, dstDevice, stream));
#endif
  return {};
}
