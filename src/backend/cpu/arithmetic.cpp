#include "cuda/arithmetic.hpp"

#include "tbb_kernels.hpp"

__global__ void addVectorsKernel(
  CUDA_KERNEL_ARGS, const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  for (uint i = index; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

__global__ void addVectorsKernel(
  CUDA_KERNEL_ARGS, const double* a, const double* b, double* c, size_t n) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  for (uint i = index; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

std::optional<StreamError> addVectors(const CudaArrayView<int64_t>& a,
  const CudaArrayView<int64_t>& b,
  CudaArrayView<int64_t>& c,
  cudaStream_t stream) {
  (void)stream;
  if (a.size() != b.size() || a.size() != c.size()) {
    return StreamError{cudaErrorInvalidValue, "Array sizes do not match"};
  }

  const size_t n = a.size();
  const uint blockSize = 256;
  const uint numBlocks = (uint(n) + blockSize - 1) / blockSize;
  const dim3 gridDim = {numBlocks, 1, 1};
  const dim3 blockDim = {blockSize, 1, 1};
  CUDA_KERNEL(addVectorsKernel, gridDim, blockDim, a.data(), b.data(), c.data(), n);
  return std::nullopt;
}

std::optional<StreamError> addVectors(const CudaArrayView<double>& a,
  const CudaArrayView<double>& b,
  CudaArrayView<double>& c,
  cudaStream_t stream) {
  (void)stream;
  if (a.size() != b.size() || a.size() != c.size()) {
    return StreamError{cudaErrorInvalidValue, "Array sizes do not match"};
  }

  const size_t n = a.size();
  const uint blockSize = 256;
  const uint numBlocks = (uint(n) + blockSize - 1) / blockSize;
  CUDA_KERNEL(addVectorsKernel, numBlocks, blockSize, a.data(), b.data(), c.data(), n);
  return std::nullopt;
}
