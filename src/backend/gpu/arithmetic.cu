#include "arithmetic.hpp"

#include <device_launch_parameters.h>

__global__ void addVectorsKernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  for (uint i = index; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

__global__ void addVectorsKernel(const double* a, const double* b, double* c, size_t n) {
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
  if (a.size() != b.size() || a.size() != c.size()) {
    return StreamError{cudaErrorInvalidValue, "Array sizes do not match"};
  }
  const size_t n = a.size();
  uint blockSize = 256;
  uint numBlocks = (uint(n) + blockSize - 1) / blockSize;
  addVectorsKernel<<<numBlocks, blockSize, 0, stream>>>(a.data(), b.data(), c.data(), n);

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) { return StreamError{error, cudaErrorMessage(error)}; }

  return std::nullopt;
}

std::optional<StreamError> addVectors(const CudaArrayView<double>& a,
  const CudaArrayView<double>& b,
  CudaArrayView<double>& c,
  cudaStream_t stream) {
  if (a.size() != b.size() || a.size() != c.size()) {
    return StreamError{cudaErrorInvalidValue, "Array sizes do not match"};
  }
  const size_t n = a.size();
  uint blockSize = 256;
  uint numBlocks = (uint(n) + blockSize - 1) / blockSize;
  addVectorsKernel<<<numBlocks, blockSize, 0, stream>>>(a.data(), b.data(), c.data(), n);

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) { return StreamError{error, cudaErrorMessage(error)}; }

  return std::nullopt;
}
