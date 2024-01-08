#include "cuda/arithmetic.hpp"

#include <algorithm>

std::optional<StreamError> addVectors(const CudaArrayView<int64_t>& a,
  const CudaArrayView<int64_t>& b,
  CudaArrayView<int64_t>& c,
  cudaStream_t stream) {
  (void)stream;
  std::transform(a.data(), a.data() + a.size(), b.data(), c.data(), std::plus<>());
  return std::nullopt;
}

std::optional<StreamError> addVectors(const CudaArrayView<double>& a,
  const CudaArrayView<double>& b,
  CudaArrayView<double>& c,
  cudaStream_t stream) {
  (void)stream;
  std::transform(a.data(), a.data() + a.size(), b.data(), c.data(), std::plus<>());
  return std::nullopt;
}
