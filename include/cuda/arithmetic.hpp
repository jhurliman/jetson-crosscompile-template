#pragma once

#include "cuda/CudaArrayView.hpp"
#include "cuda/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

std::optional<StreamError> addVectors(const CudaArrayView<int64_t>& a,
  const CudaArrayView<int64_t>& b,
  CudaArrayView<int64_t>& c,
  cudaStream_t stream);

std::optional<StreamError> addVectors(const CudaArrayView<double>& a,
  const CudaArrayView<double>& b,
  CudaArrayView<double>& c,
  cudaStream_t stream);
