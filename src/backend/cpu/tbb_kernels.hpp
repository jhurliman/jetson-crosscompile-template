#pragma once

#include <tbb/parallel_for.h>

#include <cstdint>

#define __global__

#define CUDA_KERNEL_ARGS                                                                           \
  const uint3 &threadIdx, const uint3 &blockIdx, const dim3 &blockDim, const dim3 &gridDim

#define CUDA_KERNEL(kernel, gridDim, blockDim, ...)                                                \
  launchKernelTBB(kernel, gridDim, blockDim, __VA_ARGS__)

struct dim3 {
  uint32_t x, y, z;

  dim3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) : x(x), y(y), z(z) {}
};

struct uint3 {
  uint32_t x, y, z;

  uint3(uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) : x(x), y(y), z(z) {}
};

template<typename... Args>
void launchKernelTBB(void (*kernel)(const uint3&, const uint3&, const dim3&, const dim3&, Args...),
  const dim3& gridDim,
  const dim3& blockDim,
  Args... args) {
  tbb::parallel_for(tbb::blocked_range<uint>(0, gridDim.x * blockDim.x),
    [=](const tbb::blocked_range<uint>& range) {
      for (uint globalThreadIdx = range.begin(); globalThreadIdx < range.end(); ++globalThreadIdx) {
        // Calculate the block index and thread index
        const uint3 blockIdx = {globalThreadIdx / blockDim.x, 0, 0};
        const uint3 threadIdx = {globalThreadIdx % blockDim.x, 0, 0};

        // Call the "kernel" function with the forwarded arguments
        kernel(threadIdx, blockIdx, blockDim, gridDim, args...);
      }
    });
}
