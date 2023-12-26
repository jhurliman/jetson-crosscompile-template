#include <iostream>

#include <cuda_runtime.h>

// CUDA forward declaration
void helloFromGPU(cudaStream_t stream);

int main() {
  std::cout << "Hello from CPU!" << std::endl;

  // Call a function that runs on the GPU
  helloFromGPU(nullptr);

  return 0;
}
