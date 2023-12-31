#include <cuda_runtime.h>

#include <iostream>

// CUDA forward declaration
void helloFromGPU(cudaStream_t stream);

int main() {
  std::cout << "Hello from CPU!\n";

  // Call a function that runs on the GPU
  helloFromGPU(nullptr);

  return 0;
}
