#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello from GPU!\n");
}

void helloFromGPU(cudaStream_t stream) {
    // Launch a kernel with a single thread to print from the GPU
    helloKernel<<<1, 1, 0, stream>>>();

    // Wait for the GPU to finish before returning to the host
    cudaStreamSynchronize(stream);
}
