#include "hello.hpp"

#include <iostream>

int main() {
  std::cout << "Hello from CPU!\n";

  // Call a function that runs on the GPU
  helloFromGPU(nullptr);

  return 0;
}
