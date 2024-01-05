#include "arithmetic.hpp"
#include "cuda/CudaBufferUnified.hpp"
#include "cuda/stream.hpp"

#include <iostream>

template<typename T>
std::unique_ptr<CudaBufferUnified> Vec2Cuda(const std::vector<T>& vec, cudaStream_t stream) {
  auto res = CudaBufferUnified::createFromHostData(vec.data(), vec.size() * sizeof(T), stream);
  if (!res) { throw std::runtime_error(res.error().errorMessage); }
  return std::move(res.value());
}

std::unique_ptr<CudaBufferUnified> CudaBuf(size_t size) {
  auto res = CudaBufferUnified::create(size);
  if (!res) { throw std::runtime_error(res.error().errorMessage); }
  return std::move(res.value());
}

template<typename T> CudaArrayView<T> Buf2View(CudaBufferUnified& buf) {
  const size_t n = buf.size() / sizeof(T);
  auto res = CudaArrayView<T>::fromBuffer(buf, n);
  if (!res) { throw std::runtime_error(res.error().errorMessage); }
  return std::move(res.value());
}

int main() {
  std::cout << "Adding two vectors\n";

  auto streamRes = createStream("test", StreamPriority::Normal);
  if (!streamRes) {
    std::cerr << "Error: " << streamRes.error().errorMessage << "\n";
    return 1;
  }
  cudaStream_t stream = streamRes.value();

  const auto bufA = Vec2Cuda<int64_t>({1, 2, 3, 4, 5}, stream);
  const auto bufB = Vec2Cuda<int64_t>({6, 7, 8, 9, 10}, stream);
  const auto bufC = CudaBuf(5 * sizeof(int64_t));

  const auto viewA = Buf2View<int64_t>(*bufA);
  const auto viewB = Buf2View<int64_t>(*bufB);
  auto viewC = Buf2View<int64_t>(*bufC);

  auto err = addVectors(viewA, viewB, viewC, stream);
  if (err) {
    std::cerr << "Error: " << err.value().errorMessage << "\n";
    return 1;
  }

  std::vector<int64_t> vecC(5);
  err = bufC->copyToHost(vecC.data(), 0, vecC.size(), stream);

  destroyStream(stream);

  // print the first element, then a comma and a space for the rest
  std::cout << "Result: " << vecC[0];
  for (size_t i = 1; i < vecC.size(); ++i) {
    std::cout << ", " << vecC[i];
  }
  std::cout << "\n";

  std::cout << "Expected: 7, 9, 11, 13, 15\n";
  return 0;
}
