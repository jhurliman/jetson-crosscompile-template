#include "arithmetic.hpp"
#include "cuda/CudaBufferUnified.hpp"
#include "cuda/stream.hpp"
#include "requires.hpp"

#include <catch2/catch.hpp>

#include <iostream>
#include <vector>

template<typename T>
std::unique_ptr<CudaBufferUnified> Vec2Cuda(const std::vector<T>& vec, cudaStream_t stream) {
  return REQUIRE_EXPECTED(
    CudaBufferUnified::createFromHostData(vec.data(), vec.size() * sizeof(T), stream));
}

template<typename T> CudaArrayView<T> Buf2View(CudaBufferUnified& buf) {
  return REQUIRE_EXPECTED(CudaArrayView<T>::fromBuffer(buf, buf.size() / sizeof(T)));
}

TEST_CASE("Addition works correctly", "[math]") {
  cudaStream_t stream = REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal));

  auto bufA = Vec2Cuda<int64_t>({1, 2, 3, 4, 5}, stream);
  auto bufB = Vec2Cuda<int64_t>({6, 7, 8, 9, 10}, stream);
  auto bufC = REQUIRE_EXPECTED(CudaBufferUnified::create(5 * sizeof(int64_t)));

  const auto viewA = Buf2View<int64_t>(*bufA);
  const auto viewB = Buf2View<int64_t>(*bufB);
  auto viewC = Buf2View<int64_t>(*bufC);

  REQUIRE_NO_ERROR(addVectors(viewA, viewB, viewC, stream));

  std::vector<int64_t> vecC(5);
  REQUIRE_NO_ERROR(bufC->copyToHost(vecC.data(), 0, vecC.size() * sizeof(int64_t), stream));

  bufA.reset();
  bufB.reset();
  bufC.reset();
  REQUIRE_NO_ERROR(cuda::destroyStream(stream));

  std::vector<int64_t> vecExpected{7, 9, 11, 13, 15};
  REQUIRE(vecC == vecExpected);

  REQUIRE_NO_ERROR(cuda::checkLastError());
}
