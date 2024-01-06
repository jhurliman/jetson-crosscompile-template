#include "arithmetic.hpp"
#include "cuda/CudaBufferUnified.hpp"
#include "cuda/stream.hpp"

#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

template<typename T>
std::unique_ptr<CudaBufferUnified> Vec2Cuda(const std::vector<T>& vec, cudaStream_t stream) {
  auto res = CudaBufferUnified::createFromHostData(vec.data(), vec.size() * sizeof(T), stream);
  REQUIRE(res);
  return std::move(res.value());
}

template<typename T> CudaArrayView<T> Buf2View(CudaBufferUnified& buf) {
  const size_t n = buf.size() / sizeof(T);
  auto res = CudaArrayView<T>::fromBuffer(buf, n);
  REQUIRE(res);
  return std::move(res.value());
}

std::unique_ptr<CudaBufferUnified> CudaBuf(size_t size) {
  auto res = CudaBufferUnified::create(size);
  REQUIRE(res);
  return std::move(res.value());
}

TEST_CASE("Addition works correctly", "[math]") {
  auto streamRes = createStream("test", StreamPriority::Normal);
  REQUIRE(streamRes);
  cudaStream_t stream = streamRes.value();

  auto bufA = Vec2Cuda<int64_t>({1, 2, 3, 4, 5}, stream);
  auto bufB = Vec2Cuda<int64_t>({6, 7, 8, 9, 10}, stream);
  auto bufC = CudaBuf(5 * sizeof(int64_t));

  const auto viewA = Buf2View<int64_t>(*bufA);
  const auto viewB = Buf2View<int64_t>(*bufB);
  auto viewC = Buf2View<int64_t>(*bufC);

  auto err = addVectors(viewA, viewB, viewC, stream);
  REQUIRE(!err);

  std::vector<int64_t> vecC(5);
  REQUIRE(!bufC->copyToHost(vecC.data(), 0, vecC.size() * sizeof(int64_t), stream));

  destroyStream(stream);

  std::vector<int64_t> vecExpected{7, 9, 11, 13, 15};
  REQUIRE(vecC == vecExpected);
}
