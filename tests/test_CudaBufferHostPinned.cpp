#include "cuda/CudaBufferDevice.hpp"
#include "cuda/CudaBufferHostPinned.hpp"
#include "cuda/stream.hpp"
#include "requires.hpp"

#include <catch2/catch.hpp>

TEST_CASE("Allocates CudaBufferHostPinned", "[cudabuffer]") {
  constexpr size_t COUNT = 128;

  std::unique_ptr<CudaBufferHostPinned> bufPtr;
  for (size_t i = 0; i < COUNT; i++) {
    bufPtr = REQUIRE_EXPECTED(CudaBufferHostPinned::create(i));
    REQUIRE(bufPtr);
    REQUIRE(bufPtr->size() == i);
    REQUIRE_FALSE(bufPtr->isDevice());
    if (i == 0) {
      REQUIRE(bufPtr->cudaData() == nullptr);
    } else {
      REQUIRE(bufPtr->cudaData() != nullptr);
    }
  }

  // Test large allocation (16MB)
  bufPtr = REQUIRE_EXPECTED(CudaBufferHostPinned::create(16 * 1024 * 1024));
  REQUIRE(bufPtr);
  REQUIRE(bufPtr->size() == 16 * 1024 * 1024);
  REQUIRE_FALSE(bufPtr->isDevice());
  REQUIRE(bufPtr->cudaData() != nullptr);

  bufPtr.reset();
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferHostPinned copyFromHost/copyToHost", "[cudabuffer]") {
  // To test copyFromHost we need to allocate a CudaBufferHostPinned, copy data from the host to it,
  // then copy the data back to the host and check it. Do this for a variety of offset/count values
  // and with and without a stream

  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  cudaStream_t stream = REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal));

  std::unique_ptr<CudaBufferHostPinned> buf =
    REQUIRE_EXPECTED(CudaBufferHostPinned::create(SIZE_BYTES, CudaHostPinnedFlags::Mapped));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, 1, stream));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, 1, stream));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, i, stream));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, i, stream));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      buf->memset(std::byte(0), SIZE_BYTES, stream);

      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    buf->memset(std::byte(0), SIZE_BYTES, stream);

    REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);

    buf->memset(std::byte(0), SIZE_BYTES, nullptr);

    REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE(srcVec == dstVec);
  }

  buf.reset();
  REQUIRE_NO_ERROR(cuda::destroyStream(stream));
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferHostPinned copyFrom CudaBufferHostPinned", "[cudabuffer]") {
  // To test copyFrom we need to allocate two CudaBufferHostPinneds, copy data from one to the
  // other, then copy the data back to the host and check it. Do this for a variety of offset/count
  // values and with and without a stream

  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  cudaStream_t stream = REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal));

  std::unique_ptr<CudaBufferHostPinned> bufSrc =
    REQUIRE_EXPECTED(CudaBufferHostPinned::create(SIZE_BYTES, CudaHostPinnedFlags::Mapped));
  std::unique_ptr<CudaBufferHostPinned> bufDst =
    REQUIRE_EXPECTED(CudaBufferHostPinned::create(SIZE_BYTES, CudaHostPinnedFlags::Mapped));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1, stream));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, 1, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1, stream));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i, stream));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i, stream));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      bufDst->memset(std::byte(0), SIZE_BYTES, stream);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), SIZE_BYTES, stream);
    bufDst->memset(std::byte(0), SIZE_BYTES, stream);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);

    bufSrc->memset(std::byte(0), SIZE_BYTES, stream);
    bufDst->memset(std::byte(0), SIZE_BYTES, nullptr);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  REQUIRE_NO_ERROR(cuda::destroyStream(stream));
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferHostPinned copyTo CudaBufferHostPinned", "[cudabuffer]") {
  // To test copyTo we need to allocate two CudaBufferHostPinneds, copy data from one to the other,
  // then copy the data back to the host and check it. Do this for a variety of offset/count values
  // and with and without a stream

  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  cudaStream_t stream = REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal));

  std::unique_ptr<CudaBufferHostPinned> bufSrc =
    REQUIRE_EXPECTED(CudaBufferHostPinned::create(SIZE_BYTES, CudaHostPinnedFlags::Mapped));
  std::unique_ptr<CudaBufferHostPinned> bufDst =
    REQUIRE_EXPECTED(CudaBufferHostPinned::create(SIZE_BYTES, CudaHostPinnedFlags::Mapped));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1, stream));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, 1, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1, stream));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i, stream));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i, stream));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      bufDst->memset(std::byte(0), SIZE_BYTES, stream);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), SIZE_BYTES, stream);
    bufDst->memset(std::byte(0), SIZE_BYTES, stream);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);

    bufSrc->memset(std::byte(0), SIZE_BYTES, stream);
    bufDst->memset(std::byte(0), SIZE_BYTES, nullptr);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  REQUIRE_NO_ERROR(cuda::destroyStream(stream));
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferHostPinned copyTo CudaBufferDevice", "[cudabuffer]") {
  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  cudaStream_t stream = REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal));

  std::unique_ptr<CudaBufferHostPinned> bufSrc =
    REQUIRE_EXPECTED(CudaBufferHostPinned::create(SIZE_BYTES, CudaHostPinnedFlags::Mapped));
  std::unique_ptr<CudaBufferDevice> bufDst =
    REQUIRE_EXPECTED(CudaBufferDevice::create(SIZE_BYTES, stream));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1, stream));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, 1, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1, stream));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i, stream));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i, stream));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      bufDst->memset(std::byte(0), SIZE_BYTES, stream);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), SIZE_BYTES, stream);
    bufDst->memset(std::byte(0), SIZE_BYTES, stream);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);

    bufSrc->memset(std::byte(0), SIZE_BYTES, stream);
    bufDst->memset(std::byte(0), SIZE_BYTES, nullptr);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  REQUIRE_NO_ERROR(cuda::destroyStream(stream));
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

void runCudaBufferHostPinnedCopyFromCudaBufferDevice(CudaHostPinnedFlags flags) {
  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
  }

  cudaStream_t stream = REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal));

  std::unique_ptr<CudaBufferDevice> bufSrc =
    REQUIRE_EXPECTED(CudaBufferDevice::create(SIZE_BYTES, stream));
  std::unique_ptr<CudaBufferHostPinned> bufDst =
    REQUIRE_EXPECTED(CudaBufferHostPinned::create(SIZE_BYTES, flags));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1, stream));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, 1, stream));
      REQUIRE_NO_ERROR(cuda::synchronizeStream(stream));
      REQUIRE(bufDst->hostData()[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i, stream));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, i, stream));
      REQUIRE_NO_ERROR(cuda::synchronizeStream(stream));
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), bufDst->hostData()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      bufDst->memset(std::byte(0), SIZE_BYTES, stream);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(cuda::synchronizeStream(stream));
      REQUIRE(
        std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), bufDst->hostData() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), SIZE_BYTES, stream);
    bufDst->memset(std::byte(0), SIZE_BYTES, stream);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(cuda::synchronizeStream(stream));
    REQUIRE(std::equal(srcVec.begin(), srcVec.end(), bufDst->hostData()));

    bufSrc->memset(std::byte(0), SIZE_BYTES, stream);
    bufDst->memset(std::byte(0), SIZE_BYTES, nullptr);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, nullptr));
    REQUIRE_NO_ERROR(cuda::synchronizeStream(nullptr));
    REQUIRE(std::equal(srcVec.begin(), srcVec.end(), bufDst->hostData()));
  }

  bufSrc.reset();
  bufDst.reset();
  REQUIRE_NO_ERROR(cuda::destroyStream(stream));
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferHostPinned copyFrom CudaBufferDevice", "[cudabuffer]") {
  auto flags = GENERATE(CudaHostPinnedFlags::Default,
    CudaHostPinnedFlags::Portable,
    CudaHostPinnedFlags::Mapped,
    CudaHostPinnedFlags::Portable | CudaHostPinnedFlags::Mapped);
  runCudaBufferHostPinnedCopyFromCudaBufferDevice(flags);
}

TEST_CASE("CudaBufferHostPinned memset", "[cudabuffer]") {
  constexpr size_t SIZE_BYTES = 255; // 256 - 1 since 0xFF is used as a sentinel value

  cudaStream_t stream = REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal));

  std::unique_ptr<CudaBufferHostPinned> buf =
    REQUIRE_EXPECTED(CudaBufferHostPinned::create(SIZE_BYTES, CudaHostPinnedFlags::Mapped));

  std::vector<std::byte> vec(SIZE_BYTES, std::byte(0xFF));

  for (size_t i = 0; i < SIZE_BYTES; i++) {
    REQUIRE_NO_ERROR(buf->memset(std::byte(i), i, stream));
    REQUIRE_NO_ERROR(buf->copyToHost(vec.data(), 0, i, stream));

    for (size_t j = 0; j < i; j++) {
      CAPTURE(vec);
      REQUIRE(vec[j] == std::byte(i));
    }

    // Ensure the rest of the bytes are not modified
    for (size_t j = i; j < SIZE_BYTES; j++) {
      CAPTURE(vec);
      CAPTURE(i);
      CAPTURE(j);
      REQUIRE(vec[j] != std::byte(j));
    }
  }

  buf.reset();
  REQUIRE_NO_ERROR(cuda::destroyStream(stream));
  REQUIRE_NO_ERROR(cuda::checkLastError());
}
