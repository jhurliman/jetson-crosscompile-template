#include "cuda/CudaBufferDevice.hpp"
#include "cuda/stream.hpp"
#include "nvmm/NvmmBuffer.hpp"
#include "nvmm/session.hpp"
#include "requires.hpp"

#include <catch2/catch.hpp>

TEST_CASE("Allocates NvmmBuffer", "[nvmmbuffer]") {
  constexpr size_t COUNT = 128;

  std::unique_ptr<NvmmBuffer> bufPtr;
  for (size_t i = 0; i < COUNT; i++) {
    CAPTURE(i);
    bufPtr = REQUIRE_EXPECTED(NvmmBuffer::create(i));
    REQUIRE(bufPtr);
    REQUIRE(bufPtr->size() == i);
    if (i == 0) {
      REQUIRE(bufPtr->fd() == -1);
    } else {
      REQUIRE(bufPtr->fd() >= 0);
    }
  }

  // Test large allocation (16MB)
  bufPtr = REQUIRE_EXPECTED(NvmmBuffer::create(16 * 1024 * 1024));
  REQUIRE(bufPtr);
  REQUIRE(bufPtr->size() == 16 * 1024 * 1024);
  REQUIRE(bufPtr->fd() >= 0);

  bufPtr.reset();
}

TEST_CASE("NvmmBuffer copyFromHost/copyToHost", "[nvmmbuffer]") {
  // To test copyFromHost we need to allocate a NvmmBuffer, copy data from the host to it,
  // then copy the data back to the host and check it. Do this for a variety of offset/counts

  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  std::unique_ptr<NvmmBuffer> buf = REQUIRE_EXPECTED(NvmmBuffer::create(SIZE_BYTES));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, 1));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, 1));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, i));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, i));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      buf->memset(std::byte(0), SIZE_BYTES);

      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, SIZE_VEC - i));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    buf->memset(std::byte(0), SIZE_BYTES);

    REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, SIZE_VEC));
    REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, SIZE_VEC));
    REQUIRE(srcVec == dstVec);
  }

  buf.reset();
}

TEST_CASE("NvmmBuffer copyFrom NvmmBuffer", "[nvmmbuffer]") {
  // To test copyFrom we need to allocate two NvmmBuffers, copy data from one to the other,
  // then copy the data back to the host and check it. Do this for a variety of offset/counts

  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  std::unique_ptr<NvmmBuffer> bufSrc = REQUIRE_EXPECTED(NvmmBuffer::create(SIZE_BYTES));
  std::unique_ptr<NvmmBuffer> bufDst = REQUIRE_EXPECTED(NvmmBuffer::create(SIZE_BYTES));

  const bool makeSession = GENERATE(true, false);
  NvBufferSession session = makeSession ? REQUIRE_EXPECTED(nvmm::createSession()) : nullptr;

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, 1, session));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, i, session));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      bufDst->memset(std::byte(0), SIZE_BYTES);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, SIZE_VEC - i, session));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), SIZE_BYTES);
    bufDst->memset(std::byte(0), SIZE_BYTES);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC));
    REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, session));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  if (session) { nvmm::destroySession(session); }
}

TEST_CASE("NvmmBuffer copyTo NvmmBuffer", "[nvmmbuffer]") {
  // To test copyTo we need to allocate two NvmmBuffers, copy data from one to the other,
  // then copy the data back to the host and check it. Do this for a variety of offset/counts

  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  const bool makeSession = GENERATE(true, false);
  NvBufferSession session = makeSession ? REQUIRE_EXPECTED(nvmm::createSession()) : nullptr;

  std::unique_ptr<NvmmBuffer> bufSrc = REQUIRE_EXPECTED(NvmmBuffer::create(SIZE_BYTES));
  std::unique_ptr<NvmmBuffer> bufDst = REQUIRE_EXPECTED(NvmmBuffer::create(SIZE_BYTES));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, 1, session));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, i, session));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      bufDst->memset(std::byte(0), SIZE_BYTES);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, SIZE_VEC - i, session));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), SIZE_BYTES);
    bufDst->memset(std::byte(0), SIZE_BYTES);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC));
    REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, session));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  if (session) { nvmm::destroySession(session); }
}

TEST_CASE("NvmmBuffer copyFromCuda", "[nvmmbuffer]") {
  // To test copyFromCuda we need to allocate a NvmmBuffer and a CudaBuffer, copy data from the
  // CudaBuffer to the NvmmBuffer, then copy the data to the host and check it. Do this for a
  // variety of offset/counts

  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  const bool makeStream = GENERATE(true, false);
  cudaStream_t stream =
    makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

  std::unique_ptr<NvmmBuffer> buf = REQUIRE_EXPECTED(NvmmBuffer::create(SIZE_BYTES));
  std::unique_ptr<CudaBuffer> cudaBuf =
    REQUIRE_EXPECTED(CudaBufferDevice::create(SIZE_BYTES, stream));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(cudaBuf->copyFromHost(srcVec.data() + i, i, 1, stream));
      REQUIRE_NO_ERROR(buf->copyFromCuda(*cudaBuf, i, i, 1, stream));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, 1));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(cudaBuf->copyFromHost(srcVec.data(), 0, i, stream));
      REQUIRE_NO_ERROR(buf->copyFromCuda(*cudaBuf, 0, 0, i, stream));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, i));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      buf->memset(std::byte(0), SIZE_BYTES);

      REQUIRE_NO_ERROR(cudaBuf->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(buf->copyFromCuda(*cudaBuf, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, SIZE_VEC - i));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    buf->memset(std::byte(0), SIZE_BYTES);

    REQUIRE_NO_ERROR(cudaBuf->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(buf->copyFromCuda(*cudaBuf, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, SIZE_VEC));
    REQUIRE(srcVec == dstVec);
  }
}

TEST_CASE("NvmmBuffer copyToCuda", "[nvmmbuffer]") {
  // To test copyToCuda we need to allocate a NvmmBuffer and a CudaBuffer, copy data from the
  // NvmmBuffer to the CudaBuffer, then copy the data back to the host and check it. Do this for a
  // variety of offset/counts

  constexpr size_t SIZE_BYTES = 256;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  const bool makeStream = GENERATE(true, false);
  cudaStream_t stream =
    makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

  std::unique_ptr<NvmmBuffer> buf = REQUIRE_EXPECTED(NvmmBuffer::create(SIZE_BYTES));
  std::unique_ptr<CudaBuffer> cudaBuf =
    REQUIRE_EXPECTED(CudaBufferDevice::create(SIZE_BYTES, stream));

  SECTION("Copy one byte") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, 1));
      REQUIRE_NO_ERROR(buf->copyToCuda(*cudaBuf, i, i, 1, stream));
      REQUIRE_NO_ERROR(cudaBuf->copyToHost(dstVec.data() + i, i, 1, stream));
      // dstVec[i] == i
      CAPTURE(dstVec);
      REQUIRE(dstVec[i] == std::byte(i));
    }
  }

  SECTION("Copy multiple bytes") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, i));
      REQUIRE_NO_ERROR(buf->copyToCuda(*cudaBuf, 0, 0, i, stream));
      REQUIRE_NO_ERROR(cudaBuf->copyToHost(dstVec.data(), 0, i, stream));
      // dstVec[:i] == srcVec[:i]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      buf->memset(std::byte(0), SIZE_BYTES);

      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i));
      REQUIRE_NO_ERROR(buf->copyToCuda(*cudaBuf, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(cudaBuf->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    buf->memset(std::byte(0), SIZE_BYTES);

    REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, SIZE_VEC));
    REQUIRE_NO_ERROR(buf->copyToCuda(*cudaBuf, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(cudaBuf->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);
  }
}

TEST_CASE("NvmmBuffer memset", "[nvmmbuffer]") {
  constexpr size_t SIZE_BYTES = 255; // 256 - 1 since 0xFF is used as a sentinel value

  std::unique_ptr<NvmmBuffer> buf = REQUIRE_EXPECTED(NvmmBuffer::create(SIZE_BYTES));

  std::vector<std::byte> vec(SIZE_BYTES, std::byte(0xFF));

  for (size_t i = 0; i < SIZE_BYTES; i += 3) {
    REQUIRE_NO_ERROR(buf->memset(std::byte(i), i));
    REQUIRE_NO_ERROR(buf->copyToHost(vec.data(), 0, i));

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
}
