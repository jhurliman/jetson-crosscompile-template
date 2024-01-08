#include "cuda/CudaBufferDevice.hpp"
#include "cuda/CudaBufferPitched2D.hpp"
#include "cuda/stream.hpp"
#include "requires.hpp"

#include <catch2/catch.hpp>

TEST_CASE("Allocates CudaBufferPitched2D", "[cudabuffer]") {
  constexpr size_t WIDTHS = 16;
  constexpr size_t HEIGHTS = 16;

  std::unique_ptr<CudaBufferPitched2D> bufPtr;
  for (size_t height = 0; height < HEIGHTS; height++) {
    for (size_t width = 0; width < WIDTHS; width++) {
      bufPtr = REQUIRE_EXPECTED(CudaBufferPitched2D::create(width, height));
      REQUIRE(bufPtr);
      REQUIRE(bufPtr->widthBytes() == width);
      REQUIRE(bufPtr->height() == height);
      REQUIRE(bufPtr->size() == bufPtr->pitch() * height);
      REQUIRE(bufPtr->isDevice());
      if (width * height == 0) {
        REQUIRE(bufPtr->cudaData() == nullptr);
      } else {
        REQUIRE(bufPtr->pitch() >= width);
        REQUIRE(bufPtr->cudaData() != nullptr);
      }
    }
  }

  // Test large allocation (16MB)
  bufPtr = REQUIRE_EXPECTED(CudaBufferPitched2D::create(4096, 4096));
  REQUIRE(bufPtr);
  REQUIRE(bufPtr->size() == 4096 * 4096);
  REQUIRE(bufPtr->widthBytes() == 4096);
  REQUIRE(bufPtr->height() == 4096);
  REQUIRE(bufPtr->pitch() == 4096);
  REQUIRE(bufPtr->isDevice());
  REQUIRE(bufPtr->cudaData() != nullptr);

  bufPtr.reset();
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferPitched2D copyFromHost/copyToHost", "[cudabuffer]") {
  // To test copyFromHost we need to allocate a CudaBufferPitched2D, copy data from the host to it,
  // then copy the data back to the host and check it. Do this for a variety of offset/count values
  // and with and without a stream

  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 16;
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

  std::unique_ptr<CudaBufferPitched2D> buf =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(WIDTH_BYTES, HEIGHT));

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
      buf->memset(std::byte(0), buf->size(), stream);

      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    buf->memset(std::byte(0), buf->size(), stream);

    REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);
  }

  buf.reset();
  if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferPitched2D copyFrom CudaBufferPitched2D", "[cudabuffer]") {
  // To test copyFrom we need to allocate two CudaBufferPitched2Ds, copy data from one to the
  // other, then copy the data back to the host and check it. Do this for a variety of offset/count
  // values and with and without a stream

  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 16;
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

  std::unique_ptr<CudaBufferPitched2D> bufSrc =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(WIDTH_BYTES, HEIGHT));
  std::unique_ptr<CudaBufferPitched2D> bufDst =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(WIDTH_BYTES, HEIGHT));

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
      bufDst->memset(std::byte(0), bufDst->size(), stream);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), bufSrc->size(), stream);
    bufDst->memset(std::byte(0), bufDst->size(), stream);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferPitched2D copyTo CudaBufferPitched2D", "[cudabuffer]") {
  // To test copyTo we need to allocate two CudaBufferPitched2Ds, copy data from one to the other,
  // then copy the data back to the host and check it. Do this for a variety of offset/count values
  // and with and without a stream

  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 16;
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

  std::unique_ptr<CudaBufferPitched2D> bufSrc =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(WIDTH_BYTES, HEIGHT));
  std::unique_ptr<CudaBufferPitched2D> bufDst =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(WIDTH_BYTES, HEIGHT));

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
      bufDst->memset(std::byte(0), bufDst->size(), stream);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), bufSrc->size(), stream);
    bufDst->memset(std::byte(0), bufDst->size(), stream);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferPitched2D copyTo CudaBufferDevice", "[cudabuffer]") {
  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 16;
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

  std::unique_ptr<CudaBufferPitched2D> bufSrc =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(WIDTH_BYTES, HEIGHT));
  std::unique_ptr<CudaBufferDevice> bufDst =
    REQUIRE_EXPECTED(CudaBufferDevice::create(WIDTH_BYTES * HEIGHT, stream));

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
      bufDst->memset(std::byte(0), bufDst->size(), stream);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), bufSrc->size(), stream);
    bufDst->memset(std::byte(0), bufDst->size(), stream);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferPitched2D copyFrom CudaBufferDevice", "[cudabuffer]") {
  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 16;
  constexpr size_t SIZE_VEC = 128;

  const bool makeStream = GENERATE(true, false);
  cudaStream_t stream =
    makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  std::unique_ptr<CudaBufferDevice> bufSrc =
    REQUIRE_EXPECTED(CudaBufferDevice::create(WIDTH_BYTES * HEIGHT, stream));
  std::unique_ptr<CudaBufferPitched2D> bufDst =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(WIDTH_BYTES, HEIGHT));

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
      // srcVec[:i] == dstVec[:i]
      CAPTURE(srcVec);
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
    }
  }

  SECTION("Copy multiple bytes at an offset") {
    for (size_t i = 0; i < SIZE_VEC; i++) {
      bufDst->memset(std::byte(0), bufDst->size(), stream);

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, SIZE_VEC - i, stream));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
      // srcVec[i:] == dstVec[i:]
      CAPTURE(srcVec);
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), bufSrc->size(), stream);
    bufDst->memset(std::byte(0), bufDst->size(), stream);

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, stream));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
  if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferPitched2D memset", "[cudabuffer]") {
  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 15;
  constexpr size_t SIZE_BYTES = WIDTH_BYTES * HEIGHT;

  const bool makeStream = GENERATE(true, false);
  cudaStream_t stream =
    makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

  std::unique_ptr<CudaBufferPitched2D> buf =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(WIDTH_BYTES, HEIGHT));

  std::vector<std::byte> vec(SIZE_BYTES, std::byte(0xFF));

  for (size_t height = 0; height < HEIGHT; height++) {
    for (size_t width = 0; width < WIDTH_BYTES; width++) {
      const size_t i = height * WIDTH_BYTES + width;

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
  }

  buf.reset();
  if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

TEST_CASE("CudaBufferPitched2D copyFrom2D CudaBufferPitched2D", "[cudabuffer2d]") {
  // To test copyFrom2D we allocate three CudaBufferPitched2Ds, two with matching dimensions and
  // one with different height, width, and pitch. We copy data from the first to the second, then
  // copy the data back to the host and check it. Then from the first to the third, then back to the
  // host and check it. Do this for a variety of srcX, srcY, dstX, dstY, widthBytes, and height
  // values

  std::vector<std::byte> srcVec(256);
  std::vector<std::byte> dstVec(256);
  for (size_t i = 0; i < 256; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  const bool makeStream = GENERATE(true, false);
  cudaStream_t stream =
    makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

  std::unique_ptr<CudaBufferPitched2D> bufSrc1 =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(16, 16));
  std::unique_ptr<CudaBufferPitched2D> bufSrc2 =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(7, 3));
  std::unique_ptr<CudaBufferPitched2D> bufDst =
    REQUIRE_EXPECTED(CudaBufferPitched2D::create(16, 16));

  constexpr size_t PITCH = 512;

  CHECK(bufSrc1->pitch() == PITCH);
  CHECK(bufSrc1->size() == PITCH * 16);
  CHECK(bufSrc2->pitch() == PITCH);
  CHECK(bufSrc2->size() == PITCH * 3);
  CHECK(bufDst->pitch() == PITCH);
  CHECK(bufDst->size() == PITCH * 16);

  SECTION("Copy one byte") {
    for (size_t srcY = 0; srcY < 16; srcY++) {
      for (size_t srcX = 0; srcX < 16; srcX++) {
        for (size_t dstY = 0; dstY < 16; dstY++) {
          for (size_t dstX = 0; dstX < 16; dstX++) {
            const size_t srcI = srcY * bufSrc1->pitch() + srcX;
            const size_t dstI = dstY * bufDst->pitch() + dstX;
            REQUIRE_NO_ERROR(bufSrc1->copyFromHost(srcVec.data() + srcI, srcI, 1, stream));
            REQUIRE_NO_ERROR(bufDst->copyFrom2D(*bufSrc1, srcX, srcY, dstX, dstY, 1, 1, stream));
            REQUIRE_NO_ERROR(bufDst->copyToHost2D(dstVec.data(), 16, dstX, dstY, 1, 1, stream));
            // srcVec[srcI] == dstVec[dstI]
            CAPTURE(srcX, srcY, dstX, dstY);
            CAPTURE(srcVec);
            CAPTURE(dstVec);
            REQUIRE(srcVec[srcI] == dstVec[dstI]);
          }
        }
      }
    }
  }

  bufSrc1.reset();
  bufSrc2.reset();
  bufDst.reset();
  if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
  REQUIRE_NO_ERROR(cuda::checkLastError());
}

// std::optional<StreamError> CudaBufferPitched2D::copyFromHost2D(const void* src,
//   size_t srcPitch,
//   size_t dstX,
//   size_t dstY,
//   size_t widthBytes,
//   size_t height,
//   cudaStream_t stream) {
//   void* dstPtr = static_cast<std::byte*>(data_) + dstY * pitch() + dstX;
//   CUDA_OPTIONAL(cudaMemcpy2DAsync(
//     dstPtr, pitch(), src, srcPitch, widthBytes, height, cudaMemcpyHostToDevice, stream));
//   return {};
// }

// std::optional<StreamError> CudaBufferPitched2D::copyTo2D(CudaBuffer2D& dst,
//   size_t srcX,
//   size_t srcY,
//   size_t dstX,
//   size_t dstY,
//   size_t widthBytes,
//   size_t height,
//   cudaStream_t stream) const {
//   void* dstPtr = static_cast<std::byte*>(dst.cudaData()) + dstY * dst.pitch() + dstX;
//   const void* srcPtr = static_cast<const std::byte*>(data_) + srcY * pitch() + srcX;
//   const auto copyType = dst.isDevice() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
//   CUDA_OPTIONAL(
//     cudaMemcpy2DAsync(dstPtr, dst.pitch(), srcPtr, pitch(), widthBytes, height, copyType,
//     stream));
//   return {};
// }

// std::optional<StreamError> CudaBufferPitched2D::copyToHost2D(void* dst,
//   size_t srcX,
//   size_t srcY,
//   size_t widthBytes,
//   size_t height,
//   cudaStream_t stream,
//   bool synchronize) const {
//   void* dstPtr = static_cast<std::byte*>(dst);
//   const void* srcPtr = static_cast<const std::byte*>(data_) + srcY * pitch() + srcX;
//   CUDA_OPTIONAL(cudaMemcpy2DAsync(
//     dstPtr, widthBytes, srcPtr, pitch(), widthBytes, height, cudaMemcpyDeviceToHost, stream));
//   if (synchronize) { CUDA_OPTIONAL(cudaStreamSynchronize(stream)); }
//   return {};
// }
