#include "nvmm/NvmmImageBuffer.hpp"
#include "nvmm/session.hpp"
#include "nvmm_image_testdata.hpp"
#include "requires.hpp"

#include <catch2/catch.hpp>

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)

constexpr PlaneArray PLANE_ZEROS = {0, 0, 0, 0};

TEST_CASE("Allocates NvmmImageBuffer (YUV420)", "[nvmmimagebuffer]") {
  constexpr size_t WIDTHS = 16;
  constexpr size_t HEIGHTS = 16;

  constexpr size_t MIN_PLANE_SIZE = 131072;
  constexpr size_t MIN_PITCH = 64;

  std::unique_ptr<NvmmImageBuffer> bufPtr;
  for (size_t height = 0; height < HEIGHTS; height++) {
    for (size_t width = 0; width < WIDTHS; width++) {
      bufPtr = REQUIRE_EXPECTED(NvmmImageBuffer::create(width, height));
      REQUIRE(bufPtr);

      if (height * width == 0) {
        CHECK(bufPtr->fd() == -1);
        CHECK(bufPtr->data() == nullptr);
        CHECK(bufPtr->widths() == PLANE_ZEROS);
        CHECK(bufPtr->heights() == PLANE_ZEROS);
        CHECK(bufPtr->pitches() == PLANE_ZEROS);
        CHECK(bufPtr->offsets() == PLANE_ZEROS);
        CHECK(bufPtr->sizes() == PLANE_ZEROS);
        CHECK(bufPtr->layouts() == NvmmImageBuffer::PlaneLayoutArray{});
        continue;
      }

      // Expected height and width are the input height and width rounded up to the next even number
      const size_t expectedWidth = size_t((ssize_t(width) + 1) & ~1);
      const size_t expectedHeight = size_t((ssize_t(height) + 1) & ~1);

      CHECK(bufPtr->format() == NvmmColorFormat::YUV420);
      CHECK(bufPtr->numPlanes() == 3);
      CHECK(bufPtr->width() == expectedWidth);
      CHECK(bufPtr->height() == expectedHeight);
      CHECK(bufPtr->size() == bufPtr->sizes()[0] + bufPtr->sizes()[1] + bufPtr->sizes()[2]);
      CHECK(bufPtr->fd() >= 0);
      CHECK(bufPtr->data() != nullptr);
      CHECK(bufPtr->widths() == PlaneArray{expectedWidth, expectedWidth / 2, expectedWidth / 2, 0});
      CHECK(
        bufPtr->heights() == PlaneArray{expectedHeight, expectedHeight / 2, expectedHeight / 2, 0});
      CHECK(bufPtr->pitches() == PlaneArray{MIN_PITCH, MIN_PITCH, MIN_PITCH, 0});
      CHECK(bufPtr->offsets() == PlaneArray{0, MIN_PLANE_SIZE, MIN_PLANE_SIZE * 2, 0});
      CHECK(bufPtr->sizes() == PlaneArray{MIN_PLANE_SIZE, MIN_PLANE_SIZE, MIN_PLANE_SIZE, 0});
      CHECK(bufPtr->layouts() ==
        NvmmImageBuffer::PlaneLayoutArray{BlockLinear, BlockLinear, BlockLinear, Pitch});
    }
  }

  // Test large allocation (~21MB)
  bufPtr = REQUIRE_EXPECTED(NvmmImageBuffer::create(4096, 4096));
  REQUIRE(bufPtr);
  REQUIRE(bufPtr->format() == NvmmColorFormat::YUV420);
  REQUIRE(bufPtr->numPlanes() == 3);
  REQUIRE(bufPtr->width() == 4096);
  REQUIRE(bufPtr->height() == 4096);
  CHECK(bufPtr->size() == bufPtr->sizes()[0] + bufPtr->sizes()[1] + bufPtr->sizes()[2]);
  REQUIRE(bufPtr->fd() >= 0);
  REQUIRE(bufPtr->data() != nullptr);
  REQUIRE(bufPtr->widths() == PlaneArray{4096, 2048, 2048, 0});
  REQUIRE(bufPtr->heights() == PlaneArray{4096, 2048, 2048, 0});
  REQUIRE(bufPtr->pitches() == PlaneArray{4096, 2048, 2048, 0});
  REQUIRE(bufPtr->offsets() == PlaneArray{0, 16777216, 16777216 + 4194304, 0});
  REQUIRE(bufPtr->sizes() == PlaneArray{16777216, 4194304, 4194304, 0});
  REQUIRE(bufPtr->layouts() ==
    NvmmImageBuffer::PlaneLayoutArray{BlockLinear, BlockLinear, BlockLinear, Pitch});

  bufPtr.reset();
}

TEST_CASE("Allocates NvmmImageBuffer (all)", "[nvmmimagebuffer]") {
  // Iterate over every NvmmColorFormat
  for (const auto& test : NVMM_IMAGE_ALLOCATION_TESTS) {
    std::unique_ptr<NvmmImageBuffer> bufPtr =
      REQUIRE_EXPECTED(NvmmImageBuffer::create(test.width, test.height, test.format));
    REQUIRE(bufPtr);
    CHECK(bufPtr->format() == test.format);
    CAPTURE(test.format);
    CHECK(bufPtr->numPlanes() == test.numPlanes);
    CHECK(bufPtr->width() == test.width);
    CHECK(bufPtr->height() == test.height);
    CHECK(bufPtr->size() == test.sizes[0] + test.sizes[1] + test.sizes[2] + test.sizes[3]);
    CHECK(bufPtr->fd() >= 0);
    CHECK(bufPtr->data() != nullptr);
    CHECK(bufPtr->widths() == test.widths);
    CHECK(bufPtr->heights() == test.heights);
    CHECK(bufPtr->pitches() == test.pitches);
    CHECK(bufPtr->offsets() == test.offsets);
    CHECK(bufPtr->sizes() == test.sizes);
    CHECK(bufPtr->layouts() == test.layouts);
  }
}

// TEST_CASE("CudaBufferDevice2D copyFromHost/copyToHost", "[cudabuffer]") {
//   // To test copyFromHost we need to allocate a CudaBufferDevice2D, copy data from the host to
//   it,
//   // then copy the data back to the host and check it. Do this for a variety of offset/count
//   values
//   // and with and without a stream

//   constexpr size_t WIDTH_BYTES = 16;
//   constexpr size_t HEIGHT = 16;
//   constexpr size_t SIZE_VEC = 128;

//   std::vector<std::byte> srcVec(SIZE_VEC);
//   std::vector<std::byte> dstVec(SIZE_VEC);
//   for (size_t i = 0; i < SIZE_VEC; i++) {
//     srcVec[i] = std::byte(i);
//     dstVec[i] = std::byte(0);
//   }

//   const bool makeStream = GENERATE(true, false);
//   cudaStream_t stream =
//     makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

//   std::unique_ptr<CudaBufferDevice2D> buf =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(WIDTH_BYTES, HEIGHT, stream));

//   SECTION("Copy one byte") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, 1, stream));
//       REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, 1, stream));
//       // dstVec[i] == i
//       CAPTURE(dstVec);
//       REQUIRE(dstVec[i] == std::byte(i));
//     }
//   }

//   SECTION("Copy multiple bytes") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, i, stream));
//       REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, i, stream));
//       // dstVec[:i] == srcVec[:i]
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
//     }
//   }

//   SECTION("Copy multiple bytes at an offset") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       buf->memset(std::byte(0), buf->size(), stream);

//       REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
//       // dstVec[i:] == srcVec[i:]
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() +
//       ssize_t(i)));
//     }
//   }

//   SECTION("Copy entire vector") {
//     buf->memset(std::byte(0), buf->size(), stream);

//     REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE(srcVec == dstVec);
//   }

//   buf.reset();
//   if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
//   REQUIRE_NO_ERROR(cuda::checkLastError());
// }

// TEST_CASE("CudaBufferDevice2D copyFrom CudaBufferDevice2D", "[cudabuffer]") {
//   // To test copyFrom we need to allocate two CudaBufferDevice2Ds, copy data from one to the
//   // other, then copy the data back to the host and check it. Do this for a variety of
//   offset/count
//   // values and with and without a stream

//   constexpr size_t WIDTH_BYTES = 16;
//   constexpr size_t HEIGHT = 16;
//   constexpr size_t SIZE_VEC = 128;

//   std::vector<std::byte> srcVec(SIZE_VEC);
//   std::vector<std::byte> dstVec(SIZE_VEC);
//   for (size_t i = 0; i < SIZE_VEC; i++) {
//     srcVec[i] = std::byte(i);
//     dstVec[i] = std::byte(0);
//   }

//   const bool makeStream = GENERATE(true, false);
//   cudaStream_t stream =
//     makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

//   std::unique_ptr<CudaBufferDevice2D> bufSrc =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(WIDTH_BYTES, HEIGHT, stream));
//   std::unique_ptr<CudaBufferDevice2D> bufDst =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(WIDTH_BYTES, HEIGHT, stream));

//   SECTION("Copy one byte") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1, stream));
//       REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, 1, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1, stream));
//       // dstVec[i] == i
//       CAPTURE(dstVec);
//       REQUIRE(dstVec[i] == std::byte(i));
//     }
//   }

//   SECTION("Copy multiple bytes") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i, stream));
//       // dstVec[:i] == srcVec[:i]
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
//     }
//   }

//   SECTION("Copy multiple bytes at an offset") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       bufDst->memset(std::byte(0), bufDst->size(), stream);

//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
//       // dstVec[i:] == srcVec[i:]
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() +
//       ssize_t(i)));
//     }
//   }

//   SECTION("Copy entire vector") {
//     bufSrc->memset(std::byte(0), bufSrc->size(), stream);
//     bufDst->memset(std::byte(0), bufDst->size(), stream);

//     REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE(srcVec == dstVec);
//   }

//   bufSrc.reset();
//   bufDst.reset();
//   if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
//   REQUIRE_NO_ERROR(cuda::checkLastError());
// }

// TEST_CASE("CudaBufferDevice2D copyTo CudaBufferDevice2D", "[cudabuffer]") {
//   // To test copyTo we need to allocate two CudaBufferDevice2Ds, copy data from one to the other,
//   // then copy the data back to the host and check it. Do this for a variety of offset/count
//   values
//   // and with and without a stream

//   constexpr size_t WIDTH_BYTES = 16;
//   constexpr size_t HEIGHT = 16;
//   constexpr size_t SIZE_VEC = 128;

//   std::vector<std::byte> srcVec(SIZE_VEC);
//   std::vector<std::byte> dstVec(SIZE_VEC);
//   for (size_t i = 0; i < SIZE_VEC; i++) {
//     srcVec[i] = std::byte(i);
//     dstVec[i] = std::byte(0);
//   }

//   const bool makeStream = GENERATE(true, false);
//   cudaStream_t stream =
//     makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

//   std::unique_ptr<CudaBufferDevice2D> bufSrc =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(WIDTH_BYTES, HEIGHT, stream));
//   std::unique_ptr<CudaBufferDevice2D> bufDst =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(WIDTH_BYTES, HEIGHT, stream));

//   SECTION("Copy one byte") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1, stream));
//       REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, 1, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1, stream));
//       // dstVec[i] == i
//       CAPTURE(dstVec);
//       REQUIRE(dstVec[i] == std::byte(i));
//     }
//   }

//   SECTION("Copy multiple bytes") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i, stream));
//       REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i, stream));
//       // dstVec[:i] == srcVec[:i]
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
//     }
//   }

//   SECTION("Copy multiple bytes at an offset") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       bufDst->memset(std::byte(0), bufDst->size(), stream);

//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
//       // dstVec[i:] == srcVec[i:]
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() +
//       ssize_t(i)));
//     }
//   }

//   SECTION("Copy entire vector") {
//     bufSrc->memset(std::byte(0), bufSrc->size(), stream);
//     bufDst->memset(std::byte(0), bufDst->size(), stream);

//     REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE(srcVec == dstVec);
//   }

//   bufSrc.reset();
//   bufDst.reset();
//   if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
//   REQUIRE_NO_ERROR(cuda::checkLastError());
// }

// TEST_CASE("CudaBufferDevice2D copyTo CudaBufferDevice", "[cudabuffer]") {
//   constexpr size_t WIDTH_BYTES = 16;
//   constexpr size_t HEIGHT = 16;
//   constexpr size_t SIZE_VEC = 128;

//   std::vector<std::byte> srcVec(SIZE_VEC);
//   std::vector<std::byte> dstVec(SIZE_VEC);
//   for (size_t i = 0; i < SIZE_VEC; i++) {
//     srcVec[i] = std::byte(i);
//     dstVec[i] = std::byte(0);
//   }

//   const bool makeStream = GENERATE(true, false);
//   cudaStream_t stream =
//     makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

//   std::unique_ptr<CudaBufferDevice2D> bufSrc =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(WIDTH_BYTES, HEIGHT, stream));
//   std::unique_ptr<CudaBufferDevice> bufDst =
//     REQUIRE_EXPECTED(CudaBufferDevice::create(WIDTH_BYTES * HEIGHT, stream));

//   SECTION("Copy one byte") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1, stream));
//       REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, 1, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1, stream));
//       // dstVec[i] == i
//       CAPTURE(dstVec);
//       REQUIRE(dstVec[i] == std::byte(i));
//     }
//   }

//   SECTION("Copy multiple bytes") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i, stream));
//       REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i, stream));
//       // dstVec[:i] == srcVec[:i]
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
//     }
//   }

//   SECTION("Copy multiple bytes at an offset") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       bufDst->memset(std::byte(0), bufDst->size(), stream);

//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
//       // dstVec[i:] == srcVec[i:]
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() +
//       ssize_t(i)));
//     }
//   }

//   SECTION("Copy entire vector") {
//     bufSrc->memset(std::byte(0), bufSrc->size(), stream);
//     bufDst->memset(std::byte(0), bufDst->size(), stream);

//     REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(bufSrc->copyTo(*bufDst, 0, 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE(srcVec == dstVec);
//   }

//   bufSrc.reset();
//   bufDst.reset();
//   if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
//   REQUIRE_NO_ERROR(cuda::checkLastError());
// }

// TEST_CASE("CudaBufferDevice2D copyFrom CudaBufferDevice", "[cudabuffer]") {
//   constexpr size_t WIDTH_BYTES = 16;
//   constexpr size_t HEIGHT = 16;
//   constexpr size_t SIZE_VEC = 128;

//   const bool makeStream = GENERATE(true, false);
//   cudaStream_t stream =
//     makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

//   std::vector<std::byte> srcVec(SIZE_VEC);
//   std::vector<std::byte> dstVec(SIZE_VEC);
//   for (size_t i = 0; i < SIZE_VEC; i++) {
//     srcVec[i] = std::byte(i);
//     dstVec[i] = std::byte(0);
//   }

//   std::unique_ptr<CudaBufferDevice> bufSrc =
//     REQUIRE_EXPECTED(CudaBufferDevice::create(WIDTH_BYTES * HEIGHT, stream));
//   std::unique_ptr<CudaBufferDevice2D> bufDst =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(WIDTH_BYTES, HEIGHT, stream));

//   SECTION("Copy one byte") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, 1, stream));
//       REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, 1, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, 1, stream));
//       // dstVec[i] == i
//       CAPTURE(dstVec);
//       REQUIRE(dstVec[i] == std::byte(i));
//     }
//   }

//   SECTION("Copy multiple bytes") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, i, stream));
//       // srcVec[:i] == dstVec[:i]
//       CAPTURE(srcVec);
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin(), srcVec.begin() + ssize_t(i), dstVec.begin()));
//     }
//   }

//   SECTION("Copy multiple bytes at an offset") {
//     for (size_t i = 0; i < SIZE_VEC; i++) {
//       bufDst->memset(std::byte(0), bufDst->size(), stream);

//       REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, SIZE_VEC - i, stream));
//       REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i, stream));
//       // srcVec[i:] == dstVec[i:]
//       CAPTURE(srcVec);
//       CAPTURE(dstVec);
//       REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() +
//       ssize_t(i)));
//     }
//   }

//   SECTION("Copy entire vector") {
//     bufSrc->memset(std::byte(0), bufSrc->size(), stream);
//     bufDst->memset(std::byte(0), bufDst->size(), stream);

//     REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, stream));
//     REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC, stream));
//     REQUIRE(srcVec == dstVec);
//   }

//   bufSrc.reset();
//   bufDst.reset();
//   if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
//   REQUIRE_NO_ERROR(cuda::checkLastError());
// }

// TEST_CASE("CudaBufferDevice2D memset", "[cudabuffer]") {
//   constexpr size_t WIDTH_BYTES = 16;
//   constexpr size_t HEIGHT = 15;
//   constexpr size_t SIZE_BYTES = WIDTH_BYTES * HEIGHT;

//   const bool makeStream = GENERATE(true, false);
//   cudaStream_t stream =
//     makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

//   std::unique_ptr<CudaBufferDevice2D> buf =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(WIDTH_BYTES, HEIGHT, stream));

//   std::vector<std::byte> vec(SIZE_BYTES, std::byte(0xFF));

//   for (size_t height = 0; height < HEIGHT; height += 3) {
//     for (size_t width = 0; width < WIDTH_BYTES; width++) {
//       const size_t i = height * WIDTH_BYTES + width;

//       REQUIRE_NO_ERROR(buf->memset(std::byte(i), i, stream));
//       REQUIRE_NO_ERROR(buf->copyToHost(vec.data(), 0, i, stream));

//       for (size_t j = 0; j < i; j++) {
//         CAPTURE(vec);
//         REQUIRE(vec[j] == std::byte(i));
//       }

//       // Ensure the rest of the bytes are not modified
//       for (size_t j = i; j < SIZE_BYTES; j++) {
//         CAPTURE(vec);
//         CAPTURE(i);
//         CAPTURE(j);
//         REQUIRE(vec[j] != std::byte(j));
//       }
//     }
//   }

//   buf.reset();
//   if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
//   REQUIRE_NO_ERROR(cuda::checkLastError());
// }

// TEST_CASE("CudaBufferDevice2D copyFrom2D CudaBufferDevice2D", "[cudabuffer2d]") {
//   // To test copyFrom2D we allocate three CudaBufferDevice2Ds, two with matching dimensions and
//   // one with different height, width, and pitch. We copy data from the first to the second, then
//   // copy the data back to the host and check it. Then from the first to the third, then back to
//   the
//   // host and check it. Do this for a variety of srcX, srcY, dstX, dstY, widthBytes, and height
//   // values

//   std::vector<std::byte> srcVec(256);
//   std::vector<std::byte> dstVec(256, std::byte(0xFF));
//   for (size_t i = 0; i < 256; i++) {
//     srcVec[i] = std::byte(i);
//   }

//   const bool makeStream = GENERATE(true, false);
//   cudaStream_t stream =
//     makeStream ? REQUIRE_EXPECTED(cuda::createStream("test", StreamPriority::Normal)) : nullptr;

//   std::unique_ptr<CudaBufferDevice2D> bufSrc1 =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(16, 16, stream));
//   std::unique_ptr<CudaBufferDevice2D> bufSrc2 =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(513, 17, stream));
//   std::unique_ptr<CudaBufferDevice2D> bufDst =
//     REQUIRE_EXPECTED(CudaBufferDevice2D::create(16, 16, stream));

//   CHECK(bufSrc1->pitch() == 16);
//   CHECK(bufSrc1->size() == 16 * 16);
//   CHECK(bufSrc1->capacity() == 16 * 16);
//   CHECK(bufSrc2->pitch() == 513);
//   CHECK(bufSrc2->size() == 513 * 17);
//   CHECK(bufSrc2->capacity() == 513 * 17);
//   CHECK(bufDst->pitch() == 16);
//   CHECK(bufDst->size() == 16 * 16);
//   CHECK(bufDst->capacity() == 16 * 16);

//   SECTION("Copy one byte") {
//     for (size_t srcY = 0; srcY < 16; srcY += 3) {
//       for (size_t srcX = 0; srcX < 16; srcX++) {
//         // Reset dstVec
//         for (size_t i = 0; i < 256; i++) {
//           dstVec[i] = std::byte(0xFF);
//         }

//         for (size_t dstY = 0; dstY < 16; dstY += 2) {
//           for (size_t dstX = 0; dstX < 16; dstX++) {
//             const size_t srcVecI = srcY * 16 + srcX;
//             const size_t dstVecI = dstY * 16 + dstX;
//             const size_t srcI = srcY * bufSrc1->pitch() + srcX;
//             REQUIRE_NO_ERROR(bufSrc1->copyFromHost(srcVec.data() + srcVecI, srcI, 1, stream));
//             REQUIRE_NO_ERROR(bufDst->copyFrom2D(*bufSrc1, srcX, srcY, dstX, dstY, 1, 1, stream));
//             REQUIRE_NO_ERROR(
//               bufDst->copyToHost2D(dstVec.data() + dstVecI, 16, dstX, dstY, 1, 1, stream));
//             // srcVec[srcVecI] == dstVec[dstVecI]
//             CAPTURE(srcX, srcY, dstX, dstY, srcVecI, dstVecI);
//             CAPTURE(srcVec);
//             CAPTURE(dstVec);
//             REQUIRE(srcVec[srcVecI] == dstVec[dstVecI]);
//           }
//         }
//       }
//     }
//   }

//   SECTION("Copy 2x3 bytes") {
//     // Define a fixed number of source positions to extract a 2x3 byte block from the 16x16
//     source const std::vector<std::pair<size_t, size_t>> positions = {
//       { 0,  0}, // Top-left
//       {14,  0}, // Top-right
//       { 0, 13}, // Bottom-left
//       {14, 13}, // Bottom-right
//       { 7,  7}  // Center
//     };

//     for (const auto& [srcX, srcY] : positions) {
//       for (const auto& [dstX, dstY] : positions) {
//         // Reset dstVec
//         for (size_t i = 0; i < 256; i++) {
//           dstVec[i] = std::byte(0xFF);
//         }

//         const size_t srcVecI = srcY * 16 + srcX;
//         const size_t dstVecI = dstY * 16 + dstX;
//         // Copy 2x3 bytes from host `srcVec` to CUDA `bufSrc1`
//         REQUIRE_NO_ERROR(
//           bufSrc1->copyFromHost2D(srcVec.data() + srcVecI, 16, srcX, srcY, 2, 3, stream));
//         // Copy 2x3 bytes from CUDA `bufSrc1` to CUDA `bufSrc2`
//         REQUIRE_NO_ERROR(bufSrc1->copyTo2D(*bufSrc2, srcX, srcY, srcX, srcY, 2, 3, stream));
//         // Copy 2x3 bytes from CUDA `bufSrc2` to CUDA `bufDst`
//         REQUIRE_NO_ERROR(bufDst->copyFrom2D(*bufSrc2, srcX, srcY, dstX, dstY, 2, 3, stream));
//         // Copy 2x3 bytes from CUDA `bufDst` to host `dstVec`
//         REQUIRE_NO_ERROR(
//           bufDst->copyToHost2D(dstVec.data() + dstVecI, 16, dstX, dstY, 2, 3, stream));
//         // Check all 2x3 bytes individually
//         for (size_t y = 0; y < 3; y++) {
//           for (size_t x = 0; x < 2; x++) {
//             const size_t curSrcVecI = (srcY + y) * 16 + srcX + x;
//             const size_t curDstVecI = (dstY + y) * 16 + dstX + x;
//             CAPTURE(srcX, srcY, dstX, dstY, curSrcVecI, curDstVecI);
//             CAPTURE(srcVec);
//             CAPTURE(dstVec);
//             REQUIRE(srcVec[curSrcVecI] == dstVec[curDstVecI]);
//           }
//         }
//       }
//     }
//   }

//   bufSrc1.reset();
//   bufSrc2.reset();
//   bufDst.reset();
//   if (stream) { REQUIRE_NO_ERROR(cuda::destroyStream(stream)); }
//   REQUIRE_NO_ERROR(cuda::checkLastError());
// }

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
