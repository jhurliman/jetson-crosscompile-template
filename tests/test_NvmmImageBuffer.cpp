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
      const auto& info = bufPtr->info();

      if (height * width == 0) {
        CHECK(bufPtr->fd() == -1);
        CHECK(bufPtr->data() == nullptr);
        CHECK(bufPtr->nvBufferSize() == 0);
        CHECK(info.widths == PLANE_ZEROS);
        CHECK(info.heights == PLANE_ZEROS);
        CHECK(info.pitches == PLANE_ZEROS);
        CHECK(info.offsets == PLANE_ZEROS);
        CHECK(info.sizes == PLANE_ZEROS);
        CHECK(info.layouts == PlaneLayoutArray{});
        continue;
      }

      // Expected height and width are the input height and width rounded up to the next even number
      const size_t expectedWidth = size_t((ssize_t(width) + 1) & ~1);
      const size_t expectedHeight = size_t((ssize_t(height) + 1) & ~1);

      CHECK(bufPtr->format() == NvmmColorFormat::YUV420);
      CHECK(bufPtr->numPlanes() == 3);
      CHECK(bufPtr->width() == expectedWidth);
      CHECK(bufPtr->height() == expectedHeight);
      CHECK(bufPtr->size() == info.size());
      CHECK(bufPtr->fd() >= 0);
      CHECK(bufPtr->data() != nullptr);
      CHECK(info.widths == PlaneArray{expectedWidth, expectedWidth / 2, expectedWidth / 2, 0});
      CHECK(info.heights == PlaneArray{expectedHeight, expectedHeight / 2, expectedHeight / 2, 0});
      CHECK(info.pitches == PlaneArray{MIN_PITCH, MIN_PITCH, MIN_PITCH, 0});
      CHECK(info.offsets == PlaneArray{0, MIN_PLANE_SIZE, MIN_PLANE_SIZE * 2, 0});
      CHECK(info.sizes == PlaneArray{MIN_PLANE_SIZE, MIN_PLANE_SIZE, MIN_PLANE_SIZE, 0});
      CHECK(info.layouts == PlaneLayoutArray{BlockLinear, BlockLinear, BlockLinear, Pitch});
    }
  }

  // Test large allocation (~21MB)
  bufPtr = REQUIRE_EXPECTED(NvmmImageBuffer::create(4096, 4096));
  REQUIRE(bufPtr);
  const auto& info = bufPtr->info();
  REQUIRE(bufPtr->format() == NvmmColorFormat::YUV420);
  REQUIRE(bufPtr->numPlanes() == 3);
  REQUIRE(bufPtr->width() == 4096);
  REQUIRE(bufPtr->height() == 4096);
  CHECK(bufPtr->size() == 16777216 + 4194304 + 4194304 + 0);
  REQUIRE(bufPtr->fd() >= 0);
  REQUIRE(bufPtr->data() != nullptr);
  REQUIRE(bufPtr->nvBufferSize() == 1008);
  REQUIRE(info.widths == PlaneArray{4096, 2048, 2048, 0});
  REQUIRE(info.heights == PlaneArray{4096, 2048, 2048, 0});
  REQUIRE(info.pitches == PlaneArray{4096, 2048, 2048, 0});
  REQUIRE(info.offsets == PlaneArray{0, 16777216, 16777216 + 4194304, 0});
  REQUIRE(info.sizes == PlaneArray{16777216, 4194304, 4194304, 0});
  REQUIRE(info.layouts == PlaneLayoutArray{BlockLinear, BlockLinear, BlockLinear, Pitch});

  bufPtr.reset();
}

TEST_CASE("Allocates NvmmImageBuffer (all)", "[nvmmimagebuffer]") {
  // Iterate over every NvmmColorFormat
  for (const auto& test : NVMM_IMAGE_ALLOCATION_TESTS) {
    std::unique_ptr<NvmmImageBuffer> bufPtr =
      REQUIRE_EXPECTED(NvmmImageBuffer::create(test.width, test.height, test.format));
    REQUIRE(bufPtr);
    const auto& info = bufPtr->info();
    CHECK(bufPtr->format() == test.format);
    CAPTURE(test.format);
    CHECK(bufPtr->numPlanes() == test.numPlanes);
    CHECK(bufPtr->width() == test.width);
    CHECK(bufPtr->height() == test.height);
    CHECK(bufPtr->size() == test.sizes[0] + test.sizes[1] + test.sizes[2] + test.sizes[3]);
    CHECK(bufPtr->fd() >= 0);
    CHECK(bufPtr->data() != nullptr);
    CHECK(info.widths == test.widths);
    CHECK(info.heights == test.heights);
    CHECK(info.pitches == test.pitches);
    CHECK(info.offsets == test.offsets);
    CHECK(info.sizes == test.sizes);
    CHECK(info.layouts == test.layouts);
  }
}

TEST_CASE("NvmmImageBuffer copyFromHost/copyToHost", "[cudabuffer]") {
  // To test copyFromHost we need to allocate a NvmmImageBuffer, copy data from the host to it,
  // then copy the data back to the host and check it. Do this for a variety of offset/count values

  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 16;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  std::unique_ptr<NvmmImageBuffer> buf =
    REQUIRE_EXPECTED(NvmmImageBuffer::create(WIDTH_BYTES, HEIGHT));

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
      buf->memset(std::byte(0), buf->size());

      REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i));
      REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data() + i, i, SIZE_VEC - i));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    buf->memset(std::byte(0), buf->size());

    REQUIRE_NO_ERROR(buf->copyFromHost(srcVec.data(), 0, SIZE_VEC));
    REQUIRE_NO_ERROR(buf->copyToHost(dstVec.data(), 0, SIZE_VEC));
    REQUIRE(srcVec == dstVec);
  }

  buf.reset();
}

TEST_CASE("NvmmImageBuffer copyFrom NvmmImageBuffer", "[cudabuffer]") {
  // To test copyFrom we need to allocate two NvmmImageBuffers, copy data from one to the
  // other, then copy the data back to the host and check it. Do this for a variety of offset/count
  // values, with and without a NvBufferSession

  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 16;
  constexpr size_t SIZE_VEC = 128;

  std::vector<std::byte> srcVec(SIZE_VEC);
  std::vector<std::byte> dstVec(SIZE_VEC);
  for (size_t i = 0; i < SIZE_VEC; i++) {
    srcVec[i] = std::byte(i);
    dstVec[i] = std::byte(0);
  }

  const bool makeSession = GENERATE(true, false);
  auto session = makeSession ? REQUIRE_EXPECTED(nvmm::createSession()) : nullptr;

  std::unique_ptr<NvmmImageBuffer> bufSrc =
    REQUIRE_EXPECTED(NvmmImageBuffer::create(WIDTH_BYTES, HEIGHT));
  std::unique_ptr<NvmmImageBuffer> bufDst =
    REQUIRE_EXPECTED(NvmmImageBuffer::create(WIDTH_BYTES, HEIGHT));

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
      bufDst->memset(std::byte(0), bufDst->size());

      REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data() + i, i, SIZE_VEC - i));
      REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, i, i, SIZE_VEC - i, session));
      REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data() + i, i, SIZE_VEC - i));
      // dstVec[i:] == srcVec[i:]
      CAPTURE(dstVec);
      REQUIRE(std::equal(srcVec.begin() + ssize_t(i), srcVec.end(), dstVec.begin() + ssize_t(i)));
    }
  }

  SECTION("Copy entire vector") {
    bufSrc->memset(std::byte(0), bufSrc->size());
    bufDst->memset(std::byte(0), bufDst->size());

    REQUIRE_NO_ERROR(bufSrc->copyFromHost(srcVec.data(), 0, SIZE_VEC));
    REQUIRE_NO_ERROR(bufDst->copyFrom(*bufSrc, 0, 0, SIZE_VEC, session));
    REQUIRE_NO_ERROR(bufDst->copyToHost(dstVec.data(), 0, SIZE_VEC));
    REQUIRE(srcVec == dstVec);
  }

  bufSrc.reset();
  bufDst.reset();
}

TEST_CASE("NvmmImageBuffer memset", "[cudabuffer]") {
  constexpr size_t WIDTH_BYTES = 16;
  constexpr size_t HEIGHT = 15;
  constexpr size_t SIZE_BYTES = WIDTH_BYTES * HEIGHT;

  std::unique_ptr<NvmmImageBuffer> buf =
    REQUIRE_EXPECTED(NvmmImageBuffer::create(WIDTH_BYTES, HEIGHT));

  std::vector<std::byte> vec(SIZE_BYTES, std::byte(0xFF));

  for (size_t height = 0; height < HEIGHT; height += 3) {
    for (size_t width = 0; width < WIDTH_BYTES; width++) {
      const size_t i = height * WIDTH_BYTES + width;

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
  }

  buf.reset();
}

// FIXME: copyFrom2D

// FIXME: copyFromHost2D

// FIXME: copyToHost2D

// FIXME: copyFromCuda

// FIXME: copyToCuda

// FIXME: copyFromCuda2D

// FIXME: copyToCuda2D

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
