#pragma once

#include "CudaBuffer.hpp"
#include "errors.hpp"

#include <tl/expected.hpp>

/**
 * @brief A lightweight wrapper around a CudaBuffer that provides a view of the buffer as a typed
 * array.
 */
template<typename T> class CudaArrayView {
public:
  static const size_t BYTES_PER_ELEMENT = sizeof(T);

  static tl::expected<CudaArrayView<T>, ArgumentError> fromBuffer(
    CudaBuffer& buffer, size_t n, size_t byteOffset = 0) {
    if (byteOffset % BYTES_PER_ELEMENT != 0) {
      return tl::unexpected(ArgumentError("byteOffset must be a multiple of BYTES_PER_ELEMENT"));
    }
    if (buffer.size() < byteOffset + n * BYTES_PER_ELEMENT) {
      return tl::unexpected(ArgumentError("byteOffset + n * BYTES_PER_ELEMENT out of bounds"));
    }
    return CudaArrayView<T>(buffer.cudaData(), n, byteOffset, buffer.isDevice());
  }

  size_t size() const { return n_; }

  size_t byteLength() const { return n_ * BYTES_PER_ELEMENT; }

  size_t byteOffset() const { return byteOffset_; }

  T* data() { return reinterpret_cast<T*>(static_cast<std::byte*>(data_) + byteOffset_); }

  const T* data() const {
    return reinterpret_cast<const T*>(static_cast<const std::byte*>(data_) + byteOffset_);
  }

  bool isDevice() const { return isDevice_; }

  CudaArrayView<T> subarray(size_t begin, size_t end) const {
    if (begin > end) { return CudaArrayView<T>(data_, 0, byteOffset_); }
    return CudaArrayView<T>(data_, end - begin, byteOffset_ + begin * BYTES_PER_ELEMENT);
  }

private:
  size_t n_;
  void* data_;
  size_t byteOffset_;
  bool isDevice_;

  CudaArrayView(void* data, size_t n, size_t byteOffset, bool isDevice)
    : n_(n),
      data_(data),
      byteOffset_(byteOffset),
      isDevice_(isDevice) {}
};
