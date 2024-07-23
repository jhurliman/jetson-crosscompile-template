#pragma once

#include "errors.hpp"
#include "types.hpp"

#include <tl/expected.hpp>

#include <cstddef>
#include <memory>
#include <optional>

/*
 * Wraps NvBufferMemMap and NvBufferMemUnMap calls to provide RAII semantics.
 */
class NvmmMemMap {
public:
  static tl::expected<std::unique_ptr<NvmmMemMap>, StreamError> create(
    int fd, NvmmBufferMemAccess access);

  ~NvmmMemMap();

  NvmmMemMap(const NvmmMemMap&) = delete;
  NvmmMemMap& operator=(const NvmmMemMap&) = delete;

  NvmmMemMap(NvmmMemMap&&) = default;
  NvmmMemMap& operator=(NvmmMemMap&&) = default;

  const std::byte* data() const;
  std::byte* data();

  int fd() const;
  NvmmBufferMemAccess access() const;

  std::optional<StreamError> syncForCpu();
  std::optional<StreamError> syncForDevice();

private:
  void* data_;
  int fd_;
  NvmmBufferMemAccess access_;

  NvmmMemMap(void* pVirtAddr, int fd, NvmmBufferMemAccess access);
};
