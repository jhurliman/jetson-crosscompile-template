#pragma once

#include "types.hpp"

#include <tl/expected.hpp>

#include <cstddef>
#include <memory>

/*
 * Wraps NvBufferMemMap and NvBufferMemUnMap calls to provide RAII semantics.
 */
class NvmmMemMap {
public:
  static tl::expected<std::unique_ptr<NvmmMemMap>, NvmmError> create(
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

  std::optional<NvmmError> syncForCpu();
  std::optional<NvmmError> syncForDevice();

private:
  void* data_;
  int fd_;
  NvmmBufferMemAccess access_;

  NvmmMemMap(void* pVirtAddr, int fd, NvmmBufferMemAccess access);
};
