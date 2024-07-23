#include "nvmm/NvmmMemMap.hpp"

tl::expected<std::unique_ptr<NvmmMemMap>, StreamError> NvmmMemMap::create(
  int fd, NvmmBufferMemAccess access) {
  return std::unique_ptr<NvmmMemMap>(new NvmmMemMap(nullptr, fd, access));
}

NvmmMemMap::NvmmMemMap(void* pVirtAddr, int fd, NvmmBufferMemAccess access)
  : data_(pVirtAddr),
    fd_(fd),
    access_(access) {}

NvmmMemMap::~NvmmMemMap() {}

const std::byte* NvmmMemMap::data() const {
  return static_cast<std::byte*>(data_);
}

std::byte* NvmmMemMap::data() {
  return static_cast<std::byte*>(data_);
}

int NvmmMemMap::fd() const {
  return fd_;
}

NvmmBufferMemAccess NvmmMemMap::access() const {
  return access_;
}

std::optional<StreamError> NvmmMemMap::syncForCpu() {
  return std::nullopt;
}

std::optional<StreamError> NvmmMemMap::syncForDevice() {
  return std::nullopt;
}
