#include "nvmm/NvmmMemMap.hpp"

#include "cuda/types.hpp"
#include "nvmm/types.hpp"

#include <nvbuf_utils.h>

tl::expected<std::unique_ptr<NvmmMemMap>, StreamError> NvmmMemMap::create(
  int fd, NvmmBufferMemAccess access) {
  void* pVirtAddr = nullptr;
  const auto flags = NvBufferMemFlags(access);
  const int res = NvBufferMemMap(fd, 0, flags, &pVirtAddr);
  if (res != 0) {
    return tl::make_unexpected(StreamError{cudaErrorMemoryAllocation, "NvBufferMemMap failed"});
  }
  return std::unique_ptr<NvmmMemMap>(new NvmmMemMap(pVirtAddr, fd, access));
}

NvmmMemMap::NvmmMemMap(void* pVirtAddr, int fd, NvmmBufferMemAccess access)
  : data_(pVirtAddr),
    fd_(fd),
    access_(access) {}

NvmmMemMap::~NvmmMemMap() {
  const int res = NvBufferMemUnMap(fd_, 0, &data_);
  if (res != 0) {
    // LOG(ERROR) << "NvBufferMemUnMap failed: " << res;
  }
}

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
  const int res = NvBufferMemSyncForCpu(fd_, 0, &data_);
  if (res != 0) { return StreamError{cudaErrorMemoryAllocation, "NvBufferMemSyncForCpu failed"}; }
  return std::nullopt;
}

std::optional<StreamError> NvmmMemMap::syncForDevice() {
  const int res = NvBufferMemSyncForDevice(fd_, 0, &data_);
  if (res != 0) {
    return StreamError{cudaErrorMemoryAllocation, "NvBufferMemSyncForDevice failed"};
  }
  return std::nullopt;
}
