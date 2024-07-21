#include "nvmm/session.hpp"

#include "cuda/types.hpp"

#include <nvbuf_utils.h>

namespace nvmm {

tl::expected<NvBufferSession, NvmmError> createSession() {
  const auto session = NvBufferSessionCreate();
  if (!session) {
    return tl::make_unexpected(NvmmError{cudaErrorIllegalState, "NvBufferSessionCreate failed"});
  }
  return session;
}

void destroySession(NvBufferSession session) {
  NvBufferSessionDestroy(session);
}

} // namespace nvmm
