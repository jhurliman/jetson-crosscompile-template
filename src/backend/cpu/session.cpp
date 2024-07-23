#include "nvmm/session.hpp"

// NvBufferSession is simply `nullptr` in the CPU backend

namespace nvmm {

tl::expected<NvBufferSession, NvmmError> createSession() {
  return nullptr;
}

void destroySession(NvBufferSession session) {
  (void)session;
}

} // namespace nvmm
