#pragma once

#include "errors.hpp"
#include "types.hpp"

#include <tl/expected.hpp>

namespace nvmm {

// Create an NvBuffer session used for asynchronous/parallel buffer operations
tl::expected<NvBufferSession, StreamError> createSession();

// Destroy an NvBuffer session
void destroySession(NvBufferSession session);

} // namespace nvmm
