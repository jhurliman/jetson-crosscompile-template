#pragma once

#include "cuda/types.hpp"

#include <tl/expected.hpp>

#include <optional>
#include <string_view>

namespace cuda {

std::optional<StreamError> ensureInitialized();

tl::expected<cudaStream_t, StreamError> createStream(
  const std::string_view name, const StreamPriority priority);

std::optional<StreamError> destroyStream(cudaStream_t stream);

std::optional<StreamError> synchronizeStream(cudaStream_t stream);

} // namespace cuda
