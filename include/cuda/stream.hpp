#pragma once

#include "cuda/types.hpp"

#include <tl/expected.hpp>

#include <string_view>

tl::expected<cudaStream_t, StreamError> createStream(
  const std::string_view name, const StreamPriority priority);

void destroyStream(cudaStream_t stream);
