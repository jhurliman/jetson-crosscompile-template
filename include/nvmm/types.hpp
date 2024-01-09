#pragma once

#include <optional>
#include <string>

#ifdef USE_T210

#include <nvbuf_utils.h>

#else

typedef struct _NvBufferSession* NvBufferSession;

#endif

struct NvmmError {
  int errorCode;
  std::string errorMessage;

  NvmmError(int code,
    const std::string& message,
    std::optional<std::string> filename = std::nullopt,
    std::optional<int> line = std::nullopt)
    : errorCode(code),
      errorMessage(message) {
    if (filename && line) {
      errorMessage = *filename + ":" + std::to_string(*line) + ": " + errorMessage;
    }
  }
};
