#pragma once

#include <optional>
#include <string>

struct ArgumentError {
  std::string errorMessage;

  ArgumentError(const std::string& message) : errorMessage(message) {}
};

struct StreamError {
  int errorCode;
  std::string errorMessage;

  StreamError(int code,
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

void InstallStackTraceHandler();
