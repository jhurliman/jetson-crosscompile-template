#pragma once

#include <string>

struct ArgumentError {
  std::string errorMessage;

  ArgumentError(const std::string& message) : errorMessage(message) {}
};

void InstallStackTraceHandler();
