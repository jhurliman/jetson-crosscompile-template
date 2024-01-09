#include "errors.hpp"

#include <cpptrace/cpptrace.hpp>
#include <signal.h>
#include <string.h> // for strsignal
#include <unistd.h> // for STDERR_FILENO, write()

#include <array>

static void SignalHandler(int signalNum) {
  const char* signalName = ::strsignal(signalNum);

  constexpr size_t BUFFER_SIZE = 512;
  std::array<char, BUFFER_SIZE> buffer{};
  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)
  int len =
    ::snprintf(buffer.data(), buffer.size(), "Caught signal: %d (%s)\n", signalNum, signalName);
  // NOLINTEND(cppcoreguidelines-pro-type-vararg)
  if (len > 0) { ::write(STDERR_FILENO, buffer.data(), size_t(len)); }

  cpptrace::generate_trace(2).print(); // Skip this function and the signal handler
  exit(signalNum);
}

void InstallStackTraceHandler() {
  signal(SIGILL, SignalHandler);
  signal(SIGABRT, SignalHandler);
  signal(SIGBUS, SignalHandler);
  signal(SIGFPE, SignalHandler);
  signal(SIGSEGV, SignalHandler);
}
