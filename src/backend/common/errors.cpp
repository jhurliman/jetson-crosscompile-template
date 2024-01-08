#include "errors.hpp"

#include <cpptrace/cpptrace.hpp>
#include <signal.h>
#include <string.h> // for strsignal

static void SignalHandler(int signalNum) {
  const char* signalName = strsignal(signalNum);

  char buffer[512];
  int len = snprintf(buffer, sizeof(buffer), "Caught signal: %d (%s)\n", signalNum, signalName);
  if (len > 0) { write(STDERR_FILENO, buffer, size_t(len)); }

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
