#define CATCH_CONFIG_RUNNER
#include "errors.hpp"

#include <catch2/catch.hpp>

int main(int argc, char* argv[]) {
  InstallStackTraceHandler();

  return Catch::Session().run(argc, argv);
}
