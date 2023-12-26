set(PROJECT_WARNING_FLAGS
  # Warnings as errors
  "-Werror"

  # Enable most warnings
  "-Wall"
  "-Wextra"

  # Readability, tidiness, and portability
  "-Wshadow"
  "-Wold-style-cast"
  "-Wunused"
  "-Wpedantic"

  # Correctness, type safety, and performance
  "-Woverloaded-virtual"
  "-Wnon-virtual-dtor"
  "-Wcast-align"
  "-Wdouble-promotion"

  # Security-related
  "-Walloca"
  "-Wcast-qual"
  "-Wconversion"
  "-Wsign-conversion"
  "-Wformat=2"
  "-Wformat-security"
  "-Wnull-dereference"
  "-Wstack-protector"
  "-Wvla"
  "-Warray-bounds"
  "-Warray-bounds-pointer-arithmetic"
  "-Wassign-enum"
  "-Wbad-function-cast"
  "-Wconditional-uninitialized"
  "-Wconversion"
  "-Wfloat-equal"
  "-Wformat-type-confusion"
  "-Widiomatic-parentheses"
  "-Wimplicit-fallthrough"
  "-Wloop-analysis"
  "-Wpointer-arith"
  "-Wshift-sign-overflow"
  "-Wshorten-64-to-32"
  "-Wswitch-enum"
  "-Wtautological-constant-in-range-compare"
  "-Wunreachable-code-aggressive"
  "-Wthread-safety"
  "-Wthread-safety-beta"
  "-Wcomma"

  # Disabled warnings
  "-Wno-c++98-compat"
  "-Wno-c++98-compat-pedantic"
)
