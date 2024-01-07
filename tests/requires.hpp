#pragma once

#include <tl/expected.hpp>

#include <catch2/catch.hpp>

#define REQUIRE_EXPECTED(result)                                                                   \
  std::move((result).or_else([](auto&& err) { FAIL("[ERROR] " << err.errorMessage); }).value())

#define REQUIRE_UNEXPECTED(result)                                                                 \
  std::move((result)                                                                               \
              .transform([](auto&&) { FAIL("[ERROR] Unexpected success"); })                       \
              .or_else([](auto&& err) { return std::move(err); })                                  \
              .error())

#define REQUIRE_NO_ERROR(result)                                                                   \
  do {                                                                                             \
    auto&& _result = (result);                                                                     \
    if (_result) { FAIL("[ERROR] " << _result->errorMessage); }                                    \
  } while (false)
