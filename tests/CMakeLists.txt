file(GLOB_RECURSE TEST_SOURCES "test_*.cpp" "main.cpp")

if(COMPILE_CPU)
  add_executable(unit_tests_cpu ${TEST_SOURCES})
  target_compile_options(unit_tests_cpu PRIVATE ${PROJECT_WARNING_FLAGS})
  target_include_directories(unit_tests_cpu PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                    ${PROJECT_SOURCE_DIR}/src/backend/cpu)
  target_link_libraries(unit_tests_cpu Catch2::Catch2 backend_cpu)
  add_test(NAME unit_tests_cpu COMMAND unit_tests_cpu)
endif()

if(COMPILE_CUDA)
  add_executable(unit_tests_cuda ${TEST_SOURCES})
  target_compile_options(unit_tests_cuda PRIVATE ${PROJECT_WARNING_FLAGS})
  target_include_directories(unit_tests_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                     ${PROJECT_SOURCE_DIR}/src/backend/gpu)
  target_link_libraries(unit_tests_cuda Catch2::Catch2 backend_cuda)
  add_test(NAME unit_tests_cuda COMMAND unit_tests_cuda)
endif()

if(COMPILE_T210)
  add_executable(unit_tests_t210 ${TEST_SOURCES})
  target_compile_options(unit_tests_t210 PRIVATE ${PROJECT_WARNING_FLAGS})
  target_include_directories(unit_tests_t210 PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                     ${PROJECT_SOURCE_DIR}/src/backend/gpu)
  target_link_libraries(unit_tests_t210 Catch2::Catch2 backend_t210)
  add_test(NAME unit_tests_t210 COMMAND unit_tests_t210)
endif()
