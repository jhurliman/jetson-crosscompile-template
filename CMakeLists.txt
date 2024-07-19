cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(JetsonCrossCompilation LANGUAGES CXX CUDA)

option(COMPILE_CPU "Compile the CPU backend" ON)
option(COMPILE_CUDA "Compile the CUDA backend" ON)
option(COMPILE_T194 "Compile the Jetson AGX Xavier backend" OFF)
option(COMPILE_T210 "Compile the Jetson Nano backend" OFF)
option(BUILD_TESTS "Build tests" ON)

if(NOT COMPILE_CPU
   AND NOT COMPILE_CUDA
   AND NOT COMPILE_T194
   AND NOT COMPILE_T210)
  message(FATAL_ERROR "At least one backend must be compiled")
endif()

if(COMPILE_CUDA OR COMPILE_T194 OR COMPILE_T210)
  enable_language(CUDA)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(CPM)
include(CppWarnings)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

cpmaddpackage("gh:fmtlib/fmt#10.2.1")
cpmaddpackage("gh:jeremy-rifkin/cpptrace@0.5.1")
cpmaddpackage("gh:TartanLlama/expected@1.1.0")

# Work around dependencies including with `-I` instead of `-isystem`
get_target_property(FMT_INCLUDE_DIRS fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(EXPECTED_INCLUDE_DIRS tl::expected INTERFACE_INCLUDE_DIRECTORIES)

file(GLOB_RECURSE COMMON_SOURCES "src/backend/common/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/backend/gpu/*.cpp" "src/backend/gpu/*.cu")

if(COMPILE_CPU)
  cpmaddpackage("gh:oneapi-src/oneTBB@2021.11.0")

  file(GLOB_RECURSE CPU_SOURCES "src/backend/cpu/*.cpp")
  add_library(backend_cpu STATIC ${COMMON_SOURCES} ${CPU_SOURCES})
  target_compile_options(backend_cpu PRIVATE ${PROJECT_WARNING_FLAGS})
  target_include_directories(
    backend_cpu
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu)
  target_include_directories(backend_cpu SYSTEM PRIVATE ${EXPECTED_INCLUDE_DIRS})
  target_link_libraries(backend_cpu PUBLIC cpptrace::cpptrace tl::expected TBB::tbb)
endif()

if(COMPILE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(backend_cuda STATIC ${COMMON_SOURCES} ${CUDA_SOURCES})
  target_compile_options(backend_cuda PRIVATE ${PROJECT_WARNING_FLAGS})
  target_include_directories(
    backend_cuda
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/gpu)
  target_include_directories(backend_cuda SYSTEM PRIVATE ${EXPECTED_INCLUDE_DIRS})
  target_link_libraries(backend_cuda PUBLIC cpptrace::cpptrace tl::expected CUDA::cudart
                                            CUDA::nvToolsExt)
  target_compile_definitions(
    backend_cuda PUBLIC -DUSE_GPU -DCUDA_MAJOR=${CUDAToolkit_VERSION_MAJOR}
                        -DCUDA_MINOR=${CUDAToolkit_VERSION_MINOR})
endif()

if(COMPILE_T194)
  find_package(CUDAToolkit REQUIRED)

  add_library(backend_t194 STATIC ${COMMON_SOURCES} ${CUDA_SOURCES})
  target_compile_options(backend_t194 PRIVATE ${PROJECT_WARNING_FLAGS})
  target_include_directories(
    backend_t194
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/gpu)
  target_include_directories(backend_t194 SYSTEM PRIVATE ${EXPECTED_INCLUDE_DIRS})
  target_link_libraries(backend_t194 PUBLIC cpptrace::cpptrace tl::expected CUDA::cudart
                                            CUDA::nvToolsExt)
  target_compile_definitions(
    backend_t194 PUBLIC -DUSE_GPU -DUSE_T194 -DCUDA_MAJOR=${CUDAToolkit_VERSION_MAJOR}
                        -DCUDA_MINOR=${CUDAToolkit_VERSION_MINOR})
endif()

if(COMPILE_T210)
  find_package(CUDAToolkit REQUIRED)

  add_library(backend_t210 STATIC ${COMMON_SOURCES} ${CUDA_SOURCES})
  target_compile_options(backend_t210 PRIVATE ${PROJECT_WARNING_FLAGS})
  target_include_directories(
    backend_t210
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/gpu)
  target_include_directories(backend_t210 SYSTEM PRIVATE ${EXPECTED_INCLUDE_DIRS})
  target_link_libraries(backend_t210 PUBLIC cpptrace::cpptrace tl::expected CUDA::cudart
                                            CUDA::nvToolsExt)
  target_compile_definitions(
    backend_t210 PUBLIC -DUSE_GPU -DUSE_T210 -DCUDA_MAJOR=${CUDAToolkit_VERSION_MAJOR}
                        -DCUDA_MINOR=${CUDAToolkit_VERSION_MINOR})
endif()

add_executable(myapp src/main.cpp)
target_compile_options(myapp PRIVATE ${PROJECT_WARNING_FLAGS})
target_include_directories(myapp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(myapp SYSTEM PRIVATE ${FMT_INCLUDE_DIRS})
target_link_libraries(myapp PRIVATE fmt::fmt)
if(COMPILE_T194)
  target_link_libraries(myapp PRIVATE backend_t194)
  target_include_directories(myapp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/backend/gpu)
elseif(COMPILE_T210)
  target_link_libraries(myapp PRIVATE backend_t210)
  target_include_directories(myapp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/backend/gpu)
elseif(COMPILE_CUDA)
  target_link_libraries(myapp PRIVATE backend_cuda)
  target_include_directories(myapp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/backend/gpu)
else()
  target_link_libraries(myapp PRIVATE backend_cpu)
  target_include_directories(myapp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/backend/cpu)
endif()

if(BUILD_TESTS)
  cpmaddpackage("gh:catchorg/Catch2@3.5.1")
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()

# Copy compile_commands.json to the root of `build/` so clangd and other tools can find it
add_custom_target(
  copy-compile-commands ALL
  ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR}/build
)
