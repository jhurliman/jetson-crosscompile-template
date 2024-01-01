cmake_minimum_required(VERSION 3.18)

project(JetsonCrossCompilation LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(cpp_warnings)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(test src/test.cpp src/hello.cu)
target_compile_options(test PRIVATE ${PROJECT_WARNING_FLAGS})
target_include_directories(test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(test PRIVATE CUDA::cudart)
