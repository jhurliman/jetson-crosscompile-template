cmake_minimum_required(VERSION 3.18)

project(JetsonCrossCompilation LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(cpp_warnings)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(test src/test.cpp src/hello.cu)
set_target_properties(test PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(test PRIVATE ${PROJECT_WARNING_FLAGS})
target_link_libraries(test PRIVATE CUDA::cudart_static CUDA::cuda_driver)
