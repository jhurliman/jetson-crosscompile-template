# Specify the compilers
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CUDA_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "clang++")

if(APPLE)
  # Use the only CUDA toolkit available for macOS (10.2)
  set(CUDAToolkit_ROOT "${CMAKE_CURRENT_LIST_DIR}/nvidia/cuda-10.2_macos")
  if(NOT EXISTS ${CUDAToolkit_ROOT})
    message(
      FATAL_ERROR
        "CUDAToolkit_ROOT does not exist: ${CUDAToolkit_ROOT}\nPlease run ./scripts/extract-cuda.sh")
  endif()

  # FIXME: This doesn't fully work because the CUDA test program is compiled with `-lrt` which is
  # not available on macOS
else()
  # Find the latest host (x86_64) vendored CUDA toolkit
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/nvidia/cuda-11.4_amd64")
    set(CUDAToolkit_ROOT "${CMAKE_CURRENT_LIST_DIR}/nvidia/cuda-11.4_amd64")
  elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/nvidia/cuda-10.2_amd64")
    set(CUDAToolkit_ROOT "${CMAKE_CURRENT_LIST_DIR}/nvidia/cuda-10.2_amd64")
  else()
    message(FATAL_ERROR "CUDAToolkit_ROOT does not exist. Please run ./scripts/extract-cuda.sh")
  endif()
endif()

set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "Enable separable compilation for CUDA")

# Use LLD as the linker on non-MacOS platforms
if(NOT APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld" CACHE STRING "Linker flags")
endif()

# Set compiler flags for color diagnostics
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fansi-escape-codes -fcolor-diagnostics")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fansi-escape-codes -fcolor-diagnostics")
