set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross compilers
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CUDA_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "clang++")

set(TARGET_TRIPLE "aarch64-linux-gnu")
set(CMAKE_C_COMPILER_TARGET "${TARGET_TRIPLE}")
set(CMAKE_CUDA_COMPILER_TARGET "${TARGET_TRIPLE}")
set(CMAKE_CXX_COMPILER_TARGET "${TARGET_TRIPLE}")

# Set the sysroot path
set(CMAKE_SYSROOT "${CMAKE_CURRENT_LIST_DIR}/sysroot/jetson-t194")
if(NOT EXISTS ${CMAKE_SYSROOT})
  message(
    FATAL_ERROR
      "CMAKE_SYSROOT does not exist: ${CMAKE_SYSROOT}\nPlease run ./scripts/extract-sysroot.sh --board-id t194")
endif()

set(SYSROOT_CUDA "${CMAKE_SYSROOT}/usr/local/cuda-11.4")

# Path to the GCC toolchain for the target architecture
set(GCC_TOOLCHAIN "${CMAKE_CURRENT_LIST_DIR}/sysroot/bootlin-toolchain-gcc-93")
if(NOT EXISTS ${GCC_TOOLCHAIN})
  message(
    FATAL_ERROR
      "GCC_TOOLCHAIN does not exist: ${GCC_TOOLCHAIN}\nPlease run ./scripts/extract-sysroot.sh --board-id t194")
endif()

# Path to the host (x86_64) vendored CUDA toolkit
set(CUDAToolkit_ROOT "${CMAKE_CURRENT_LIST_DIR}/nvidia/cuda-11.4_amd64")
if(NOT EXISTS ${CUDAToolkit_ROOT})
  message(
    FATAL_ERROR
      "CUDAToolkit_ROOT does not exist: ${CUDAToolkit_ROOT}\nPlease run ./scripts/extract-cuda.sh --cuda 11.4")
endif()

set(CUDA_TOOLKIT_INCLUDE "${SYSROOT_CUDA}/include")
set(CUDA_CUDART "${SYSROOT_CUDA}/lib64/libcudart.so")
set(CUDA_CUDART_LIBRARY "${SYSROOT_CUDA}/lib64/libcudart.so")
set(CUDA_NVTX_LIBRARY "${SYSROOT_CUDA}/lib64/libnvToolsExt.so")

# CMake 3.22 is not using `CMAKE_EXE_LINKER_FLAGS` when compiling a CUDA test
# program, causing the linker to fail. This is a workaround
set(CMAKE_CUDA_FLAGS "-fuse-ld=lld")

# sm_72 is the compute capability of the Jetson AGX Xavier (t194)
set(CMAKE_CUDA_ARCHITECTURES "72")

set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "Enable separable compilation for CUDA")

# Adjust the default behavior of the FIND_XXX() commands: search programs in the host environment
# only.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search headers and libraries in the target environment only.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
  -isystem ${CMAKE_SYSROOT}/usr/include/c++/9 \
  -isystem ${CMAKE_SYSROOT}/usr/include/${TARGET_TRIPLE}/c++/9")

# Set the default linker flags to use the LLD linker, find the CUDA libraries,
# and find the GCC toolchain libraries in the sysroot
set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld \
  -L${SYSROOT_CUDA}/lib64 \
  -B${CMAKE_SYSROOT}/lib/gcc/${TARGET_TRIPLE}/9 \
  -L${CMAKE_SYSROOT}/lib/gcc/${TARGET_TRIPLE}/9 \
  -L${CMAKE_SYSROOT}/lib/${TARGET_TRIPLE}"
  CACHE STRING "Linker flags")

# Prevent CMake from adding `-isystem ${CMAKE_SYSROOT}/usr/include` when compiling CUDA files, which
# messes up the include order for <cmath> and <math.h> and breaks compilation
set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}
                                            ${CMAKE_SYSROOT}/usr/include)

# Set compiler flags for color diagnostics
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fansi-escape-codes -fcolor-diagnostics")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fansi-escape-codes -fcolor-diagnostics")
