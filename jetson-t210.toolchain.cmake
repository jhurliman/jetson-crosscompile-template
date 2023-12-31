set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross compilers
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

# Set the sysroot path
set(CMAKE_SYSROOT "${CMAKE_CURRENT_LIST_DIR}/sysroot/jetson-t210")
if(NOT EXISTS ${CMAKE_SYSROOT})
  message(FATAL_ERROR "CMAKE_SYSROOT does not exist: ${CMAKE_SYSROOT}\nPlease run ./scripts/extract-sysroot.sh")
endif()

set(SYSROOT_CUDA "${CMAKE_SYSROOT}/usr/local/cuda-10.2")

# Path to the GCC toolchain for the target architecture
set(GCC_TOOLCHAIN "${CMAKE_CURRENT_LIST_DIR}/sysroot/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu")
if(NOT EXISTS ${GCC_TOOLCHAIN})
  message(FATAL_ERROR "GCC_TOOLCHAIN does not exist: ${GCC_TOOLCHAIN}\nPlease run ./scripts/extract-sysroot.sh")
endif()

set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/nvidia/cuda-10.2_amd64")
set(CUDAToolkit_ROOT "${CMAKE_CURRENT_LIST_DIR}/nvidia/cuda-10.2_amd64")
if(NOT EXISTS ${CUDAToolkit_ROOT})
  message(FATAL_ERROR "CUDAToolkit_ROOT does not exist: ${CUDAToolkit_ROOT}\nPlease run ./scripts/extract-cuda.sh")
endif()

set(CUDA_TOOLKIT_INCLUDE "${SYSROOT_CUDA}/include")
set(CUDA_CUDART_LIBRARY "${SYSROOT_CUDA}/lib64/libcudart.so")

# Specify Clang as the CUDA compiler
set(CMAKE_CUDA_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_COMPILER_FORCED ON)

# Set Clang flags for CUDA
set(CMAKE_CUDA_FLAGS "--target=aarch64-linux-gnu --cuda-gpu-arch=sm_53" CACHE STRING "CUDA flags")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "CUDA architectures" FORCE)

# Adjust the default behavior of the FIND_XXX() commands:
# search programs in the host environment only.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search headers and libraries in the target environment only.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Set compiler/linker flags for cross-compiling
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=aarch64-linux-gnu --sysroot=${CMAKE_SYSROOT} --gcc-toolchain=${GCC_TOOLCHAIN}" CACHE STRING "C flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --target=aarch64-linux-gnu --sysroot=${CMAKE_SYSROOT} --gcc-toolchain=${GCC_TOOLCHAIN} -isystem ${SYSROOT_CUDA}/include" CACHE STRING "C++ flags")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld -L${SYSROOT_CUDA}/lib64" CACHE STRING "Linker flags")

# Set compiler flags for color diagnostics
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fansi-escape-codes -fcolor-diagnostics")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fansi-escape-codes -fcolor-diagnostics")
