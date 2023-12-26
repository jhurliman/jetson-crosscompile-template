# Specify the compilers
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

# Specify Clang as the CUDA compiler
set(CMAKE_CUDA_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_COMPILER_FORCED ON)

# Set Clang flags for CUDA
set(CMAKE_CUDA_FLAGS "--cuda-gpu-arch=sm_53" CACHE STRING "CUDA flags")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "CUDA architectures" FORCE)

# Set linker flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld" CACHE STRING "Linker flags")

# Set compiler flags for color diagnostics
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fansi-escape-codes -fcolor-diagnostics")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fansi-escape-codes -fcolor-diagnostics")
