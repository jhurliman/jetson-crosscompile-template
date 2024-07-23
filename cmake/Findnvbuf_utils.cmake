# - Try to find the NVIDIA Tegra Multimedia API (r32 nvbuf_utils variant)
# Once done this will define
#  nvbuf_utils_FOUND - System has nvbuf_utils
#  nvbuf_utils_INCLUDE_DIRS - The nvbuf_utils include directories
#  nvbuf_utils_LIBRARIES - The libraries needed to use nvbuf_utils
#  nvbuf_utils_DEFINITIONS - Compiler switches required for using nvbuf_utils

find_package(PkgConfig)

find_path(nvbuf_utils_INCLUDE_DIR nvbuf_utils.h
          HINTS /usr/src/jetson_multimedia_api/include)

find_library(nvbuf_utils_LIBRARY NAMES nvbuf_utils
             HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)

set(nvbuf_utils_LIBRARIES ${nvbuf_utils_LIBRARY})
set(nvbuf_utils_INCLUDE_DIRS ${nvbuf_utils_INCLUDE_DIR})
set(nvbuf_utils_DEFINITIONS -DNVMMAPI_SUPPORTED -DUSE_NVBUF_UTILS)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set nvbuf_utils_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(nvbuf_utils DEFAULT_MSG
                                  nvbuf_utils_LIBRARY nvbuf_utils_INCLUDE_DIR)

mark_as_advanced(nvbuf_utils_INCLUDE_DIR nvbuf_utils_LIBRARY)
