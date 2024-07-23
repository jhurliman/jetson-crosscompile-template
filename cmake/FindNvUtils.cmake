# - Try to find the NVIDIA Tegra Multimedia API (r34+ NvUtils variant)
# Once done this will define
#  NvUtils_FOUND - System has NvUtils
#  NvUtils_INCLUDE_DIRS - The NvUtils include directories
#  NvUtils_LIBRARIES - The libraries needed to use NvUtils
#  NvUtils_DEFINITIONS - Compiler switches required for using NvUtils

find_package(PkgConfig)

find_path(NvUtils_INCLUDE_DIR nvbufsurftransform.h
          HINTS /usr/src/jetson_multimedia_api/include)

find_library(NVBUFSURFACE_LIBRARY NAMES nvbufsurface
             HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)

find_library(NVBUFSURFTRANSFORM_LIBRARY NAMES nvbufsurftransform
             HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)

set(NvUtils_LIBRARIES ${NVBUFSURFACE_LIBRARY} ${NVBUFSURFTRANSFORM_LIBRARY})
set(NvUtils_INCLUDE_DIRS ${NvUtils_INCLUDE_DIR})
set(NvUtils_DEFINITIONS -DNvUtils_SUPPORTED -DUSE_NVUTILS)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NvUtils_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(NvUtils DEFAULT_MSG
                                  NVBUFSURFACE_LIBRARY NVBUFSURFTRANSFORM_LIBRARY NvUtils_INCLUDE_DIR)

mark_as_advanced(NvUtils_INCLUDE_DIR NVBUFSURFACE_LIBRARY NVBUFSURFTRANSFORM_LIBRARY)
