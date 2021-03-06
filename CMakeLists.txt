CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src init.c)
FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(of "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(of luaT TH)

FIND_PACKAGE(CUDA 4.0 REQUIRED)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_35")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
SET(src-cuda init.cu)
CUDA_ADD_LIBRARY(cuof MODULE ${src-cuda})
TARGET_LINK_LIBRARIES(cuof luaT THC TH ${OpenCV_LIBS})

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(cuof PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

INSTALL(TARGETS cuof
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/cuof")







