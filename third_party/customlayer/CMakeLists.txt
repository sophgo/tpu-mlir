cmake_minimum_required(VERSION 3.5)
project(backend_custom)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${SRC_DIR}/include)
include_directories(${SRC_DIR}/include/kernel)

file(GLOB_RECURSE SRC_FILES
    ${SRC_DIR}/src/*.cpp
    ${SRC_DIR}/src/*.c
)

add_library(backend_custom SHARED ${SRC_FILES})
set_target_properties(backend_custom PROPERTIES LINK_FLAGS "-g")

install(TARGETS backend_custom LIBRARY DESTINATION lib)

install(FILES nntoolchain/lib/libbackend_1684.so
              nntoolchain/lib/libbackend_1684x.so
              nntoolchain/lib/libbackend_1686.so
              nntoolchain/lib/libbmlib.so.0
              nntoolchain/lib/libbmlib.so
              nntoolchain/lib/libcmodel_1684.so
              nntoolchain/lib/libcmodel_1684x.so
              nntoolchain/lib/libcmodel_1686.so
              nntoolchain/lib/libcmodel.so
              nntoolchain/lib/libbmrt.so
              nntoolchain/lib/libcpuop.so
              nntoolchain/lib/libbm1684x_kernel_module.so
              # cvitek
              CV18xx/lib/libcvikernel.so
              CV18xx/lib/libcviruntime.so
              CV18xx/lib/libcvicmodel.so
              CV18xx/lib/libunwind.so.8
        DESTINATION lib)
