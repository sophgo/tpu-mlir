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
