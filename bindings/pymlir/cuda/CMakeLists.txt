cmake_minimum_required(VERSION 3.18)
project(cuda_helper LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60;61;70;75;80;86)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

add_library(cuda_helper STATIC cuda_helper.cu)

target_include_directories(cuda_helper PUBLIC
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(cuda_helper PROPERTIES
                      LINKER_LANGUAGE CUDA)
