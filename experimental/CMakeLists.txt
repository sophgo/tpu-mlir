# optin(MLIR_DIR)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include/tpu-mlir)
add_subdirectory(lib)
