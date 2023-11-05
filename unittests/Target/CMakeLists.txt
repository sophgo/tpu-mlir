add_tpumlir_unittest(TPUMLIRIRSG2260Tests sg2260_test.cpp)

target_link_libraries(
  TPUMLIRIRSG2260Tests #
  PRIVATE
  MLIRIR
  TPUMLIRSG2260Dialect
  MLIRFuncDialect
)

target_include_directories(TPUMLIRIRSG2260Tests
  PUBLIC
  ${PROJECT_SOURCE_DIR}/experimental/include/
  ${CMAKE_BINARY_DIR}/experimental/include/
)
