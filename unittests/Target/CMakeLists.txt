add_tpumlir_unittest(TPUMLIRIRBM1690Tests bm1690_test.cpp)

target_link_libraries(
  TPUMLIRIRBM1690Tests #
  PRIVATE
  MLIRIR
  TPUMLIRBM1690Dialect
  MLIRFuncDialect
)

target_include_directories(TPUMLIRIRBM1690Tests
  PUBLIC
  ${PROJECT_SOURCE_DIR}/experimental/include/
  ${CMAKE_BINARY_DIR}/experimental/include/
)
