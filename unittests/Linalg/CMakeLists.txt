add_tpumlir_unittest(MLIRIRLinalgTests linalg.cpp)
target_link_libraries(
  MLIRIRLinalgTests #
  PRIVATE
  MLIRIR
  TPUDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRPass)

target_include_directories(MLIRIRLinalgTests
  PUBLIC
  ${PROJECT_SOURCE_DIR}/test/lib/Dialect
)
