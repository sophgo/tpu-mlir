add_tpumlir_unittest(
  MLIRIRLinalgTests
  linalg.cpp
  PARTIAL_SOURCES_INTENDED
)

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


add_tpumlir_unittest(
  MLIRIRStructuredTransformTest
  StructuredTransformTest.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(
  MLIRIRStructuredTransformTest #
  PRIVATE
  TPUMLIRCodegenPass
  MLIRIR
  MLIRParser
  MLIRPass)

target_include_directories(MLIRIRStructuredTransformTest
  PUBLIC
  ${PROJECT_SOURCE_DIR}/test/lib/Dialect
  ${PROJECT_SOURCE_DIR}/experimental/include
)
