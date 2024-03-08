add_tpumlir_unittest(
  TPUMLIRLinalgTests
  linalg.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(
  TPUMLIRLinalgTests #
  PRIVATE
  MLIRIR
  TPUDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRPass)

target_include_directories(TPUMLIRLinalgTests
  PUBLIC
  ${PROJECT_SOURCE_DIR}/test/lib/Dialect
)


add_tpumlir_unittest(
  TPUMLIRStructuredTransformTest
  StructuredTransformTest.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(
  TPUMLIRStructuredTransformTest #
  PRIVATE
  TPUMLIRCodegenPass
  MLIRIR
  MLIRParser
  MLIRPass)

target_include_directories(TPUMLIRStructuredTransformTest
  PUBLIC
  ${PROJECT_SOURCE_DIR}/test/lib/Dialect
  ${PROJECT_SOURCE_DIR}/experimental/include
)

add_tpumlir_unittest(
  TPUMLIRBenefitTest
  BenefitTest.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(
  TPUMLIRBenefitTest #
  PRIVATE
  TPUMLIRCodegenPass
  MLIRIR
  MLIRParser
  MLIRPass
  TPUMLIRBM1690Dialect
)

target_include_directories(TPUMLIRBenefitTest
  PUBLIC
  ${PROJECT_SOURCE_DIR}/experimental/include
  ${CMAKE_BINARY_DIR}/experimental/include
)
