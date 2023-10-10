add_tpumlir_unittest(MLIRIRLinalgTests linalg.cpp)
target_link_libraries(
  MLIRIRLinalgTests #
  PRIVATE
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRPass)
