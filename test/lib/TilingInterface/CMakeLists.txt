add_mlir_library(TPUMLIRTilingInterfaceTestPasses
  TestTilingInterface.cpp

  EXCLUDE_FROM_LIBMLIR
  LINK_LIBS
  PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTilingInterfaceImpl)
