add_mlir_public_c_api_library(TPUMLIRCAPI
  Dialects.cpp
  RegisterEverything.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  TPUMLIRTop
  TPUMLIRTpu
  MLIRFuncDialect
  MLIRQuantDialect
)
