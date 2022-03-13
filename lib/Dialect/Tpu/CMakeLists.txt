file(GLOB _sources IR/*.cpp)

add_mlir_library(SophgoTpu
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Tops
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Tpu

  DEPENDS
  MLIRTopsOpsIncGen
  MLIRTpuOpsIncGen
  MLIRInferenceInterfaceIncGen
  MLIRQuantizeInterfaceIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFunc
  MLIRQuant
  MLIRSideEffectInterfaces
  SophgoSupport
)

