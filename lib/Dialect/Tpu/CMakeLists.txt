file(GLOB _sources IR/*.cpp Transforms/*.cpp WeightReorder/*.cpp)

add_mlir_library(SophgoTpu
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Top
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Tpu

  DEPENDS
  MLIRTopOpsIncGen
  MLIRTpuOpsIncGen
  MLIRTpuPassIncGen
  MLIRInferenceInterfaceIncGen
  MLIRWeightReorderInterfaceIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFunc
  MLIRQuant
  MLIRSideEffectInterfaces
  MLIRInferenceInterface
  MLIRWeightReorderInterface
  SophgoSupport
)

