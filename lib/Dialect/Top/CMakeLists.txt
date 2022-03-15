file(GLOB _sources IR/*.cpp Transforms/*.cpp)

add_mlir_library(SophgoTop
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Top

  DEPENDS
  MLIRTopOpsIncGen
  MLIRTopPassIncGen
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
  MLIRInferenceInterface
  MLIRQuantizeInterface
  SophgoSupport
)

