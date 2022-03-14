file(GLOB _sources IR/*.cpp Transforms/*.cpp)

add_mlir_library(SophgoTops
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Tops

  DEPENDS
  MLIRTopsOpsIncGen
  MLIRTopsPassIncGen
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

