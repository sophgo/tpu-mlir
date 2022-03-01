file(GLOB _sources IR/*.cpp Transforms/*.cpp)

add_mlir_library(SophgoTops
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Tops

  DEPENDS
  MLIRTopsOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  SophgoSupport
)

