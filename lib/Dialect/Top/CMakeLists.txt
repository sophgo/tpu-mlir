file(GLOB _sources
          IR/*.cpp
          Interfaces/*.cpp
          Canonicalize/*.cpp
          Transforms/*.cpp
          Lowering/*.cpp
          Lowering/BM1684/*.cpp
          Lowering/BM1686/*.cpp
    )

add_mlir_library(SophgoTop
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Top

  DEPENDS
  MLIRTopOpsIncGen
  MLIRTopPassIncGen
  MLIRInferenceInterfaceIncGen
  MLIRLoweringInterfaceIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFunc
  MLIRQuant
  MLIRRewrite
  MLIRTransforms
  MLIRSideEffectInterfaces
  MLIRInferenceInterface
  MLIRLoweringInterface
  SophgoSupport
  SophgoTrait
)

