file(GLOB _sources
          IR/*.cpp
          Inference/*.cpp
          Canonicalize/*.cpp
          Transforms/*.cpp
          Quantization/*.cpp
          Quantization/BM1684/*.cpp
          Quantization/BM1686/*.cpp
    )

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
  MLIRRewrite
  MLIRTransforms
  MLIRSideEffectInterfaces
  MLIRInferenceInterface
  MLIRQuantizeInterface
  SophgoSupport
)

