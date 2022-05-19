file(GLOB _sources
          IR/*.cpp
          Inference/*.cpp
          Interfaces/BM1684/*.cpp
          Interfaces/BM1686/*.cpp
          Interfaces/Common/*.cpp
          Transforms/*.cpp
          Transforms/LayerGroup/*.cpp)

add_mlir_library(SophgoTpu
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Top
  ${PROJECT_SOURCE_DIR}/include/sophgo/Dialect/Tpu

  DEPENDS
  MLIRTopOpsIncGen
  MLIRTpuOpsIncGen
  MLIRTpuPassIncGen
  MLIRTpuAttrGen
  MLIRInferenceInterfaceIncGen
  MLIRWeightReorderInterfaceIncGen
  MLIRGlobalGenInterfaceIncGen
  MLIRFlopsInterfaceIncGen
  MLIRLocalGenInterfaceIncGen
  Builder

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
  MLIRWeightReorderInterface
  MLIRGlobalGenInterface
  MLIRFlopsInterface
  MLIRLocalGenInterface
  SophgoSupport
  SophgoTrait
  SophgoBackend
  SophgoBuilder
)

