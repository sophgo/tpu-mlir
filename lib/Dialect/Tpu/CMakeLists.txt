file(GLOB _sources
          IR/*.cpp
          Inference/*.cpp
          WeightReorder/*.cpp
          WeightReorder/BM1684/*.cpp
          WeightReorder/BM1686/*.cpp
          Codegen/*.cpp
          Codegen/BM1684/*.cpp
          Codegen/BM1686/*.cpp
          Transforms/*.cpp)

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
  MLIRCodegenInterfaceIncGen
  MLIRFlopsInterfaceIncGen

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
  MLIRCodegenInterface
  MLIRFlopsInterface
  SophgoSupport
  SophgoTrait
  SophgoBackend
  SophgoBuilder
)

