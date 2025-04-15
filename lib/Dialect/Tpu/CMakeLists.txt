file(GLOB _sources
          IR/*.cpp
          Interfaces/BM1684/*.cpp
          Interfaces/BM1684X/*.cpp
          Interfaces/CV18xx/*.cpp
          Interfaces/Common/*.cpp
          Canonicalize/*.cpp
          Transforms/*.cpp
          Transforms/BM168x/*.cpp
          Transforms/CV18xx/*.cpp
          Transforms/LayerGroup/*.cpp
          Transforms/LayerGroup/opt3/*.cpp
          Transforms/Codegen/*.cpp
          Transforms/Codegen/Dynamic/*.cpp
          Transforms/AddressAssign/*.cpp
          Transforms/ProcessorOptimize/*.cpp
          Transforms/WeightReorder/*.cpp
          Transforms/WeightReorder/BM1684/*.cpp
          Transforms/WeightReorder/BM1684X/*.cpp
          Transforms/WeightReorder/CV18xx/*.cpp
          Transforms/DevParallel/*.cpp
          Transforms/CoreParallel/*.cpp
          Transforms/OptPostProcessor/*.cpp
)

add_llvm_library(TPUMLIRTpu
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Top
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Tpu

  DEPENDS
  CV18xx_Builder
  MLIRTopOpsIncGen
  MLIRTpuOpsIncGen
  MLIRTpuPassIncGen
  MLIRTpuAttrGen
  MLIRInferenceInterfaceIncGen
  MLIRGlobalGenInterfaceIncGen
  MLIRLocalGenInterfaceIncGen
  MLIRDynLocalGenInterfaceIncGen
  MLIRDynGlobalGenInterfaceIncGen
  MLIRTypeInterfaceIncGen
  MLIRIndexingMapsInterfaceIncGen
  # MLIRTpuRewriteIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRRewrite
  MLIRTransforms
  MLIRInferenceInterface
  MLIRGlobalGenInterface
  MLIRLocalGenInterface
  MLIRDynLocalGenInterface
  MLIRDynGlobalGenInterface
  MLIRIndexingMapsInterface
  MLIRTypeInterface
  TPUMLIRSupport
  TPUMLIRTrait
  TPUMLIRBackend
  TPUMLIRBuilder
  ortools
)
