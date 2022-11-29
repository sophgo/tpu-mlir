file(GLOB _sources
          IR/*.cpp
          Interfaces/BM1684/*.cpp
          Interfaces/BM1684X/*.cpp
          Interfaces/CV18xx/*.cpp
          Interfaces/Common/*.cpp
          Canonicalize/*.cpp
          Transforms/*.cpp
          Transforms/BM168x/*.cpp
          Transforms/BM1684/*.cpp
          Transforms/CV18xx/*.cpp
          Transforms/LayerGroup/*.cpp)

add_llvm_library(TPUMLIRTpu
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Top
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Tpu

  DEPENDS
  MLIRTopOpsIncGen
  MLIRTpuOpsIncGen
  MLIRTpuPassIncGen
  MLIRTpuAttrGen
  MLIRInferenceInterfaceIncGen
  MLIRGlobalGenInterfaceIncGen
  MLIRLocalGenInterfaceIncGen
  MLIRTypeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRRewrite
  MLIRTransforms
  MLIRSideEffectInterfaces
  MLIRInferenceInterface
  MLIRGlobalGenInterface
  MLIRLocalGenInterface
  MLIRTypeInterface
  TPUMLIRSupport
  TPUMLIRTrait
  TPUMLIRBackend
  TPUMLIRBuilder
  ssl
  crypto
)
