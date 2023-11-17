file(GLOB _sources
          IR/*.cpp
          Interfaces/*.cpp
          Canonicalize/*.cpp
          Transforms/*.cpp
          Transforms/ProcessorOptimize/*.cpp
    )

add_mlir_library(TPUMLIRTop
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Top

  DEPENDS
  MLIRTopAttrGen
  MLIRTopOpsIncGen
  MLIRTopPassIncGen
  MLIRInferenceInterfaceIncGen
  MLIRFlopsInterfaceIncGen
  MLIRShapeInterfaceIncGen

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
  MLIRFlopsInterface
  MLIRShapeInterface
  TPUMLIRSupport
  TPUMLIRTrait
)
