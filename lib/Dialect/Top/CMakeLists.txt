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

# Exclude files with conflicting static function/class names from unity builds.
if(CMAKE_UNITY_BUILD)
  foreach(_file ${_sources})
    # ProcessorOptimize files — include Common.h with inline class definitions
    if(_file MATCHES "ProcessorOptimize/")
      set_source_files_properties(${_file} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
    # StructOptimize — mlir::Value / llvm::json::Value name ambiguity
    if(_file MATCHES "Transforms/StructOptimize\\.cpp$")
      set_source_files_properties(${_file} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
  endforeach()
endif()
