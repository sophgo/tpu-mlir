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
  MLIRInplaceInterface
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

# Exclude files with conflicting static function/class names from unity builds.
# These files define identically-named static functions or inline classes
# that would cause ODR violations when combined into a single translation unit.
if(CMAKE_UNITY_BUILD)
  foreach(_file ${_sources})
    # BM1684X: is_sign, SpecAssign, has_pad
    if(_file MATCHES "BM1684X/(AddConst|Mul|MulConst|SubConst|MaxPoolWithMask|Pooling1D|Pooling2D|Pooling3D)\\.cpp$")
      set_source_files_properties(${_file} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
    # BM1684: isInt8
    if(_file MATCHES "BM1684/(Max|Min)\\.cpp$")
      set_source_files_properties(${_file} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
    # Common: normlize_bf16, normlize_f32, inference, cvi_int8_to_bf16, active_func
    if(_file MATCHES "Common/(GroupNorm|GroupNormTrain|InstanceNorm|LayerNorm|LayerNormTrain|RMSNorm|PixelNorm|Cast|DtypeCast|Active|ActiveCast|GRU)\\.cpp$")
      set_source_files_properties(${_file} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
    # CV18xx WeightReorder: transposeBiasFp32, transposeConvolutionFilter, packWeight, transpose_row_col
    if(_file MATCHES "WeightReorder/CV18xx/(Conv2D|Conv3D|Deconv|GRU|LSTM)\\.cpp$")
      set_source_files_properties(${_file} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
    # ProcessorOptimize files — include Common.h headers with inline class definitions
    if(_file MATCHES "ProcessorOptimize/")
      set_source_files_properties(${_file} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
  endforeach()
endif()
