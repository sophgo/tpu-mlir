# This file allows users to call find_package(MLIR) and pick up our targets.

# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(MLIR_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(MLIR_INSTALL_PREFIX "${MLIR_INSTALL_PREFIX}" PATH)
get_filename_component(MLIR_INSTALL_PREFIX "${MLIR_INSTALL_PREFIX}" PATH)
get_filename_component(MLIR_INSTALL_PREFIX "${MLIR_INSTALL_PREFIX}" PATH)

find_package(LLVM REQUIRED CONFIG
             HINTS "${MLIR_INSTALL_PREFIX}/lib/cmake/llvm")

set(MLIR_EXPORTED_TARGETS "mlir-tblgen;MLIRAnalysis;MLIRPresburger;MLIRAsmParser;MLIRBytecodeReader;MLIRBytecodeWriter;MLIRFuncDialect;MLIRPDLDialect;MLIRPDLInterpDialect;MLIRQuantDialect;MLIRQuantUtils;MLIRDialect;MLIRIR;MLIRCallInterfaces;MLIRCastInterfaces;MLIRControlFlowInterfaces;MLIRCopyOpInterface;MLIRDataLayoutInterfaces;MLIRDerivedAttributeOpInterface;MLIRInferIntRangeInterface;MLIRInferTypeOpInterface;MLIRParallelCombiningOpInterface;MLIRSideEffectInterfaces;MLIRTilingInterface;MLIRVectorInterfaces;MLIRViewLikeInterface;MLIRLoopLikeInterface;MLIRParser;MLIRPass;MLIRRewrite;MLIRSupport;MLIRTableGen;MLIROptLib;MLIRTblgenLib;MLIRTransformUtils;MLIRTransforms;MLIRExecutionEngineUtils;MLIRCAPIDebug;obj.MLIRCAPIDebug;MLIRCAPIFunc;obj.MLIRCAPIFunc;MLIRCAPIQuant;obj.MLIRCAPIQuant;MLIRCAPIInterfaces;obj.MLIRCAPIInterfaces;MLIRCAPIIR;obj.MLIRCAPIIR;MLIRCAPIRegisterEverything;obj.MLIRCAPIRegisterEverything;MLIRCAPITransforms;obj.MLIRCAPITransforms;MLIRMlirOptMain;MLIRPythonSources;MLIRPythonSources.Dialects;MLIRPythonSources.Core;MLIRPythonCAPI.HeaderSources;MLIRPythonSources.Dialects.builtin;MLIRPythonSources.Dialects.builtin.ops_gen;MLIRPythonSources.Dialects.func;MLIRPythonSources.Dialects.func.ops_gen;MLIRPythonSources.Dialects.quant;MLIRPythonExtension.Core;MLIRPythonExtension.RegisterEverything;MLIRPythonExtension.Dialects.Quant.Pybind")
set(MLIR_CMAKE_DIR "${MLIR_INSTALL_PREFIX}/lib/cmake/mlir")
set(MLIR_INCLUDE_DIRS "${MLIR_INSTALL_PREFIX}/include")
set(MLIR_TABLEGEN_EXE "mlir-tblgen")
set(MLIR_PDLL_TABLEGEN_EXE "mlir-pdll")
set(MLIR_INSTALL_AGGREGATE_OBJECTS "1")
set(MLIR_ENABLE_BINDINGS_PYTHON "ON")

# For mlir_tablegen()
set(MLIR_INCLUDE_DIR "/workspace/third-party/llvm-project/build/tools/mlir/include")
set(MLIR_MAIN_SRC_DIR "/workspace/third-party/llvm-project/mlir")

set_property(GLOBAL PROPERTY MLIR_ALL_LIBS "MLIRAnalysis;MLIRPresburger;MLIRAsmParser;MLIRBytecodeReader;MLIRBytecodeWriter;MLIRFuncDialect;MLIRPDLDialect;MLIRPDLInterpDialect;MLIRQuantDialect;MLIRQuantUtils;MLIRDialect;MLIRIR;MLIRCallInterfaces;MLIRCastInterfaces;MLIRControlFlowInterfaces;MLIRCopyOpInterface;MLIRDataLayoutInterfaces;MLIRDerivedAttributeOpInterface;MLIRInferIntRangeInterface;MLIRInferTypeOpInterface;MLIRParallelCombiningOpInterface;MLIRSideEffectInterfaces;MLIRTilingInterface;MLIRVectorInterfaces;MLIRViewLikeInterface;MLIRLoopLikeInterface;MLIRParser;MLIRPass;MLIRRewrite;MLIRSupport;MLIRTableGen;MLIROptLib;MLIRTblgenLib;MLIRTransformUtils;MLIRTransforms;MLIRExecutionEngineUtils;MLIRCAPIDebug;obj.MLIRCAPIDebug;MLIRCAPIFunc;obj.MLIRCAPIFunc;MLIRCAPIQuant;obj.MLIRCAPIQuant;MLIRCAPIInterfaces;obj.MLIRCAPIInterfaces;MLIRCAPIIR;obj.MLIRCAPIIR;MLIRCAPIRegisterEverything;obj.MLIRCAPIRegisterEverything;MLIRCAPITransforms;obj.MLIRCAPITransforms;MLIRMlirOptMain")
set_property(GLOBAL PROPERTY MLIR_DIALECT_LIBS "MLIRFuncDialect;MLIRPDLDialect;MLIRPDLInterpDialect;MLIRQuantDialect;MLIRQuantUtils")
set_property(GLOBAL PROPERTY MLIR_CONVERSION_LIBS "")
set_property(GLOBAL PROPERTY MLIR_TRANSLATION_LIBS "")

# Provide all our library targets to users.
# More specifically, configure MLIR so that it can be directly included in a top
# level CMakeLists.txt, but also so that it can be imported via `find_package`.
# This is based on how LLVM handles exports.
if(NOT TARGET MLIRSupport)
  include("${MLIR_CMAKE_DIR}/MLIRTargets.cmake")
endif()

# By creating these targets here, subprojects that depend on MLIR's
# tablegen-generated headers can always depend on these targets whether building
# in-tree with MLIR or not.
if(NOT TARGET mlir-tablegen-targets)
  add_custom_target(mlir-tablegen-targets)
endif()
if(NOT TARGET mlir-headers)
  add_custom_target(mlir-headers)
endif()
if(NOT TARGET mlir-generic-headers)
  add_custom_target(mlir-generic-headers)
endif()
if(NOT TARGET mlir-doc)
  add_custom_target(mlir-doc)
endif()
