#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mlir-tblgen" for configuration "Release"
set_property(TARGET mlir-tblgen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mlir-tblgen PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mlir-tblgen"
  )

list(APPEND _IMPORT_CHECK_TARGETS mlir-tblgen )
list(APPEND _IMPORT_CHECK_FILES_FOR_mlir-tblgen "${_IMPORT_PREFIX}/bin/mlir-tblgen" )

# Import target "MLIRPresburger" for configuration "Release"
set_property(TARGET MLIRPresburger APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRPresburger PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRPresburger.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRPresburger )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRPresburger "${_IMPORT_PREFIX}/lib/libMLIRPresburger.a" )

# Import target "MLIRAnalysis" for configuration "Release"
set_property(TARGET MLIRAnalysis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRAnalysis PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRAnalysis.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRAnalysis )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRAnalysis "${_IMPORT_PREFIX}/lib/libMLIRAnalysis.a" )

# Import target "MLIRAsmParser" for configuration "Release"
set_property(TARGET MLIRAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRAsmParser PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRAsmParser.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRAsmParser )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRAsmParser "${_IMPORT_PREFIX}/lib/libMLIRAsmParser.a" )

# Import target "MLIRBytecodeReader" for configuration "Release"
set_property(TARGET MLIRBytecodeReader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRBytecodeReader PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRBytecodeReader.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRBytecodeReader )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRBytecodeReader "${_IMPORT_PREFIX}/lib/libMLIRBytecodeReader.a" )

# Import target "MLIRBytecodeWriter" for configuration "Release"
set_property(TARGET MLIRBytecodeWriter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRBytecodeWriter PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRBytecodeWriter.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRBytecodeWriter )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRBytecodeWriter "${_IMPORT_PREFIX}/lib/libMLIRBytecodeWriter.a" )

# Import target "MLIRObservers" for configuration "Release"
set_property(TARGET MLIRObservers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRObservers PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRObservers.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRObservers )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRObservers "${_IMPORT_PREFIX}/lib/libMLIRObservers.a" )

# Import target "MLIRDebug" for configuration "Release"
set_property(TARGET MLIRDebug APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRDebug PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRDebug.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRDebug )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRDebug "${_IMPORT_PREFIX}/lib/libMLIRDebug.a" )

# Import target "MLIRFuncDialect" for configuration "Release"
set_property(TARGET MLIRFuncDialect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRFuncDialect PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRFuncDialect.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRFuncDialect )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRFuncDialect "${_IMPORT_PREFIX}/lib/libMLIRFuncDialect.a" )

# Import target "MLIRIRDL" for configuration "Release"
set_property(TARGET MLIRIRDL APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRIRDL PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRIRDL.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRIRDL )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRIRDL "${_IMPORT_PREFIX}/lib/libMLIRIRDL.a" )

# Import target "MLIRPDLDialect" for configuration "Release"
set_property(TARGET MLIRPDLDialect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRPDLDialect PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRPDLDialect.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRPDLDialect )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRPDLDialect "${_IMPORT_PREFIX}/lib/libMLIRPDLDialect.a" )

# Import target "MLIRPDLInterpDialect" for configuration "Release"
set_property(TARGET MLIRPDLInterpDialect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRPDLInterpDialect PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRPDLInterpDialect.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRPDLInterpDialect )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRPDLInterpDialect "${_IMPORT_PREFIX}/lib/libMLIRPDLInterpDialect.a" )

# Import target "MLIRQuantDialect" for configuration "Release"
set_property(TARGET MLIRQuantDialect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRQuantDialect PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRQuantDialect.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRQuantDialect )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRQuantDialect "${_IMPORT_PREFIX}/lib/libMLIRQuantDialect.a" )

# Import target "MLIRQuantUtils" for configuration "Release"
set_property(TARGET MLIRQuantUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRQuantUtils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRQuantUtils.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRQuantUtils )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRQuantUtils "${_IMPORT_PREFIX}/lib/libMLIRQuantUtils.a" )

# Import target "MLIRTosaDialect" for configuration "Release"
set_property(TARGET MLIRTosaDialect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRTosaDialect PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRTosaDialect.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRTosaDialect )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRTosaDialect "${_IMPORT_PREFIX}/lib/libMLIRTosaDialect.a" )

# Import target "MLIRDialect" for configuration "Release"
set_property(TARGET MLIRDialect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRDialect PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRDialect.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRDialect )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRDialect "${_IMPORT_PREFIX}/lib/libMLIRDialect.a" )

# Import target "MLIRIR" for configuration "Release"
set_property(TARGET MLIRIR APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRIR PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRIR.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRIR )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRIR "${_IMPORT_PREFIX}/lib/libMLIRIR.a" )

# Import target "MLIRCallInterfaces" for configuration "Release"
set_property(TARGET MLIRCallInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCallInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCallInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCallInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCallInterfaces "${_IMPORT_PREFIX}/lib/libMLIRCallInterfaces.a" )

# Import target "MLIRCastInterfaces" for configuration "Release"
set_property(TARGET MLIRCastInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCastInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCastInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCastInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCastInterfaces "${_IMPORT_PREFIX}/lib/libMLIRCastInterfaces.a" )

# Import target "MLIRControlFlowInterfaces" for configuration "Release"
set_property(TARGET MLIRControlFlowInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRControlFlowInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRControlFlowInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRControlFlowInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRControlFlowInterfaces "${_IMPORT_PREFIX}/lib/libMLIRControlFlowInterfaces.a" )

# Import target "MLIRCopyOpInterface" for configuration "Release"
set_property(TARGET MLIRCopyOpInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCopyOpInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCopyOpInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCopyOpInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCopyOpInterface "${_IMPORT_PREFIX}/lib/libMLIRCopyOpInterface.a" )

# Import target "MLIRDataLayoutInterfaces" for configuration "Release"
set_property(TARGET MLIRDataLayoutInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRDataLayoutInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRDataLayoutInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRDataLayoutInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRDataLayoutInterfaces "${_IMPORT_PREFIX}/lib/libMLIRDataLayoutInterfaces.a" )

# Import target "MLIRDerivedAttributeOpInterface" for configuration "Release"
set_property(TARGET MLIRDerivedAttributeOpInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRDerivedAttributeOpInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRDerivedAttributeOpInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRDerivedAttributeOpInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRDerivedAttributeOpInterface "${_IMPORT_PREFIX}/lib/libMLIRDerivedAttributeOpInterface.a" )

# Import target "MLIRDestinationStyleOpInterface" for configuration "Release"
set_property(TARGET MLIRDestinationStyleOpInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRDestinationStyleOpInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRDestinationStyleOpInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRDestinationStyleOpInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRDestinationStyleOpInterface "${_IMPORT_PREFIX}/lib/libMLIRDestinationStyleOpInterface.a" )

# Import target "MLIRInferIntRangeInterface" for configuration "Release"
set_property(TARGET MLIRInferIntRangeInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRInferIntRangeInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRInferIntRangeInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRInferIntRangeInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRInferIntRangeInterface "${_IMPORT_PREFIX}/lib/libMLIRInferIntRangeInterface.a" )

# Import target "MLIRInferTypeOpInterface" for configuration "Release"
set_property(TARGET MLIRInferTypeOpInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRInferTypeOpInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRInferTypeOpInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRInferTypeOpInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRInferTypeOpInterface "${_IMPORT_PREFIX}/lib/libMLIRInferTypeOpInterface.a" )

# Import target "MLIRLoopLikeInterface" for configuration "Release"
set_property(TARGET MLIRLoopLikeInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRLoopLikeInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRLoopLikeInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRLoopLikeInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRLoopLikeInterface "${_IMPORT_PREFIX}/lib/libMLIRLoopLikeInterface.a" )

# Import target "MLIRMemorySlotInterfaces" for configuration "Release"
set_property(TARGET MLIRMemorySlotInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRMemorySlotInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRMemorySlotInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRMemorySlotInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRMemorySlotInterfaces "${_IMPORT_PREFIX}/lib/libMLIRMemorySlotInterfaces.a" )

# Import target "MLIRParallelCombiningOpInterface" for configuration "Release"
set_property(TARGET MLIRParallelCombiningOpInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRParallelCombiningOpInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRParallelCombiningOpInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRParallelCombiningOpInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRParallelCombiningOpInterface "${_IMPORT_PREFIX}/lib/libMLIRParallelCombiningOpInterface.a" )

# Import target "MLIRRuntimeVerifiableOpInterface" for configuration "Release"
set_property(TARGET MLIRRuntimeVerifiableOpInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRRuntimeVerifiableOpInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRRuntimeVerifiableOpInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRRuntimeVerifiableOpInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRRuntimeVerifiableOpInterface "${_IMPORT_PREFIX}/lib/libMLIRRuntimeVerifiableOpInterface.a" )

# Import target "MLIRShapedOpInterfaces" for configuration "Release"
set_property(TARGET MLIRShapedOpInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRShapedOpInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRShapedOpInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRShapedOpInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRShapedOpInterfaces "${_IMPORT_PREFIX}/lib/libMLIRShapedOpInterfaces.a" )

# Import target "MLIRSideEffectInterfaces" for configuration "Release"
set_property(TARGET MLIRSideEffectInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRSideEffectInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRSideEffectInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRSideEffectInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRSideEffectInterfaces "${_IMPORT_PREFIX}/lib/libMLIRSideEffectInterfaces.a" )

# Import target "MLIRTilingInterface" for configuration "Release"
set_property(TARGET MLIRTilingInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRTilingInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRTilingInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRTilingInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRTilingInterface "${_IMPORT_PREFIX}/lib/libMLIRTilingInterface.a" )

# Import target "MLIRVectorInterfaces" for configuration "Release"
set_property(TARGET MLIRVectorInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRVectorInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRVectorInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRVectorInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRVectorInterfaces "${_IMPORT_PREFIX}/lib/libMLIRVectorInterfaces.a" )

# Import target "MLIRViewLikeInterface" for configuration "Release"
set_property(TARGET MLIRViewLikeInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRViewLikeInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRViewLikeInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRViewLikeInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRViewLikeInterface "${_IMPORT_PREFIX}/lib/libMLIRViewLikeInterface.a" )

# Import target "MLIRValueBoundsOpInterface" for configuration "Release"
set_property(TARGET MLIRValueBoundsOpInterface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRValueBoundsOpInterface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRValueBoundsOpInterface.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRValueBoundsOpInterface )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRValueBoundsOpInterface "${_IMPORT_PREFIX}/lib/libMLIRValueBoundsOpInterface.a" )

# Import target "MLIRInferIntRangeCommon" for configuration "Release"
set_property(TARGET MLIRInferIntRangeCommon APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRInferIntRangeCommon PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRInferIntRangeCommon.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRInferIntRangeCommon )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRInferIntRangeCommon "${_IMPORT_PREFIX}/lib/libMLIRInferIntRangeCommon.a" )

# Import target "MLIRParser" for configuration "Release"
set_property(TARGET MLIRParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRParser PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRParser.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRParser )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRParser "${_IMPORT_PREFIX}/lib/libMLIRParser.a" )

# Import target "MLIRPass" for configuration "Release"
set_property(TARGET MLIRPass APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRPass PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRPass.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRPass )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRPass "${_IMPORT_PREFIX}/lib/libMLIRPass.a" )

# Import target "MLIRRewrite" for configuration "Release"
set_property(TARGET MLIRRewrite APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRRewrite PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRRewrite.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRRewrite )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRRewrite "${_IMPORT_PREFIX}/lib/libMLIRRewrite.a" )

# Import target "MLIRSupport" for configuration "Release"
set_property(TARGET MLIRSupport APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRSupport PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRSupport.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRSupport )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRSupport "${_IMPORT_PREFIX}/lib/libMLIRSupport.a" )

# Import target "MLIRTableGen" for configuration "Release"
set_property(TARGET MLIRTableGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRTableGen PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRTableGen.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRTableGen )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRTableGen "${_IMPORT_PREFIX}/lib/libMLIRTableGen.a" )

# Import target "MLIROptLib" for configuration "Release"
set_property(TARGET MLIROptLib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIROptLib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIROptLib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIROptLib )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIROptLib "${_IMPORT_PREFIX}/lib/libMLIROptLib.a" )

# Import target "MLIRTblgenLib" for configuration "Release"
set_property(TARGET MLIRTblgenLib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRTblgenLib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRTblgenLib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRTblgenLib )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRTblgenLib "${_IMPORT_PREFIX}/lib/libMLIRTblgenLib.a" )

# Import target "MLIRPluginsLib" for configuration "Release"
set_property(TARGET MLIRPluginsLib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRPluginsLib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRPluginsLib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRPluginsLib )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRPluginsLib "${_IMPORT_PREFIX}/lib/libMLIRPluginsLib.a" )

# Import target "MLIRTransformUtils" for configuration "Release"
set_property(TARGET MLIRTransformUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRTransformUtils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRTransformUtils.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRTransformUtils )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRTransformUtils "${_IMPORT_PREFIX}/lib/libMLIRTransformUtils.a" )

# Import target "MLIRTransforms" for configuration "Release"
set_property(TARGET MLIRTransforms APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRTransforms PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRTransforms.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRTransforms )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRTransforms "${_IMPORT_PREFIX}/lib/libMLIRTransforms.a" )

# Import target "MLIRExecutionEngineUtils" for configuration "Release"
set_property(TARGET MLIRExecutionEngineUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRExecutionEngineUtils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRExecutionEngineUtils.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRExecutionEngineUtils )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRExecutionEngineUtils "${_IMPORT_PREFIX}/lib/libMLIRExecutionEngineUtils.a" )

# Import target "MLIRCAPIDebug" for configuration "Release"
set_property(TARGET MLIRCAPIDebug APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCAPIDebug PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCAPIDebug.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCAPIDebug )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCAPIDebug "${_IMPORT_PREFIX}/lib/libMLIRCAPIDebug.a" )

# Import target "obj.MLIRCAPIDebug" for configuration "Release"
set_property(TARGET obj.MLIRCAPIDebug APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(obj.MLIRCAPIDebug PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_OBJECTS_RELEASE "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIDebug/Debug.cpp.o"
  )

list(APPEND _IMPORT_CHECK_TARGETS obj.MLIRCAPIDebug )
list(APPEND _IMPORT_CHECK_FILES_FOR_obj.MLIRCAPIDebug "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIDebug/Debug.cpp.o" )

# Import target "MLIRCAPIFunc" for configuration "Release"
set_property(TARGET MLIRCAPIFunc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCAPIFunc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCAPIFunc.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCAPIFunc )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCAPIFunc "${_IMPORT_PREFIX}/lib/libMLIRCAPIFunc.a" )

# Import target "obj.MLIRCAPIFunc" for configuration "Release"
set_property(TARGET obj.MLIRCAPIFunc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(obj.MLIRCAPIFunc PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_OBJECTS_RELEASE "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIFunc/Func.cpp.o"
  )

list(APPEND _IMPORT_CHECK_TARGETS obj.MLIRCAPIFunc )
list(APPEND _IMPORT_CHECK_FILES_FOR_obj.MLIRCAPIFunc "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIFunc/Func.cpp.o" )

# Import target "MLIRCAPIQuant" for configuration "Release"
set_property(TARGET MLIRCAPIQuant APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCAPIQuant PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCAPIQuant.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCAPIQuant )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCAPIQuant "${_IMPORT_PREFIX}/lib/libMLIRCAPIQuant.a" )

# Import target "obj.MLIRCAPIQuant" for configuration "Release"
set_property(TARGET obj.MLIRCAPIQuant APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(obj.MLIRCAPIQuant PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_OBJECTS_RELEASE "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIQuant/Quant.cpp.o"
  )

list(APPEND _IMPORT_CHECK_TARGETS obj.MLIRCAPIQuant )
list(APPEND _IMPORT_CHECK_FILES_FOR_obj.MLIRCAPIQuant "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIQuant/Quant.cpp.o" )

# Import target "MLIRCAPIInterfaces" for configuration "Release"
set_property(TARGET MLIRCAPIInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCAPIInterfaces PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCAPIInterfaces.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCAPIInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCAPIInterfaces "${_IMPORT_PREFIX}/lib/libMLIRCAPIInterfaces.a" )

# Import target "obj.MLIRCAPIInterfaces" for configuration "Release"
set_property(TARGET obj.MLIRCAPIInterfaces APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(obj.MLIRCAPIInterfaces PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_OBJECTS_RELEASE "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIInterfaces/Interfaces.cpp.o"
  )

list(APPEND _IMPORT_CHECK_TARGETS obj.MLIRCAPIInterfaces )
list(APPEND _IMPORT_CHECK_FILES_FOR_obj.MLIRCAPIInterfaces "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIInterfaces/Interfaces.cpp.o" )

# Import target "MLIRCAPIIR" for configuration "Release"
set_property(TARGET MLIRCAPIIR APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCAPIIR PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCAPIIR.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCAPIIR )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCAPIIR "${_IMPORT_PREFIX}/lib/libMLIRCAPIIR.a" )

# Import target "obj.MLIRCAPIIR" for configuration "Release"
set_property(TARGET obj.MLIRCAPIIR APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(obj.MLIRCAPIIR PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_OBJECTS_RELEASE "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/AffineExpr.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/AffineMap.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/BuiltinAttributes.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/BuiltinTypes.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/Diagnostics.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/DialectHandle.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/IntegerSet.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/IR.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/Pass.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/Support.cpp.o"
  )

list(APPEND _IMPORT_CHECK_TARGETS obj.MLIRCAPIIR )
list(APPEND _IMPORT_CHECK_FILES_FOR_obj.MLIRCAPIIR "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/AffineExpr.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/AffineMap.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/BuiltinAttributes.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/BuiltinTypes.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/Diagnostics.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/DialectHandle.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/IntegerSet.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/IR.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/Pass.cpp.o;${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIIR/Support.cpp.o" )

# Import target "MLIRCAPIRegisterEverything" for configuration "Release"
set_property(TARGET MLIRCAPIRegisterEverything APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCAPIRegisterEverything PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCAPIRegisterEverything.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCAPIRegisterEverything )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCAPIRegisterEverything "${_IMPORT_PREFIX}/lib/libMLIRCAPIRegisterEverything.a" )

# Import target "obj.MLIRCAPIRegisterEverything" for configuration "Release"
set_property(TARGET obj.MLIRCAPIRegisterEverything APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(obj.MLIRCAPIRegisterEverything PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_OBJECTS_RELEASE "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIRegisterEverything/RegisterEverything.cpp.o"
  )

list(APPEND _IMPORT_CHECK_TARGETS obj.MLIRCAPIRegisterEverything )
list(APPEND _IMPORT_CHECK_FILES_FOR_obj.MLIRCAPIRegisterEverything "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPIRegisterEverything/RegisterEverything.cpp.o" )

# Import target "MLIRCAPITransforms" for configuration "Release"
set_property(TARGET MLIRCAPITransforms APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRCAPITransforms PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRCAPITransforms.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRCAPITransforms )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRCAPITransforms "${_IMPORT_PREFIX}/lib/libMLIRCAPITransforms.a" )

# Import target "obj.MLIRCAPITransforms" for configuration "Release"
set_property(TARGET obj.MLIRCAPITransforms APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(obj.MLIRCAPITransforms PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_OBJECTS_RELEASE "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPITransforms/Passes.cpp.o"
  )

list(APPEND _IMPORT_CHECK_TARGETS obj.MLIRCAPITransforms )
list(APPEND _IMPORT_CHECK_FILES_FOR_obj.MLIRCAPITransforms "${_IMPORT_PREFIX}/lib/objects-Release/obj.MLIRCAPITransforms/Passes.cpp.o" )

# Import target "MLIRMlirOptMain" for configuration "Release"
set_property(TARGET MLIRMlirOptMain APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLIRMlirOptMain PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMLIRMlirOptMain.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLIRMlirOptMain )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLIRMlirOptMain "${_IMPORT_PREFIX}/lib/libMLIRMlirOptMain.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
