set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    ${cuda_test_libs}
    MLIRTestFuncToLLVM
    MLIRAffineTransformsTestPasses
    MLIRArithTestPasses
    MLIRBufferizationTestPasses
    MLIRControlFlowTestPasses
    MLIRDLTITestPasses
    MLIRFuncTestPasses
    MLIRGPUTestPasses
    MLIRLinalgTestPasses
    MLIRLoopLikeInterfaceTestPasses
    MLIRMathTestPasses
    MLIRMemRefTestPasses
    MLIRNVGPUTestPasses
    MLIRSCFTestPasses
    MLIRShapeTestPasses
    MLIRSPIRVTestPasses
    MLIRTensorTestPasses
    MLIRTestAnalysis
    MLIRTestDialect
    MLIRTestDynDialect
    MLIRTestIR
    MLIRTestOneToNTypeConversionPass
    MLIRTestPass
    MLIRTestPDLL
    MLIRTestReducer
    MLIRTestRewrite
    MLIRTestTransformDialect
    MLIRTestTransforms
    MLIRTilingInterfaceTestPasses
    MLIRVectorTestPasses
    MLIRTestVectorToSPIRV
    MLIRLLVMTestPasses
    )
endif()

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${test_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

add_llvm_executable(tpuc-opt-exp
  tpuc-opt-exp.cpp

  DEPENDS
  ${LIBS}
  SUPPORT_PLUGINS
  )
target_link_libraries(tpuc-opt-exp PRIVATE ${LIBS})
llvm_update_compile_flags(tpuc-opt-exp)

mlir_check_all_link_libraries(tpuc-opt-exp)
