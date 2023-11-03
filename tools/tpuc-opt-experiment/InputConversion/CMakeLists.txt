set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name InputConversion)
add_public_tablegen_target(InputConversionPassIncGen)
file(GLOB _sources
          *.cpp
    )

add_mlir_library(MLIRInputConversion
  ${_sources}

  DEPENDS
    InputConversionPassIncGen
    GlobalOptimizationPassIncGen

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRLinalgTransforms
    MLIRMLProgramDialect
    MLIRFuncDialect
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRShapeOpsTransforms
    MLIRShapeToStandard
    MLIRTransforms
)
