set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GlobalOpt)
add_public_tablegen_target(GlobalOptimizationPassIncGen)
file(GLOB _sources
          *.cpp
    )

add_mlir_library(MLIRGlobalOptimization
  ${_sources}

  DEPENDS
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
