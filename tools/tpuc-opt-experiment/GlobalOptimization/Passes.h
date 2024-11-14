#pragma once
#include "Utils/PassUtils.h"
#include "include/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
namespace mlir {
void buildGlobalOptimizationPassPipeline(OpPassManager &mainPassManager,
                                         bool dynamic_mode = false);

std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass();
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createRemoveZeroExtentTensorsPass();

std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createEraseUnusedLinalgOperands();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGeneralizeLinalgNamedOpsPass();

std::unique_ptr<OperationPass<ModuleOp>> createStripDebugOpPass();

std::unique_ptr<OperationPass<ModuleOp>> createVerifyInputLegalityPass();

std::unique_ptr<OperationPass<ModuleOp>>
createTensorPadToTensorInsertSlicePass(bool skipSingleLinalgOpUses = false);

std::unique_ptr<OperationPass<ModuleOp>> createInterchangeGenericOpsPass();

std::unique_ptr<OperationPass<ModuleOp>> createCollapseDimsPass();

// fusion the elementwise base on linalg-on-tensor
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFusionOfTensorOpsPass(bool fuseMultiUse = false,
                            unsigned multiUseFusionIteration = 2);

std::unique_ptr<OperationPass<ModuleOp>>
createSubgraphSplitPass(bool dynamic_mode = false);
#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "GlobalOptimization/Passes.h.inc"
} // namespace  mlir
