#pragma once
#include "include/Utils.h"
#include "Utils/PassUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"

namespace  mlir
{
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager);

std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass();
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createRemoveZeroExtentTensorsPass();

std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createEraseUnusedLinalgOperands();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "GlobalOptimization/Passes.h.inc"
} // namespace  mlir

