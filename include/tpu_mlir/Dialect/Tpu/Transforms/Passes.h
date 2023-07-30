//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"


namespace tpu_mlir {
namespace tpu {

std::unique_ptr<OperationPass<ModuleOp>> createOpReorderPass();
std::unique_ptr<OperationPass<ModuleOp>> createWeightReorderPass();
std::unique_ptr<OperationPass<ModuleOp>> createSubnetDividePass();
std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass();
std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass();
std::unique_ptr<OperationPass<ModuleOp>> createStripIOQuant();
std::unique_ptr<OperationPass<ModuleOp>> createLayerGroupPass();
std::unique_ptr<OperationPass<ModuleOp>> createDynCodegenPass();
std::unique_ptr<OperationPass<ModuleOp>> createChipOptimizePass();
std::unique_ptr<OperationPass<ModuleOp>> createOpDividePass();
std::unique_ptr<OperationPass<ModuleOp>> createDistributePass();
std::unique_ptr<OperationPass<ModuleOp>> createParallelPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h.inc"

} // namespace top
} // namespace mlir
