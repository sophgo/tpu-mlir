//===-- Passes.h - ----------------------------------- ----------*- C++ -*-===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"


namespace tpu_mlir {
namespace top {

std::unique_ptr<OperationPass<ModuleOp>> createInitPass();
std::unique_ptr<OperationPass<ModuleOp>> createDeinitPass();
std::unique_ptr<OperationPass<ModuleOp>> createProcessorAssignPass();
std::unique_ptr<OperationPass<ModuleOp>> createProcessorOptimizePass();
std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass();
std::unique_ptr<OperationPass<ModuleOp>> createQDQConvertPass();
std::unique_ptr<OperationPass<ModuleOp>> createExtraOptimizePass();
std::unique_ptr<OperationPass<ModuleOp>> createFusePreprocessPass();
std::unique_ptr<OperationPass<ModuleOp>> createAddPostprocessPass();
std::unique_ptr<OperationPass<ModuleOp>> createShapeInferPass();

void WeightFolder(Operation *op);
#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h.inc"

} // namespace top
} // namespace tpu_mlir
