//===-- Passes.h - ----------------------------------- ----------*- C++ -*-===//
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

using namespace mlir;
namespace tpu_mlir {
namespace top {

std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass();
std::unique_ptr<OperationPass<ModuleOp>> createLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createMarkFLOPsPass();
std::unique_ptr<OperationPass<ModuleOp>> createSaveWeightPass();
#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h.inc"

} // namespace top
} // namespace mlir
