//===-- Passes.h - ----------------------------------- ----------*- C++ -*-===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"

using namespace mlir;
namespace sophgo {
namespace top {

std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass();
std::unique_ptr<OperationPass<ModuleOp>> createLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createMarkFLOPsPass();
std::unique_ptr<OperationPass<ModuleOp>> createSaveWeightPass();
#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "sophgo/Dialect/Top/Transforms/Passes.h.inc"

} // namespace top
} // namespace mlir
