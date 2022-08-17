//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"

using namespace mlir;
namespace tpu_mlir {
namespace tpu {

std::unique_ptr<OperationPass<ModuleOp>> createWeightReorderPass();
std::unique_ptr<OperationPass<ModuleOp>> createSubnetDividePass();
std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass();
std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass();
std::unique_ptr<OperationPass<FuncOp>> createLayerGroupPass();
std::unique_ptr<OperationPass<FuncOp>> createStripIOQuant();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h.inc"

} // namespace top
} // namespace mlir
