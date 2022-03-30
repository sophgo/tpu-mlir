//===-- Passes.h - TOSA optimization pass declarations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the optimization passes for the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"

using namespace mlir;
namespace sophgo {
namespace tpu {

std::unique_ptr<OperationPass<ModuleOp>> createWeightReorderPass();
std::unique_ptr<OperationPass<ModuleOp>> createSubnetDividePass();
std::unique_ptr<OperationPass<ModuleOp>> createAddressAsignPass();
std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "sophgo/Dialect/Tpu/Transforms/Passes.h.inc"

} // namespace top
} // namespace mlir
