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

#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"

namespace mlir {
namespace tops {

std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass();
std::unique_ptr<OperationPass<ModuleOp>> createQuantizePass();
#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "sophgo/Dialect/Tops/Transforms/Passes.h.inc"

} // namespace tops
} // namespace mlir
