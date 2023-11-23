//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace tpu_mlir {
using namespace mlir;

void registerDepencyDialect(DialectRegistry &registry);

void registerCodegenInterfaces(DialectRegistry &registry);

std::unique_ptr<Pass> createDecomposeLinalgGenericPass();
std::unique_ptr<Pass> createPrintAffineDimsPass();
std::unique_ptr<Pass> createTileAndFuseGreedilyPass();
std::unique_ptr<Pass> createBufferizePass();
std::unique_ptr<Pass> createInstrctionSelctionPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "tpu-mlir/Transforms/Passes.h.inc"

} // namespace tpu_mlir

// enumerate the affine type
// match the longest or the best benefit
// How to represent benefit dimensional
//
