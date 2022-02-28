//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/InitAll.h"
#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

void mlir::sophgo::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::tops::TopsDialect>();
  registry.insert<mlir::StandardOpsDialect>();
}

void mlir::sophgo::registerAllPasses() {
  mlir::registerCanonicalizerPass();
}

