//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/InitAll.h"
#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Dialect/Top/Transforms/Passes.h"
#include "sophgo/Dialect/Tpu/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"

namespace sophgo {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect, top::TopDialect, tpu::TpuDialect,
                  mlir::quant::QuantizationDialect>();
}

void registerAllPasses() {
  registerCanonicalizerPass();
  top::registerTopPasses();
  tpu::registerTpuPasses();
}
} // namespace sophgo
