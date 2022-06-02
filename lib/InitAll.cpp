//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/InitAll.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"

namespace tpu_mlir {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect, top::TopDialect, tpu::TpuDialect,
                  mlir::quant::QuantizationDialect>();
}

void registerAllPasses() {
  registerCanonicalizerPass();
  registerConversionPasses();
  top::registerTopPasses();
  tpu::registerTpuPasses();
}
} // namespace tpu_mlir
