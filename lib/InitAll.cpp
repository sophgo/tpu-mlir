//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace tpu_mlir {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::tosa::TosaDialect, mlir::func::FuncDialect, top::TopDialect,
              tpu::TpuDialect, mlir::quant::QuantizationDialect>();
}

void registerAllPasses() {
  registerCanonicalizer();
  mlir::registerConversionPasses();
  top::registerTopPasses();
  tpu::registerTpuPasses();
}
} // namespace tpu_mlir
