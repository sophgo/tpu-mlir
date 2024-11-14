//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"
#include "tpu_mlir/Conversion/Passes.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include <mlir/Dialect/Linalg/Passes.h>

namespace tpu_mlir {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::tosa::TosaDialect, mlir::func::FuncDialect, top::TopDialect,
              tpu::TpuDialect, mlir::quant::QuantizationDialect,
              mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
}

void registerAllPasses() {
  registerCanonicalizer();
  mlir::registerConversionPasses();
  top::registerTopPasses();
  tpu::registerTpuPasses();
}

void registerToolPasses() { tpu::registerTruncIO(); }

} // namespace tpu_mlir
