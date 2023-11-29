//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu-mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tpu_mlir {

void registerDepencyDialect(DialectRegistry &registry) {
  // clang-format off
  registry.insert<func::FuncDialect,
                  affine::AffineDialect,
                  linalg::LinalgDialect,
                  memref::MemRefDialect,
                  scf::SCFDialect,
                  tensor::TensorDialect,
                  arith::ArithDialect>();
  // clang-format on
}

void registerCodegenInterfaces(DialectRegistry &registry) {
  linalg::registerTilingInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
}

} // namespace tpu_mlir
} // namespace mlir