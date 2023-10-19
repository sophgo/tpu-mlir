//===- tpuc-test.cpp - MLIR Optimizer Driver ------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Main entry function for tpuc-test for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

namespace tpu_mlir {
namespace test {
void registerTestTilingInterface();
} // namespace test
} // namespace tpu_mlir

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  tpu_mlir::test::registerTestTilingInterface();
  mlir::DialectRegistry registry;
  using namespace mlir;
  // clang-format off
  registry.insert<linalg::LinalgDialect,
                  func::FuncDialect,
                  scf::SCFDialect,
                  tensor::TensorDialect,
                  arith::ArithDialect>();
  // clang-format on

  return asMainReturnCode(
      MlirOptMain(argc, argv, "TPU MLIR test driver\n", registry));
}
