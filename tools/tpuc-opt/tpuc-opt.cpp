//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/InitAll.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
using namespace mlir;

int main(int argc, char **argv) {
  tpu_mlir::registerAllPasses();

  DialectRegistry registry;
  tpu_mlir::registerAllDialects(registry);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "TPU MLIR module optimizer driver\n", registry,
                  /*preloadDialectsInContext=*/false));
}
