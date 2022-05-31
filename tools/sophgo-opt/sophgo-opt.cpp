//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "sophgo/InitAll.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
using namespace mlir;

int main(int argc, char **argv) {
  sophgo::registerAllPasses();

  DialectRegistry registry;
  sophgo::registerAllDialects(registry);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Sophgo MLIR module optimizer driver\n", registry,
                  /*preloadDialectsInContext=*/false));
}
