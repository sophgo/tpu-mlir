//===-- Passes.td - TOSA pass declarations ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_DIALECT_TOPS_PASSES
#define SOPHGO_DIALECT_TOPS_PASSES

include "mlir/Pass/PassBase.td"

def ImportCalibration : Pass<"import-calibration", "ModuleOp"> {
  let summary = "Import calibration table by sophgo-opt";
  let constructor = "mlir::tops::createImportCalibrationPass()";
  let options = [
    Option<"calibrationTable", "calibration-table", "std::string", /*default=*/"",
           "calibration table file path">,
  ];
  let dependentDialects = ["tops::TopsDialect"];
}

#endif


