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

def ImportCalibrationTable : Pass<"import-calibration-table", "ModuleOp"> {
  let summary = "Import calibration table by sophgo-opt";
  let constructor = "mlir::tops::createImportCalibrationTablePass()";
  let options = [
    Option<"tableFile", "file", "std::string", /*default=*/"",
           "calibration table file path">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"true",
           "true for asymmetric quantization, or false for symmetric">,
  ];
  let dependentDialects = ["tops::TopsDialect", "quant::QuantizationDialect"];
}

#endif


