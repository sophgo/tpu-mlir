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

#ifndef SOPHGO_DIALECT_TOPPASSES
#define SOPHGO_DIALECT_TOPPASSES

include "mlir/Pass/PassBase.td"

def ImportCalibrationTable : Pass<"import-calibration-table", "ModuleOp"> {
  let summary = "Import calibration table by sophgo-opt";
  let constructor = "createImportCalibrationTablePass()";
  let options = [
    Option<"tableFile", "file", "std::string", /*default=*/"",
           "calibration table file path">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"true",
           "true for asymmetric quantization, or false for symmetric">,
  ];
  let dependentDialects = ["top::TopDialect", "mlir::quant::QuantizationDialect"];
}

def Lowering : Pass<"lowering", "ModuleOp"> {
  let summary = "lowering mlir by sophgo-opt";
  let constructor = "createLoweringPass()";
  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"",
           "default quantization mode: INT8/BF16/F32">,
    Option<"chip", "chip", "std::string", /*default=*/"",
           "chip: cv183x/cv182x/bm1684/bm1686">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"false",
           "true for asymmetric quantization, or false for symmetric">,
  ];
  let dependentDialects = ["TopDialect", "::sophgo::tpu::TpuDialect", "mlir::quant::QuantizationDialect"];
}

def MarkFLOPs : Pass<"mark-FLOPs", "ModuleOp"> {
  let summary = "mark FLOPs in mlir by sophgo-opt";
  let constructor = "createMarkFLOPsPass()";
  let dependentDialects = ["TopDialect"];
}

def SaveWeight : Pass<"save-weight", "ModuleOp"> {
  let summary = "save weight by sophgo-opt";
  let constructor = "createSaveWeightPass()";
  let dependentDialects = ["TopDialect"];
}

#endif


