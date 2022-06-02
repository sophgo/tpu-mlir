//===-- Passes.td - Calibration pass declarations ----------*- tablegen -*-===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOP Dialect to the TPU Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_DIALECT_TOPPASSES
#define TPU_MLIR_DIALECT_TOPPASSES

include "mlir/Pass/PassBase.td"

def ImportCalibrationTable : Pass<"import-calibration-table", "ModuleOp"> {
  let summary = "Import calibration table by tpuc-opt";
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
  let summary = "lowering mlir by tpuc-opt";
  let constructor = "createLoweringPass()";
  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"",
           "default quantization mode: INT8/BF16/F32">,
    Option<"chip", "chip", "std::string", /*default=*/"",
           "chip: cv183x/cv182x/bm1684/bm1686">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"false",
           "true for asymmetric quantization, or false for symmetric">,
  ];
  let dependentDialects = ["TopDialect", "::tpu_mlir::tpu::TpuDialect", "mlir::quant::QuantizationDialect"];
}

def MarkFLOPs : Pass<"mark-FLOPs", "ModuleOp"> {
  let summary = "mark FLOPs in mlir by tpuc-opt";
  let constructor = "createMarkFLOPsPass()";
  let dependentDialects = ["TopDialect"];
}

def SaveWeight : Pass<"save-weight", "ModuleOp"> {
  let summary = "save weight by tpuc-opt";
  let constructor = "createSaveWeightPass()";
  let dependentDialects = ["TopDialect"];
}

#endif
