//===-- Passes.td - Calibration pass declarations ----------*- tablegen -*-===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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
           "chip: cv183x/cv182x/bm1684/bm1684x">,
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
