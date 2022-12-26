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

def Init : Pass<"init", "ModuleOp"> {
  let summary = "init module";
  let constructor = "createInitPass()";
  let dependentDialects = ["TopDialect"];
}

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

def QDQConvert : Pass<"convert-qdq-to-calibrated-dialect", "ModuleOp"> {
  let summary = "Convert from qdq model to regular quantized model";
  let constructor = "createQDQConvertPass()";
  let dependentDialects = ["top::TopDialect", "mlir::quant::QuantizationDialect"];
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
