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

def FusePreprocess : Pass<"fuse-preprocess", "ModuleOp"> {
  let summary = "Fuse preprocess in cvimodels";
  let constructor = "createFusePreprocessPass()";
  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"",
        "default quantization mode: INT8/BF16">,
    Option<"customization_format", "customization_format", "std::string", /*default=*/"",
           "cv18xx pixel_format">,
  ];
  let dependentDialects = ["TopDialect", "mlir::quant::QuantizationDialect"];
}

def AlignInput : Pass<"align-input", "ModuleOp"> {
  let summary = "Fuse align-input in cvimodels";
  let constructor = "createAlignInputPass()";
  let dependentDialects = ["TopDialect", "mlir::quant::QuantizationDialect"];
}
#endif
