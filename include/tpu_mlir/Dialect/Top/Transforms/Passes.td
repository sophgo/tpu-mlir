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

def Deinit : Pass<"deinit", "ModuleOp"> {
  let summary = "deinit module";
  let constructor = "createDeinitPass()";
  let dependentDialects = ["TopDialect"];
}

def ChipAssign : Pass<"chip-assign", "ModuleOp"> {
  let summary = "Assign chip type";
  let constructor = "createChipAssignPass()";
  let dependentDialects = ["TopDialect"];
  let options = [
    Option<"chip", "chip", "std::string", /*default=*/"",
           "chip: cv183x/cv182x/bm1684/bm1684x/bm1686">,
  ];
}

def ChipOptimize : Pass<"chip-top-optimize", "ModuleOp"> {
  let summary = "Before lowering, do some extra Op conversions for different chips";
  let constructor = "createChipOptimizePass()";
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

def AfterOptimize : Pass<"after-optimize", "ModuleOp"> {
  let summary = "after top optimize in mlir by tpuc-opt";
  let constructor = "createAfterOptimizePass()";
  let dependentDialects = ["TopDialect"];
}

def ShapeInfer : Pass<"shape-infer", "ModuleOp"> {
  let summary = "do shape inference for each op";
  let constructor = "createShapeInferPass()";
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
  let options = [
    Option<"chip", "chip", "std::string", /*default=*/"",
           "chip: cv181x/cv182x/cv183x">,
    Option<"customization_format", "customization_format", "std::string", /*default=*/"",
           "cv18xx pixel_format">,
  ];
  let dependentDialects = ["TopDialect", "mlir::quant::QuantizationDialect"];
}

def PostHandle : Pass<"post-handle", "ModuleOp"> {
  let summary = "post handle in mlir by tpuc-opt";
  let constructor = "createPostHandlePass()";
  let dependentDialects = ["TopDialect"];
  let options = [
      Option<"type", "type", "std::string", /*default=*/"",
             "type of post-handle.">
  ];
}
#endif
