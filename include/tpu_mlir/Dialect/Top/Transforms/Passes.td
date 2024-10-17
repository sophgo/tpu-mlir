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
  let options = [
    Option<"freq", "freq", "uint64_t", /*default=*/"",
           "chip frequance">,
    Option<"weight_in_mem", "weight_in_mem", "bool", /*default=*/"false",
           "whether to save weight in memory instead of hard disk.">,
    Option<"level", "level", "uint64_t", /*default=*/"",
           "log level">,
  ];
}

def Deinit : Pass<"deinit", "ModuleOp"> {
  let summary = "deinit module";
  let constructor = "createDeinitPass()";
  let dependentDialects = ["TopDialect"];
  let options = [
    Option<"no_save_weight", "no_save_weight", "bool", /*default=*/"false",
           "whether to save weight.">,
  ];
}

def ProcessorAssign : Pass<"processor-assign", "ModuleOp"> {
  let summary = "Assign chip type";
  let constructor = "createProcessorAssignPass()";
  let dependentDialects = ["TopDialect"];
  let options = [
    Option<"chip", "chip", "std::string", /*default=*/"",
           "chip: cv183x/cv182x/cv186x/bm1684/bm1684x/bm1688/bm1690/mars3">,
    Option<"mode", "mode", "std::string", /*default=*/"",
           "default quantization mode: INT8/BF16/F16/F32/F8/F8E4M3/F8E5M2">,
    Option<"num_device", "num_device", "int64_t", /*default=*/"1",
           "num of devices to distributed.">,
    Option<"num_core", "num_core", "int64_t", /*default=*/"1",
           "core_num=1: Set how many cores will be used to run model in parallel.">,
    Option<"addr_mode", "addr_mode", "std::string", /*default=*/"\"auto\"", "addr assign mode">,
  ];
}

def ProcessorOptimize : Pass<"processor-top-optimize", "ModuleOp"> {
  let summary = "Before lowering, do some extra Op conversions for different chips";
  let constructor = "createProcessorOptimizePass()";
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

def ExtraOptimize : Pass<"extra-optimize", "ModuleOp"> {
  let summary = "after top optimize in mlir by tpuc-opt";
  let constructor = "createExtraOptimizePass()";
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
           "set input pixel_format">,
    Option<"align", "align", "bool", /*default=*/"false",
           "whether input align, only for cv18xx">,
  ];
  let dependentDialects = ["TopDialect", "mlir::quant::QuantizationDialect"];
}

def AddPostprocess : Pass<"add-postprocess", "ModuleOp"> {
  let summary = "post handle in mlir by tpuc-opt";
  let constructor = "createAddPostprocessPass()";
  let dependentDialects = ["TopDialect"];
  let options = [
      Option<"type", "type", "std::string", /*default=*/"",
             "type of add-postprocess.">
  ];
}
#endif
