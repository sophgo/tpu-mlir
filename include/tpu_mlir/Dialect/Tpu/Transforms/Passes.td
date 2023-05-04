//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_DIALECT_TOPPASSES
#define TPU_MLIR_DIALECT_TOPPASSES

include "mlir/Pass/PassBase.td"

def OpReorder : Pass<"op-reorder", "FuncOp"> {
  let summary = "op reorder in tpu by tpuc-opt";
  let constructor = "createOpReorderPass()";
  let dependentDialects = ["TpuDialect"];
}

def WeightReorder : Pass<"weight-reorder", "ModuleOp"> {
  let summary = "weight reorder in tpu by tpuc-opt";
  let constructor = "createWeightReorderPass()";
  let dependentDialects = ["TpuDialect"];
}

def SubnetDivide : Pass<"subnet-divide", "ModuleOp"> {
  let summary = "subnet divide in tpu by tpuc-opt";
  let constructor = "createSubnetDividePass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
      Option<"dynamic", "dynamic", "bool", /*default=*/"false",
             "dynamic compiler or not.">
  ];
}

def LayerGroup : Pass<"layer-group", "FuncOp"> {
  let summary = "convert to layer group in tpu by tpuc-opt";
  let constructor = "createLayerGroupPass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"opt", "opt", "int64_t", /*default=*/"2",
           "opt=1: group layers as many as possible. opt=2: dynamic programming layer group">,
  ];
}

def AddressAssign : Pass<"address-assign", "ModuleOp"> {
  let summary = "assign address in tpu by tpuc-opt";
  let constructor = "createAddressAssignPass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"reuse_addr", "reuse_addr", "bool", /*default=*/"true",
           "reuse tensor memory.">,
    Option<"merge_weight", "merge_weight", "bool", /*default=*/"false",
           "merge weight memory.">,
    Option<"compress_weight", "compress_weight", "bool", /*default=*/"true",
           "compress weight memory.">,
    Option<"weight_map_file", "weight_map_file", "std::string", /*default=*/"\"_weight_map.csv\"",
           "record weight offset with its name into a csv map file.">,
  ];
}

def Codegen : Pass<"codegen", "ModuleOp"> {
  let summary = "codegen in tpu by tpuc-opt";
  let constructor = "createCodegenPass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"model_file", "model_file", "std::string", /*default=*/"",
           "save to model file">
  ];
}

def StripIOQuant : Pass<"strip-io-quant", "ModuleOp"> {
  let summary = "remove input & output fp32<->int8 converiton in int8model";
  let constructor = "createStripIOQuant()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"quant_input", "quant_input", "bool", /*default=*/"false",
           "strip input quant.">,
    Option<"quant_output", "quant_output", "bool", /*default=*/"false",
           "strip output quant.">,
  ];
}

def ChipOptimize : Pass<"chip-optimize", "ModuleOp"> {
  let summary = "aplly passes in tpu by tpuc-opt";
  let constructor = "createChipOptimizePass()";
  let dependentDialects = ["TpuDialect"];
}

#endif
