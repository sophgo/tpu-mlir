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

def FutureUpdate : Pass<"future-update", "ModuleOp"> {
  let summary = "add some op for updating in future";
  let constructor = "createFutureUpdatePass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"rank", "rank", "int64_t", /*default=*/"0",
           "lora rank">,
    Option<"weight_list", "weight_list", "std::string", /*default=*/"\"none\"", "which weight to update">,
  ];
}

def OpReorder : Pass<"op-reorder", "ModuleOp"> {
  let summary = "op reorder in tpu by tpuc-opt";
  let constructor = "createOpReorderPass()";
  let dependentDialects = ["TpuDialect"];
}

def WeightFold : Pass<"weight-fold", "ModuleOp"> {
  let summary = "fold weight if all input of an operation is weight";
  let constructor = "createWeightFoldPass()";
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

def LayerGroup : Pass<"layer-group", "ModuleOp"> {
  let summary = "convert to layer group in tpu by tpuc-opt";
  let constructor = "createLayerGroupPass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"opt", "opt", "int64_t", /*default=*/"2",
           "opt=1: group layers as many as possible. opt=2: dynamic programming layer group">,
    Option<"group_by_cores", "group_by_cores", "std::string", /*default=*/"\"auto\"", "whether force group by cores">,
    Option<"compress_mode", "compress_mode", "std::string", /*default=*/"\"none\"", "compress mode">,
    Option<"lgcache", "lgcache", "std::string", /*default=*/"\"true\"", "whether to dump cut_results">,
  ];
}

def DevParallel : Pass<"dev-parallel", "ModuleOp"> {
  let summary = "distribute module to multi modules to run in multi devices";
  let constructor = "createDevParallelPass()";
  let dependentDialects = ["TpuDialect"];
}

def CoreParallel : Pass<"core-parallel", "ModuleOp"> {
  let summary = "split the operation to fine-grained and run it in parallel on TPU";
  let constructor = "createCoreParallelPass()";
  let dependentDialects = ["TpuDialect"];
}

def DDRInterleave : Pass<"ddr-interleave", "ModuleOp"> {
  let summary = "DDR interleave mode";
  let constructor = "createDDRInterleavePass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"interleave_region", "interleave-region", "int64_t", /*default=*/"1",
           "interleave_region:1 The DDRC is configured to balance performance across most use cases.">,
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
           "save to model file">,
    Option<"embed_debug_info", "embed_debug_info", "bool", /*default=*/"false",
           "embed debug and profiling data to model file.">,
    Option<"model_version", "model_version", "std::string", /*default=*/"\"lastest\"",
           "model version.">,
    Option<"bmodel_only", "bmodel_only", "bool", /*default=*/"false",
           "dump bmodel only.">,
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
    Option<"quant_input_list", "quant_input_list", "std::string", /*default=*/"",
           "choose index to strip input quant.">,
    Option<"quant_output_list", "quant_output_list", "std::string", /*default=*/"",
           "choose index to strip output quant.">,
  ];
}

def ProcessorOptimize : Pass<"processor-tpu-optimize", "ModuleOp"> {
  let summary = "aplly passes in tpu by tpuc-opt";
  let constructor = "createProcessorOptimizePass()";
  let dependentDialects = ["TpuDialect"];
}

def OpDivide : Pass<"op-divide", "ModuleOp"> {
  let summary = "divide large global op to save global memory";
  let constructor = "createOpDividePass()";
  let dependentDialects = ["TpuDialect"];
}

def ShapeOptimize : Pass<"shape-optimize", "ModuleOp"> {
  let summary = "optimize bad shape in tpu by tpuc-opt";
  let constructor = "createShapeOptimizePass()";
  let dependentDialects = ["TpuDialect"];
}

def ShowAddress : Pass<"show-address", "ModuleOp"> {
  let summary = "print final mlir address by tpuc-opt";
  let constructor = "createShowAddressPass()";
  let dependentDialects = ["TpuDialect"];
}

def TruncIO : Pass<"trunc-io", "ModuleOp"> {
  let summary = "truncate final mlir according to inputs/outputs, "
    "keeping the structure as far as possible.";
  let constructor = "createTruncIOPass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"inputs", "inputs", "std::string", /*default=*/"",
           "new inputs">,
    Option<"outputs", "outputs", "std::string", /*default=*/"",
           "new outputs">,
  ];
}

#endif
