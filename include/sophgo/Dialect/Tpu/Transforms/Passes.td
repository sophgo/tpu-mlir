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

def WeightReorder : Pass<"weight-reorder", "ModuleOp"> {
  let summary = "weight reorder in tpu by sophgo-opt";
  let constructor = "createWeightReorderPass()";
  let dependentDialects = ["TpuDialect"];
}

def SubnetDivide : Pass<"subnet-divide", "ModuleOp"> {
  let summary = "subnet divide in tpu by sophgo-opt";
  let constructor = "createSubnetDividePass()";
  let dependentDialects = ["TpuDialect"];
}

def LayerGroup : Pass<"layer-group", "FuncOp"> {
  let summary = "convert to layer group in tpu by sophgo-opt";
  let constructor = "createLayerGroupPass()";
  let dependentDialects = ["TpuDialect"];
}

def AddressAsign : Pass<"address-asign", "ModuleOp"> {
  let summary = "assign address in tpu by sophgo-opt";
  let constructor = "createAddressAsignPass()";
  let dependentDialects = ["TpuDialect"];
}

def Codegen : Pass<"codegen", "ModuleOp"> {
  let summary = "codegen in tpu by sophgo-opt";
  let constructor = "createCodegenPass()";
  let dependentDialects = ["TpuDialect"];
  let options = [
    Option<"model_file", "model_file", "std::string", /*default=*/"",
           "save to model file">,
  ];
}

#endif


