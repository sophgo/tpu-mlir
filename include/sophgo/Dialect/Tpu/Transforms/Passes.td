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

def Codegen : Pass<"codegen", "ModuleOp"> {
  let summary = "codegen in tpu by sophgo-opt";
  let constructor = "createCodegenPass()";
  let dependentDialects = ["TpuDialect"];
}

#endif


