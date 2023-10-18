// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef INPUTCONVERSION_PASSES
#define INPUTCONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AutoInputConversionPipeline :
    Pass<"auto-input-conversion", "ModuleOp"> {
  let summary = "Analyzes and runs appropriate input pipeline.";
  let description = [{
    Analyzes the input program to determine which input pipeline, or part of input
    conversion to run, then run that conversion.
  }];
  let constructor = "mlir::createAutoInputConversionPipelinePass()";
}

#endif // INPUTCONVERSION_PASSES
