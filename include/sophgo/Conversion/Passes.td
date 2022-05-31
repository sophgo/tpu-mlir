//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_CONVERSION_PASSES
#define SOPHGO_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TopTFLiteToTpu
//===----------------------------------------------------------------------===//

def ConvertTopTFLiteToTpu : Pass<"toptflite-to-tpu", "ModuleOp"> {
  let summary = "Lower TOP quantize operations to TPU operations";
  let description = [{

    Convert operations from the TopTFLite quantize dialect into operations from
    the Tpu dialects.

  }];
  let constructor = "sophgo::createLowerTopTFLitePass()";
  let dependentDialects = [
    "top::TopDialect",
    "tpu::TpuDialect",
  ];
}

#endif // SOPHGO_CONVERSION_PASSES
