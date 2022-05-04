//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_CONVERSION_PASSES
#define SOPHGO_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TopTFLiteToTpu
//===----------------------------------------------------------------------===//

def ConvertTopTFLiteToTpu : Pass<"toptflite-to-tpu"> {
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
