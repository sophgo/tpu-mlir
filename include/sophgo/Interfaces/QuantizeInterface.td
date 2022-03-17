//===- TilingInterface.td - Interface for tiling operations *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an interface to allow operations to generate a tiled
// implementation of themselves.
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_QUANTIZEINTERFACE
#define SOPHGO_QUANTIZEINTERFACE

include "mlir/IR/OpBase.td"

def QuantizeInterface : OpInterface<"QuantizeInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"quantize_int8_bm1684",
        /*args=*/(ins)
      >,
  ];
}
#endif // SOPHGO_QUANTIZEINTERFACE
