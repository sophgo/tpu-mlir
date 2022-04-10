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

#ifndef SOPHGO_CODEGENINTERFACE
#define SOPHGO_CODEGENINTERFACE

include "mlir/IR/OpBase.td"

def CodegenInterface : OpInterface<"CodegenInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_int8_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_int8_bm1686",
        /*args=*/(ins)
      >,
  ];
}
#endif // SOPHGO_CODEGENINTERFACE
