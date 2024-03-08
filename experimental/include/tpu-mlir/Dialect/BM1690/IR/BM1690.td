//===-- BM1690.td - BM1690 dialect operation definitions *- tablegen -*----===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.

//
//===----------------------------------------------------------------------===//

#ifndef TPUMLIR_DIALECT_BM1690_IR_BM1690
#define TPUMLIR_DIALECT_BM1690_IR_BM1690

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// BM1690 dialect definition.
//===----------------------------------------------------------------------===//

def BM1690_Dialect : Dialect {
  let name = "bm1690";
  let cppNamespace = "::tpu_mlir::bm1690";
  let description = [{The BM1690 Atomic instruciton.}];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
       void registerTypes();
  }];
}



#endif // BM1690
