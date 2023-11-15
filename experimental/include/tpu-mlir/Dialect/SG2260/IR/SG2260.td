//===-- SG2260.td - SG2260 dialect operation definitions *- tablegen -*----===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.

//
//===----------------------------------------------------------------------===//

#ifndef TPUMLIR_DIALECT_SG2260_IR_SG2260
#define TPUMLIR_DIALECT_SG2260_IR_SG2260

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SG2260 dialect definition.
//===----------------------------------------------------------------------===//

def SG2260_Dialect : Dialect {
  let name = "sg2260";
  let cppNamespace = "::tpu_mlir::sg2260";
  let description = [{The SG2260 Atomic instruciton.}];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
       void registerTypes();
  }];
}



#endif // SG2260
