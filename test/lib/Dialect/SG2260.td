//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SG2260_OPS
#define SG2260_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// define sg2260 dialect
def SG2260_Dialect : Dialect {
  let name = "sg2260";
  let cppNamespace = "::sg2260";
  let summary = "SG2260 instruction dialect.";
}

class SG2260_Op<string mnemonic, list<Trait> traits = []>
    : Op<SG2260_Dialect, mnemonic, traits>;


// %R2.3333 = sg2260.usemem : memref<43x22x44xf32, address=23ef22, layout=[2,3,4]>
def MemRefOp : SG2260_Op<"memref",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let results = (outs AnyMemRef:$result);
}

def DMARegDefProperties : Property<"DMARegDef"> {
  let convertToAttribute = [{getPropertiesAsAttribute($_ctxt, $_storage)}];
  let convertFromAttribute =
      [{ return setPropertiesFromAttribute($_storage, $_attr, $_diag); }];
  let hashProperty = [{ computeHash($_storage); }];
}
// %B2.dd.dd, %dmaId23 = sg2260.dma.tensor %G0e34f, %tiuId32 {
//   mask = 5fea339e,
// } (memref<23x32xi64, strided=[23,44]>) -> (memref<45x89x77xi32>)
def GDMAOp : SG2260_Op<"dma",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins
    DMARegDefProperties:$prop,
    AnyMemRef:$source,
    I64:$dep_id
  );
  let results = (outs AnyType:$target, I64:$id);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def Matrix2RegDefProperties : Property<"Matrix2RegDef"> {
  let convertToAttribute = [{getPropertiesAsAttribute($_ctxt, $_storage)}];
  let convertFromAttribute =
      [{ return setPropertiesFromAttribute($_storage, $_attr, $_diag); }];
  let hashProperty = [{ computeHash($_storage); }];
}

// %R2.dd.dd, %TID23 = sg2260.matmul2.generic %R2.333.2323, %c2.3232, %DID120 {
//   partial_sum = true,
//   partial_sum = true,
// } (memref<23x32xi64, strided=[23,44]>, f32) -> (memref<45x89x77xi32>)
def MatMulOp : SG2260_Op<"matmul2",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins
    Matrix2RegDefProperties:$prop,
    AnyMemRef:$left,
    AnyMemRef:$right,
    I64:$dep_id
  );
  let results = (outs AnyType:$result, I64:$id);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // SG2260_OPS
