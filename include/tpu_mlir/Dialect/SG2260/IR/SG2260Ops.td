//===-- SG2260.td - SG2260 operation definitions *- tablegen -*------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.

//
//===----------------------------------------------------------------------===//

#ifndef TPUMLIR_DIALECT_SG2260_IR_SG2260OPS
#define TPUMLIR_DIALECT_SG2260_IR_SG2260OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tpu_mlir/Dialect/SG2260/IR/SG2260.td"
include "tpu_mlir/Dialect/SG2260/IR/SG2260Types.td"


class SG2260_Op<string mnemonic, list<Trait> traits = []>
    : Op<SG2260_Dialect, mnemonic, traits>;

def SG2260_MatMulOp : SG2260_Op<"matmul",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Matrix Multiplication.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$left,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$right,
    Optional<AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>>:$bias,
    SG2260_DMAIdType:$dependency,
    DefaultValuedAttr<BoolAttr, "false">:$left_is_transposed,
    DefaultValuedAttr<BoolAttr, "false">:$add_restult,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    ShortMatrix2RegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

#endif // SG2260
