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
include "SG2260.td"
include "SG2260Types.td"


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
    DefaultValuedAttr<BoolAttr, "false">:$add_result,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    ShortMatrix2RegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_ConvOp : SG2260_Op<"conv",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Convolution.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$kernel,
    Optional<AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>>:$bias,
    SG2260_DMAIdType:$dependency,
    SG2260_IntArrayAttr2:$kernel_shape,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{1, 1}">:$strides,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$dilations,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$input_insert0,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$kernel_insert0,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$add_result,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    ShortConvRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

#endif // SG2260
