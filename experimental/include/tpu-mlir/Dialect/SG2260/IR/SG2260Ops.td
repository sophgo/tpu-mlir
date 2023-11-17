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

// TIU operartions  ------------------------------------------------------------

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
    DefaultValuedAttr<SG2260_PaddingModeAttr, "sg2260::PaddingMode::constant">:$pad_mode,
    ShortConvRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

// DMA operartions  ------------------------------------------------------------

class SG2260_DMATensor<string eu_type> : SG2260_Op<"dma.tensor" # eu_type ,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{DMA Tensor.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$source,
    SG2260_TIUIdType:$dependency,
    DMATensorRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $target, SG2260_DMAIdType : $id);
  let hasVerifier = 1;
}

def SG2260_DMATensorOp : SG2260_DMATensor<"">;
def SG2260_DMATensorTransOp : SG2260_DMATensor<".trans">;
def SG2260_DMATensorBroadcastOp : SG2260_DMATensor<".broadcast">;

class SG2260_ArithmeticOp<string mnemonic> : SG2260_Op<"arithmetic." # mnemonic,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Arithmetic.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$rhs,
    Optional<AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>>:$shift,
    SG2260_DMAIdType:$dependency,

    DefaultValuedAttr<BoolAttr, "false">:$is_saturation,
    DefaultValuedAttr<SG2260_RoundModeAttr, "sg2260::RoundMode::HalfAwayFromZero">:$round_mode,
    ShortARRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

class SG2260_BinaryOp<string mnemonic> : SG2260_ArithmeticOp<mnemonic> {
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$rhs,
    SG2260_DMAIdType:$dependency,
    ShortARRegDefProperties:$reg
    );
}

def SG2260_AndOp: SG2260_BinaryOp<"and"> {
  let description = [{And Operation.}];
}
def SG2260_OrOp: SG2260_BinaryOp<"or"> {
  let description = [{Or Operation.}];
}
def SG2260_XorOp: SG2260_BinaryOp<"xor"> {
  let description = [{Xor Operation.}];
}
def SG2260_AddOp: SG2260_ArithmeticOp<"add"> {
  let description = [{Add Operation.}];
}



#endif // SG2260
