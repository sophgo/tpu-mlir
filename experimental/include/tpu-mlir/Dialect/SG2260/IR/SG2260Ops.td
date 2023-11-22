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

class AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;
def VCOpTypeAttr: AnyStrAttrOf<["VC_ADD","VC_SUB","VC_MUL","VC_DIV","VC_MAX","VC_MIN","VC_AND",
  "VC_OR","VC_XOR","VC_SG","VC_SE","VC_SL","VC_ADD_SATU","VC_SUB_SATU","VC_MUL_SATU"]>;

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
    ShortMMRegDefProperties:$reg
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
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$insertions,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$add_result,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<SG2260_PaddingModeAttr, "sg2260::PaddingMode::constant">:$pad_mode,
    ShortConvRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

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
def SG2260_MaxOp: SG2260_BinaryOp<"max"> {
  let description = [{Max Operation.}];
}
def SG2260_MinOp: SG2260_BinaryOp<"min"> {
  let description = [{Min Operation.}];
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
def SG2260_SubOp: SG2260_ArithmeticOp<"sub"> {
  let description = [{Sub Operation.}];
}
def SG2260_MulOp: SG2260_ArithmeticOp<"mul"> {
  let description = [{Mul Operation.}];
}

// def SG2260_MacOp: SG2260_ArithmeticOp<"mac"> {
//   let description = [{Mac Operation.}];
//   AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
//   AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$rhs,
//   SG2260_DMAIdType:$dependency,

//   DefaultValuedAttr<I32Attr, 0>:$lshift,
//   DefaultValuedAttr<I32Attr, 0>:$rshift,
//   DefaultValuedAttr<Sg_RoundModeAttr, "sg2260::RoundMode::HalfAwayFromZero">:$round_mode,
//   ShortARRegDefProperties:$reg
// }
class SG2260_UnaryOp<string mnemonic> : SG2260_ArithmeticOp<mnemonic> {
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
    SG2260_DMAIdType:$dependency,
    ShortARRegDefProperties:$reg
    );
}
def SG2260_CopyOp: SG2260_UnaryOp<"copy"> {
  let description = [{Copy Operation.}];
}
def SG2260_AbsOp: SG2260_UnaryOp<"abs"> {
  let description = [{Abs Operation.}];
}
def SG2260_NotOp: SG2260_UnaryOp<"not"> {
  let description = [{Not Operation.}];
}
def SG2260_CLZOp: SG2260_UnaryOp<"clz"> {
  let description = [{Count leading zero Operation. Get first zero}];
}
def SG2260_CLOOp: SG2260_UnaryOp<"clo"> {
  let description = [{Count leading ones Operation. Get first one}];
}

def SG2260_TaylorOp : SG2260_Op<"sfu.taylor",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Taylor";
  let description = [{taylor exponent}];
  let arguments = (ins
    AnyMemRef:$input,
    AnyMemRef:$table,
    DefaultValuedAttr<I64Attr, "10">:$length,
    SG2260_DMAIdType:$dependency,
    ShortSFURegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_NormalOp : SG2260_Op<"sfu.normal",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Normal";
  let description = [{generate float exponent part}];
  let arguments = (ins
    AnyMemRef:$input,
    SG2260_DMAIdType:$dependency,
    ShortSFURegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_RsqrtOp : SG2260_Op<"sfu.rsqrt",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Rsqrt";
  let description = [{Fast Inverse Square Root. wiki: https://en.wikipedia.org/wiki/Fast_inverse_square_root }];
  let arguments = (ins
    AnyMemRef:$input,
    DefaultValuedAttr<I64Attr, "3">:$num_iter,
    SG2260_DMAIdType:$dependency,
    ShortSFURegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

class SG2260_RQDQOp<string mnemonic> : SG2260_Op<"rqdq." # mnemonic,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{RQ/DQ.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger]>:$input,
    AnyTypeOf<[AnyMemRef, AnyFloat]>:$scale,
    OptionalAttr<AnyAttrOf<[I16Attr, F32Attr]>>:$offset,
    SG2260_DMAIdType:$dependency,

    DefaultValuedAttr<BoolAttr, "false">:$is_saturation,
    DefaultValuedAttr<SG2260_RoundModeAttr, "sg2260::RoundMode::HalfAwayFromZero">:$round_mode0,
    DefaultValuedAttr<SG2260_RoundModeAttr, "sg2260::RoundMode::HalfAwayFromZero">:$round_mode1,
    ShortRQDQRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}
def SG2260_RequantFpOp : SG2260_RQDQOp<"requant_fp"> {
  let summary = "RequantFp";
  let description = [{float requant. output = int(float(input) * scale + offset)}];
}
def SG2260_DequantFpOp : SG2260_RQDQOp<"dequant_fp"> {
  let summary = "DequantFp";
  let description = [{float dequant. output = float(input - offset) * scale}];
}

class SG2260_RQDQIntOp<string mnemonic> : SG2260_Op<"rqdq." # mnemonic,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{RQ/DQ.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger]>:$multiplier,
    OptionalAttr<I8Attr>:$shift,
    OptionalAttr<I16Attr>:$zero_point,
    SG2260_DMAIdType:$dependency,

    DefaultValuedAttr<BoolAttr, "false">:$is_saturation,
    DefaultValuedAttr<SG2260_RoundModeAttr, "sg2260::RoundMode::HalfAwayFromZero">:$round_mode,
    ShortRQDQRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}
def SG2260_RequantIntOp : SG2260_RQDQIntOp<"requant_int"> {
  let summary = "RequantInt";
  let description = [{Int requant. output = input * multiplier >> shift + zero_point}];
}
def SG2260_DequantIntOp : SG2260_RQDQIntOp<"dequant_int"> {
  let summary = "DequantInt";
  let description = [{Int dequant. output = (input - zero_point) * multiplier >> shift)}];
}

class SG2260_MaxMinPoolOp<string mnemonic> : SG2260_Op<"maxminpool." # mnemonic,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{MaxMin Pool.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_DMAIdType:$dependency,
    SG2260_IntArrayAttr2:$kernel_shape,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{1, 1}">:$strides,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$dilations,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$insertions,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<SG2260_PaddingModeAttr, "sg2260::PaddingMode::constant">:$pad_mode,
    ShortPorDRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_MaxPoolOp : SG2260_MaxMinPoolOp<"max_pool"> {
  let description = [{Max Pool.}];
}

def SG2260_MinPoolOp : SG2260_MaxMinPoolOp<"min_pool"> {
  let description = [{Min Pool.}];
}

def SG2260_AvgPoolOp : SG2260_Op<"avg_pool",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Avg Pooling.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    F64Attr:$scale,
    SG2260_DMAIdType:$dependency,
    SG2260_IntArrayAttr2:$kernel_shape,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{1, 1}">:$strides,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$dilations,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$insertions,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<SG2260_PaddingModeAttr, "sg2260::PaddingMode::constant">:$pad_mode,
    ShortPorDRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_DepthwiseOp : SG2260_Op<"depthwise",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Depthwise Conv.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$kernel,
    Optional<AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>>:$bias,
    SG2260_DMAIdType:$dependency,
    SG2260_IntArrayAttr2:$kernel_shape,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{1, 1}">:$strides,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$dilations,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$insertions,
    DefaultValuedAttr<SG2260_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<SG2260_PaddingModeAttr, "sg2260::PaddingMode::constant">:$pad_mode,
    ShortPorDRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_VectorCorrOp : SG2260_Op<"VectorCorrelation",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Vector Correlation.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$left,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$right,
    SG2260_DMAIdType:$dependency,
    VCOpTypeAttr:$op_type,
    ShortVCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_CWTransOp : SG2260_Op<"CWTrans",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{CW Transpose.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_DMAIdType:$dependency,
    BoolAttr:$is_cw_trans,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_LaneBroadOp : SG2260_Op<"LaneBroad",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Lane Broadcast.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_DMAIdType:$dependency,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_LaneCopyOp : SG2260_Op<"LaneCopy",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Lane Copy.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_DMAIdType:$dependency,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_StaticBroadOp : SG2260_Op<"StaticBroad",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Static Broadcast.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_DMAIdType:$dependency,
    I64Attr:$output_c,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_StaticDistOp : SG2260_Op<"StaticDist",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Static Distribute.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_DMAIdType:$dependency,
    I64Attr:$output_c,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_TIUSendMsgOp : SG2260_Op<"SendMsg",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Send Msg.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_DMAIdType:$dependency,
    I64Attr:$msg_id,
    I64Attr:$wait_cnt,
    SYSRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_TIUIdType : $id);
  let hasVerifier = 1;
}

def SG2260_TIUWaitMsgOp : SG2260_Op<"WaitMsg",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Wait Msg.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_DMAIdType:$dependency,
    I64Attr:$msg_id,
    I64Attr:$send_cnt,
    SYSRegDefProperties:$reg
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

def SG2260_GDMAMatrixMoveOp : SG2260_Op<"GDMAMatrixMove",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA Matrix Move.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_TIUIdType:$dependency,
    DefaultValuedAttr<BoolAttr, "false">:$transpose,
    GDMAMatrixRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_DMAIdType : $id);
  let hasVerifier = 1;
}

def SG2260_GDMAGeneralMoveOp : SG2260_Op<"GDMAGeneralMove",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA General Move.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_TIUIdType:$dependency,
    I64Attr:$count,
    GDMAGeneralRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_DMAIdType : $id);
  let hasVerifier = 1;
}

def SG2260_GDMAGeneralBroadcastOp : SG2260_Op<"GDMAGeneralBroadcast",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA General Broadcast.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    SG2260_TIUIdType:$dependency,
    I64Attr:$count,
    I64Attr:$output_c,
    GDMAGeneralRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_DMAIdType : $id);
  let hasVerifier = 1;
}

def SG2260_GDMAGatherOp : SG2260_Op<"GDMAGather",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA Gather.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$index,
    SG2260_TIUIdType:$dependency,
    I64Attr:$const_val,
    GDMAGatherRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_DMAIdType : $id);
  let hasVerifier = 1;
}

def SG2260_GDMAScatterOp : SG2260_Op<"GDMAScatter",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA Scatter.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$index,
    SG2260_TIUIdType:$dependency,
    I64Attr:$const_val,
    GDMAScatterRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_DMAIdType : $id);
  let hasVerifier = 1;
}

def SG2260_GDMAScatterAddOp : SG2260_Op<"GDMAScatterAdd",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA Scatter.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$index,
    SG2260_TIUIdType:$dependency,
    I64Attr:$const_val,
    GDMAScatterRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, SG2260_DMAIdType : $id);
  let hasVerifier = 1;
}

class SG2260_LinOp<string mnemonic> : SG2260_Op<"lin." # mnemonic,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$mhs,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$rhs,
    SG2260_DMAIdType: $dependency,
    ShortLINRegDefProperties:$reg
    );
    let results = (outs AnyMemRef : $target, SG2260_TIUIdType : $id);
    let hasVerifier = 1;
  }

def SG2260_MacOp: SG2260_LinOp<"mac"> {
  let description = [{LIN: Mac }];
}
def SG2260_AddSqrOp: SG2260_LinOp<"add_sqr"> {
  let description = [{LIN: ADD_SQR}];
}
def SG2260_SubSqrOp: SG2260_LinOp<"sub_sqr"> {
  let description = [{LIN: SUB_SQR}];
}


#endif // SG2260
