//===-- BM1690.td - BM1690 operation definitions *- tablegen -*------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.

//
//===----------------------------------------------------------------------===//

#ifndef TPUMLIR_DIALECT_BM1690_IR_BM1690OPS
#define TPUMLIR_DIALECT_BM1690_IR_BM1690OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "BM1690.td"
include "BM1690Types.td"
include "StructuredOpsInterfaces.td"

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

class BM1690_Op<string mnemonic, list<Trait> traits = []>
    : Op<BM1690_Dialect, mnemonic,
      traits # [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
      // instance of getAsmResultNames
  let extraClassDefinition = [{
     void $cppClass::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
       ::tpu_mlir::bm1690::getAsmResultNames(setNameFn, getResults());
     }
  }];
}

// TIU operartions  ------------------------------------------------------------

def BM1690_MatMulOp : BM1690_Op<"matmul",
  [DeclareOpInterfaceMethods<StructuredInterface>]> {
  let summary = "";
  let description = [{Matrix Multiplication.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$left,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$right,
    Optional<AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>>:$bias,
    BM1690_DMAIdType:$dependency,
    DefaultValuedAttr<BoolAttr, "false">:$left_is_transposed,
    DefaultValuedAttr<BoolAttr, "false">:$add_result,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    ShortMMRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}

def BM1690_ConvOp : BM1690_Op<"conv"> {
  let summary = "";
  let description = [{Convolution.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$kernel,
    Optional<AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>>:$bias,
    BM1690_DMAIdType:$dependency,
    BM1690_IntArrayAttr2:$kernel_shape,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{1, 1}">:$strides,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$dilations,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$insertions,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$add_result,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<BM1690_PaddingModeAttr, "bm1690::PaddingMode::constant">:$pad_mode,
    ShortConvRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

class BM1690_ArithmeticOp<string mnemonic>
  : BM1690_Op<"arithmetic." # mnemonic, [DeclareOpInterfaceMethods<StructuredInterface>]> {
  let summary = "";
  let description = [{Arithmetic.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$rhs,
    Optional<AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>>:$shift,
    BM1690_DMAIdType:$dependency,

    DefaultValuedAttr<BoolAttr, "false">:$is_saturation,
    DefaultValuedAttr<BM1690_RoundModeAttr, "bm1690::RoundMode::HalfAwayFromZero">:$round_mode,
    ShortARRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
}

class BM1690_BinaryOp<string mnemonic> : BM1690_ArithmeticOp<mnemonic> {
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$rhs,
    BM1690_DMAIdType:$dependency,
    ShortARRegDefProperties:$reg
    );
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}
def BM1690_MaxOp: BM1690_BinaryOp<"max"> {
  let description = [{Max Operation.}];
}
def BM1690_MinOp: BM1690_BinaryOp<"min"> {
  let description = [{Min Operation.}];
}
def BM1690_AndOp: BM1690_BinaryOp<"and"> {
  let description = [{And Operation.}];
}
def BM1690_OrOp: BM1690_BinaryOp<"or"> {
  let description = [{Or Operation.}];
}
def BM1690_XorOp: BM1690_BinaryOp<"xor"> {
  let description = [{Xor Operation.}];
}
def BM1690_AddOp: BM1690_ArithmeticOp<"add"> {
  let description = [{Add Operation.}];
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}
def BM1690_SubOp: BM1690_ArithmeticOp<"sub"> {
  let description = [{Sub Operation.}];
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}
def BM1690_MulOp: BM1690_ArithmeticOp<"mul"> {
  let description = [{Mul Operation.}];
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}

// def BM1690_MacOp: BM1690_ArithmeticOp<"mac"> {
//   let description = [{Mac Operation.}];
//   AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
//   AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$rhs,
//   BM1690_DMAIdType:$dependency,

//   DefaultValuedAttr<I32Attr, 0>:$lshift,
//   DefaultValuedAttr<I32Attr, 0>:$rshift,
//   DefaultValuedAttr<Sg_RoundModeAttr, "bm1690::RoundMode::HalfAwayFromZero">:$round_mode,
//   ShortARRegDefProperties:$reg
// }
class BM1690_UnaryOp<string mnemonic> : BM1690_ArithmeticOp<mnemonic> {
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
    BM1690_DMAIdType:$dependency,
    ShortARRegDefProperties:$reg
    );
}
def BM1690_CopyOp: BM1690_UnaryOp<"copy"> {
  let description = [{Copy Operation.}];
}
def BM1690_AbsOp: BM1690_UnaryOp<"abs"> {
  let description = [{Abs Operation.}];
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}
def BM1690_NotOp: BM1690_UnaryOp<"not"> {
  let description = [{Not Operation.}];
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}
def BM1690_CLZOp: BM1690_UnaryOp<"clz"> {
  let description = [{Count leading zero Operation. Get first zero}];
}
def BM1690_CLOOp: BM1690_UnaryOp<"clo"> {
  let description = [{Count leading ones Operation. Get first one}];
}

def BM1690_TaylorOp : BM1690_Op<"sfu.taylor",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Taylor";
  let description = [{taylor exponent}];
  let arguments = (ins
    AnyMemRef:$input,
    AnyMemRef:$table,
    DefaultValuedAttr<I64Attr, "10">:$length,
    BM1690_DMAIdType:$dependency,
    ShortSFURegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_NormalOp : BM1690_Op<"sfu.normal",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Normal";
  let description = [{generate float exponent part}];
  let arguments = (ins
    AnyMemRef:$input,
    BM1690_DMAIdType:$dependency,
    ShortSFURegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_RsqrtOp : BM1690_Op<"sfu.rsqrt",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Rsqrt";
  let description = [{Fast Inverse Square Root. wiki: https://en.wikipedia.org/wiki/Fast_inverse_square_root }];
  let arguments = (ins
    AnyMemRef:$input,
    DefaultValuedAttr<I64Attr, "3">:$num_iter,
    BM1690_DMAIdType:$dependency,
    ShortSFURegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

class BM1690_RQDQOp<string mnemonic> : BM1690_Op<"rqdq." # mnemonic,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{RQ/DQ.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger]>:$input,
    AnyTypeOf<[AnyMemRef, AnyFloat]>:$scale,
    OptionalAttr<AnyAttrOf<[I16Attr, F32Attr]>>:$offset,
    BM1690_DMAIdType:$dependency,

    DefaultValuedAttr<BoolAttr, "false">:$is_saturation,
    DefaultValuedAttr<BM1690_RoundModeAttr, "bm1690::RoundMode::HalfAwayFromZero">:$round_mode0,
    DefaultValuedAttr<BM1690_RoundModeAttr, "bm1690::RoundMode::HalfAwayFromZero">:$round_mode1,
    ShortRQDQRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}
def BM1690_RequantFpOp : BM1690_RQDQOp<"requant_fp"> {
  let summary = "RequantFp";
  let description = [{float requant. output = int(float(input) * scale + offset)}];
}
def BM1690_DequantFpOp : BM1690_RQDQOp<"dequant_fp"> {
  let summary = "DequantFp";
  let description = [{float dequant. output = float(input - offset) * scale}];
}

class BM1690_RQDQIntOp<string mnemonic> : BM1690_Op<"rqdq." # mnemonic,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{RQ/DQ.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger]>:$multiplier,
    OptionalAttr<I8Attr>:$shift,
    OptionalAttr<I16Attr>:$zero_point,
    BM1690_DMAIdType:$dependency,

    DefaultValuedAttr<BoolAttr, "false">:$is_saturation,
    DefaultValuedAttr<BM1690_RoundModeAttr, "bm1690::RoundMode::HalfAwayFromZero">:$round_mode,
    ShortRQDQRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}
def BM1690_RequantIntOp : BM1690_RQDQIntOp<"requant_int"> {
  let summary = "RequantInt";
  let description = [{Int requant. output = input * multiplier >> shift + zero_point}];
}
def BM1690_DequantIntOp : BM1690_RQDQIntOp<"dequant_int"> {
  let summary = "DequantInt";
  let description = [{Int dequant. output = (input - zero_point) * multiplier >> shift)}];
}

class BM1690_MaxMinPoolOp<string mnemonic, list<Trait> traits = [DeclareOpInterfaceMethods<StructuredInterface>]>
  : BM1690_Op<"pooling." # mnemonic, traits> {
  let summary = "";
  let description = [{MaxMin Pool.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_DMAIdType:$dependency,
    BM1690_IntArrayAttr2:$kernel_shape,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{1, 1}">:$strides,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$dilations,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$insertions,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<BM1690_PaddingModeAttr, "bm1690::PaddingMode::constant">:$pad_mode,
    ShortPorDRegDefProperties:$reg
    );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}

def BM1690_MaxPoolOp : BM1690_MaxMinPoolOp<"max"> {
  let description = [{Max Pool.}];
}

def BM1690_MinPoolOp : BM1690_MaxMinPoolOp<"min"> {
  let description = [{Min Pool.}];
}

def BM1690_AvgPoolOp : BM1690_Op<"pooling.avg",
  [DeclareOpInterfaceMethods<StructuredInterface>]> {
  let summary = "";
  let description = [{Avg Pooling.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    F64Attr:$scale,
    BM1690_DMAIdType:$dependency,
    BM1690_IntArrayAttr2:$kernel_shape,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{1, 1}">:$strides,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$dilations,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$insertions,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<BM1690_PaddingModeAttr, "bm1690::PaddingMode::constant">:$pad_mode,
    ShortPorDRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}

def BM1690_DepthwiseOp : BM1690_Op<"depthwise", [DeclareOpInterfaceMethods<StructuredInterface>]> {
  let summary = "";
  let description = [{Depthwise Conv.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$kernel,
    Optional<AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>>:$bias,
    BM1690_DMAIdType:$dependency,
    BM1690_IntArrayAttr2:$kernel_shape,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{1, 1}">:$strides,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$dilations,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0}">:$insertions,
    DefaultValuedAttr<BM1690_IntArrayAttr2, "SmallVector<int64_t>{0, 0, 0, 0}">:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<BM1690_PaddingModeAttr, "bm1690::PaddingMode::constant">:$pad_mode,
    ShortPorDRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let extraClassDeclaration = [{ static TPUISATraits getStuctutedTraits(::mlir::MLIRContext*); }];
}

def BM1690_VectorCorrOp : BM1690_Op<"VectorCorrelation",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Vector Correlation.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$left,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$right,
    BM1690_DMAIdType:$dependency,
    VCOpTypeAttr:$op_type,
    ShortVCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_CWTransOp : BM1690_Op<"CWTrans",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{CW Transpose.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_DMAIdType:$dependency,
    BoolAttr:$is_cw_trans,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_LaneBroadOp : BM1690_Op<"LaneBroad",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Lane Broadcast.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_DMAIdType:$dependency,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_LaneCopyOp : BM1690_Op<"LaneCopy",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Lane Copy.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_DMAIdType:$dependency,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_StaticBroadOp : BM1690_Op<"StaticBroad",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Static Broadcast.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_DMAIdType:$dependency,
    I64Attr:$output_c,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_StaticDistOp : BM1690_Op<"StaticDist",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Static Distribute.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_DMAIdType:$dependency,
    I64Attr:$output_c,
    ShortCWBCRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_TIUSendMsgOp : BM1690_Op<"SendMsg",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Send Msg.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_DMAIdType:$dependency,
    I64Attr:$msg_id,
    I64Attr:$wait_cnt,
    SYSRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

def BM1690_TIUWaitMsgOp : BM1690_Op<"WaitMsg",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{Wait Msg.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_DMAIdType:$dependency,
    I64Attr:$msg_id,
    I64Attr:$send_cnt,
    SYSRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_TIUIdType : $id);
  let hasVerifier = 1;
}

// DMA operartions  ------------------------------------------------------------

class BM1690_DMATensor<string eu_type> : BM1690_Op<"dma.tensor" # eu_type ,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{DMA Tensor.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$source,
    BM1690_TIUIdType:$dependency,
    DMATensorRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $target, BM1690_DMAIdType : $id);
  let hasVerifier = 1;
}

def BM1690_DMATensorOp : BM1690_DMATensor<"">;
def BM1690_DMATensorTransOp : BM1690_DMATensor<".trans">;
def BM1690_DMATensorBroadcastOp : BM1690_DMATensor<".broadcast">;

def BM1690_GDMAMatrixMoveOp : BM1690_Op<"GDMAMatrixMove",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA Matrix Move.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_TIUIdType:$dependency,
    DefaultValuedAttr<BoolAttr, "false">:$transpose,
    GDMAMatrixRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_DMAIdType : $id);
  let hasVerifier = 1;
}

def BM1690_GDMAGeneralMoveOp : BM1690_Op<"GDMAGeneralMove",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA General Move.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_TIUIdType:$dependency,
    I64Attr:$count,
    GDMAGeneralRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_DMAIdType : $id);
  let hasVerifier = 1;
}

def BM1690_GDMAGeneralBroadcastOp : BM1690_Op<"GDMAGeneralBroadcast",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA General Broadcast.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    BM1690_TIUIdType:$dependency,
    I64Attr:$count,
    I64Attr:$output_c,
    GDMAGeneralRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_DMAIdType : $id);
  let hasVerifier = 1;
}

def BM1690_GDMAGatherOp : BM1690_Op<"GDMAGather",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA Gather.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$index,
    BM1690_TIUIdType:$dependency,
    I64Attr:$const_val,
    GDMAGatherRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_DMAIdType : $id);
  let hasVerifier = 1;
}

def BM1690_GDMAScatterOp : BM1690_Op<"GDMAScatter",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA Scatter.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$index,
    BM1690_TIUIdType:$dependency,
    I64Attr:$const_val,
    GDMAScatterRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_DMAIdType : $id);
  let hasVerifier = 1;
}

def BM1690_GDMAScatterAddOp : BM1690_Op<"GDMAScatterAdd",
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "";
  let description = [{GDMA Scatter.}];
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$input,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$index,
    BM1690_TIUIdType:$dependency,
    I64Attr:$const_val,
    GDMAScatterRegDefProperties:$reg
  );
  let results = (outs AnyMemRef : $result, BM1690_DMAIdType : $id);
  let hasVerifier = 1;
}

class BM1690_LinOp<string mnemonic> : BM1690_Op<"lin." # mnemonic,
  [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$lhs,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$mhs,
    AnyTypeOf<[AnyMemRef, AnyInteger, AnyFloat]>:$rhs,
    BM1690_DMAIdType: $dependency,
    ShortLINRegDefProperties:$reg
    );
    let results = (outs AnyMemRef : $target, BM1690_TIUIdType : $id);
    let hasVerifier = 1;
  }

def BM1690_MacOp: BM1690_LinOp<"mac"> {
  let description = [{LIN: Mac }];
}
def BM1690_AddSqrOp: BM1690_LinOp<"add_sqr"> {
  let description = [{LIN: ADD_SQR}];
}
def BM1690_SubSqrOp: BM1690_LinOp<"sub_sqr"> {
  let description = [{LIN: SUB_SQR}];
}


#endif // BM1690
