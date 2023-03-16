//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

// =============================================================================
//
// Defines TPU Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TPU_OPS
#define TPU_MLIR_TPU_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tpu_mlir/Interfaces/LocalGenInterface.td"
include "tpu_mlir/Interfaces/GlobalGenInterface.td"
include "tpu_mlir/Interfaces/InferenceInterface.td"
include "tpu_mlir/Interfaces/TypeInterface.td"
include "tpu_mlir/Interfaces/DynLocalGenInterface.td"
include "tpu_mlir/Interfaces/DynGlobalGenInterface.td"
include "tpu_mlir/Traits/Traits.td"

// =============================================================================
//
// Defines Tpu Dialect.
//
//===----------------------------------------------------------------------===//

def Tpu_Dialect : Dialect {
  let name = "tpu";
  let summary = "A tpu dialect for the SOPHGO AI chips";
  let cppNamespace = "::tpu_mlir::tpu";
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Tpu Attributes.
//===----------------------------------------------------------------------===//

class Tpu_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Tpu_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// A string attribute whose value are one of the values in `cases`.
class AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;


def ArgModeAttr: AnyStrAttrOf<["ArgMin","ArgMax"]>;
def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceL2","ReduceL1","ReduceSum","ReduceProd"]>;
def RoiAlignModeAttr: AnyStrAttrOf<["Avg","Max"]>;
def NonZeroOrderAttr: AnyStrAttrOf<["ColMajor","RowMajor"]>;
def DetectionOutputCodeTypeAttr: AnyStrAttrOf<["CORNER", "CENTER_SIZE", "CORNER_SIZE"]>;
def Tpu_LayerGroupAttr : Tpu_Attr<"LayerGroup", "lg"> {
  let summary = "Structure of layer group parameters";
  let parameters = (ins
    "int64_t":$out_addr,
    "int64_t":$out_size,
    "int64_t":$buffer_addr,
    "int64_t":$buffer_size,
    "bool":$eu_align,
    "DenseI64ArrayAttr":$h_idx,
    "DenseI64ArrayAttr":$h_slice,
    "DenseI64ArrayAttr":$n_idx,
    "DenseI64ArrayAttr":$n_slice,
    "int64_t":$id,
    "int64_t":$stage,
    "int64_t":$group_type
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Tpu_DequantMode: I32EnumAttr<"DequantMode",
    "dequant mode supported by DequantOp",
    [
      I32EnumAttrCase<"Normal", 0>,
      I32EnumAttrCase<"TFLite", 1>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_DequantModeAttr : EnumAttr<Tpu_Dialect, Tpu_DequantMode, "dq_mode">;

def Tpu_RequantMode: I32EnumAttr<"RequantMode",
    "requant mode supported by RequantOp",
    [
      I32EnumAttrCase<"TFLite_LShift", 0>,
      I32EnumAttrCase<"TFLite", 1>,  // * Multi >> 31 >> shift, == QDM
      I32EnumAttrCase<"MultiplierShift", 2>, // * Multi >> shift
      I32EnumAttrCase<"OnlyShift", 3>, // >> shift
      I32EnumAttrCase<"QDM", 4>       // similar to TFLite
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RequantModeAttr : EnumAttr<Tpu_Dialect, Tpu_RequantMode, "rq_mode">;

def Tpu_RoundMode: I32EnumAttr<"RoundMode",
    "round mode supported by Round",
    [
      I32EnumAttrCase<"HalfAwayFromZero", 0>,
      I32EnumAttrCase<"HalfUp", 1>,
      I32EnumAttrCase<"HalfDown", 2>,
      I32EnumAttrCase<"HalfToEven", 3>,
      I32EnumAttrCase<"HalfToOdd", 4>,
      I32EnumAttrCase<"HalfTowardsZero", 5>,
      I32EnumAttrCase<"TowardsZero", 6>,
      I32EnumAttrCase<"Up", 7>,
      I32EnumAttrCase<"Down", 8>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RoundModeAttr : EnumAttr<Tpu_Dialect, Tpu_RoundMode, "round_mode">;

def Tpu_PoolMode: I32EnumAttr<"PoolMode",
    "pooling mode supported by PoolOp",
    [
      I32EnumAttrCase<"Avg", 0>,
      I32EnumAttrCase<"Max", 1>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_PoolModeAttr : EnumAttr<Tpu_Dialect, Tpu_PoolMode, "pool_mode">;

def Tpu_LutBF16Mode : I32EnumAttr<"LutBF16Mode",
    "bf16 look up table mode",
    [
      I32EnumAttrCase<"Other", 0>,
      I32EnumAttrCase<"Mantissa", 1>,
      I32EnumAttrCase<"Slope", 2>,
      I32EnumAttrCase<"Log", 3>,
      I32EnumAttrCase<"Exp", 4>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_LutBF16ModeAttr : EnumAttr<Tpu_Dialect, Tpu_LutBF16Mode, "lut_mode">;

def Tpu_ActiveMode : I32EnumAttr<"ActiveMode",
    "Activation mode for ActiveOp, for sigmoid/exp, e.g.",
    [
      I32EnumAttrCase<"TANH", 0>,
      I32EnumAttrCase<"SIGMOID", 1>,
      I32EnumAttrCase<"RELU", 2>,
      I32EnumAttrCase<"EXP", 3>,
      I32EnumAttrCase<"ELU", 4>,
      I32EnumAttrCase<"SQRT", 5>,
      I32EnumAttrCase<"SQUARE", 6>,
      I32EnumAttrCase<"RSQRT", 7>,
      I32EnumAttrCase<"ABSVAL", 8>,
      I32EnumAttrCase<"LN", 9>,
      I32EnumAttrCase<"ROUND", 10>,
      I32EnumAttrCase<"CEIL", 11>,
      I32EnumAttrCase<"FLOOR", 12>,
      I32EnumAttrCase<"SIN", 13>,
      I32EnumAttrCase<"COS", 14>,
      I32EnumAttrCase<"IS_FINITE", 15>,
      I32EnumAttrCase<"MISH", 16>,
      I32EnumAttrCase<"SWISH", 17>,
      I32EnumAttrCase<"HSWISH", 18>,
      I32EnumAttrCase<"SILU", 19>,
      I32EnumAttrCase<"ARCSIN", 20>,
      I32EnumAttrCase<"ARCCOS", 21>,
      I32EnumAttrCase<"ARCSINH", 22>,
      I32EnumAttrCase<"ARCCOSH", 23>,
      I32EnumAttrCase<"ARCTANH", 24>,
      I32EnumAttrCase<"SINH", 25>,
      I32EnumAttrCase<"COSH", 26>,
      I32EnumAttrCase<"TAN", 27>,
      I32EnumAttrCase<"SIGN", 28>,
      I32EnumAttrCase<"GELU", 29>,
      I32EnumAttrCase<"ERF", 30>,
      I32EnumAttrCase<"HSIGMOID", 31>,
      I32EnumAttrCase<"LOG_SIGMOID", 32>,
      I32EnumAttrCase<"SOFT_PLUS", 33>,
      I32EnumAttrCase<"SOFT_SIGN", 34>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_ActiveModeAttr : EnumAttr<Tpu_Dialect, Tpu_ActiveMode, "active_mode">;

def Tpu_ResizeMode : I32EnumAttr<"ResizeMode",
    "Resize mode",
    [
      I32EnumAttrCase<"nearest", 0>,
      I32EnumAttrCase<"linear", 1>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_ResizeModeAttr : EnumAttr<Tpu_Dialect, Tpu_ResizeMode, "mode">;

def Tpu_ResizeCoordMode : I32EnumAttr<"ResizeCoordMode",
    "Resize coord mode",
    [
      I32EnumAttrCase<"align_corners", 0>,
      I32EnumAttrCase<"half_pixel", 1>,
      I32EnumAttrCase<"pytorch_half_pixel", 2>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_ResizeCoordModeAttr : EnumAttr<Tpu_Dialect, Tpu_ResizeCoordMode, "coord_mode">;

def Tpu_RunMode: I32EnumAttr<"RunMode",
    "tpu dialect run mode for each subnet",[
      I32EnumAttrCase<"TPU_STATIC",  0>,
      I32EnumAttrCase<"TPU_DYNAMIC", 1>,
      I32EnumAttrCase<"CPU",         2>,
      I32EnumAttrCase<"SCF",         3>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RunModeAttr : EnumAttr<Tpu_Dialect, Tpu_RunMode, "run_mode">;

//===----------------------------------------------------------------------===//
// Tpu Types.
//===----------------------------------------------------------------------===//

def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// Tpu Operations.
//===----------------------------------------------------------------------===//

class Tpu_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Tpu_Dialect, mnemonic, !listconcat(traits,[TpuTypeRestrict])> ;

class Tpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tpu_Dialect, mnemonic, !listconcat(traits,
       [TpuTypeRestrict,
       DeclareOpInterfaceMethods<GlobalGenInterface>,
       DeclareOpInterfaceMethods<InferenceInterface>,
       DeclareOpInterfaceMethods<DynGlobalGenInterface>])> ;

def Tpu_BufferOp: Tpu_BaseOp<"Buffer"> {
  let summary = "buffer operator";

  let description = [{
    A global buffer for operation, and free after op
  }];

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    static mlir::Value create(mlir::Operation * OwnerOp,
                              mlir::RankedTensorType& type);
  }];
}

class Tpu_ConvOp<string mnemonic, list<Trait> traits = []> : Tpu_Op<mnemonic,
    !listconcat(traits, [SupportFuseRelu,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH", "LocalGenSupport", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface, ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>])> {
  let summary = "convolution operator";

  let description = [{
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$filter,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads, // top,left,bottom,right
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<I64ArrayAttr>:$inserts,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    //new param
    BoolAttr:$with_bias,
    DefaultValuedAttr<BoolAttr, "false">:$coeff_merged,
    DefaultValuedAttr<I64Attr, "0">:$use_3ic_optimize,
    DefaultValuedAttr<I64Attr, "0">:$kernel_zp,
    OptionalAttr<I64ArrayAttr>:$multiplier,
    OptionalAttr<I64ArrayAttr>:$rshift,
    DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    // fuse leakyRelu
    OptionalAttr<BoolAttr>:$do_leaky_relu,
    OptionalAttr<F64Attr>:$neg_slope,
    OptionalAttr<SI32Attr>:$multiplier_pos,
    OptionalAttr<SI32Attr>:$multiplier_neg,
    OptionalAttr<I64Attr>:$rshift_pos,
    OptionalAttr<I64Attr>:$rshift_neg
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    conv_attr_t parseParam();
  }];
}

def Tpu_Conv1DOp : Tpu_ConvOp<"Conv1D">;
def Tpu_Conv2DOp : Tpu_ConvOp<"Conv2D">;
def Tpu_Conv3DOp : Tpu_ConvOp<"Conv3D",[
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>]> {
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$filter,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads, // front,top,left,back,bottom,right
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<I64ArrayAttr>:$inserts,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    //new param
    BoolAttr:$with_bias,
    DefaultValuedAttr<I64Attr, "0">:$kernel_zp,
    // OptionalAttr<I64ArrayAttr>:$multiplier,
    // OptionalAttr<I64ArrayAttr>:$rshift,
    // Tpu_RequantModeAttr:$quant_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
}

class Tpu_PoolOp <string mnemonic> : Tpu_Op<mnemonic,
  [SupportFuseRelu,
   DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport","BackwardH","assign_sec_info"]>,
   DeclareOpInterfaceMethods<DynLocalGenInterface, ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>]> {
  let summary = "pool operator";

  let description = [{
    This performs an  pooling over the given input tensor. A sliding
    window of size given by <kernel size> is passed over the input tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    Tpu_PoolModeAttr:$pool_mode,
    DefaultValuedAttr<I64Attr, "0">:$pad_value,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    /// symmetric quantize param
    OptionalAttr<SI32Attr>:$multiplier,
    OptionalAttr<I64Attr>:$rshift,
    /// asymmetric quantize param
    OptionalAttr<F64Attr>:$scale,
    OptionalAttr<F64Attr>:$offset,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    pool_attr_t parseParam();
  }];
}

def Tpu_Pool1DOp:Tpu_PoolOp<"Pool1D">;
def Tpu_Pool2DOp:Tpu_PoolOp<"Pool2D">;
def Tpu_Pool3DOp:Tpu_PoolOp<"Pool3D">;

def Tpu_MaxPoolWithMaskOp: Tpu_Op<"MaxPoolWithMask",
  [SupportFuseRelu,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "max pool with operator";

  let description = [{
    This performs an  max pooling over the given input tensor. A sliding
    window of size given by <kernel size> is passed over the input tensor.
    get output tensor and mask tensor
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group
  );

  let results = (outs AnyRankedTensor:$output, AnyRankedTensor:$mask);
  let extraClassDeclaration = [{
    pool_attr_t parseParam();
  }];
}

def Tpu_PoolMaskOp: Tpu_Op<"PoolMask"> {
  let summary = "pool mask operator";

  let description = [{
    pooling mask on input
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$scale
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_AddOp: Tpu_Op<"Add", [
  SupportFuseRelu, SupportEarlyStride,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "add operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff,
    // early stride param
    OptionalAttr<BoolAttr>:$do_early_stride,
    OptionalAttr<I32Attr>:$early_stride_h,
    OptionalAttr<I32Attr>:$early_stride_w,
    // quant param
    OptionalAttr<I64ArrayAttr>:$multipliers,
    OptionalAttr<I64ArrayAttr>:$rshifts,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_AddConstOp: Tpu_Op<"AddConst",
    [SupportFuseRelu, InOutSameShape,
    DeclareOpInterfaceMethods<LocalGenInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "add const operator";

  let description = [{
    Elementwise add of input1 and input2. Input2 is constant.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizeMethod = 1;
}

def Tpu_SubOp: Tpu_Op<"Sub",
    [SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
     DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "sub operator";

  let description = [{
    Elementwise subtraction of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff,
    // quant param
    OptionalAttr<I64ArrayAttr>:$multipliers,
    OptionalAttr<I64ArrayAttr>:$rshifts,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_SubConstOp: Tpu_Op<"SubConst",
    [SupportFuseRelu, InOutSameShape,
    DeclareOpInterfaceMethods<LocalGenInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "sub const operator";

  let description = [{
    Elementwise subtraction of input1 and input2. Input1 or Input2 is constant.
    as necessary.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$const_val,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MulOp: Tpu_Op<"Mul",
    [SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "mul operator";

  let description = [{
    Elementwise mul of input1 and input2. Input1 and input2 are tensors.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MaxOp: Tpu_Op<"Max", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "max operator";

  let description = [{
    Elementwise max of input1 and input2. All inputs and outputs must have the same data type.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff,
    // quant param
    OptionalAttr<I64ArrayAttr>:$multipliers,
    OptionalAttr<I64ArrayAttr>:$rshifts,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MinOp: Tpu_Op<"Min", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "min operator";

  let description = [{
    Elementwise min of input1 and input2. All inputs and outputs must have the same data type.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff,
    // quant param
    OptionalAttr<I64ArrayAttr>:$multipliers,
    OptionalAttr<I64ArrayAttr>:$rshifts,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MaxConstOp: Tpu_Op<"MaxConst", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "max_const operator";

  let description = [{
    max of one input and one const.
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    OptionalAttr<I64ArrayAttr>:$multipliers,
    OptionalAttr<I64ArrayAttr>:$rshifts,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_MinConstOp: Tpu_Op<"MinConst", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "min_const operator";

  let description = [{
    min of one input and one const.
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    OptionalAttr<I64ArrayAttr>:$multipliers,
    OptionalAttr<I64ArrayAttr>:$rshifts,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_ActiveOp: Tpu_Op<"Active",
  [DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>, InOutSameShape]>{
  let summary = "Active operator";

  let description = [{
     The operator for activation function
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Tpu_ActiveModeAttr:$mode,
    OptionalAttr<F64ArrayAttr>:$coeffs,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ClipOp: Tpu_Op<"Clip",
  [DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
   DeclareOpInterfaceMethods<DynLocalGenInterface>, InOutSameShape]>{
  let summary = "Clip operator";
  let description = [{
     The operator limits the given input to a certain range.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$min,
    F64Attr:$max,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MulConstOp: Tpu_Op<"MulConst", [SupportFuseRelu, InOutSameShape,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "mul const operator";

  let description = [{
    Elementwise mul of input1 and input2. Input2 is constant.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ReciprocalOp: Tpu_Op<"Reciprocal", [SupportFuseRelu, InOutSameShape,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "ConstantBinary (Div) operator";

  let description = [{
    Y = const_val / X
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<F64Attr, "1.0">: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_Depth2SpaceOp: Tpu_Op<"Depth2Space"> {

  let summary = "Depth2Space operator";

  let description = [{
    Refer to `https://github.com/onnx/onnx/blob/main/docs/Operators.md#depthtospace`
    [n, c, h, w] => [n, c / (block_h * block_w), h * block_h, w * block_w];
    if inversed, [n, c, h, w] => [n, c * block_h * block_w, h / block_h, w / block_w];

    if DCR(depth-column-row), channel ordered by block_h * block_w * c;
    else CRD(column-row-depth), channel ordered by c * block_h * block_w;

    The format of input or output is NCHW or NHWC.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64Attr:$block_h,
    I64Attr:$block_w,
    BoolAttr:$is_CRD,
    BoolAttr:$is_inversed,
    DefaultValuedAttr<BoolAttr, "true">:$in_is_NCHW,
    DefaultValuedAttr<BoolAttr, "true">:$out_is_NCHW,
    DefaultValuedAttr<BoolAttr, "false">:$swap_cr
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LutOp: Tpu_Op<"Lut",
    [DeclareOpInterfaceMethods<LocalGenInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>,
    InOutSameShape]>{
  let summary = "Lut operator";

  let description = [{
    lookup table in index [0-255], y[i] = table(x[i])
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$table,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LutBF16Op: Tpu_Op<"LutBF16",
    [DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>,
    InOutSameShape]>{
  let summary = "LutBF16 operator";

  let description = [{
    input and output is BF16, input BF16 split as exponent and mantissa,
    get output by exponent table and mantissa table
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$table,
    AnyTensorOrNone:$mantissa,
    DefaultValuedAttr<F64Attr, "8">:$max_range,
    DefaultValuedAttr<F64Attr, "-8">:$min_range,
    DefaultValuedAttr<Tpu_LutBF16ModeAttr, "tpu::LutBF16Mode::Other">:$lut_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MatMulOp: Tpu_Op<"MatMul", [
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "AllowDataSplit"]>,
    SupportFuseRelu]> {
  let summary = "matmul operator";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$right,
    AnyTensorOrNone:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$left_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$right_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$hdim_is_batch,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<I64ArrayAttr, "{1}">:$multipliers,
    DefaultValuedAttr<I64ArrayAttr, "{0}">:$rshifts,
    DefaultValuedAttr<I64Attr, "0">:$right_zp,
    DefaultValuedAttr<I64Attr, "0">:$input_zp,
    DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    matmul_attr_t parseParam();
  }];
}

def Tpu_ReluOp: Tpu_Op<"Relu",
  [DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  InOutSameShape]>{
  let summary = "Relu operator";

  let description = [{
     ReLU with a scalar maximum value.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ReshapeOp:Tpu_Op<"Reshape",
    [DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Reshape operation";
  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
  }];
  let arguments = (ins
    AnyRankedTensor:$input
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ReverseOp:Tpu_Op<"Reverse", [InOutSameShape]> {
  let summary = "Reverse operation";
  let description = [{
    Reverse on input
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_CastOp:Tpu_Op<"Cast", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>,
  InOutSameShape]> {
  let summary = "Cast operation";
  let description = [{
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    OptionalAttr<BoolAttr>:$extra_input,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizer = 1;
}

def Tpu_LoadOp:Tpu_Op<"Load",
  [DeclareOpInterfaceMethods<LocalGenInterface, ["assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>, InOutSameShape]> {
  let summary = "Load operation";
  let description = [{
    load input or weight from gmem to lmem;
    if do_bcast, [1,1,1,w] will load to [1,npu,1,w]
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$do_bcast,
    DefaultValuedAttr<I64Attr, "0">:$use_3ic_optimize,
    DefaultValuedAttr<I64Attr, "0">:$lmem_type,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_StoreOp:Tpu_Op<"Store",
  [DeclareOpInterfaceMethods<LocalGenInterface, ["assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>, InOutSameShape]> {
  let summary = "Store operation";
  let description = [{
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RequantIntOp:Tpu_Op<"RequantInt", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>, InOutSameShape]> {
  let summary = "requant operation";
  let description = [{
    Requant 32/16/8 bit data to int8 or uint8 data, by int multiplier and int shift
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    SI32Attr:$multiplier,
    I64Attr:$rshift,
    Tpu_RequantModeAttr:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RequantIntAxisOp:Tpu_Op<"RequantIntAxis", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>, InOutSameShape]> {
  let summary = "requant operation";
  let description = [{
    Requant 32/16/8 bit data to int8 or uint8 data, PerAxis(or PerChannel)
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$quant,
    Tpu_RequantModeAttr:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RequantFpOp:Tpu_Op<"RequantFp", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>, InOutSameShape]> {
  let summary = "requant float operation";
  let description = [{
    Requant 32/16/8 bit data to int8 or uint8 data, by float scale and float offset
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$scale,
    DefaultValuedAttr<F64Attr, "0.0">:$offset,
    Tpu_RequantModeAttr:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RequantFpAxisOp:Tpu_Op<"RequantFpAxis", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>, InOutSameShape]> {
  let summary = "requant float operation";
  let description = [{
    Requant 32/16/8 bit data to int8 or uint8 data, PerAxis(or PerChannel)
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$quant,
    Tpu_RequantModeAttr:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_DequantIntOp:Tpu_Op<"DequantInt", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>, InOutSameShape]> {
  let summary = "dequant operation";
  let description = [{
    Dequant 8 bit data to 32/16 bit data
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    SI32Attr:$multiplier,
    I64Attr:$shift,
    DefaultValuedAttr<I64Attr, "0">:$lshift,
    Tpu_DequantModeAttr:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_DequantIntAxisOp:Tpu_Op<"DequantIntAxis", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>, InOutSameShape]> {
  let summary = "dequant operation";
  let description = [{
    Dequant 8 bit data to 32/16 bit data, PerAxis(or PerChannel)
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$quant,
    DefaultValuedAttr<I64Attr, "0">:$lshift,
    Tpu_DequantModeAttr:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_GroupOp:Tpu_BaseOp<"Group"> {
  let summary = "Group operation";
  let description = [{
    Make ops in one group to inferece by local mem
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    I64Attr:$nsecs,
    I64Attr:$hsecs,
    I64Attr:$swpipl_stage_num,
    I64Attr:$group_type,
    DefaultValuedAttr<I64ArrayAttr, "{0}">:$flow
  );
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let regions = (region SizedRegion<1>:$body);
}

def Tpu_YieldOp : Tpu_BaseOp<"Yield", [Terminator, HasParent<"GroupOp">]> {
  let summary = "Yield values to parent operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Tpu_SoftmaxOp: Tpu_Op<"Softmax",[
    DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>,
    DeclareOpInterfaceMethods<TypeInterface>,
    InOutSameShape]> {
  let summary = "softmax operator";

  let description = [{
    Integrates some operations related to softmax.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$table,
    AnyTensorOrNone:$slope_table,
    AnyTensorOrNone:$reciprocal_table,
    AnyTensorOrNone:$reciprocal_mantissa_table,
    AnyTensorOrNone:$buffer,
    SI32Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$log,
    DefaultValuedAttr<F64Attr, "1.0">:$beta
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LeakyReluOp: Tpu_Op<"LeakyRelu",
   [DeclareOpInterfaceMethods<LocalGenInterface>,
   DeclareOpInterfaceMethods<DynLocalGenInterface>,
    InOutSameShape]> {
  let summary = "leakyrelu operation";
  let description = [{
    The LeakyRelu operation multiples alpha with negative values, and the others keep changeless
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    OptionalAttr<F64Attr>:$alpha,
    // quantize param
    OptionalAttr<SI32Attr>:$multiplier,
    OptionalAttr<SI32Attr>:$multiplier_neg,
    OptionalAttr<I64Attr>:$rshift,
    OptionalAttr<I64Attr>:$rshift_neg,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ConcatOp:Tpu_Op<"Concat", [
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Concatate operation";
  let description = [{
  Concatenates the given sequence of seq tensors in the given dimension.
  All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    I64Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$only_merge,
    // param for cv18xx
    OptionalAttr<I64ArrayAttr>:$multipliers,
    OptionalAttr<I64ArrayAttr>:$rshifts,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // for group
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MulShiftOp: Tpu_Op<"MulShift", [
    DeclareOpInterfaceMethods<LocalGenInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "MulShift operator";

  let description = [{
      Y = int8(X-zx) * multiplier >> rshift + zy)
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    SI32Attr:$multiplier,
    I64Attr:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_PermuteOp: Tpu_Op<"Permute"> {

  let summary = "Permute operator";

  let description = [{
      Perform permute on input.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64ArrayAttr:$order,
    AnyTensorOrNone:$buffer,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    permute_attr_t parseParam();
  }];
}

def Tpu_ShuffleChannelOp: Tpu_Op<"ShuffleChannel"> {

  let summary = "ShuffleChannel operator";

  let description = [{
      Perform ShuffleChannel on input.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64Attr:$group,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_UpsampleOp: Tpu_Op<"Upsample", [
    SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH", "LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Upsample operation";
  let description = [{
    Perform nearest upsample on input.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$scale_h,
    I64Attr:$scale_w,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MaxUnpoolOp: Tpu_Op<"MaxUnpool", [
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]>  {
  let summary = "MaxUnpool operation";
  let description = [{
    Perform MaxUnpool on input.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$mask,
    I64Attr:$scale_h,
    I64Attr:$scale_w,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_PadOp:Tpu_Op<"Pad", [
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH", "LocalGenSupport"]>]> {
  let summary = "Pad operation";
  let description = [{
    This operation pads a tensor according to the paddings you specify.
    paddings is an integer tensor with shape [n, 2], where n is the rank of tensor.
    For each dimension D of input, paddings[D, 0] indicates how many values to add
    before the contents of tensor in that dimension, and paddings[D, 1] indicates
    how many values to add after the contents of tensor in that dimension.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    // for cv18xx reflect mode
    AnyTensorOrNone:$left_select,
    AnyTensorOrNone:$right_select,
    I64ArrayAttr:$paddings,
    DefaultValuedAttr<F64Attr, "0.0">:$val,
    DefaultValuedAttr<I64Attr, "0">:$mode
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_DivOp: Tpu_Op<"Div", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>]> {
  let summary = "div operator";

  let description = [{
    Performs element-wise binary division.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_SliceOp: Tpu_Op<"Slice", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardN", "BackwardH", "LocalGenSupport"]>]> {
  let summary = "Slice operator";
  let description = [{
    Slice Operation on input.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    AnyTensorOrNone:$buffer,
    I64ArrayAttr:$offset,
    I64ArrayAttr:$steps
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    slice_attr_t parseParam();
  }];
}

def Tpu_StridedSliceOp: Tpu_Op<"StridedSlice"> {
  let summary = "Strided Slice operator";

  let description = [{
    Strided Slice Operation on input.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$starts,
    AnyRankedTensor:$ends,
    AnyRankedTensor:$strides,
    I64Attr:$begin_mask,
    I64Attr:$end_mask,
    I64Attr:$ellipsis_mask,
    I64Attr:$new_axis_mask,
    I64Attr:$shrink_axis_mask
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_SplitOp: Tpu_Op<"Split"> {
  let summary = "Split operator";

  let description = [{
    Split input tensor into a list of tensors.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64Attr:$axis,
    I64Attr:$num
  );
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
}

def Tpu_TopKOp:Tpu_Op<"TopK"> {
  let summary = "TopK operation";
  let description = [{
    Integrates some operations related to topk.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis,
    I64Attr:$K,
    DefaultValuedAttr<BoolAttr, "true">:$largest,
    DefaultValuedAttr<BoolAttr, "true">:$sorted
  );
  let results = (outs
    AnyTensorOrNone:$values,
    AnyTensorOrNone:$indices
  );
}

def Tpu_NonZeroOp:Tpu_Op<"NonZero"> {
  let summary = "NonZero operation";
  let description = [{
    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$buffer,
    NonZeroOrderAttr:$order
  );
  let results = (outs
    AnyRankedTensor:$output
  );
}

def Tpu_DeconvOp: Tpu_Op<"Deconv",[
    SupportFuseRelu,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH", "LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "deconvolution operator";

  let description = [{
    "Perform deconvolution operation."
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$filter,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    //new param
    BoolAttr:$with_bias,
    OptionalAttr<I64ArrayAttr>:$multiplier,
    OptionalAttr<I64ArrayAttr>:$rshift,
    DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);

  let extraClassDeclaration = [{
    deconv_attr_t parseParam();
  }];
}

def Tpu_SqueezeOp: Tpu_Op<"Squeeze"> {
  let summary = "Squeeze operator";

  let description = [{
    The operator squeeze the input shapes by given axis.
  }];

  let arguments = (ins
    AnyRankedTensor:$inputs,
    I64ArrayAttr:$axes
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ScaleOp: Tpu_Op<"Scale", [
  SupportFuseRelu, InOutSameShape,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Scale operator";

  let description = [{
    Y = X * S + B,
    where the shape of X/Y is [n, c, h, w] and the shape of S/B is [1, c, 1, 1].
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$scale,
    AnyRankedTensor:$bias,
    AnyTensorOrNone:$lshift,

    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LRNOp: Tpu_Op<"LRN", [InOutSameShape]> {
  let summary = "Local Response Normalization";

  let description = [{
    It normalizes over local input regions. The local region is defined across the channels.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$table,
    AnyTensorOrNone:$mantissa,
    I64Attr:$size,
    DefaultValuedAttr<F64Attr, "0.0001">:$alpha,
    DefaultValuedAttr<F64Attr, "0.75">:$beta,
    DefaultValuedAttr<F64Attr, "1.0">:$bias
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_GRUOp: Tpu_Op<"GRU"> {
  let summary = "GRU operator";

  let description = [{
    Perform RNN GRU operation.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$filter,
    AnyTensorOrNone:$recurrence,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$initial_h,
    AnyTensorOrNone:$buffer,
    AnyTensorOrNone:$sigmoid_table,
    AnyTensorOrNone:$sigmoid_slope_table,
    AnyTensorOrNone:$tanh_table,
    AnyTensorOrNone:$tanh_slope_table,
    I64Attr: $hidden_size,
    BoolAttr: $bidirectional,
    DefaultValuedAttr<BoolAttr, "true">:$linear_before_reset,
    DefaultValuedAttr<BoolAttr, "false">:$batch_first
  );

  let results = (outs
    AnyTensorOrNone:$Y,
    AnyTensorOrNone:$Y_h);

  let extraClassDeclaration = [{
    gru_attr_t parseParam();
  }];
}

def Tpu_LSTMOp: Tpu_Op<"LSTM"> {
  let summary = "LSTM operator";

  let description = [{
    Perform RNN LSTM operation.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$filter,
    AnyTensorOrNone:$recurrence,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$initial_h,
    AnyTensorOrNone:$initial_c,
    AnyTensorOrNone:$buffer,
    I64Attr: $hidden_size,
    BoolAttr: $bidirectional,
    DefaultValuedAttr<BoolAttr, "false">:$batch_first
  );

  let results = (outs
    AnyTensorOrNone:$Y,
    AnyTensorOrNone:$Y_h,
    AnyTensorOrNone:$Y_c);
  let extraClassDeclaration = [{
    lstm_attr_t parseParam();
  }];
}

def Tpu_LSTMCVIOp: Tpu_Op<"LSTMCVI"> {
  let summary = "LSTM operator for cv18xx";

  let description = [{
    Perform RNN LSTM operation.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$recurrence,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$initial_h,
    AnyTensorOrNone:$initial_c,
    AnyTensorOrNone:$sigmoid_table,
    AnyTensorOrNone:$sigmoid_slope_table,
    AnyTensorOrNone:$tanh_table,
    AnyTensorOrNone:$tanh_slope_table,
    BoolAttr: $bidirectional,
    DefaultValuedAttr<BoolAttr, "false">:$batch_first
  );

  let results = (outs
    AnyTensorOrNone:$Y,
    AnyTensorOrNone:$Y_h,
    AnyTensorOrNone:$Y_c);
  let extraClassDeclaration = [{
    lstm_attr_t parseParam();
  }];
}

def Tpu_TileOp:Tpu_Op<"Tile", [
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Tile operation";
  let description = [{
    Returns a tensor with the same type as the input, with a new shape
    specified by the shape argument.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis,
    I64Attr:$tile
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_GatherOp: Tpu_Op<"Gather", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Gather operator";
  let description = [{
    Perform Gather operation on the given axis.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$indices,

    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_PReluOp : Tpu_Op<"PReluOp", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  InOutSameShape]> {
  let summary = "PReluOp operator";
  let description = [{
     f(x) = slope * x   for x < 0
     f(x) = x           for x >= 0
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$slope,
    DefaultValuedAttr<SI32Attr, "0">:$rshift,
    OptionalAttr<SI32Attr>:$rshift_pos,
    OptionalAttr<SI32Attr>:$multiplier_pos,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_GenericCpuOp : Tpu_Op<"GenericCpu", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "generic cpu operator";
  let description = [{
    Generic Cpu Op.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    StrAttr:$cpu_op_name,
    OptionalAttr<DictionaryAttr>:$param
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_InterpOp: Tpu_Op<"Interp"> {
  let summary = "Interp operation";
  let description = [{
    Perform Interp on input.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$scale_h,
    F64Attr:$scale_w,
    Tpu_ResizeModeAttr:$mode,
    Tpu_ResizeCoordModeAttr:$coord_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ReduceOp: Tpu_Op<"Reduce"> {
  let summary = "Reduce operator";
  let description = [{
      Computes the mean/max/prod/sum of the input tensor's element along the provided axes.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$buffer, // cv18xx reciprocal_table
    AnyTensorOrNone:$reciprocal_mantissa_table,
    I64ArrayAttr:$axes,
    I64Attr:$keepdims,
    ReduceModeAttr:$mode,
    // for cv18xx
    OptionalAttr<I64ArrayAttr>:$multiplier,
    OptionalAttr<I64ArrayAttr>:$rshift
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    reduce_attr_t parseParam();
  }];
}

def Tpu_ArgOp: Tpu_Op<"Arg", [DeclareOpInterfaceMethods<TypeInterface>]>  {
  let summary = "Arg operator";
  let description = [{
    Computes the indices of the min/max/ of the input tensor's element along the provided axis.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis,
    I64Attr:$keepdims,
    ArgModeAttr:$mode
  );

  let results = (outs
    AnyRankedTensor:$indices,
    AnyTensorOrNone:$values
  );
}

def Tpu_WhereOp: Tpu_Op<"Where", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Where operator";
  let description = [{
    Return elements, either from X or Y, depending on condition.
  }];
  let arguments = (ins
    AnyRankedTensor:$cond,
    AnyTensorOrNone:$tbrn,
    AnyTensorOrNone:$fbrn,
    DefaultValuedAttr<BoolAttr, "false">:$x_is_const,
    DefaultValuedAttr<BoolAttr, "false">:$y_is_const,
    DefaultValuedAttr<F64Attr, "0.0">:$x_const_val,
    DefaultValuedAttr<F64Attr, "0.0">:$y_const_val
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MaskedFillOp: Tpu_Op<"MaskedFill", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "MaskedFill operator";
  let description = [{
    Return elements, either from X or Y, depending on condition.
  }];
  let arguments = (ins
    AnyRankedTensor:$cond,
    AnyRankedTensor:$brn,
    BoolAttr:$inversed,
    F64Attr:$const_val
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_CompareOp: Tpu_Op<"Compare", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Compare operator";
  let description = [{
    Returns the tensor resulted from performing the compare
    operation elementwise on the input tensors A and B
  }];
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    CompareModeAttr:$mode
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_CompareConstOp: Tpu_Op<"CompareConst", [
  InOutSameShape,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "CompareConst operator";
  let description = [{
    Returns the tensor resulted from performing the compare
    operation elementwise on the input tensors A and Const
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    CompareModeAttr:$mode,
    F64Attr:$const_val,
    BoolAttr:$inversed
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LayerNormOp : Tpu_Op<"LayerNorm", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>
  ]> {
  let summary = "LayerNorm operation";
  let description = [{
    layer normalization
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$table,            // cv18xx
    AnyTensorOrNone:$mantissa_table,   // cv18xx
    I64ArrayAttr:$normalized_shape,
    SI32Attr:$axis,
    F64Attr:$eps
  );
  let results = (outs
  	AnyRankedTensor:$output,
  	AnyTensorOrNone:$mean,
  	AnyTensorOrNone:$rstd
  );
}

def Tpu_InstanceNormOp : Tpu_Op<"InstanceNorm", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>
  ]> {
  let summary = "InstanceNorm operation";
  let description = [{
    instance normalization
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$table,            // cv18xx
    AnyTensorOrNone:$mantissa_table,   // cv18xx
    F64Attr:$eps
  );
  let results = (outs
  	AnyRankedTensor:$output
  );
}

def Tpu_GroupNormOp : Tpu_Op<"GroupNorm", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>
  ]> {
  let summary = "GroupNorm operation";
  let description = [{
    group normalization
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$table,            // cv18xx
    AnyTensorOrNone:$mantissa_table,   // cv18xx
    I64Attr:$num_groups,
    F64Attr:$eps
  );
  let results = (outs
  	AnyRankedTensor:$output
  );
}

def Tpu_PixelNormOp : Tpu_Op<"PixelNorm", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<TypeInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>
  ]> {
  let summary = "PixelNorm operation";
  let description = [{
    pixel normalization (normalize along c-axis)
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$table,           // cv18xx
    AnyTensorOrNone:$mantissa_table,  // cv18xx
    F64Attr:$eps
  );
  let results = (outs
  	AnyRankedTensor:$output
  );
}

def Tpu_CopyOp: Tpu_Op<"Copy"> {
  let summary = "TG copy operator.";

  let description = [{
    Inputs:
      `input`          : required, the activation memref.

    Attributes:
      `input_stride`    : required, input data stride(saved as I64ArrayAttr).
      `output_stride`   : required, output data stride(saved as I64ArrayAttr).

    Result:
      `output`          : result tensor.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64ArrayAttr:$shape,
    I64ArrayAttr:$input_stride,
    I64ArrayAttr:$output_stride,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);

}

def Tpu_CscOp : Tpu_Op<"Csc"> {
  let summary = "Color space convert for model's inputs";
  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:

      `y_align`         : width alignment of channel y.
      `w_align`         : width alignment of channel uv.
      `channel_align`   : alignment of channel.
      `pixel_type`      : required, 1--i420 2--nv12 3--nv21


    Result:
      `output`          : result tensor.
  }];
  let arguments = (
    ins AnyRankedTensor:$input,
    OptionalAttr<I32ArrayAttr>:$channel_order,
    StrAttr:$pixel_format,
    DefaultValuedAttr<BoolAttr, "true">:$aligned,
    DefaultValuedAttr<I64Attr, "1">:$pixel_type,
    DefaultValuedAttr<I64Attr, "64">:$y_align,
    DefaultValuedAttr<I64Attr, "64">:$w_align,
    DefaultValuedAttr<I64Attr, "64">:$channel_align
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ScaleLutOp : Tpu_Op<"ScaleLut", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  InOutSameShape]> {
  let summary = "scale lut operator.";

  let description = [{
    Inputs:
      `input`          : required, the variadic activation memref.
      `table`          : required, the lookup table
      `sign`           : if output is signed

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    AnyRankedTensor:$table,
    F64ArrayAttr:$scale,
    F64ArrayAttr:$bias,
    DefaultValuedAttr<BoolAttr, "true">:$sign,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def TPU_SwapChannelOp: Tpu_Op<"SwapChannel" ,[
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  InOutSameShape]> {
  let summary = "SwapChannel operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `channel_order`   : required, channel swap order

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64ArrayAttr:$channel_order
  );

  let results = (outs AnyRankedTensor:$output);
}

def TPU_SwapDimInnerOp: Tpu_Op<"SwapDimInner" ,[
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  InOutSameShape]> {
  let summary = "SwapDimInner operator.";

  let description = [{
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64ArrayAttr:$offset
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ScatterElementsOp: Tpu_Op<"ScatterElements">  {
  let summary = "ScatterElements op";
  let description = [{
    ScatterElements takes three inputs data, updates, and indices
    of the same rank r >= 1 and an optional attribute axis that
    identifies an axis of data (by default, the outer-most axis,
    that is axis 0). The output of the operation is produced by
    creating a copy of the input data, and then updating its
    value to values specified by updates at specific index
    positions specified by indices. Its output shape is the
    same as the shape of data.

    Inputs:
      `input`       : Tensor of rank r >= 1.
      `indices`     : Tensor of int32/int64 indices, of r >= 1 (same rank
                      as input). All index values are expected to be within
                      bounds [-s, s-1] along axis of size s. It is an error
                      if any of the index values are out of bounds..
      `updates`     : Tensor of rank r >=1 (same rank and shape as indices).

    Outputs:
      `output`      : Tensor of rank r >= 1 (same rank as input).
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$indices,
    AnyRankedTensor:$updates,
    I64Attr:$axis
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ScatterNDOp: Tpu_Op<"ScatterND", [
  DeclareOpInterfaceMethods<TypeInterface>]>  {
  let summary = "ScatterND operator";
  let description = [{
    The output of the operation is produced by creating a copy of the input data,
    and then updating its value to values specified by updates at
    specific index positions specified by indices.

    Inputs:
      `input_data`           : Tensor of rank r >= 1.
      `indices`      : Tensor of rank q >= 1.
      `updates`     : Tensor of rank q + r - indices_shape[-1] - 1.

    Outputs:
      `output`       : Tensor of rank r >= 1.
  }];

  let arguments = (ins
    AnyRankedTensor:$input_data,
    AnyRankedTensor:$indices,
    AnyRankedTensor:$updates
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RoiAlignOp: Tpu_Op<"RoiAlign"> {
  let summary = "RoiAlign operator";
  let description = [{
    RoiAlign consumes an input tensor X and region of interests
    (rois) to apply pooling across each RoI.

    Inputs:
      `input`         : Input data tensor, 4-D tensor.
      `rois`          : RoIs (Regions of Interest) to pool over;
                        rois is 2-D input of shape (num_rois, 4)
                        given as [[x1, y1, x2, y2], ...]. .
      `batch_indices` : 1-D tensor with each element denoting
                        the index of the corresponding image in
                        the batch.

    Outputs:
      `output`        : RoI pooled output, 4-D tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$rois,
    RoiAlignModeAttr:$mode,
    I64Attr:$output_height,
    I64Attr:$output_width,
    I64Attr:$sampling_ratio,
    F64Attr:$spatial_scale,
    BoolAttr:$align_corners
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_YoloDetectionOp : Tpu_Op<"YoloDetection"> {
  let summary = "YoloDetection operator";
  let description = [{
    Perform yolo detection on feature map
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    I64Attr:$net_input_h,
    I64Attr:$net_input_w,
    F64Attr:$nms_threshold,
    F64Attr:$obj_threshold,
    I64Attr:$keep_topk,
    DefaultValuedAttr<BoolAttr, "false">:$spp_net,
    DefaultValuedAttr<BoolAttr, "false">:$tiny,
    DefaultValuedAttr<BoolAttr, "false">:$yolo_v4,
    DefaultValuedAttr<I64Attr, "80">:$class_num,
    DefaultValuedAttr<StrAttr, "">:$anchors,
    DefaultValuedAttr<I64Attr, "0">:$flag,
    DefaultValuedAttr<I64Attr, "3">:$num_boxes,
    DefaultValuedAttr<I64Attr, "3">:$mask_group_size,
    DefaultValuedAttr<ConfinedAttr<I64ArrayAttr, [ArrayCount<3>]>, "0">:$scale,
    DefaultValuedAttr<ConfinedAttr<I64ArrayAttr, [ArrayCount<9>]>, "0">:$mask
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_DetectionOutputOp : Tpu_Op<"DetectionOutput"> {
  let summary = "DetectionOutput operation";
  let description = [{
    Intended for use with MultiBox detection method to generate prior.
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    I64Attr:$num_classes,
    DefaultValuedAttr<I64Attr, "0">:$background_label_id,
    F64Attr:$nms_threshold,
    I64Attr:$top_k,
    DetectionOutputCodeTypeAttr:$code_type,
    I64Attr:$keep_top_k,
    F64Attr:$confidence_threshold,
    DefaultValuedAttr<BoolAttr, "true">:$share_location,
    DefaultValuedAttr<F64Attr, "0">:$variance_encoded_in_target,
    DefaultValuedAttr<F64Attr, "1">:$eta,
    DefaultValuedAttr<I64Attr, "1">:$onnx_nms
  );
  let results = (outs AnyTensor:$output);
}
#endif // TPU_OPS
