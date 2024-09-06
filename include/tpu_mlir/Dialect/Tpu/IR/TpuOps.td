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
//include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "tpu_mlir/Interfaces/LocalGenInterface.td"
include "tpu_mlir/Interfaces/GlobalGenInterface.td"
include "tpu_mlir/Interfaces/InferenceInterface.td"
include "tpu_mlir/Interfaces/TypeInterface.td"
include "tpu_mlir/Interfaces/DynLocalGenInterface.td"
include "tpu_mlir/Interfaces/DynGlobalGenInterface.td"
include "tpu_mlir/Interfaces/IndexingMapsInterface.td"
include "tpu_mlir/Interfaces/InplaceInterface.td"
include "tpu_mlir/Traits/Traits.td"

// =============================================================================
//
// Defines Tpu Dialect.
//
//===----------------------------------------------------------------------===//

def Tpu_Dialect : Dialect {
  let name = "tpu";
  let summary = "A tpu dialect for the SOPHGO Deep Learning processors";
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
def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual", "NotEqual", "Not", "And", "Xor"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceL2","ReduceL1","ReduceSum","ReduceProd"]>;
def RoiAlignModeAttr: AnyStrAttrOf<["Avg","Max"]>;
def NonZeroOrderAttr: AnyStrAttrOf<["ColMajor","RowMajor"]>;
def DetectionOutputCodeTypeAttr: AnyStrAttrOf<["CORNER", "CENTER_SIZE", "CORNER_SIZE"]>;
def YoloVersionAttr: AnyStrAttrOf<["yolov3", "yolov3_tiny", "yolov3_spp", "yolov4", "yolov5","yolov8"]>;
def MatchTemplateModeAttr: AnyStrAttrOf<["TM_CCOEFF_NORMED", "TM_SQDIFF"]>;
def BinaryShiftAttr: AnyStrAttrOf<["Add","Sub", "Mul"]>;

def Tpu_LayerGroupAttr : Tpu_Attr<"LayerGroup", "lg"> {
  let summary = "Structure of layer group parameters";
  let parameters = (ins
    "int64_t":$out_addr,
    "int64_t":$out_size,
    "int64_t":$buffer_addr,
    "int64_t":$buffer_size,
    "bool":$eu_align,
    "bool":$can_merge,
    "DenseI64ArrayAttr":$in_hslice_offset,
    "DenseI64ArrayAttr":$n_idx,
    "DenseI64ArrayAttr":$n_slice,
    "DenseI64ArrayAttr":$c_idx,
    "DenseI64ArrayAttr":$c_slice,
    "DenseI64ArrayAttr":$d_idx,
    "DenseI64ArrayAttr":$d_slice,
    "DenseI64ArrayAttr":$h_idx,
    "DenseI64ArrayAttr":$h_slice,
    "DenseI64ArrayAttr":$w_idx,
    "DenseI64ArrayAttr":$w_slice,
    "int64_t":$id,
    "int64_t":$stage,
    "int64_t":$slice_idx,
    "int64_t":$group_type
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Tpu_CompressAttr : Tpu_Attr<"Compress", "do_compress"> {
  let summary = "Structure of compress or decompress parameters";
  let parameters = (ins
    "bool":$do_compress,
    "bool":$do_decompress,
    "int32_t":$bias0,
    "int32_t":$bias1,
    "bool":$is_signed,
    "bool":$zero_guard
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
      I32EnumAttrCase<"QDM", 4>,       // similar to TFLite
      I32EnumAttrCase<"OnlyScale", 5>       // requant to fp8 by scale and dtype convert
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RequantModeAttr : EnumAttr<Tpu_Dialect, Tpu_RequantMode, "rq_mode">;

def Tpu_PaddingMode: I32EnumAttr<"PaddingMode",
    "requant mode supported by PadOp",
    [
      I32EnumAttrCase<"constant", 0>,
      I32EnumAttrCase<"reflect", 1>,
      I32EnumAttrCase<"symmetric", 2>,
      I32EnumAttrCase<"edge", 3>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_PaddingModeAttr : EnumAttr<Tpu_Dialect, Tpu_PaddingMode, "Pad_Mode">;


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

def Tpu_DevPattern: I32EnumAttr<"DevPattern",
    "Patterns of distribution in multi devices",
    [
      I32EnumAttrCase<"MatMulColumn", 0>,
      I32EnumAttrCase<"MatMulRow", 1>,
      I32EnumAttrCase<"MatMulMerge", 2>,
      I32EnumAttrCase<"MatMulSliceMerge", 3>,
      I32EnumAttrCase<"MatMulTopK", 4>,
      I32EnumAttrCase<"MatMulSliceMerge2", 5>,
      I32EnumAttrCase<"MatMulSliceMerge3", 6>,
      I32EnumAttrCase<"AttentionSliceMerge", 7>,
      I32EnumAttrCase<"AttentionSliceMerge2", 8>,
      I32EnumAttrCase<"EmbeddingSliceMerge", 9>,
      I32EnumAttrCase<"FAttentionSliceMerge", 10>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_DevPatternAttr : EnumAttr<Tpu_Dialect, Tpu_DevPattern, "dev_pattern">;

def Tpu_CorePattern: I32EnumAttr<"CorePattern",
    "Patterns of distribution in multi cores",
    [
      I32EnumAttrCase<"Common", 0>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_CorePatternAttr : EnumAttr<Tpu_Dialect, Tpu_CorePattern, "core_pattern">;

// def Tpu_DevBeginMethod: I32EnumAttr<"DevBeginMethod",
//     "Begin method of distribution",
//     [
//       I32EnumAttrCase<"BeginFromCopy", 0>,
//       I32EnumAttrCase<"BeginFromSplit", 1>
//     ]> {
//   let genSpecializedAttr = 0;
//   let cppNamespace = "::tpu_mlir::tpu";
// }
// def Tpu_DevBeginMethodAttr : EnumAttr<Tpu_Dialect, Tpu_DevBeginMethod, "begin_method">;
// def Tpu_DevBeginMethodArrayAttr : TypedArrayAttrBase<Tpu_DevBeginMethodAttr, "Array of BeginMethod">;
//
// def Tpu_DevEndMethod: I32EnumAttr<"DevEndMethod",
//     "End method of distribution",
//     [
//       I32EnumAttrCase<"EndToSum", 0>,
//       I32EnumAttrCase<"EndToTopK", 1>,
//       I32EnumAttrCase<"EndToConcat", 2>
//     ]> {
//   let genSpecializedAttr = 0;
//   let cppNamespace = "::tpu_mlir::tpu";
// }
// def Tpu_DevEndMethodAttr : EnumAttr<Tpu_Dialect, Tpu_DevEndMethod, "end_method">;
// def Tpu_DevEndMethodArrayAttr : TypedArrayAttrBase<Tpu_DevEndMethodAttr, "Array of EndMethod">;

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
      I32EnumAttrCase<"LOG2", 35>,
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
      I32EnumAttrCase<"asymmetric", 3>,
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
      I32EnumAttrCase<"SWITCH",      3>,
      I32EnumAttrCase<"MERGE",       4>,
      I32EnumAttrCase<"LOOP",       5>,
      I32EnumAttrCase<"UNKNOW",       6>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RunModeAttr : EnumAttr<Tpu_Dialect, Tpu_RunMode, "run_mode">;

def Tpu_ImageOutFormat : I32EnumAttr<"ImageOutFormat",
  "attr used for tpu.yuv2rgbformulaOp",
  [
    I32EnumAttrCase<"FLOAT32", 0>,
    I32EnumAttrCase<"UINT8", 1>,
  ]
> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}

def Tpu_ImageOutFormatAttr : EnumAttr<Tpu_Dialect, Tpu_ImageOutFormat, "image_out_format">;

def Tpu_Yuv2rgbFormula : I32EnumAttr<"Yuv2rgbFormula",
  "formula mode used for tpu.yuv2rgbformulaOp",
  [
    I32EnumAttrCase<"_601_limited", 0>,
    I32EnumAttrCase<"_601_full", 1>,
  ]
> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}

def Tpu_Yuv2rgbFormulaAttr : EnumAttr<Tpu_Dialect, Tpu_Yuv2rgbFormula, "yuv2rgb_formula_mode">;

def Tpu_BufferType: I32EnumAttr<"BufferType",
    "buffer op type",[
      I32EnumAttrCase<"GMEM",    0>,
      I32EnumAttrCase<"L2",      1>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}

def Tpu_BufferTypeAttr : EnumAttr<Tpu_Dialect, Tpu_BufferType, "buffer_type">;

//===----------------------------------------------------------------------===//
// Tpu Types.
//===----------------------------------------------------------------------===//

def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;

def Tpu_CPInterleaveAttr : Tpu_Attr<"CPInterleave", "cp_interleave"> {
  let summary = "Core Parallel interleave configuration.";
  let parameters = (ins
    "int32_t":$region_id,
    "int64_t":$offset,
    "int64_t":$address
    );
  let assemblyFormat =
      "`<` `region_id` `=` $region_id`,` `offset` `=` $offset`,` `address` `=` $address `>`";
}

def Tpu_DDRInterleaveAttr : Tpu_Attr<"DDRInterleave", "ddr_interleave"> {
  let summary = "DDRC interleave configuration.";
  let parameters = (ins
    "AffineMap":$layout,
    ArrayRefParameter<"int64_t",
      "interleave dimension"
      >:$interleave_dim,
    "int64_t":$address
    );
  let assemblyFormat =
      "`<``{` `map` `=` $layout`,` `interleave_dim` `=` `[` $interleave_dim `]` `,` `address` `=` $address `}``>`";
}

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

  let arguments = (ins
    DefaultValuedAttr<Tpu_BufferTypeAttr, "tpu::BufferType::GMEM">:$buffer_type
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    static mlir::Value create(mlir::Operation * OwnerOp,
                              mlir::RankedTensorType& type,
                              tpu::BufferType buffer_type = tpu::BufferType::GMEM);
  }];
}

def Tpu_Conv2DOp: Tpu_Op<"Conv2D", [SupportFuseRelu,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface,
      ["BackwardH", "BackwardW", "LocalGenSupport", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface,
      ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>]> {
  let summary = "convolution 2d operator";

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
    DefaultValuedAttr<I64Attr, "1">:$weight_is_coeff,
    DefaultValuedAttr<BoolAttr, "false">:$coeff_merged,
    DefaultValuedAttr<I64Attr, "0">:$use_3ic_optimize,
    DefaultValuedAttr<I64Attr, "0">:$kernel_zp,
    OptionalAttr<I64Attr>:$use_winograd, // 0 not use, 1 used in toptotpu, 2 apply nodechip
    OptionalAttr<I64ArrayAttr>:$multiplier,
    OptionalAttr<I64ArrayAttr>:$rshift,
    DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfUp">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    // fuse leakyRelu
    OptionalAttr<BoolAttr>:$do_leaky_relu,
    OptionalAttr<F64Attr>:$neg_slope,
    OptionalAttr<SI32Attr>:$multiplier_pos,
    OptionalAttr<SI32Attr>:$multiplier_neg,
    OptionalAttr<SI32Attr>:$rshift_pos,
    OptionalAttr<SI32Attr>:$rshift_neg,
    OptionalAttr<F64ArrayAttr>:$out_f8_scales,
    //nnvlc
    DefaultValuedAttr<BoolAttr, "true">:$support_compress,
    OptionalAttr<Tpu_CompressAttr>:$compress_info
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    conv_attr_t parseParam();
    void assign_fw_param(void *param);
  }];
}

def Tpu_Conv3DOp: Tpu_Op<"Conv3D", [SupportFuseRelu,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface,
      ["LocalGenSupport", "BackwardH", "BackwardW", "BackwardD", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface,
      ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>]> {
  let summary = "convolution 2d operator";

  let description = [{}];

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
    OptionalAttr<I64ArrayAttr>:$multiplier,
    OptionalAttr<I64ArrayAttr>:$rshift,
    DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfUp">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    OptionalAttr<F64Attr>:$out_f8_scale
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    conv_attr_t parseParam();
    void assign_fw_param(void *param);
  }];
}

class Tpu_PoolOp <string mnemonic, list<Trait> traits = []> : Tpu_Op<mnemonic,
  !listconcat(traits, [SupportFuseRelu,
   DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "BackwardW", "assign_sec_info"]>,
   DeclareOpInterfaceMethods<DynLocalGenInterface, ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>,
   DeclareOpInterfaceMethods<IndexingMapsInterface>])> {
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
    DefaultValuedAttr<BoolAttr, "false">:$is_adaptive,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    /// symmetric quantize param
    OptionalAttr<SI32Attr>:$multiplier,
    OptionalAttr<SI32Attr>:$rshift,
    /// asymmetric quantize param
    OptionalAttr<F64Attr>:$scale,
    OptionalAttr<F64Attr>:$offset,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$first_round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    /// fp8 quantize param
    OptionalAttr<F64Attr>:$fp8_out_scale
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    pool_attr_t parseParam();
    void assign_fw_param(void *param);
  }];
}

def Tpu_Pool1DOp:Tpu_PoolOp<"Pool1D">;
def Tpu_Pool2DOp:Tpu_PoolOp<"Pool2D">;
def Tpu_Pool3DOp:Tpu_PoolOp<"Pool3D",[
  DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardD"]>]>{

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$buffer, //for BM1684 Pool3D
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
    OptionalAttr<SI32Attr>:$rshift,
    /// asymmetric quantize param
    OptionalAttr<F64Attr>:$scale,
    OptionalAttr<F64Attr>:$offset,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$first_round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    /// fp8 quantize param
    OptionalAttr<F64Attr>:$fp8_out_scale
  );
}

def Tpu_MaxPoolWithMaskOp: Tpu_Op<"MaxPoolWithMask",
  [SupportFuseRelu,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "BackwardW", "assign_sec_info"]>,
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
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "BackwardW", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    OptionalAttr<F64ArrayAttr>:$f8_scales, // for fp8 quant
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_AddConstOp: Tpu_Op<"AddConst", [
  SupportFuseRelu, InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    DefaultValuedAttr<SI32Attr, "0">:$rshift,
    DefaultValuedAttr<F64Attr, "1.0">:$f8_scale,  // input scale
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizeMethod = 1;
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_BinaryShiftOp: Tpu_Op<"BinaryShift", [
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "Binary with shift operator";

  let description = [{
      Z = saturation(X +/-/* Y >> -shift)
  }];

  let arguments = (
    ins AnyRankedTensor:$input1,
    AnyRankedTensor:$input2,
    BinaryShiftAttr:$mode,
    SI32Attr:$shift,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "true">:$saturation,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_BinaryConstShiftOp: Tpu_Op<"BinaryConstShift", [
    InOutSameShape, SupportElementwise,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "Binary Const with shift operator";

  let description = [{
      Y = saturation(X +/-/* scale >> -shift)
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    SI32Attr:$scale,
    BinaryShiftAttr:$mode,
    SI32Attr:$shift,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "true">:$saturation,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let hasCanonicalizeMethod = 1;
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RopeOp: Tpu_Op<"Rope", [
    SupportElementwise,
    DeclareOpInterfaceMethods<LocalGenInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "Rope Operator";

  let description = [{
     rope operator.
  }];

  let arguments = (
    ins AnyRankedTensor:$input1,
    AnyRankedTensor:$input2,
    AnyRankedTensor:$input3,
    DefaultValuedAttr<BoolAttr, "false">:$is_permute_optimize,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$mul1_round_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$mul2_round_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$add_round_mode,
    DefaultValuedAttr<SI32Attr, "0">:$mul1_shift,
    DefaultValuedAttr<SI32Attr, "0">:$mul2_shift,
    DefaultValuedAttr<SI32Attr, "0">:$add_shift,
    DefaultValuedAttr<BoolAttr, "true">:$mul1_saturation,
    DefaultValuedAttr<BoolAttr, "true">:$mul2_saturation,
    DefaultValuedAttr<BoolAttr, "true">:$add_saturation,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  // let hasCanonicalizeMethod = 1;
  let results = (outs AnyRankedTensor:$output);
}
def Tpu_PreprocessOp: Tpu_Op<"Preprocess"> {
  let summary = "FusePreprocess, it's just a placeholder op.";
  let description = [{
    It may be divided to permute + slice + scale/scale_lut ops.
  }];
  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$quant_mode,
    StrAttr:$customization_format,
    StrAttr:$channel_order,
    I64ArrayAttr:$resize_dims,
    F64ArrayAttr:$scale,
    F64ArrayAttr:$mean,
    DefaultValuedAttr<BoolAttr, "true">:$sign
  );
  let hasCanonicalizeMethod = 1;
  let results = (outs AnyTensor:$output);
}

def Tpu_MeanStdScaleOp: Tpu_Op<"MeanStdScale",
  [DeclareOpInterfaceMethods<TypeInterface>,DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>]> {
  let summary = "MeanStdScale, it's for preprocess.";
  let description = [{
    for preprocess, do multiplier&rshift quantize.
  }];
  let arguments = (
    ins AnyTensor:$input,
    AnyRankedTensor:$f32_param,
    StrAttr:$quant_mode,
    StrAttr:$customization_format,
    StrAttr:$channel_order,
    DefaultValuedAttr<BoolAttr, "true">:$sign,
    F64ArrayAttr:$scale,
    F64ArrayAttr:$std,
    F64ArrayAttr:$mean,
    F64ArrayAttr:$zero_points,
    I64ArrayAttr:$resize_dims,
    I32ArrayAttr:$multi,
    I32ArrayAttr:$rshift,
    I32ArrayAttr:$offset,
    StrAttr:$rounding_mode
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_PackRawOp: Tpu_Op<"PackRaw"> {
  let summary = "Preprocess for raw image.";
  let description = [{
    1. preprocess raw image from 3 byte 2 pixel to 2 byte 1 pixel,
      for each row pixel fill 0 in top 4 bits;
      for each img block pack to 4 channels pattern;
    2. cast to float then normalize and clip by black & white level;
    3. cast to odtype then pad img to align padding param and move to ddr;
  }];
  let arguments = (
    ins AnyTensor:$input,
    F64Attr:$white_level,
    F64Attr:$black_level,
    F64Attr:$threshold,
    I64ArrayAttr:$channel_order
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_DepackRawOp: Tpu_Op<"DepackRaw",
    [DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Postprocess for raw image.";
  let description = [{
    1. depack channel (b, bh * bw, ih + ph, iw + pw) -> (b, 1, oh * bh, ow * bw)
    2. cast to INT16 then postprocess img to raw pattern ( 3 byte 2 pixel )
    3. (b, bh * bw, ih + ph, iw + pw) 16bit -> (b, 1, oh * bh, ow * bw * 3 / 2) 8bit
  }];
  let arguments = (
    ins AnyTensor:$input,
    I64Attr:$padding_h,
    I64Attr:$padding_w,
    F64Attr:$white_level,
    F64Attr:$black_level,
    I64ArrayAttr:$channel_order
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_SubOp: Tpu_Op<"Sub",
    [SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    OptionalAttr<F64ArrayAttr>:$f8_scales, // for fp8 quant
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_SubConstOp: Tpu_Op<"SubConst", [
  SupportFuseRelu, InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    DefaultValuedAttr<SI32Attr, "0">:$rshift,
    DefaultValuedAttr<F64Attr, "1.0">:$f8_scale,  // input scale
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MulOp: Tpu_Op<"Mul",
    [SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    DefaultValuedAttr<SI32Attr, "0">:$rshift,
    DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
    OptionalAttr<F64ArrayAttr>:$out_f8_scales,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_MaxOp: Tpu_Op<"Max", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_MinOp: Tpu_Op<"Min", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_MaxConstOp: Tpu_Op<"MaxConst", [
   InOutSameShape, SupportElementwise,
   DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<SI32Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_MinConstOp: Tpu_Op<"MinConst", [
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<SI32Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_ActiveOp: Tpu_Op<"Active",[
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
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
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_ClipOp: Tpu_Op<"Clip",
  [DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
   DeclareOpInterfaceMethods<IndexingMapsInterface>,
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

def Tpu_ShapeClipOp: Tpu_Op<"ShapeClip", [
    ShapeProducer, ShapeConsumer]> {
  let summary = "ShapeClip operation running on CPU";
  let description = [{
     The operator limits the given input to a certain range.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    F32Attr:$min,
    F32Attr:$max
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_Yuv2rgbFormulaOp: Tpu_Op<"Yuv2rgbFormula">{
  let summary = "Yuv2rgbFormulaOp operator";
  let description = [{
     Yuv2rgbFormulaOp operator.
  }];

  let arguments = (ins
    AnyTensor:$YUV,
    UI32Attr:$src_format,
    UI32Attr:$dst_format,
    Tpu_ImageOutFormatAttr:$image_format,
    Tpu_Yuv2rgbFormulaAttr:$formula_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MulConstOp: Tpu_Op<"MulConst", [
  SupportFuseRelu, InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<TypeInterface>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    DefaultValuedAttr<SI32Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let hasCanonicalizer = 1;

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_ReciprocalOp: Tpu_Op<"Reciprocal", [
  SupportFuseRelu, InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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

def Tpu_LutOp: Tpu_Op<"Lut", [
    InOutSameShape, SupportElementwise,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>]>{
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
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_LutBF16Op: Tpu_Op<"LutBF16", [
    InOutSameShape, SupportElementwise,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]>{
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
    SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "AllowDataSplit"]>,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$output_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$hdim_is_batch,
    DefaultValuedAttr<BoolAttr, "true">:$keep_dims,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<I64ArrayAttr, "{1}">:$multipliers,
    DefaultValuedAttr<I64ArrayAttr, "{0}">:$rshifts,
    DefaultValuedAttr<I64Attr, "0">:$right_zp,
    DefaultValuedAttr<I64Attr, "0">:$input_zp,
    DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
    DefaultValuedAttr<I64Attr, "1">:$left_reuse,
    OptionalAttr<F64ArrayAttr>:$out_f8_scales, // for fp8 quant
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    OptionalAttr<BoolAttr>:$multi_core,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_rq,
    AnyTensorOrNone:$multi,
    AnyTensorOrNone:$buffer,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode
  );

  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizeMethod = 1;
  let extraClassDeclaration = [{
    matmul_attr_t parseParam();
    void assign_fw_param(void *param);
  }];
}

def Tpu_A16MatMulOp: Tpu_Op<"A16MatMul",
    [DeclareOpInterfaceMethods<IndexingMapsInterface>]> {
  let summary = "w8a16 / w4a16 matmul operator";

  let description = [{
    The special matrix multiplication designed for LLM Linear Layer.
    Weight is saved in int8 with f16 per-channel quant scale.

    y_f16 = x_f16 x (quantized_w.to(f16) * scale_f16)
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    AnyTensorOrNone:$scale, // scales for weight quant
    AnyTensorOrNone:$zp, // zero points for weight quant
    AnyTensorOrNone:$bias, // note: bias has to be the last operand
    I64Attr:$weight_bits,
    DefaultValuedAttr<BoolAttr, "true">:$sign,
    DefaultValuedAttr<BoolAttr, "false">:$w_transpose,
    // do per-group quant if q_group_size > 0, otherwise per-channel
    DefaultValuedAttr<I64Attr, "0">:$q_group_size
  );

  let hasCanonicalizeMethod = 1;
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_AttentionOp: Tpu_Op<"Attention", [
    SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "AllowDataSplit"]>,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Attention operator";

  let description = [{
    Performs a multi head attention block. https://en.wikipedia.org/wiki/Attention_(machine_learning)
    This block has Q_w, K_w,V_w, O_w and mask
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$keys,
    AnyTensorOrNone:$values,
    AnyTensor:$queries_weight,
    AnyTensorOrNone:$queries_bias,
    AnyTensorOrNone:$keys_weight,
    AnyTensorOrNone:$keys_bias,
    AnyTensorOrNone:$values_weight,
    AnyTensorOrNone:$values_bias,
    AnyTensor:$out_weight,
    AnyTensorOrNone:$out_bias,
    AnyTensorOrNone:$musk,
    AnyTensorOrNone:$table,
    DefaultValuedAttr<I64ArrayAttr, "{0}">:$quant_param,
    F64Attr:$scale,
    I64Attr:$head,
    I64Attr:$dim,
    DefaultValuedAttr<I64Attr, "0">:$has_bias
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_FAttentionOp: Tpu_Op<"FAttention", [
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Flash Attention operator";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.
  }];

  let arguments = (ins
    AnyTensor:$queries,
    AnyTensor:$keys,
    AnyTensor:$values,
    AnyTensorOrNone:$mask,
    AnyTensorOrNone:$buffer,
    F64Attr:$scale,
    I64Attr:$batch,
    I64Attr:$q_head,
    I64Attr:$kv_head,
    I64Attr:$dim,
    I64Attr:$mq,
    I64Attr:$mk
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_RandnLikeOp: Tpu_Op<"RandnLike"> {
  let summary = "RandnLike operator";
  let description = [{
    create a tensor with the same shape as input, and fill with value from normal distribution
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$randn_data
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_RangeOp: Tpu_Op<"Range", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Range operator";
  let description = [{
    range op
  }];

  let arguments = (ins
    AnyTensor:$start,
    AnyTensor:$limit,
    AnyTensor:$delta
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_ReluOp: Tpu_Op<"Relu", [
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]>{
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

def Tpu_ReshapeOp:Tpu_Op<"Reshape", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "AllowDataSplit"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Reshape operation";
  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$shape,
    Optional<AnyTensorOrNone>:$buffer
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ShapeReshapeOp:Tpu_Op<"ShapeReshape",[
  ShapeProducer, ShapeConsumer]> {
  let summary = "ShapeReshape operation";
  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$shape
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ShapeAssignOp: Tpu_Op<"ShapeAssign", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "ShapeAssign operator";
  let description = [{
    reshape for dynamic shape
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$shape
  );

  let results = (outs AnyTensor:$output);
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

def Tpu_ShapeReverseOp:Tpu_Op<"ShapeReverse", [
  ShapeProducer,ShapeConsumer]> {
  let summary = "ShapeReverse operation";
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
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Cast operation";
  let description = [{
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    OptionalAttr<BoolAttr>:$extra_input,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    DefaultValuedAttr<BoolAttr, "true">:$with_scale, //for BM1684
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode
  );
  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_DtypeCastOp:Tpu_Op<"DtypeCast", [
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Cast F32 to F16";
  let description = [{
    Cast F32 to F16
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    OptionalAttr<BoolAttr>:$extra_input,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    DefaultValuedAttr<BoolAttr, "true">:$with_scale, //for BM1684
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ShapeCastOp:Tpu_Op<"ShapeCast", [
    ShapeProducer, ShapeConsumer,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Shape Cast operation";
  let description = [{
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
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
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    DefaultValuedAttr<BoolAttr, "true">:$support_compress,
    OptionalAttr<Tpu_CompressAttr>:$compress_info
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LoadToL2MOp:Tpu_Op<"LoadToL2M",
  [DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>, InOutSameShape]> {
  let summary = "Load operation";
  let description = [{
    load weight from gmem to l2mem;
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$id,
    I64Attr:$l2m_addr
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void codegen_only_for_LoadToL2MOp();
  }];
}

def Tpu_StoreOp:Tpu_Op<"Store",
  [DeclareOpInterfaceMethods<LocalGenInterface, ["assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>, InOutSameShape]> {
  let summary = "Store operation";
  let description = [{
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$input_gmem,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    DefaultValuedAttr<BoolAttr, "true">:$support_compress,
    OptionalAttr<Tpu_CompressAttr>:$compress_info
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RequantIntOp:Tpu_Op<"RequantInt", [
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "requant operation";
  let description = [{
    Requant 32/16/8 bit data to int8 or uint8 data, by int multiplier and int shift
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    SI32Attr:$multiplier,
    SI32Attr:$rshift,
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
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    DefaultValuedAttr<SI32Attr, "1">:$rq_axis,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_rq_axis
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RequantFpOp:Tpu_Op<"RequantFp", [
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "requant float operation";
  let description = [{
    Requant 32/16/8 bit data to int8 or uint8 or fp8 data, by float scale and float offset, when to fp8 data, offset is not used
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$scale,
    DefaultValuedAttr<F64Attr, "0.0">:$offset,
    Tpu_RequantModeAttr:$quant_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$first_round_mode,
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
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$first_round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_DequantIntOp:Tpu_Op<"DequantInt", [
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]> {
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
    I64Attr:$dsecs,
    I64Attr:$wsecs,
    I64Attr:$csecs,
    I64Attr:$swpipl_stage_num,
    I64Attr:$group_type,
    // store timestep_idx(negative) and op_id(positive)
    DefaultValuedAttr<I64ArrayAttr, "{0}">:$flow,
    // only store op_id
    DefaultValuedAttr<I64ArrayAttr, "{}">:$self_up_overlap_op,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$self_down_overlap_op,
    // store timestep_idx(negative) and op_id(positive)
    DefaultValuedAttr<I64ArrayAttr, "{}">:$other_up_overlap_op,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$other_down_overlap_op,
    //nnvlc
    DefaultValuedAttr<BoolAttr, "true">:$support_compress,
    //core ids that run the group
    DefaultValuedAttr<I64ArrayAttr, "{}">:$run_core_id,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$core_slice_ncdhw
  );
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let regions = (region SizedRegion<1>:$body);
}

def Tpu_GroupParallelOp : Tpu_BaseOp<"GroupParallel"> {
  let summary = "Mutiple regions run in parallel.";
  let description = [{
    This operation is composed of numerous regions, with each region corresponding
    to a subgraph. These subgraphs share identical computational patterns and are
    distributed across various cores of a multi-core TPU for processing.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);

  let regions = (region VariadicRegion<SizedRegion<1>>:$parallel);
}

def Tpu_CoreBeginOp:Tpu_BaseOp<"CoreBegin"> {
  let summary = "Begin op parallel to multi cores";
  let description = [{
    Begin of pattern to multi cores
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Tpu_CorePatternAttr:$pattern
  );
  let results = (outs
   Variadic<AnyRankedTensor>:$outputs);
}

def Tpu_CoreEndOp:Tpu_BaseOp<"CoreEnd"> {
  let summary = "End op parallel to multi cores";
  let description = [{
    End of pattern to multi cores
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Tpu_CorePatternAttr:$pattern
  );
  let results = (outs
   Variadic<AnyRankedTensor>:$outputs);
}

def Tpu_CoreParallelOp:Tpu_BaseOp<"CoreParallel"> {
  let summary = "Parallel execution region in multi cores";
  let description = [{
    The ops in one parallel should run in parallel.
  }];
  let arguments = (ins
    Variadic<AnyTensorOrNone>:$inputs,
    I64Attr:$offset,
    I64Attr:$size
  );
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let regions = (region SizedRegion<1>:$body);
}

def Tpu_SplitOp:Tpu_BaseOp<"Split"> {
  let summary = "split tensor to continues pieces";
  let description = [{
    The ops in one parallel should run in parallel.
  }];
  let arguments = (ins
    AnyRankedTensor:$input
  );
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
}

def Tpu_JoinOp:Tpu_BaseOp<"Join"> {
  let summary = "Join tensor to continues pieces";
  let description = [{
    The ops in one parallel should run in parallel.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_DevBeginOp:Tpu_BaseOp<"DevBegin"> {
  let summary = "Begin distribution tensors to multi device";
  let description = [{
    Tensors split to distributed device
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Tpu_DevPatternAttr:$pattern,
    // TypedArrayAttrBase<Tpu_DevBeginMethodAttr, "Array of BeginMethod">:$begin_methods,
    I64ArrayAttr:$begin_methods,
    DefaultValuedAttr<I64Attr, "0">:$num_head,
    DefaultValuedAttr<BoolAttr, "false">:$done

  );
  let results = (outs
   Variadic<AnyRankedTensor>:$outputs);
}

def Tpu_DevEndOp:Tpu_BaseOp<"DevEnd"> {
  let summary = "End distribution tensors from outputs";
  let description = [{
    Tensors from distributed device connect together
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Tpu_DevPatternAttr:$pattern,
    // TypedArrayAttrBase<Tpu_DevEndMethodAttr, "Array of EndMethod">:$end_methods
    I64ArrayAttr:$end_methods
  );
  let results = (outs
   Variadic<AnyRankedTensor>:$outputs);
}

def Tpu_YieldOp : Tpu_BaseOp<"Yield",
  [Terminator, HasParent<"GroupOp, CoreParallelOp, IfOp, LoopOp, GroupParallelOp">]> {
  let summary = "Yield values to parent operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyTensor>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

}

def Tpu_SoftmaxOp: Tpu_Op<"Softmax",[
    DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>,
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
    DefaultValuedAttr<F64Attr, "1.0">:$beta,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LeakyReluOp: Tpu_Op<"LeakyRelu", [
   InOutSameShape, SupportElementwise,
   DeclareOpInterfaceMethods<IndexingMapsInterface>,
   DeclareOpInterfaceMethods<LocalGenInterface>,
   DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
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
    OptionalAttr<SI32Attr>:$rshift,
    OptionalAttr<SI32Attr>:$rshift_neg,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_ConcatOp:Tpu_Op<"Concat", [
    DeclareOpInterfaceMethods<InplaceInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "AllowDataSplit"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Concatate operation";
  let description = [{
  Concatenates the given sequence of seq tensors in the given dimension.
  All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    SI32Attr:$axis,
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
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_ShapePackOp:Tpu_Op<"ShapePack", [
    ShapeProducer, ShapeConsumer]> {
  let summary = "Shape Concatate operation";
  let description = [{
  Concatenates the given sequence of seq tensors in the given dimension.
  All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    SI32Attr:$axis
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ShapePowOp : Tpu_Op<"ShapePow" , [
    ShapeProducer, ShapeConsumer]> {
  let summary = "ShapePow operation running on CPU";
  let description = [{
    output = input ^ n
  }];
  let arguments = (ins
    AnyTensor:$input,
    F32Attr: $exponent
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_MulShiftOp: Tpu_Op<"MulShift", [
    InOutSameShape, SupportElementwise,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "MulShift operator";

  let description = [{
      Y = int8(X-zx) * multiplier >> rshift + zy)
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    SI32Attr:$multiplier,
    SI32Attr:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_PermuteOp: Tpu_Op<"Permute", [
  DeclareOpInterfaceMethods<IndexingMapsInterface>]> {

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

def Tpu_ShapeTransposeOp: Tpu_Op<"ShapeTransposeOp",[
  ShapeProducer, ShapeConsumer]> {

  let summary = "ShapeTransposeOp operator";

  let description = [{
      Perform permute on input.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64ArrayAttr:$order
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH", "BackwardW", "LocalGenSupport"]>,
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "BackwardW", "assign_sec_info"]>,
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH", "BackwardW", "LocalGenSupport"]>]> {
  let summary = "Pad operation";
  let description = [{
    This operation pads a tensor according to the paddings you specify.
    paddings is an integer tensor with shape [2, n], where n is the rank of tensor.
    For each dimension D of input, paddings[0, D] indicates how many values to add
    before the contents of tensor in that dimension, and paddings[1, D] indicates
    how many values to add after the contents of tensor in that dimension.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    // for cv18xx reflect mode
    AnyTensorOrNone:$left_select,
    AnyTensorOrNone:$right_select,
    AnyTensorOrNone:$buffer,
    I64ArrayAttr:$paddings,
    DefaultValuedAttr<F64Attr, "0.0">:$val,
    DefaultValuedAttr<Tpu_PaddingModeAttr, "tpu::PaddingMode::constant">:$mode
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_DivOp: Tpu_Op<"Div", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_SliceOp: Tpu_Op<"Slice", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardN", "BackwardH", "BackwardW", "LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>
  ]> {
  let summary = "Slice operator";
  let description = [{
    Slice Operation on input.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$offsetT,
    AnyTensorOrNone:$endsT,
    AnyTensorOrNone:$stepsT,
    AnyTensorOrNone:$buffer, // only used in BM1684
    I64ArrayAttr:$offset,
    I64ArrayAttr:$steps,
    I64ArrayAttr:$ends,
    DefaultValuedAttr<I64ArrayAttr, "{1}">:$axes
  );
  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    slice_attr_t parseParam();
    void assign_fw_param(void *param);
  }];
}

def Tpu_ShapeSliceOp: Tpu_Op<"ShapeSlice", [
  ShapeProducer, ShapeConsumer,
  DeclareOpInterfaceMethods<TypeInterface>
  ]> {
  let summary = "ShapeSlice operator";
  let description = [{
    Slice Operation on shape-type tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$offsetT,
    AnyTensorOrNone:$endsT,
    AnyTensorOrNone:$stepsT,
    I64ArrayAttr:$offset,
    I64ArrayAttr:$steps,
    I64ArrayAttr:$ends
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    slice_attr_t parseParam();
    void assign_fw_param(void *param);
  }];
}

def Tpu_ShapeUnsqueezeOp: Tpu_Op<"ShapeUnsqueeze", [
    ShapeProducer, ShapeConsumer]> {
  let summary = "Onnx-Style ShapeUnsqueeze operator";

  let description = [{
    ShapeUnsqueeze Operation on input.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$axes
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ShapeSqueezeOp: Tpu_Op<"ShapeSqueeze", [
    ShapeProducer, ShapeConsumer,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Onnx-Style ShapeSqueeze operator";

  let description = [{
    ShapeSqueeze Operation on input.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$axes
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_UnsqueezeOp: Tpu_Op<"Unsqueeze"> {
  let summary = "Onnx-Style Unsqueeze operator";

  let description = [{
    Unsqueeze Operation on input.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$axes
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_SqueezeOp: Tpu_Op<"Squeeze", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>]> {
  let summary = "Onnx-Style Squeeze operator";

  let description = [{
    Squeeze Operation on input.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$axes
  );
  let results = (outs AnyRankedTensor:$output);
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

def Tpu_TopKOp:Tpu_Op<"TopK"> {
  let summary = "TopK operation";
  let description = [{
    Integrates some operations related to topk.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    Optional<AnyTensor>:$kT,
    I64Attr:$axis,
    I64Attr:$K,
    DefaultValuedAttr<BoolAttr, "true">:$largest,
    DefaultValuedAttr<BoolAttr, "true">:$sorted,
    DefaultValuedAttr<BoolAttr, "false">:$values_used_only,
    AnyTensorOrNone:$buffer_val,
    AnyTensorOrNone:$buffer_idx
  );
  let results = (outs
    AnyTensorOrNone:$values,
    AnyTensorOrNone:$indices
  );
}


def Tpu_NonZeroOp:Tpu_Op<"NonZero", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH", "BackwardW", "LocalGenSupport", "AllowDataSplit"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface, ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>]> {
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
    OptionalAttr<I64ArrayAttr>:$output_padding,
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
    void assign_fw_param(void *param);
  }];
}

def Tpu_Deconv3DOp: Tpu_Op<"Deconv3D",[
    SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>]> {
  let summary = "3D deconvolution operator";

  let description = [{
    "Perform 3D deconvolution operation."
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
    OptionalAttr<I64ArrayAttr>:$output_padding,
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
    void assign_fw_param(void *param);
  }];
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

def Tpu_LRNOp: Tpu_Op<"LRN", [
  InOutSameShape,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>]> {
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
    AnyTensorOrNone:$filter,
    AnyTensorOrNone:$recurrence,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$initial_h,
    AnyTensorOrNone:$initial_c,
    AnyTensorOrNone:$cont,
    AnyTensorOrNone:$buffer,
    AnyTensorOrNone:$sigmoid_table,
    AnyTensorOrNone:$sigmoid_slope_table,
    AnyTensorOrNone:$tanh_table,
    AnyTensorOrNone:$tanh_slope_table,
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

def Tpu_MatchTemplateOp: Tpu_Op<"MatchTemplate", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Opencv MatchTemplate operator";

  let description = [{
    Perform opencv MatchTemplate operation.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$match,
    AnyTensorOrNone:$table,
    AnyTensorOrNone:$mantissa_table,
    MatchTemplateModeAttr:$mode
  );

  let results = (outs AnyRankedTensor:$output);
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
    Optional<AnyTensor>:$tileT,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$tile,
    AnyTensorOrNone:$buffer
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
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
    AnyTensorOrNone:$buffer, //for BM1684
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<BoolAttr, "true">:$if_neg_index
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_GatherElementsOp: Tpu_Op<"GatherElements", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "GatherElements operator";
  let description = [{
    Perform GatherElements operation on the given axis.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,
    AnyTensorOrNone:$indices_coeff,
    AnyTensorOrNone:$buffer,
    DefaultValuedAttr<I64Attr, "2">:$axis
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_GatherNDOp: Tpu_Op<"GatherND", [
  DeclareOpInterfaceMethods<TypeInterface>]>  {
  let summary = "GatherND operator";
  let description = [{
    This operator is the inverse of ScatterND.
  }];

  let arguments = (ins
    AnyRankedTensor:$input_data,
    AnyRankedTensor:$indices,
    OptionalAttr<I64Attr>:$indice_dims,
    DefaultValuedAttr<I64Attr, "0">:$batch_dims
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_PReluOp : Tpu_Op<"PReluOp", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    Variadic<AnyTensorOrNone>:$inputs,
    StrAttr:$cpu_op_name,
    OptionalAttr<DictionaryAttr>:$param
  );

  let results = (outs Variadic<AnyTensorOrNone>:$outputs);
}

def Tpu_InterpOp: Tpu_Op<"Interp",[DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Interp operation";
  let description = [{
    Perform Interp on input.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$shapeT,
    AnyTensorOrNone:$buffer,
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
    BoolAttr:$keepdims,
    ReduceModeAttr:$mode,
    // for cv18xx
    OptionalAttr<I64ArrayAttr>:$multiplier,
    OptionalAttr<I64ArrayAttr>:$rshift
  );

  let hasCanonicalizeMethod = 1;
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    reduce_attr_t parseParam();
  }];
}

def Tpu_ShapeReduceOp: Tpu_Op<"ShapeReduce", [
    ShapeProducer, ShapeConsumer,DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "ShapeReduce operator";

  let description = [{
    ShapeReduce Operation on input.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$axes,
    BoolAttr:$keepdims,
    ReduceModeAttr:$mode,
    DefaultValuedAttr<F64Attr, "1.0">:$scale
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    reduce_attr_t parseParam();
    void assign_fw_param(void *param);
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
    BoolAttr:$keepdims,
    ArgModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "true">:$select_last_index
  );

  let results = (outs
    AnyRankedTensor:$indices,
    AnyTensorOrNone:$values
  );
}

def Tpu_WhereOp: Tpu_Op<"Where", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
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
    DefaultValuedAttr<F64Attr, "0.0">:$y_const_val,
    AnyTensorOrNone:$buffer
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
  let hasCanonicalizeMethod = 1;
}

def Tpu_CompareOp: Tpu_Op<"Compare", [
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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
    AnyRankedTensor:$output
  );
}

def Tpu_IndexPutOp : Tpu_Op<"IndexPut", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "IndexPut operation";
  let description = [{
    aten::index_put_
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$indices,
    AnyRankedTensor:$values,
    AnyTensorOrNone:$buffer, //for accumulate is True
    DefaultValuedAttr<BoolAttr, "false">:$accumulate
  );
  let results = (outs
    AnyRankedTensor:$output
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
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
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

def GridSamplerPadModeAttr: AnyStrAttrOf<["zeros","border","reflection"]>;
def Tpu_GridSamplerOp : Tpu_Op<"GridSampler"> {
  let summary = "GridSampler operation";
  let description = [{
     Given an input and a flow-field grid, computes the output
     using input values and pixel locations from grid.

      Attributes:
        `mode`              : required, interpolation mode to calculate output values, Int attribute [0, 1]
                                representing 'bilinear' | 'nearest' respectively.
        `padding_mode`      : required, padding mode for outside grid values, Int attribute [0, 1, 2],
                                representing 'zero' | 'boundary' | 'reflection' respectively.
        `align_corners`     : required, Geometrically, we consider the pixels of the input as
                                squares rather than points. If set to True, the extrema (-1 and 1) are
                                considered as referring to the center points of the input's corner pixels.
                                If set to False, they are instead considered as referring to the corner
                                points of the input's corner pixels, making the sampling more resolution agnostic.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$grid,
    I64Attr:$mode,
    I64Attr:$padding_mode,
    BoolAttr:$align_corners,
    DefaultValuedAttr<F64Attr, "0">:$mean,
    DefaultValuedAttr<F64Attr, "1">:$scale,
    DefaultValuedAttr<BoolAttr, "false">:$need_permute,
    AnyTensorOrNone:$buffer
  );

  let results = (outs AnyTensor:$output);
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
  DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>,
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

def Tpu_ScatterElementsOp: Tpu_Op<"ScatterElements", [
  DeclareOpInterfaceMethods<TypeInterface>]>  {
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
    AnyTensorOrNone:$indices_coeff,
    AnyTensorOrNone:$buffer,
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
      `reduction`   : Type of reduction to apply: none (0 default), add(1), sub(2), max(3), min(4), mul(5).

    Outputs:
      `output`       : Tensor of rank r >= 1.
  }];

  let arguments = (ins
    AnyRankedTensor:$input_data,
    AnyRankedTensor:$indices,
    AnyRankedTensor:$updates,
    AnyTensorOrNone:$buffer,
    DefaultValuedAttr<I32Attr, "0">:$reduction
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
    F64ArrayAttr:$anchors,
    YoloVersionAttr:$version,
    DefaultValuedAttr<I64Attr, "80">:$class_num,
    DefaultValuedAttr<I64Attr, "3">:$num_boxes,
    DefaultValuedAttr<BoolAttr, "false">:$agnostic_nms,
    AnyTensorOrNone:$buffer
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
    DefaultValuedAttr<I64Attr, "1">:$onnx_nms,
    AnyTensorOrNone:$buffer

  );
  let results = (outs AnyTensor:$output);
}

def Tpu_ShapeOp : Tpu_Op<"Shape", [
    ShapeProducer,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Shape operation";
  let description = [{
    Takes a tensor as input and outputs an 1D int tensor containing the shape
     of the input tensor.
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Tpu_ConstantFillOp: Tpu_Op<"ConstantFill", [
    ShapeConsumer,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "constant fill operator";
  let description = [{
    fill the constant value
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$value
  );

  let results = (outs
    AnyTensor:$output
  );
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_Host2DeviceOp : Tpu_Op<"Host2Device", [
    ShapeConsumer,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Host2Device Operation";
  let description = [{
    takes data from host mem to device mem
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Tpu_Device2HostOp : Tpu_Op<"Device2Host", [
    ShapeProducer,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Device2Host Operation";
  let description = [{
    takes data from device mem to host mem
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Tpu_IfOp : Tpu_Op<"If", [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
              RecursiveMemoryEffects,
               NoRegionArguments]> {
  let summary = "if operation";
  let description = [{
    If conditional
  }];
  let arguments = (ins AnyTensor:$cond);
  let results = (outs Variadic<AnyRankedTensor>:$output);
  let regions = (region SizedRegion<1>:$then_branch,
    SizedRegion<1>:$else_branch);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 1;
    }
    static int getNumberOfResults() {
      return -1;
    }
    static std::vector<int> getTypeMap() {
      return {-1};
    }
  int64_t getSubgraphRegionIdx(const std::string& name) {
    if (name == "then_branch") return 0;
    if (name == "else_branch") return 1;
    llvm_unreachable("region with the specified name does not exist");
  }
  }];
}

def Tpu_DeformGatherOp : Tpu_Op<"DeformGather"> {
  let summary = "Deform gather operator";

  let description = [{
     The deform gather operator for deform_conv2d.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$offset,
    AnyTensorOrNone:$mask,
    AnyTensorOrNone:$buffer,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$dilations,
    I64ArrayAttr:$pads, // top,left,bottom,right
    DefaultValuedAttr<BoolAttr, "false">:$use_mask,
    DefaultValuedAttr<I64Attr, "1">:$deform_group
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    deform_gather_attr_t parseParam();
  }];
}

def Tpu_CustomOp: Tpu_Op<"Custom", [
  DeclareOpInterfaceMethods<TypeInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "BackwardW", "AllowDataSplit"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Custom operator";
  let description = [{
    Custom operator
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    AnyTensorOrNone:$buffer,
    StrAttr:$name,
    DictArrayAttr:$params,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
    );

  let results = (outs Variadic<AnyTensor>:$outputs);
}

def Tpu_Weight2ActivationOp: Tpu_Op<"Weight2Activation"> {
  let summary = "Weight to activation operator";

  let description = [{
    Convert weight tensor to activation tensor
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Tpu_Space2BatchOp: Tpu_Op<"Space2Batch"> {

  let summary = "Space2Batch operator";

  let description = [{
    Refer to `https://www.tensorflow.org/api_docs/python/tf/space_to_batch`
    h_padding = h + pad_top + pad_bottom,
    w_padding = w + pad_left + pad_right,
    [n, c, h, w] => [n, c, h_padding, w_padding]
      =>[n * block_h * block_w, c, h_padding / block_h, w / block_w];
    h_padding and w_padding should satisfy:
    h_padding % block_h = 0, w_padding % block_w = 0
    The format of input or output is NCHW.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64Attr:$block_h,
    I64Attr:$block_w,
    I64ArrayAttr:$pads, // top, bottom, left, right
    AnyTensorOrNone:$buffer
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_Batch2SpaceOp: Tpu_Op<"Batch2Space"> {

  let summary = "Batch2Space operator";

  let description = [{
    Refer to `https://www.tensorflow.org/api_docs/python/tf/batch_to_space`
    h_croping = h * block_h - crop_top - crop_bottom,
    w_croping = w * block_w - crop_left - crop_right,
    [n, c, h, w] => [n / (block_h * block_w), c, h * block_h, w * block_w]
      => [n / (block_h * block_w), c, h_croping, w_croping];
    The format of input or output is NCHW.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64Attr:$block_h,
    I64Attr:$block_w,
    I64ArrayAttr:$crops, // top, bottom, left, right
    AnyTensorOrNone:$buffer
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_IdentityOp : Tpu_Op<"Identity"> {
  let summary = "identity operator";

  let description = [{
     identity operator.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$input
  );

  let results = (outs Variadic<AnyRankedTensor>:$output);
}

def Tpu_NmsOp : Tpu_Op<"Nms"> {
  let summary = " NMS operator";
  let description = [{
      tpu nms
  }];
  let arguments = (ins
    Variadic<AnyTensor>: $inputs,
    I64Attr: $center_point_box,
    I64Attr: $max_output_size,
    AnyTensorOrNone:$buffer
  );

  let results = (outs AnyTensor:$output);
}


def Tpu_RMSNormOp : Tpu_Op<"RMSNorm", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "RMSNorm operation";
  let description = [{
    A simplification of the original layer normalization (LayerNorm).
    Only normalize the last dimension of tensor.

    RMSNorm(x) = gamma * x / sqrt(mean(x^2) + epsilon)
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$gamma,
    F64Attr:$eps
  );

  let results = (outs
    AnyTensor:$output
  );
}

def Tpu_LoopOp : Tpu_Op<"Loop", [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
              SingleBlockImplicitTerminator<"tpu::YieldOp">,
              RecursiveMemoryEffects]> {
  let summary = "Loop operation";
  let description = [{
    Generic Looping construct, support while/do_while/for/forerver etc:
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, NoneType]>:$M,
                   AnyTypeOf<[AnyTensor, NoneType]>:$cond,
                  Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$v_final_and_scan_outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return -1;
    }
    static int getNumberOfResults() {
      return -1;
    }
    static std::vector<int> getTypeMap() {
      return {22};
    }

    mlir::Operation::result_range v_final();
    mlir::Operation::result_range scan_outputs();
    int64_t getSubgraphRegionIdx(const std::string& name) {
      if (name == "body") return 0;
      llvm_unreachable("region with the specified name does not exist");
    }
  }];
}


def Tpu_AutoIncreaseOp: Tpu_Op<"AutoIncrease",
    [SameOperandsAndResultType]> {
  let summary = "Auto increase ";
  let description = [{
    increase by 1 in-place
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$const_val
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_D2DOp: Tpu_Op<"D2D"> {
  let summary = "Copy WeightOp to Device Mem ";
  let description = [{
    for to alloc address, and can modify the data(with WeightOp to init it)
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$const_val
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ShapeArithOp: Tpu_Op<"ShapeArith", [
    ShapeProducer, ShapeConsumer]> {
  let summary = "Cpu data operation";

  let description = [{
    Arithmetic implementation for simple data in host memory such as 'Shape, Index, Stride' etc.
  }];

  let arguments = (ins
    Variadic<AnyTensor>: $inputs,
    StrAttr:$type
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_CumSumOp: Tpu_Op<"CumSum"> {
  let summary = "CumSum operator";

  let description = [{
    Returns the cumulative sum of elements of input in the dimension dim.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$dim,
    I64Attr:$axis
  );
  let results = (outs
    AnyTensor:$output);
}

def Tpu_LayerNormTrainOp : Tpu_Op<"LayerNormTrain"> {
  let summary = "LayerNorm operation for train";
  let description = [{
    layer normalization
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$normalized_shape,
    SI32Attr:$axis,
    F64Attr:$eps
  );
  let results = (outs
    AnyRankedTensor:$output,
    AnyRankedTensor:$mean,
    AnyRankedTensor:$variance
  );
}

def Tpu_LayerNormBwdOp : Tpu_Op<"LayerNormBwd"> {
  let summary = "LayerNorm operation for train";
  let description = [{
    layer normalization
  }];
  let arguments = (ins
    AnyTensor:$grad_out,
    AnyTensor:$input,
    AnyTensor:$mean,
    AnyTensor:$variance,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$normalized_shape
  );
  let results = (outs
    AnyTensorOrNone:$grad_input,
    AnyTensorOrNone:$grad_weight,
    AnyTensorOrNone:$grad_bias
  );
}


def Tpu_BatchNormTrainOp: Tpu_Op<"BatchNormTrain"> {
  let summary = "BatchNormalization operation";
  let description = [{
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    ```math
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    ```

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$mean,
    AnyTensor:$variance,
    AnyTensorOrNone:$gamma,
    AnyTensorOrNone:$beta,
    DefaultValuedAttr<F64Attr, "1e-05">:$epsilon,
    DefaultValuedAttr<F64Attr, "0.1">:$momentum
  );
  let results = (outs
    AnyTensor:$output,
    AnyTensor:$mean_out,
    AnyTensor:$variance_out
  );
}


def Tpu_BatchNormBwdOp: Tpu_Op<"BatchNormBwd"> {
  let summary = "BatchNormalization operation";
  let description = [{
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    ```math
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    ```

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).
  }];
  let arguments = (ins
    AnyTensor:$grad_out,
    AnyTensor:$input,
    AnyTensorOrNone:$weight_opt,
    AnyTensorOrNone:$saved_mean,
    AnyTensorOrNone:$saved_invstd,
    DefaultValuedAttr<F64Attr, "1e-05">:$epsilon
  );
  let results = (outs
    AnyTensor:$grad_in,
    AnyTensorOrNone:$weight_grad,
    AnyTensorOrNone:$bias_grad
  );
}

def Tpu_EmbDenseBwdOp : Tpu_Op<"EmbDenseBwd"> {
  let summary = "EmbDenseBwd operation for train";
  let description = [{
    layer normalization
  }];
  let arguments = (ins
    AnyTensor:$grad_output,
    AnyTensor:$indices,
    SI32Attr:$num_weights,
    SI32Attr:$padding_idx,
    BoolAttr:$scale_grad_by_freq
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Tpu_WeightReorderOp: Tpu_Op<"WeightReorder"> {
  let summary = "WeightReorder operator";

  let description = [{
      reorder Weight.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I64Attr, "0">:$reorder_mode
  );
  let results = (outs AnyTensor:$output);
}


def Tpu_SoftmaxBwdOp: Tpu_Op<"SoftmaxBwd"> {
  let summary = "softmax backward operator";

  let description = [{
    Integrates some operations related to softmax backward.
  }];

  let arguments = (ins
    AnyTensor:$grad_output,
    AnyTensor:$output,
    SI32Attr:$dim
  );

  let results = (outs AnyRankedTensor:$grad_input);
}



def Tpu_ConvBwdWeightOp:Tpu_Op<"ConvBwd_Weight", [SupportFuseRelu,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface,
      ["BackwardH", "BackwardW", "LocalGenSupport", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface,
      ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>]> {
  let summary = "Convolution Backward operator";
  let description = [{Gradient of Weight in Convolution Backward}];
  let arguments = (ins
  AnyTensor:$input,
  AnyTensor:$gradout,
  AnyTensor:$gradout_transpose,
  I64Attr:$groups,
  I64ArrayAttr:$input_shape,
  I64ArrayAttr:$grad_out_shape,
  I64ArrayAttr:$kernel_shape,
  I64ArrayAttr:$stride,
  I64ArrayAttr:$dilations,
  I64ArrayAttr:$padding,
  BoolAttr:$grad_bias_enable
  );
  let results =(outs
  AnyTensor:$output);
  let extraClassDeclaration = [{
    convbwd_weight_attr_t parseParam();
  }];
}

def Tpu_TriluOp:Tpu_Op<"Trilu", [
  DeclareOpInterfaceMethods<LocalGenInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>
  ]> {
  let summary = "Trilu operation";
  let description = [{
     Returns the upper or lower triangular part of input.
  }];
  let arguments = (ins
    AnyTensor:$input,
    SI32Attr:$upper,
    SI32Attr:$diagonal
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_SortOp:Tpu_Op<"Sort", [DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Sort operation";
  let description = [{
    Integrates some operations related to Sort.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$buffer,
    I64Attr:$axis,
    DefaultValuedAttr<BoolAttr, "true">:$descending
  );
  let results = (outs
    AnyTensorOrNone:$values,
    AnyTensor:$indices
  );
}
def Tpu_SliceMergeOp: Tpu_BaseOp<"SliceMerge"> {
  let summary = "SliceMerge";

  let description = [{
    When there are multiple slices in the layer-group,
    this operation combines the store op output of each slice to output to the yield op.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MeanRstdOp: Tpu_Op<"MeanRstd">{
  let summary = "Compute Mean, Rstd, Running_mean, Running_var in batchnorm train op ";
  let description = [{
    mean(x),1/sqrt(var+eps),(1-momentum)*running_mean + momentum*mean,(1-momentum)*running_var + momentum*var
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$running_mean,
    AnyTensor:$running_var,
    AnyTensor:$weight,
    AnyTensor:$bias,
    F64Attr:$eps,
    F64Attr:$momentum
  );
  let results = (outs
    AnyTensor:$mean,
    AnyTensor:$rstd,
    AnyTensor:$running_mean_update,
    AnyTensor:$running_var_update,
    AnyTensor:$scale,
    AnyTensor:$bias_new
    );
}

def Tpu_OutBufferOp : Tpu_BaseOp<"OutBuffer"> {
  let summary = "OutBuffer for store op";

  let description = [{
     This operation stores storeOp output data on DDR when some tentor in layer-group
     is not output to returnOp but storeAndLoad is needed.
  }];

  let arguments = (ins
    DefaultValuedAttr<BoolAttr, "false">:$need_dump
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MoveOp:Tpu_BaseOp<"Move"> {
  let summary = "local mem move op for layer group lmem alloc";
  let description = [{
    local mem move op for layer group lmem alloc
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    StrAttr:$name,
    I64ArrayAttr:$move_src_add,
    I64ArrayAttr:$move_dest_add,
    I64ArrayAttr:$move_size,
    I64Attr:$ts_id
  );
  let results = (outs Variadic<AnyRankedTensor>:$outputs);

  let extraClassDeclaration = [{
    void codegen_only_for_moveOp(std::vector<int64_t>& move_src_add,
                                                    std::vector<int64_t>& move_dest_add,
                                                    std::vector<int64_t>& move_size);
  }];
}

def Tpu_GroupNormTrainOp : Tpu_Op<"GroupNormTrain"> {
  let summary = "GroupNorm operation";
  let description = [{
    group normalization
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    I64Attr:$num_groups,
    F64Attr:$eps
  );
  let results = (outs
    AnyTensor:$output,
    AnyTensor:$mean,
    AnyTensor:$rstd
  );
}
def Tpu_LogicalAndOp :Tpu_Op<"LogicalAnd">{
  let summary = "logical and operation";
  let description = [{logical and operation between two variables}];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );
  let results = (outs
    AnyTensor:$output
  );
}
def Tpu_ConvbwdOp :Tpu_Op<"Convbwd", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "convolution backward";
  let description = [{calculate grad_input,grad_weight,grad_bias of convolution operation}];
  let arguments = (ins
    AnyTensor:$grad_out,
    AnyTensor:$input,
    AnyTensor:$kernel,
    AnyTensorOrNone:$buffer,
    I64Attr:$groups,
    I64ArrayAttr:$input_shape,
    I64ArrayAttr:$grad_out_shape,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$dilations,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$inserts,
    BoolAttr:$grad_input_enable,
    BoolAttr:$grad_weight_enable,
    BoolAttr:$grad_bias_enable
  );
  let results = (outs
    AnyTensorOrNone:$grad_input,
    AnyTensorOrNone:$grad_weight,
    AnyTensorOrNone:$grad_bias
  );
  let extraClassDeclaration = [{
    convbwd_attr_t parseParam();
  }];
}

def Tpu_MaskRCNNRPNGetBboxesOp: Tpu_Op<"MaskRCNN_RPNGetBboxes", [
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "RPN_get_bboxes gen by PPL";

  let description = [{
    MaskRCNN_RPN_get_bboxes, the sub-block with 1st NMS between RPN_head and 1st ROIAlign
  }];

  let arguments = (ins
AnyTensor:$cls_scores_0,
    AnyTensor:$cls_scores_1,
    AnyTensor:$cls_scores_2,
    AnyTensor:$cls_scores_3,
    AnyTensor:$cls_scores_4,
    AnyTensor:$bbox_preds_0,
    AnyTensor:$bbox_preds_1,
    AnyTensor:$bbox_preds_2,
    AnyTensor:$bbox_preds_3,
    AnyTensor:$bbox_preds_4,
    AnyTensor:$max_shape,
    AnyTensor:$mlvl_anchors_0,
    AnyTensor:$mlvl_anchors_1,
    AnyTensor:$mlvl_anchors_2,
    AnyTensor:$mlvl_anchors_3,
    AnyTensor:$mlvl_anchors_4,
    AnyTensorOrNone:$batch_mlvl_scores,
    AnyTensorOrNone:$batch_mlvl_anchors,
    AnyTensorOrNone:$batch_mlvl_rpn_bbox_pred,
    AnyTensorOrNone:$batch_mlvl_proposals,
    AnyTensorOrNone:$batch_mlvl_ids,
    AnyTensorOrNone:$glb_buffer_tmp_scores_stretched,
    AnyTensorOrNone:$glb_buffer_ranked_scores,
    AnyTensorOrNone:$glb_buffer_rank_inds_int32,
    AnyTensorOrNone:$glb_buffer_rank_inds_u32,
    AnyTensorOrNone:$glb_topk_inds,
    AnyTensorOrNone:$glb_buffer_gather_1,
    AnyTensorOrNone:$glb_buffer_gather_2,
    AnyTensorOrNone:$glb_buffer_rpn_bbox_permuted,
    AnyTensorOrNone:$glb_buffer_nonzero,
    AnyTensorOrNone:$result_valid_ind,
    AnyTensorOrNone:$glb_buffer_gather_boxes,
    AnyTensorOrNone:$glb_buffer_gather_scores,
    AnyTensorOrNone:$keep_3nch,
    AnyTensorOrNone:$keep_u32_1h,
    AnyTensorOrNone:$glb_buffer_boxes,
    AnyTensorOrNone:$glb_buffer_scores,
    AnyTensorOrNone:$glb_buffer_nms,
    AnyTensorOrNone:$gather_mlvl_proposals,
    AnyTensorOrNone:$gather_mlvl_scores,
    AnyTensorOrNone:$gather_mlvl_ids,
    AnyTensorOrNone:$glb_buffer_result_list,
    F64Attr:$delta2bbox_mean_0,
    F64Attr:$delta2bbox_mean_1,
    F64Attr:$delta2bbox_mean_2,
    F64Attr:$delta2bbox_mean_3,
    F64Attr:$delta2bbox_std_0,
    F64Attr:$delta2bbox_std_1,
    F64Attr:$delta2bbox_std_2,
    F64Attr:$delta2bbox_std_3,
    F64Attr:$delta2bbox_max_scalar_c,
    F64Attr:$iou_threshold,
    F64Attr:$conf_threshold,
    I64Attr:$MAX_LENGTH_STATIC_STRECHED,
    I64Attr:$NUM_INDEXES,
    I64Attr:$NUM_CLASSES,
    I64Attr:$CHANNEL_RPN_BBOXES,
    I64Attr:$CHANNEL_RPN_SCORES,
    I64Attr:$NMS_PRE,
    I64Attr:$HARDWARE_FACTOR_TOPK,
    I64Attr:$NMS_MAX_LENGTH,
    I64Attr:$TOPK_ONNX_NMS,
    I64Attr:$H_RPN_DYN_MAX,
    I64Attr:$W_RPN_DYN_MAX,
    I64Attr:$MAX_PER_IMG
  );

  let results = (outs AnyTensor:$result_list);
}

def Tpu_MaskRCNNBboxPoolerOp: Tpu_Op<"MaskRCNN_BboxPooler", [
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Bbox_Pooler gen by PPL";

  let description = [{
    MaskRCNN_Bbox_Pooler, the 1st ROIAlign in MaskRCNN
  }];

  let arguments = (ins
    AnyTensor:$ptr_feat0,
    AnyTensor:$ptr_feat1,
    AnyTensor:$ptr_feat2,
    AnyTensor:$ptr_feat3,
    AnyTensor:$rois_multi_batch,
    AnyTensorOrNone:$ptr_tmp_res,
    AnyTensorOrNone:$ptr_rois_tmp,
    I64Attr:$ROI_NUM_LEVELS,
    I64Attr:$ROI_H,
    I64Attr:$ROI_W,
    I64Attr:$CHANNEL_ROI,
    I64Attr:$ROI_SLICE,
    I64Attr:$ROI_PH,
    I64Attr:$ROI_PW,
    I64Attr:$ROI_LEN
  );
  let results = (outs
    AnyTensor:$result_res,
    AnyTensor:$result_rois
  );
}

def Tpu_MaskRCNNGetBboxBOp: Tpu_Op<"MaskRCNN_GetBboxB", [
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "GetBboxB operator";

  let description = [{
     MaskRCNN_GetBboxB, the 2nd onnx_nms in MaskRCNN
  }];

  let arguments = (ins
          AnyTensor:$ptr_rois,
          AnyTensor:$ptr_bbox,
          AnyTensor:$ptr_score,
          AnyTensor:$max_val,
          AnyTensor:$scale_factor,
          AnyTensorOrNone:$means,
          AnyTensorOrNone:$stds,
          AnyTensorOrNone:$res_bbox,
          AnyTensorOrNone:$res_bbox1,
          AnyTensorOrNone:$res_bbox0,
          AnyTensorOrNone:$res_score0,
          AnyTensorOrNone:$res_score1,
          AnyTensorOrNone:$res_score2,
          AnyTensorOrNone:$res_score3,
          AnyTensorOrNone:$res_label2,
          AnyTensorOrNone:$result_list,
          AnyTensorOrNone:$keep_3nch,
          AnyTensorOrNone:$keep_u32_1h,
          AnyTensorOrNone:$glb_buffer_boxes,
          AnyTensorOrNone:$glb_buffer_scores,
          AnyTensorOrNone:$glb_buffer_nms,
          AnyTensorOrNone:$glb_buffer_nonzero,
          AnyTensorOrNone:$result_valid_ind,
          AnyTensorOrNone:$glb_lables,
          AnyTensorOrNone:$glb_lables_expand,
          F64Attr:$threshold_score_eq,
          F64Attr:$wh_ratio_log,
          F64Attr:$nms_iou_thr,
          F64Attr:$delta2bbox_means,
          F64Attr:$delta2bbox_stds_0,
          F64Attr:$delta2bbox_stds_1,
          I64Attr:$NUM_INDEXES,
          I64Attr:$NUM_CLASSES,
          I64Attr:$TOPK_ONNX_NMS,
          I64Attr:$NUM_CLASSES_getBboxB,
          I64Attr:$MAX_NMS_LENGTH_GetBboxB,
          I64Attr:$MAX_PER_IMG,
          I64Attr:$MAX_PER_IMG_GetBboxB
  );

  let results = (outs
    AnyTensor:$result_det_bboxes,
    AnyTensor:$result_det_labels
  );
}

def Tpu_MaskRCNNMaskPoolerOp: Tpu_Op<"MaskRCNN_MaskPooler", [
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Mask_Pooler gen by PPL";

  let description = [{
    MaskRCNN_Mask_Pooler, the 2st ROIAlign in MaskRCNN
  }];

  let arguments = (ins
    AnyTensor:$x_0,
    AnyTensor:$x_1,
    AnyTensor:$x_2,
    AnyTensor:$x_3,
    AnyTensor:$det_bboxes_multi_batch,
    AnyTensor:$det_labels_multi_batch,
    AnyTensor:$scale_factor,
    AnyTensorOrNone:$ptr_rois_buff,
    AnyTensorOrNone:$result_filled_det_bboxes,
    AnyTensorOrNone:$result_filled_det_labels,
    AnyTensorOrNone:$ptr_tmp_res,
    AnyTensorOrNone:$ptr_rois_tmp,
    I64Attr:$ROI_NUM_LEVELS,
    I64Attr:$ROI_H,
    I64Attr:$ROI_W,
    I64Attr:$CHANNEL_ROI,
    I64Attr:$ROI_SLICE,
    I64Attr:$ROI_PH,
    I64Attr:$ROI_PW,
    I64Attr:$ROI_LEN
  );
  let results = (outs AnyTensor:$result_res);
}
#endif // TPU_OPS
