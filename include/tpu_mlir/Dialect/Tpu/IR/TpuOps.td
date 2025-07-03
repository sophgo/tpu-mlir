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
      I32EnumAttrCase<"TGELU", 36>, // TGELU(x) = 0.5 * x * (1.0 + tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
      I32EnumAttrCase<"QGELU", 37>, // QGELU(x) = x * sigmoid(1.702 * x)
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
    1.Op Introduction
    A global buffer for operation, and free after op

    2.Math formula
    ```math
        output = \text{buffer}(\text{data})
    ```

    3.activation and weight
    none

    4.attribute
    buffer_type: the type of buffer to be used for the operation.
                 include Global memory buffer(GMEM)y, Local memory buffer(LMEM), Tensor memory buffer(TMEM);
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
    1.Op Introduction
    The Tpu_Conv2DOp operation implements a 2D convolution, which is a fundamental operation in many neural networks,particularly in convolutional neural networks (CNNs).
    This operation takes an input tensor (often representing an image or feature map) and applies a set of learnable filters (kernels) to produce an output tensor.

    2.Math formula
    ```math
        output(N, C_{out}, H, W) = \sum_{C_{in}} input(N, C_{in}, H + sH * kH, W + sW * kW) * filter(C_{in}, C_{out}, kH, kW) + bias(C_{out})
    ```
    where, kH and kW are the height and width of the filter (kernel), sH and sW are the vertical and horizontal strides.
            N is a batch size, C denotes a number of channels, H is a height of input, and W is width.

    3.activation and weight
    input(act.): input tensor;
    filter(w.): the learnable weights of the convolution 2d operation.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attribute
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    stride: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    groups: (optional)Number of blocked connections from input channels to output channels. Default: 1.;
    dilation: controls the spacing between the kernel points;
    inserts: additional parameters that may be used for specific optimizations or configurations.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    //new param
    with_bias: whether to include a bias term in the convolution operation.;
    weight_is_coeff: whether the weights (filters) are coefficients.;
    coeff_merged: whether the coefficients have been merged.;
    use_3ic_optimize: whether to use 3-input channel optimization.;
      This Options call `use_3ic_optimize` is useful for speed up convolution computation:
      use_3ic_optimize & 0x3 == 1  means merge kh to ic
      use_3ic_optimize & 0x3 == 2  means merge kw to ic
      use_3ic_optimize & 0x3 == 3  means merge kh and kw to ic
      use_3ic_optimize & 0x10 != 0 means using tiu to do channel broadcast instead of gdma
      use_3ic_optimize & 0x20 != 0 means input bcast addr use buffer (e.g. current op is used by more than one ops)
      use_3ic_optimize & 0x20 != 0 means using tiu to do channel broadcast instead of gdma and input bcast addr use buffer
                                         (e.g. current op is used by more than one ops)
    kernel_zp: the zero-point for the kernel.It is used in quantized models to adjust the range of the weights.;
    use_winograd: This attribute indicates whether to use the Winograd algorithm for convolution.;
    multiplier: This parameter is used for scaling the output values.;
    rshift: the number of bits to right-shift the quantized values.;
    quant_mode: It determines how the output tensor should be quantized.;
    round_mode: This parameter specifies the rounding mode to be used during quantization.;
    ginfo: This attribute contains information about layer grouping, which can be useful for organizing layers in a neural network.;
    // fuse leakyRelu
    do_leaky_relu: whether to apply the Leaky ReLU activation function after the convolution operation.;
    neg_slope: sets the slope for the negative part of the Leaky ReLU function, determining how much the output can be negative.;
    multiplier_pos: This parameter specifies the multiplier for the positive part of the output, used in the context of quantization.;
    multiplier_neg: This parameter specifies the multiplier for the negative part of the output, used in the context of quantization.;
    rshift_pos: This attribute defines the right shift for the positive output values during quantization.;
    rshift_neg: This attribute defines the right shift for the negative output values during quantization.;
    out_f8_scales: This parameter contains the scaling factors for the output in FP8 (8-bit floating point) format.;
    //nnvlc
    support_compress: whether the operation supports compression.;
    compress_info: contains information about the compression method used.;
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
    OptionalAttr<BoolAttr>:$do_kernel_rotate,
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
  let hasCanonicalizer = 1;
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

  let description = [{
    1.Op Introduction
    The Tpu_Conv2DOp operation implements a 2D convolution, which is a fundamental operation in many neural networks,particularly in convolutional neural networks (CNNs).
    This operation takes an input tensor (often representing an image or feature map) and applies a set of learnable filters (kernels) to produce an output tensor.

    2.Math formula
    ```math
        output(N, C_{out}, D, H, H) = \sum_{C_{in}} input(N, C_{in}, D + sD * kD, H + sH * kH, W + sW * kW) * filter(C_{in}, C_{out}, kD, kH, kW) + bias(C_{out})
    ```
    where, kD, kH and kW are the depth, height and width of the filter (kernel), sH and sW are the vertical and horizontal strides.
            N is a batch size, C denotes a number of channels, H is a height of input, and W is width.

    3.activation and weight
    input(act.): input tensor;
    filter(w.): the learnable weights of the convolution 2d operation.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attribute
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    stride: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    groups: (optional)Number of blocked connections from input channels to output channels. Default: 1.;
    dilation: controls the spacing between the kernel points;
    inserts: additional parameters that may be used for specific optimizations or configurations.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    //new param
    with_bias: whether to include a bias term in the convolution operation.;
    kernel_zp: the zero-point for the kernel.It is used in quantized models to adjust the range of the weights.;
    multiplier: This parameter is used for scaling the output values.;
    rshift: the number of bits to right-shift the quantized values.;
    quant_mode: It determines how the output tensor should be quantized.;
    round_mode: This parameter specifies the rounding mode to be used during quantization.;
    ginfo: This attribute contains information about layer grouping, which can be useful for organizing layers in a neural network.;
    out_f8_scale: This parameter contains the scaling factors for the output in FP8 (8-bit floating point) format.;

  }];

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
    1.Op Introduction
    This performs an  pooling over the given input tensor. A sliding
    window of size given by <kernel size> is passed over the input tensor.

    2.Math formula
    ```math
        output(N, C, H', W') = Pool(max(input(N, C, H, W)))
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    stride: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    pool_mode: ;
    pad_value: whether to retain the dimensions of the input tensor in the output.
                If true, will have the same number of dimensions as the input tensor.;
    is_adaptive: whether the pooling operation is adaptive.
                 If true, adjusts the kernel size based on the input size to produce a specified output size.
    count_include_pad: whether to include the padded values in the pooling count.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    ceil_mode: whether to use ceiling or floor when calculating the output size.;
    /// symmetric quantize param
    multipliers: an array of multipliers used for quantization, It allows for scaling the input values before the addition operation.;
    rshift: right shift values corresponding to each input tensor.;
    /// asymmetric quantize param
    scale: Scalar;
    offset: Scalar;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
    first_round_mode: the rounding behavior applied to the scaled value before the offset is added.;
    layer_group: multiple layers to be processed together or treated as a single unit during computation.;
    /// fp8 quantize param
    fp8_out_scale: scaling factor used for the output tensor when utilizing 8-bit floating-point (FP8) representation.;
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
    OptionalAttr<BoolAttr>:$ceil_mode,
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
    1.Op Introduction
    This performs an  max pooling over the given input tensor. A sliding
    window of size given by <kernel size> is passed over the input tensor.
    get output tensor and mask tensor

    2.Math formula
    ```math
        output(N, C, H', W') = MaxPool(max(input(N, C, H, W)))
        maskOutput(N, C, H', W') = MaxPoolWithMask(argmax(input(N, C, H, W)))
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    strides: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    layer_group: multiple layers to be processed together or treated as a single unit during computation.;
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
    1.Op Introduction
    pooling mask on input

    2.Math formula
    ```math
        output1(N, C, H', W') = Pool(max(input(N, C, H, W)))
        maskOutput(N, C, H', W') = PoolMask(argmax(input(N, C, H, W)))
        output = scale * output1
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    scale: a scaling factor is applied to the output of the pooling operation, can adjust the intensity or magnitude of the output mask.;
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$scale
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_AddOp: Tpu_Op<"Add", [
  SupportFuseRelu, SupportEarlyStride,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "BackwardH", "BackwardW", "assign_sec_info", "DumpQuantAgnosticAttrs"]>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "add operator";

  let description = [{
    1.Op Introduction
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.

    2.Math formula
    ```math
        output = ReLU((input1 + input2; dim))
    ```
    Axis of size 1 will be broadcast if necessary.

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    coeff: It is an array and allows for scaling the output of the addition operation.;
    // early stride param
    do_early_stride: whether to apply early stride optimization during the addition operation.;
    early_stride_h: the height of the early stride.;
    early_stride_w: the width of the early stride.;
    // quant param
    multipliers: an array of multipliers used for quantization, It allows for scaling the input values before the addition operation.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    f8_scales: scaling factors for FP8 (8-bit floating point) quantization.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Elementwise add of input1 and input2. Input2 is constant.

    2.Math formula
    ```math
        output = input + const_val
    ```
    Where input1, input2, ..., inputN are the input tensors.

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    const_val: the constant value to be added to each element of the input tensor(positive, negative, or zero).;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    // quant param
    multiplier: Floating-point multiplication operations are usually converted to fixed-point multiplication operations.;
    rshift: the number of bits to right-shift the quantized values.;
    f8_scale: the scaling factor for FP8 (8-bit floating point) quantization.;
    ginfo: contains layer grouping information.;
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
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    void assign_fw_param(void *param);
  }];
}

def Tpu_BinaryShiftOp: Tpu_Op<"BinaryShift", [
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "Binary with shift operator";

  let description = [{
    1.Op Introduction
    The BinaryShift operator is designed to perform binary operations on two input tensors with an additional shift operation.

    2.Math formula
    ```math
        output = saturation(input1 +/-/* input2 >> -shift)
    ```

    3.activation and weight
    input1(act.): input tensor;
    input2(act.): input tensor;

    4.attribute
    shift: a shift value applied to the quantized data before scaling.;
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    is_reverse: whether the subtraction operation is performed in reverse order.;
    saturation: whether the output should be saturated.
                When set to true, the output will be clamped to a predefined range to prevent overflow or underflow during the operation.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
    ginfo: associated with layer grouping information.;
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["assign_sec_info"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "Binary Const with shift operator";

  let description = [{
    1.Op Introduction
    The BinaryConstShift operator is a specialized tensor operation that combines binary arithmetic with constant scaling and shifting.

    2.Math formula
    ```math
        output = saturation(input +/-/* scale >> -shift)
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    scale: a scaling factor multiplies the input tensor.;
    shift: a shift value applied to the quantized data before scaling.;
    is_reverse: whether the subtraction operation is performed in reverse order.;
    saturation: whether the output should be saturated.
                When set true, the output will be clamped to a predefined range to prevent overflow or underflow during the operation.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
    ginfo: associated with layer grouping information.;
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

  let hasCanonicalizer = 1;
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RopeOp: Tpu_Op<"Rope", [
    SupportElementwise,
    DeclareOpInterfaceMethods<LocalGenInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "Rope Operator";

  let description = [{
    1.Op Introduction
    rope operator.

    2.Math formula
    ```math
        output=saturation((input1 x shift(input2, mul1_shift))⊕(input3 x shift(input2, mul2_shift))) + shift(input3, dd_shift)
    ```
    The operator ⊕ represents the addition of the two multiplicative results.
    The function shift(input,shift_value) applies a shift to the input tensor based on the provided shift value.
    The saturation function ensures that the output remains within a defined range, preventing overflow or underflow.

    3.activation and weight
    input1(act.): input tensor;
    input2(act.): input tensor;
    input3(act.): input tensor;

    4.attribute
    is_permute_optimize:whether to apply optimization for permuting the input tensors.;
    mul1_round_mode: the rounding mode to be used for the first multiplication operation.;
    mul2_round_mode: Similar to mul1_round_mode, this attribute defines the rounding mode for the second multiplication operation.;
    add_round_mode: the rounding mode for the addition operation.;
    mul1_shift: the number of bits to shift the result of the first multiplication.;
    mul2_shift: Similar to mul1_shift, this attribute defines the number of bits to shift for the second multiplication operation.;
    add_shift: the number of bits to shift the result of the addition operation.;
    mul1_saturation: whether the output of the first multiplication should be saturated.
                     When set to true, the result will be clamped to prevent overflow or underflow.;
    mul2_saturation: Similar to mul1_saturation, this attribute specifies whether saturation should be applied to the second multiplication's output.;
    add_saturation: whether to apply saturation to the output of the addition operation.;
    ginfo: associated with layer grouping information.;
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

  let results = (outs AnyRankedTensor:$output);
}
def Tpu_PreprocessOp: Tpu_Op<"Preprocess"> {
  let summary = "FusePreprocess, it's just a placeholder op.";
  let description = [{
    1.Op Introduction
    It may be divided to permute + slice + scale/scale_lut ops.

    2.Math formula
    ```math
        output = \text{Preprocess}(input, \text{resize_dims}, \text{mean}, \text{scale}, \text{channel_order})
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    quant_mode: It determines how the output tensor should be quantized.;
    customization_format: define how input data should be structured or formatted before being processed by the model.;
    channel_order: the order of the color channels in the output tensor.(e.g., RGB, BGR).;
    resize_dims: the target dimensions to which the input tensor should be resized.;
    scale: a scaling factor applied to the attention scores before they are passed through the softmax function.;
    mean: the mean values that should be subtracted from the input tensor during normalization.;
    sign: whether the preprocessing operation should include a sign adjustment.;
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
  let hasCanonicalizer = 1;
  let results = (outs AnyTensor:$output);
}

def Tpu_MeanStdScaleOp: Tpu_Op<"MeanStdScale",
  [DeclareOpInterfaceMethods<TypeInterface>,DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>]> {
  let summary = "MeanStdScale, it's for preprocess.";
  let description = [{
    1.Op Introduction
    for preprocess, do multiplier&rshift quantize.

    2.Math formula
    ```math
            output = (input - mean) / std * scale + zero_points
    ```

    3.activation and weight
    input(act.): input tensor;
    f32_param(w.): a weight tensor parameter of type float32.;

    4.attributes
    quant_mode: the mode or method used for quantization during the requantization operation.;
    customization_format: custom format for the input data.;
    channel_order: The order of color channels in the input tensor.;
    sign: if output is signed.;
    scale: scalar.;
    std: standard deviation values for each channel.;
    mean: mean values to subtract from each channel for normalization.;
    zero_points: zero point values for each channel.;
    resize_dims: resize the input tensor' dimensions.;
    multi: used during the matrix multiplication operation.;
    rshift: right shift values corresponding to each input tensor.;
    offset: Scalar;
    rounding_mode: The rounding method to use during quantization.;
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

def Tpu_PackRawOp: Tpu_Op<"PackRaw", [
  DeclareOpInterfaceMethods<TypeInterface>]>  {
  let summary = "Preprocess for raw image.";
  let description = [{
    1.Op Introduction
    (1) preprocess raw image from 3 byte 2 pixel to 2 byte 1 pixel,
        for each row pixel fill 0 in top 4 bits;
        for each img block pack to 4 channels pattern;
    (2) cast to float then normalize and clip by black & white level;
    (3) cast to odtype then pad img to align padding param and move to ddr;

    2.Math formula
    (1)Pixel Conversion:
    For each pixel in the input tensor, the conversion can be represented as:
    ```math
        packed_pixel = (input & 0x3FF) (keeping lower 10 bits) (with top 4 bits set to 0);
    ```
    Here, input_pixel is the original pixel value, and packed_pixel is the newly formatted pixel value.
    (2)Normalization:
    The normalization step can be described as:
    ```math
        normalized_value = (packed_pixel - black_level) / (white_level - black_level);
    ```
    This formula ensures that the pixel values are scaled between 0 and 1 based on the specified black and white levels.
    (3)Clipping:
    After normalization, the values are clipped to ensure they remain within the valid range:
    ```math
        clipped_value = clip(normalized_value, 0, 1);
    ```
    (4)Output Casting and Padding:
    Finally, the output tensor is formed by casting the clipped values to the desired output data type and applying padding:
    ```math
        output = pad(cast(clipped_value), padding_params);
    ```

    3.activation and weight
    input(act.): input tensor;
    high_table(w.): defines the upper bounds for pixel values during normalization.;
    low_table(w.): defines the lower bounds for pixel values.;

    4.attribute
    white_level: the maximum intensity level for normalization.;
    black_level: the minimum intensity level for normalization.;
    threshold: a threshold value that can be used during the normalization or clipping stages.;
    channel_order: the order of the color channels in the output tensor.(e.g., RGB, BGR).;
  }];
  let arguments = (
    ins AnyTensor:$input,
    AnyTensorOrNone:$high_table,
    AnyTensorOrNone:$low_table,
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
    1.Op Introduction
    (1) depack channel (b, bh * bw, ih + ph, iw + pw) -> (b, 1, oh * bh, ow * bw)
    (2) cast to INT16 then postprocess img to raw pattern ( 3 byte 2 pixel )
    (3) (b, bh * bw, ih + ph, iw + pw) 16bit -> (b, 1, oh * bh, ow * bw * 3 / 2) 8bit

    2.Math formula
    (1)Depacking:
    The depacking operation can be represented as:
    ```math
        depacked_tensor = reshape(input, (b, 1, oh · bh,ow · bw));
    ```
    Here, input is the packed tensor, and depacked_tensor is the resulting tensor after reshaping.
    (2)Casting to INT16:
    After reshaping, the values are cast to 16-bit integers:
    ```math
        int16_tensor = cast(depacked_tensor, int16);
    ```
    (3)Postprocessing to Raw Pattern:
    The conversion from INT16 back to the raw image format can be described as:
    ```math
        raw_image = int16_tensor x (255 / (white_level - black_level));
    ```
    This scaling ensures that the pixel values are appropriately adjusted based on the specified white and black levels.
    (4)Final Output Tensor:
    Finally, the output tensor is formed by packing the processed values into the required format:
    ```math
        output=reshape(raw_image, (b, 1, oh · bh, ow · bw · 2/3));
    ```
    This ensures that the output tensor has the correct dimensions and format for further processing.

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    padding_h: the height of the padding applied to the input image.;
    padding_w: the width of the padding applied to the input image.;
    white_level: the maximum intensity level for normalization.;
    black_level: the minimum intensity level for normalization.;
    channel_order: the order of the color channels in the output tensor.(e.g., RGB, BGR).;
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

def Tpu_Mmap2RgbmapOp: Tpu_Op<"Mmap2Rgbmap"> {
  let summary = "Postprocess for isp mmap2rgbmap.";
  let description = [{
    1.Op Introduction
    Postprocess for isp mmap2rgbmap.

    2.Math formula
    ```math
        output = Postprocess(input)
    ```

    3.activation and weight
    input(act.): input tensor;
  }];
  let arguments = (
    ins AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_SubOp: Tpu_Op<"Sub",
    [SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info", "DumpQuantAgnosticAttrs"]>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "sub operator";

  let description = [{
    1.Op Introduction
    Elementwise subtraction of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.

    2.Math formula
    ```math
        output = ReLU((input1 - input2; dim))
    ```
    Axis of size 1 will be broadcast if necessary.

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    is_reverse: whether the subtraction operation is performed in reverse order.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    coeff: It is an array and allows for scaling the output of the addition operation.;
    // quant param
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    f8_scales: scaling factors for FP8 (8-bit floating point) quantization.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Elementwise subtraction of input1 and input2. Input1 or Input2 is constant.
    as necessary.

    2.Math formula
    ```math
        output = input - const_val or const_val - input
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    const_val: the constant value to be added to each element of the input tensor(positive, negative, or zero).;
    is_reverse: This boolean attribute indicates whether the subtraction operation is performed in reverse order.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    // quant param
    multiplier: applied during the concatenation process to adjust the values of the input tensors.;
    rshift: right shift values corresponding to each input tensor.;
    f8_scale: scaling factors for FP8 (8-bit floating point) quantization.;
    ginfo: associated with layer grouping information.;
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info", "DumpQuantAgnosticAttrs"]>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "mul operator";

  let description = [{
    1.Op Introduction
    Elementwise multiplication of input1 and input2. input1 and input2 are tensors.

    2.Math formula
    ```math
        output = ReLU((input1 * input2))
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    // quant param
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    quant_mode: It determines how the output tensor should be quantized.;
    out_f8_scales: This parameter contains the scaling factors for the output in FP8 (8-bit floating point) format.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Elementwise max of input1 and input2. All inputs and outputs must have the same data type.

    2.Math formula
    ```math
        output = max(input1,input2, ..., inputN)
    ```
    Where input1, input2, ..., inputN are the input tensors.

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    coeff: It is an array and allows for scaling the output of the addition operation.;
    // quant param
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Elementwise min of input1 and input2. All inputs and outputs must have the same data type.

    2.Math formula
    ```math
        output = min(input1,input2, ..., inputN)
    ```
    Where input1, input2, ..., inputN are the input tensors.

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    coeff: It is an array and allows for scaling the output of the addition operation.;
    // quant param
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    max of one input and one const.

    2.Math formula
    ```math
        output = Max(input, const_val)
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    const_val: the constant value to be added to each element of the input tensor(positive, negative, or zero).;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    // quant param
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    min of one input and one const.

    2.Math formula
    ```math
        output = Min(input, const_val)
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    const_val: the constant value to be added to each element of the input tensor(positive, negative, or zero).;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    // quant param
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    The operator for activation function.

    2.Math formula
    ```math
            output = Active(input),for example:Relu, Silu...
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    mode: the specific activation mode or function to be applied by the operator.;
    coeffs: an array of float64 values as coefficients for activation functions.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    The operator limits the given input to a certain range.

    2.Math formula
    ```math
            output[i] = min      if input[i] < min;
                        input[i] if input[i] >= min && input[i] <= max;
                        max      if input[i] > max;

    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    min: the minimum value that the elements of the input tensor can take.;
    max: the maximum value that the elements of the input tensor can take.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    The operator limits the given input to a certain range.

    2.Math formula
    ```math
            output[i] = min      if input[i] < min;
                        input[i] if input[i] >= min && input[i] <= max;
                        max      if input[i] > max;
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    min: the minimum value that the elements of the input tensor can take.;
    max: the maximum value that the elements of the input tensor can take.;
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
    1.Op Introduction
    Yuv2rgbFormulaOp operator.

    2.Math formula
    ```math
            (R) = (Y + 1.402 x (V - 128))
            (G) = (Y - 0.344136 x (U - 128) - 0.714136 x (V - 128))
            (B) = (Y + 1.772 x (U - 128))

    ```

    3.activation and weight
    YUV(act.): input tensor.;

    4.attributes
    src_format: the source format of the input YUV data.;
    dst_format: the desired destination format for the output RGB data.;
    image_format: how the YUV data should be processed and how the output RGB data should be structured.;
    formula_mode: the mode of the conversion formula used for the YUV to RGB transformation. ;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
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
    1.Op Introduction
    Elementwise mul of input1 and input2. Input2 is constant.

    2.Math formula
    ```math
        output = input * const_val
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    const_val: the constant value to be added to each element of the input tensor(positive, negative, or zero).;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    // quant param
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    The Reciprocal operator is a tensor operation that performs division of a constant scalar value by an input tensor.

    2.Math formula
    ```math
        output = const_val / input
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    const_val: specifies the constant value to be added to each element of the input tensor(positive, negative, or zero).;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Refer to `https://github.com/onnx/onnx/blob/main/docs/Operators.md#depthtospace`
    [n, c, h, w] => [n, c / (block_h * block_w), h * block_h, w * block_w];
    if inversed, [n, c, h, w] => [n, c * block_h * block_w, h / block_h, w / block_w];
    if DCR(depth-column-row), channel ordered by block_h * block_w * c;
    else CRD(column-row-depth), channel ordered by c * block_h * block_w;
    The format of input or output is NCHW or NHWC.

    2.Math formula

    (1)Standard Transformation:
    Given an input tensor of shape ( (N, C, H, W) ):
    The output tensor after applying the Depth2Space operation can be calculated as:
    ```math
        {output}(N_i, C_j', H_k, W_l) = input(N_i, C_j, k / block_h, l / block_w)
    ```
    where k / block_h and l / block_w are rounded down.

    (2)Inverse Transformation:
    Given an input tensor of shape ( (N, C, H, W) ):
    The output tensor after applying the Depth2Space operation can be calculated as:
    ```math
        {output}(N_i, C_j, H_k, W_l) = input(N_i, C_j', (k x block_h + j / (C / (block_h x block_w))), (l x block_w + j % (C / (block_h x block_w))))
    ```
    where C / (block_h x block_w) is rounded down.

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    block_h: The height of the blocks used to rearrange the depth into spatial dimensions.;
    block_w: The width of the blocks used to rearrange the depth into spatial dimensions.;
    is_CRD: whether the channel ordering is in Column-Row-Depth format.;
    is_inversed: whether the channel ordering is in Column-Row-Depth format.;
    in_is_NCHW: whether the input tensor is in NCHW format.;
    out_is_NCHW: whether the output tensor should be in NCHW format.;
    swap_cr: swaps the height and width dimensions in the output tensor.;
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
    1.Op Introduction
    lookup table in index [0-255], y[i] = table(x[i])

    2.Math formula
    ```math
            output[i] = table(input[i])
    ```
    3.activation and weight
    input(act.): input tensor;
    table(w.): map input values to corresponding output values.;

    4.attributes
    ginfo: associated with layer grouping information.;
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

    1.Op Introduction
    input and output is BF16, input BF16 split as exponent and mantissa,
    get output by exponent table and mantissa table
    BF16 mode, lookup table in index [0-255], y[i] = table(x[i]).

    2.Math formula
    ```math
            output[i] = table(input[i])
    ```
    3.activation and weight
    input(act.): input tensor;
    table(w.): map input values to corresponding output values.;
    mantissa(w.): the mantissa component of the lookup table weights.;

    4.attributes
    max_range: the maximum value in the range of valid input values.;
    min_range: the minimum value in the range of valid input values.;
    lut_mode: the mode of operation for the lookup table.;
    ginfo: associated with layer grouping information.;
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "AllowDataSplit", "DumpQuantAgnosticAttrs"]>,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "matmul operator";

  let description = [{
    1.Op Introduction
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.

    2.Math formula
    ```math
        output = input x right + bias
    ```

    3.activation and weight
    input(act.): the first input tensor;
    right(act.): the second input tensor;
    bias(w.): an optional tensor can be added to the result of the matrix multiplication. ;
    multi(w.): used during the matrix multiplication operation.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attribute
    right_transpose: whether to transpose the right input tensor before performing the multiplication.;
    left_transpose: whether to transpose the input tensor before performing the multiplication.;
    output_transpose: whether to transpose the output tensor after the multiplication.;
    hdim_is_batch: whether the first dimension of the input tensor represents the batch size.;
    keep_dims: whether to keep the dimensions of the output tensor the same as the input tensors.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    multipliers: an integer array that specifies scaling factors to be applied to the results of the matrix multiplication.;
    rshifts: an integer array that indicates the right shift values.;
    right_zp: the zero point for the right input tensor.;
    input_zp: the zero point for the left input tensor.;
    quant_mode: It determines how the output tensor should be quantized.;
    ginfo: associated with layer grouping information.;
    multi_core: executed using multiple cores.;
    fuse_rq: fuse the requantization step with the matrix multiplication operation.;
    round_mode: the rounding mode to be used during the casting operation.;
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
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    matmul_attr_t parseParam();
    matmul_attr_t dynparseParam();
    uint64_t getL2BufferSize();
    void assign_fw_param(void *param);
  }];
}

def Tpu_A16MatMulOp: Tpu_Op<"A16MatMul",
    [DeclareOpInterfaceMethods<IndexingMapsInterface>]> {
  let summary = "w8a16 / w4a16 matmul operator";

  let description = [{
    1.Op Introduction
    The special matrix multiplication designed for LLM Linear Layer.
    Weight is saved in int8 with f16 per-channel quant scale.

    2.Math formula
    ```math
            y_f16 = x_f16 x (quantized_w.to(f16) * scale_f16)
    ```

    3.activation and weight
    input(act.): input tensor.;
    weight(w.): weight tensor.;
    scale(w.): scalar.;
    zp(w.): zero points for weight quant.;
    bias(w.): an optional tensor can be added to the result of the matrix multiplication. ;

    4.attributes
    weight_bits: the bit-width used to represent the weight values.;
    sign: if output is signed.;
    w_transpose: whether the weight tensor should be transposed;
    q_group_size: the group size for per-group quantization.;
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

  let hasCanonicalizer = 1;
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_AttentionOp: Tpu_Op<"Attention", [
    SupportFuseRelu,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "AllowDataSplit", "DumpQuantAgnosticAttrs"]>,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Attention operator";

  let description = [{
    1.Op Introduction
    Performs a multi head attention block. https://en.wikipedia.org/wiki/Attention_(machine_learning)
    This block has Q_w, K_w,V_w, O_w and mask

    2.Math formula
    ```math
        Attention(Q, K, V) = softmax(((Q x K^T) / \sqrt{d_k}) + musk) x V;
        head_i = Attention(Q x queries_weight, K x keys_weight, V x values_weight);
        MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) x out_weight + out_bias;
        output = MultiHead(input x queries_weight + queries_bias, input x keys_weight + keys_bias, input x values_weight + values_bias).
    ```

    3.activation and weight
    input(act.): input tensor.;
    keys(act.): The keys are derived from the input data and help the model determine which parts of the input are relevant for each query.;
    values(act.): The values are the actual information that will be aggregated based on the attention scores computed from the queries and keys.;
    queries_weight(w.): Queries are the features that the model uses to ask questions about the input data.;
    queries_bias(w.): added to the query representations after the weight transformation.;
    keys_weight(w.): This weight tensor transforms the input into key representations.;
    keys_bias(w.): added to the key representations after the weight transformation, providing further adjustment.;
    values_weight(w.): used to transform the input into value representations.;
    values_bias(w.): added to the value representations after the weight transformation.;
    out_weight(w.): used to transform the concatenated output of the attention heads into the final output representation.;
    out_bias(w.): added to the output representation after the final weight transformation.;
    musk(w.):  apply masking during the attention computation, Masks can prevent the model from attending to certain positions in the input.;
    table(w.): additional computations or transformations during the softmax operation.;

    4.attribute
    quant_param: used to reduce the precision of the numbers used in computations.;
    scale: a scaling factor applied to the attention scores before they are passed through the softmax function.;
    head: the number of attention heads to use in the multi-head attention mechanism.;
    dim: the size of the input features or the size of the query, key, and value vectors.;;
    has_bias: whether the attention mechanism includes bias terms in its computations.;
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
    AnyTensorOrNone:$mask,
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
    1.Op Introduction
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.

    2.Math formula
    ```math
        Attention(Q, K, V) = softmax(((Q x K^T) / \sqrt{d_k}) + musk) x V;
        head_i = Attention(Q x queries_weight, K x keys_weight, V x values_weight);
        MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) x out_weight + out_bias;
        output = MultiHead(input x queries_weight + queries_bias, input x keys_weight + keys_bias, input x values_weight + values_bias).
    ```

    3.activation and weight
    queries(act.): input tensor.;
    keys(act.): The keys are derived from the input data and help the model determine which parts of the input are relevant for each query.;
    values(act.): The values are the actual information that will be aggregated based on the attention scores computed from the queries and keys.;
    mask(w.): apply masking during the attention computation, Masks can prevent the model from attending to certain positions in the input.;;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attributes
    scale: scalar.;
    batch: batch size.;
    q_head: the number of query heads in the multi-head attention mechanism.;
    kv_head: the number of key/value heads.;
    dim: the size of the input features or the size of the query, key, and value vectors.;
    mq: a dimension or a modifier related to the query matrix.;
    mk: the key matrix.;
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
    1.Op Introduction
    create a tensor with the same shape as input, and fill with value from normal distribution

    2.Math formula
    ```math
        output = randn(shape(input))
    ```

    3.activation and weight
    input(act.): input tensor.;
    randn_data(w.): the characteristics of the normal distribution.;
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
    1.Op Introduction
    range op.
    generates a sequence of evenly spaced values within a specified range.

    2.Math formula
    ```math
        output = [x | x = start + n * delta, n is an integer, and start ≤ x < limit]
    ```

    3.activation and weight
    start(act.): The starting value of the sequence. This can be a tensor or None. If None, it defaults to 0.;
    limit(w.): The exclusive upper limit of the sequence.;
    delta(w.): The increment between each value;
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
    1.Op Introduction
    ReLU with a scalar maximum value.

    2.Math formula
    ```math
        output = ReluOp(input) -> (0, 1)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    ginfo: associated with layer grouping information.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
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
    1.Op Introduction
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.

    2.Math formula
    ```math
        output = ReshapeOp(input, shape)
    ```

    3.activation and weight
    input(act.): input tensor.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attribute
    shape: 0: keep dim from input; -1: left dim from input.;
    flatten_start_dim: the starting dimension from which to begin flattening the input tensor.;
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$shape,
    Optional<AnyTensorOrNone>:$buffer,
    DefaultValuedAttr<I64Attr, "-1">:$flatten_start_dim
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ShapeReshapeOp:Tpu_Op<"ShapeReshape",[
  ShapeProducer, ShapeConsumer]> {
  let summary = "ShapeReshape operation";
  let description = [{
    1.Op Introduction
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.

    2.Math formula
    ```math
        output = ShapeReshapeOp(input, shape)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    shape: 0: keep dim from input; -1: left dim from input.;
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
    1.Op Introduction
    reshape for dynamic shape

    2.Math formula
    ```math
        output = ShapeAssign(input, shape)
    ```

    3.activation and weight
    input(act.): input tensor.;
    shape(w.): 0: keep dim from input; -1: left dim from input.;
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
    1.Op Introduction
    Reverse on input.

    2.Math formula
    ```math
        output = ReverseOp(input, axis)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    axis: the dimension of reverse;
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
    1.Op Introduction
    Reverse on input

    2.Math formula
    ```math
        output = ShapeReverse(input, axis)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    axis: the dimension of reverse;
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
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Cast operation";
  let description = [{
    1.Op Introduction
    The Tpu_CastOp is a tensor operation that performs type casting on the input tensor.

    2.Math formula
    output = Cast(input);

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    extra_input: whether additional input is required for the casting operation.;
    ginfo: contains layer grouping information.;
    with_scale: whether the casting operation should include a scaling factor.;
    round_mode: the rounding mode to be used during the casting operation.;
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
    1.Op Introduction
    Cast F32 to F16
    The Tpu_DtypeCastOp is a specialized operation designed to cast floating-point tensors from 32-bit precision (F32) to 16-bit precision (F16).

    2.Math formula
    FLOAT16(output) = DtypeCastOp (FLOAT32(input));

    3.activation
    input(act.): input tensor;

    4.attribute
    extra_input: whether additional input is required for the casting operation.;
    ginfo: contains layer grouping information.;
    with_scale: whether the casting operation should include a scaling factor.;
    round_mode: the rounding mode to be used during the casting operation.;
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
    1.Op Introduction
    designed to perform shape casting, which allows the transformation of the shape of an input tensor while preserving its data type.

    2.Math formula
    output = reshape(input, new_shape);

    3.activation
    input(act.): input tensor;
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LoadOp:Tpu_Op<"Load",
  [DeclareOpInterfaceMethods<LocalGenInterface, ["assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>, InOutSameShape]> {
  let summary = "Load operation";
  let description = [{
    1.Op Introduction
    load input or weight from gmem to lmem;
    if do_bcast, [1,1,1,w] will load to [1,npu,1,w]

    2.Math formula
    output = Load(input, do_bcast, use_3ic_optimize, lmem_type, support_compress);

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    do_bcast: whether broadcasting.;
    use_3ic_optimize: whether the 3-IC (Three Input Channels) optimization;
    lmem_type: the type of local memory (lmem) to be used when loading data from global memory (gmem).;
    ginfo: associated with layer grouping information.;
    support_compress: whether compression is supported for the load operation.;
    compress_info: used in conjunction with support_compress to specify details like compression schemes, ratios, or any relevant metadata.;
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
    1.Op Introduction
    load weight from gmem to l2mem;

    2.Math formula
    output = Load(input, id, l2m_addr);

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    id: differentiate or index various instances of the load operation, ensuring correctly matched.;
    l2m_addr: the address or offset within the local L2 memory (l2mem).;
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$buffer,
    I64Attr:$id
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    void codegen_only_for_LoadToL2MOp(std::pair<int, int>& core_num_idx);
  }];
}

def Tpu_StoreOp:Tpu_Op<"Store",
  [DeclareOpInterfaceMethods<LocalGenInterface, ["assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>, InOutSameShape]> {
  let summary = "Store operation";
  let description = [{
    1.Op Introduction
    store weight from gmem to l2mem;

    2.Math formula
    output = store(input, input_gmem, ginfo, support_compress, compress_info, l2m_addr)

    3.activation and weight
    input(act.): input tensor;
    input_gmem(w.): the input tensor containing the weights stored in global memory (gmem).;

    4.attributes
    ginfo: associated with layer grouping information.;
    support_compress: whether compression is supported for the load operation.;
    compress_info: used in conjunction with support_compress to specify details like compression schemes, ratios, or any relevant metadata.;
    l2m_addr: the address or offset within the local L2 memory (l2mem).;
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$buffer,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
    DefaultValuedAttr<BoolAttr, "true">:$support_compress,
    OptionalAttr<Tpu_CompressAttr>:$compress_info
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_RequantIntOp:Tpu_Op<"RequantInt", [
  InOutSameShape, SupportElementwise,
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface, ["DumpQuantAgnosticAttrs"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "requant operation";
  let description = [{
    1.Op Introduction
    Requant 32/16/8 bit data to int8 or uint8 data, by int multiplier and int shift;

    2.Math formula
    int8/uint8(output) = RequantIntOp (int32/16/8(input));

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    multiplier: Floating-point multiplication operations are usually converted to fixed-point multiplication operations.;
    rshift: the number of bits to right-shift the quantized values.;
    quant_mode: It determines how the output tensor should be quantized.;
    round_mode: This parameter specifies the rounding mode to be used during quantization.;
    ginfo: contains layer grouping information.;
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
    1.Op Introduction
    Requant 32/16/8 bit data to int8 or uint8 data, PerAxis(or PerChannel);

    2.Math formula
    int8/uint8(output) = RequantIntAxisOp(int32/16/8(input), quant);

    3.activation and weight
    input(act.): input tensor;
    quant(act.): the quantization parameters tensor.;

    4.attribute
    quant_mode: It determines how the output tensor should be quantized.;
    round_mode: This parameter specifies the rounding mode to be used during quantization.;
    ginfo: contains layer grouping information.;
    rq_axis: the axis along which the requantization is performed.;
    fuse_rq_axis: whether to fuse the requantization operation with subsequent operations along the specified axis.;
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
    1.Op Introduction
    Requant 32/16/8 bit data to int8 or uint8 or fp8 data, by float scale and float offset, when to fp8 data, offset is not used;

    2.Math formula
    int8/uint8/fp8(output) = round(float32/float16/float8(input) x scale + offset);

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    scale: Scalar;
    offset: Scalar;
    quant_mode: It determines how the output tensor should be quantized.;
    round_mode: This parameter specifies the rounding mode to be used during quantization.;
    first_round_mode: the rounding behavior applied to the scaled value before the offset is added.;
    ginfo: contains layer grouping information.;
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
    1.Op Introduction
    Requant 32/16/8 bit data to int8 or uint8 data, PerAxis(or PerChannel);

    2.Math formula
    int8/uint8(output) = RequantFpAxisOp(float32/float16/float8(input),quant);

    3.activation and weight
    input(act.): input tensor;
    quant(act.): This attribute represents the quantization parameters tensor.
                 It contains the values used for requantization, such as multipliers and shifts,
                 which are specific to each axis or channel.;

    4.attribute
    quant_mode: It determines how the output tensor should be quantized.;
    round_mode: This parameter specifies the rounding mode to be used during quantization.;
    first_round_mode: the rounding behavior applied to the scaled value before the offset is added.;
    ginfo: contains layer grouping information.;
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
    1.Op Introduction
    Dequant 8 bit data to 32/16 bit data.

    2.Math formula
    32/16bit(output) = DequantIntOp((8bit(input), shift) x multiplier) ≪ lshift;

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    multiplier: Floating-point multiplication operations are usually converted to fixed-point multiplication operations.;
    shift: a shift value applied to the quantized data before scaling.;
    lshift: a left shift operation applied to the dequantized data after scaling.;
    quant_mode: It determines how the output tensor should be quantized.;
    round_mode: This parameter specifies the rounding mode to be used during quantization.;
    ginfo: contains layer grouping information.;
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
    1.Op Introduction
    Dequant 8 bit data to 32/16 bit data, PerAxis(or PerChannel)

    2.Math formula
    32/16bit(output) = DequantIntAxisOp((8bit(input), quant, shift) x multiplier) ≪ lshift;

    3.activation and weight
    input(act.): input tensor;
    quant(act.): This attribute represents the quantization parameters tensor.
                 It contains the values used for requantization, such as multipliers and shifts,
                 which are specific to each axis or channel.;

    4.attribute
    lshift: a left shift operation applied to the dequantized data after scaling.;
    quant_mode: It determines how the output tensor should be quantized.;
    round_mode: This parameter specifies the rounding mode to be used during quantization.;
    ginfo: contains layer grouping information.;
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
    1.Op Introduction
    Make ops in one group to inferece by local mem

    2.Math formula
        output = Group(nsecs, hsecs, dsecs, wsecs, csecs; swpipl_stage_num, group_type, flow, self_up_overlap_op, self_down_overlap_op, other_up_overlap_op, other_down_overlap_op, support_compress, run_core_id, core_slice_ncdhw)

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    nsecs: the number of sections in the "n" (batch or number) dimension.;
    hsecs: the number of sections in the "h" (height) dimension.;
    dsecs: the number of sections in the "d" (depth) dimension.;
    wsecs: the number of sections in the "w" (width) dimension.;
    csecs: the number of sections in the "c" (channel) dimension.;
    swpipl_stage_num: the number of stages in the pipeline, which related to swappable operations or pipelined processing.;
    group_type: the type of the grouping strategy.;
    flow: store both negative timestep indices and positive operation identifiers.;
    self_up_overlap_op: overlapped in the upward (or previous) direction within the same group.;
    self_down_overlap_op: overlapped in the downward (or subsequent) direction within the same group.;
    other_up_overlap_op: Holds the operation identifiers in other groups overlapping in the upward direction.;
    other_down_overlap_op: Holds the operation identifiers in other groups overlapping in the downward direction.;
    support_compress: whether compression is supported for the load operation.;
    run_core_id: an array of core IDs for running.;
    core_slice_ncdhw: the tensor's dimensions (n, c, d, h, w) are partitioned.;
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
    1.Op Introduction
    This operation is composed of numerous regions, with each region corresponding
    to a subgraph. These subgraphs share identical computational patterns and are
    distributed across various cores of a multi-core TPU for processing.

    2.Math formula
        output = GroupParallel(input)

    3.activation and weight
    inputs(act.): input tensor.;
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);

  let regions = (region VariadicRegion<SizedRegion<1>>:$parallel);
}

def Tpu_CoreBeginOp:Tpu_BaseOp<"CoreBegin"> {
  let summary = "Begin op parallel to multi cores";
  let description = [{
    1.Op Introduction
    Begin of pattern to multi cores

    2.Math formula
        output = CoreBegin(input, pattern)

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    pattern: the core operation pattern used for initializing parallel execution across multiple cores.;
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
    1.Op Introduction
    End of pattern to multi cores

    2.Math formula
        output = CoreEnd(input, pattern)

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    pattern: the core operation pattern used for initializing parallel execution across multiple cores.;
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
    1.Op Introduction
    The ops in one parallel should run in parallel.

    2.Math formula
        output = CoreParallel(input, offset, size)

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    offset: scalar;
    size: the number of elements.;
  }];
  let arguments = (ins
    Variadic<AnyTensorOrNone>:$inputs,
    I64Attr:$offset,
    I64Attr:$size
  );
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let regions = (region SizedRegion<1>:$body);
}

def Tpu_CorrelationOp: Tpu_Op<"Correlation"> {
  let summary = "Custom operator correlation";

  let description = [{
  Multiply the sliced left_feature and right_feature based on max_disp;
  then perform a reduce operation;
  and finally concatenate the results.

  2.Math formula
  for i in range(max_disp):
    if i > 0:
        output[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
    else:
        output[:, i, :, :] = (left_feature * right_feature).mean(dim=1)

  3.activation and weight
  input(act.): input tensor;

  4.attribute
  max_disp: The number of slicing iterations, which is also the size of the output dimension C.
  num_groups: The number of batch groups.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    DefaultValuedAttr<I64Attr, "0">:$max_disp,
    DefaultValuedAttr<I64Attr, "1">:$num_groups
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_SplitOp:Tpu_BaseOp<"Split"> {
  let summary = "split tensor to continues pieces";
  let description = [{
    1.Op Introduction
    The ops in one parallel should run in parallel.

    2.Math formula
    ```math
        output = input[i * split_size: (i + 1) * split_size] for i = 0, 1, ... num - 1
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins
    AnyRankedTensor:$input
  );
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
}

def Tpu_JoinOp:Tpu_BaseOp<"Join"> {
  let summary = "Join tensor to continues pieces";
  let description = [{
    1.Op Introduction
    The ops in one parallel should run in parallel.

    2.Math formula
        output = concat(input)

    3.activation and weight
    inputs(act.): input tensor.;
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_DevBeginOp:Tpu_BaseOp<"DevBegin"> {
  let summary = "Begin distribution tensors to multi device";
  let description = [{
    1.Op Introduction
    Tensors split to distributed device

    2.Math formula
        output = DevBegin(inputs, pattern, begin_methods, num_head, done)

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    pattern: the core operation pattern used for initializing parallel execution across multiple cores.;
    begin_methods: an array of strategies used to begin the distribution process across devices.;
    num_head: the number of primary channels that are processed first or in parallel.;
    done: a flag whether the distribution initialization process is completed.;
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
    1.Op Introduction
    Tensors from distributed device connect together.

    2.Math formula
        output = DevBegin(inputs, pattern, begin_methods, num_head, done)

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    pattern: the core operation pattern used for initializing parallel execution across multiple cores.;
    end_methods: an array of strategies used to end the distribution process across devices.;
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
    1.Op Introduction
    yields values to its parent operation.

    2.Math formula
        output_i = input_i i = 1, 2, ..., N

    3.activation and weight
    operands(act.): a variadic number of input tensors.;
  }];

  let arguments = (ins Variadic<AnyTensor>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

}

def Tpu_SoftmaxOp: Tpu_Op<"Softmax",[
    DeclareOpInterfaceMethods<IndexingMapsInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>,
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>,
    InOutSameShape]> {
  let summary = "softmax operator";

  let description = [{
    1.Op Introduction
    Integrates some operations related to softmax.

    2.Math formula
    ```math
            \text{output}[i] = \frac{e^{\text{input}[i]}}{\sum_{j} e^{\text{input}[j]}}
    ```

    3.activation and weight
    input(act.): input tensor.;
    table(w.): additional computations or transformations during the softmax operation.;
    slope_table(w.): contain scaling factors or slopes that can adjust the output.;
    reciprocal_table(w.): holds precomputed reciprocal values that can be used to optimize the division operations;
    reciprocal_mantissa_table(w.): store the mantissa values of the reciprocal.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attributes
    axis: the dimension of the input tensor.;
    log: when set to true, indicates that the output should be computed in log space.;
    beta: scaling factor.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
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
    1.Op Introduction
    The LeakyRelu operation multiples alpha with negative values, and the others keep changeless.

    2.Math formula
    ```math
            output = alpha * input, if input < 0
            output = input, if input >= 0
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    alpha: a scalar factor.;
    multiplier: Floating-point multiplication operations are usually converted to fixed-point multiplication operations.;
    multiplier_neg: specify the multiplier for negative input values.;
    rshift: the number of bits to right-shift the quantized values.;
    rshift_neg: specifies the number of bits to right-shift the quantized negative values.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
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
    1.Op Introduction
    Concatenates the given sequence of seq tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    2.Math formula
    output = Concat(input1, input2, axis)
           = input1[axis] + input2[axis];

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    axis: the dimension along which the input tensors will be concatenated.;
    only_merge: whether the operation should only perform a merge of tensors without additional processing.;
    // param for cv18xx
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    // for group
    ginfo: contains layer group information.;
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
    1.Op Introduction
    Concatenates the given sequence of seq tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    2.Math formula
    output = ShapePack(input, axis)

    3.activation and weight
    inputs(act.): input tensor;

    4.attribute
    axis: the dimension of the input tensor.;
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
    1.Op Introduction
    perform an element-wise power calculation on the input tensor.

    2.Math formula
        output = input ^ n

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    exponent:;
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
    DeclareOpInterfaceMethods<DynLocalGenInterface>]> {

  let summary = "MulShift operator";

  let description = [{
    1.Op Introduction
    performs an element-wise multiplication of the input tensor (after adjusting for a zero-point offset)

    2.Math formula
        output = int8(input-zx) * multiplier >> rshift + zy

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    multiplier: Floating-point multiplication operations are usually converted to fixed-point multiplication operations.;
    rshift: the number of bits to right-shift the quantized values.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Perform permute on input.

    2.Math formula
    ```math
        output(...dim2, dim1, dim0) = PermuteOp(input(dim0, dim1, dim2...order))
    ```

    3.activation and weight
    input(act.): input tensor.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attribute
    order: An array of integers specifying the permutation order of the input tensor's dimensions.;
    ginfo: associated with layer grouping information.;
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

def Tpu_ShapeTransposeOp: Tpu_Op<"ShapeTranspose",[
  ShapeProducer, ShapeConsumer]> {

  let summary = "ShapeTransposeOp operator";

  let description = [{
    1.Op Introduction
    Perform permute on input.

    2.Math formula
    ```math
        output(dim1, dim0) = TransposeOp(input(dim0, dim1))
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    order: the order in which the non-zero indices should be returned.;
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
    1.Op Introduction
    Perform ShuffleChannel on input.

    2.Math formula
    ```math
        output(N, C, H, W) = input(N, Shuffle(C), H, W)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    group: An integer specifying the number of groups to divide the channels into.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Perform nearest upsample on input.

    2.Math formula
    ```math
            output[i, j] = Upsample(input[i / scale_h, j / scale_w])
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    scale_h: the scaling factor for the height (number of rows) of the input tensor.;
    scale_w: the scaling factor for the width (number of columns) of the input tensor.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Perform MaxUnpool on input.

    2.Math formula
    ```math
        output1(N, C, H', W') = input(N, C, H, W) #if (h,w) is the index of the max value in the mask, otherwise is 0;
        output = scale * output1
    ```

    3.activation and weight
    input(act.): input tensor;
    mask(act.): place the pooled values in the output tensor.;

    4.attribute
    scale_h: scaling factor for the height dimension of the output tensor.;
    scale_w: scaling factor for the width dimension of the output tensor.;
    ginfo: group information for layer grouping;
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
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH", "BackwardW", "LocalGenSupport"]>,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Pad operation";
  let description = [{
    1.Op Introduction
    This operation pads a tensor according to the paddings you specify.
    paddings is an integer tensor with shape [2, n], where n is the rank of tensor.
    For each dimension D of input, paddings[0, D] indicates how many values to add
    before the contents of tensor in that dimension, and paddings[1, D] indicates
    how many values to add after the contents of tensor in that dimension.

    2.Math formula
    ```math
        output = input(padding, val, mode)
    ```

    3.activation and weight
    input(act.): input tensor.;
    paddingsT(act.):  the padding values for each dimension.;
    // for cv18xx reflect mode
    left_select(w.):reflect the tensor values when adding padding on the left side.;
    right_select(w.):reflect the tensor values when adding padding on the right side.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attribute
    paddings: defines how much padding to add before and after the contents of the input tensor for each dimension. ;
    val: the value to be used for padding the input tensor. ;
    mode: the padding mode include constant(Pads with a constant value); reflect(Pads with a reflection of the tensor values); replicate(Pads by replicating the edge values of the tensor).;
    with_insert_zero : When with_insert_zero is true, it means that the PadOp first performs the insert zero operation, and then conducts the Padding expansion. In this case, the mode can only be PaddingMode::constant. When with_insert_zero is false, the insert zero operation is not performed.
    inserts : When with_insert_zero is true, the number of zeros inserted between points in the h and w dimensions.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    Optional<AnyTensorOrNone>:$paddingsT,
    // for cv18xx reflect mode
    AnyTensorOrNone:$left_select,
    AnyTensorOrNone:$right_select,
    AnyTensorOrNone:$buffer,
    I64ArrayAttr:$paddings,
    DefaultValuedAttr<F64Attr, "0.0">:$val,
    DefaultValuedAttr<Tpu_PaddingModeAttr, "tpu::PaddingMode::constant">:$mode,

    DefaultValuedAttr<BoolAttr, "false">:$with_insert_zero,
    OptionalAttr<I64ArrayAttr>:$insert_zeros
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
    1.Op Introduction
    Performs element-wise binary division.

    2.Math formula
    ```math
        output = input/const_val or const_val/input
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    is_reverse: whether the subtraction operation is performed in reverse order.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    // quant param
    multipliers: applied during the concatenation process to adjust the values of the input tensors.;
    rshifts: an array of right shift values corresponding to each input tensor.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Slice Operation on input.

    2.Math formula
    ```math
            output[i] = input[offset[j] : ends[j] : steps[j]]
    ```

    3.activation and weight
    input(act.): input tensor.;
    offsetT(w.): the starting indices for each slice along the specified axes.;
    endsT(w.): the ending indices for each slice along the specified axes.;
    stepsT(w.): the step sizes for each slice along the specified axes.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attribute
    offset: An array of the starting indices for slicing along each axis.;
    steps: An array of the step sizes for slicing along each axis.;
    ends: An array of the ending indices for slicing along each axis.;
    axes: An array of the axes along which to perform the slicing operation.;
    hasparamConvert_axes: whether parameter conversion is needed for the specified axes.;
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
    DefaultValuedAttr<I64ArrayAttr, "{1}">:$axes,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$hasparamConvert_axes
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
    1.Op Introduction
    Slice Operation on shape-type tensor.

    2.Math formula
    ```math
            output[i] = input[offset[j] : ends[j] : steps[j]]
    ```

    3.activation and weight
    input(act.): input tensor.;
    offsetT(w.): the starting indices for each slice along the specified axes.;
    endsT(w.): the ending indices for each slice along the specified axes.;
    stepsT(w.): the step sizes for each slice along the specified axes.;

    4.attribute
    offset: An array of the starting indices for slicing along each axis.;
    steps: An array of the step sizes for slicing along each axis.;
    ends: An array of the ending indices for slicing along each axis.;
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
    1.Op Introduction
    ShapeUnsqueeze Operation on input.

    2.Math formula
        output = ShapeUnsqueeze(input, axes)

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    axes: An array of the axes along which to perform the slicing operation.;
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
    1.Op Introduction
    ShapeSqueeze Operation on input.

    2.Math formula
        output = ShapeSqueeze(input, axes)

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    axes: An array of the axes along which to perform the slicing operation.;
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
    1.Op Introduction
    Unsqueeze Operation on input.

    2.Math formula
    ```math
            output = unsqueeze(input, axes)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    axes: the dimensions (axes) of the input tensor that should be squeezed (removed).;
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
    1.Op Introduction
    Squeeze Operation on input.

    2.Math formula
    ```math
            output = squeeze(input, axes)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    axes: the dimensions (axes) of the input tensor that should be squeezed (removed).;
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
    1.Op Introduction
    Strided Slice Operation on input.

    2.Math formula
    ```math
            output[i] = input[starts[j] + i * strides[j]]
    ```

    3.activation and weight
    input(act.): input tensor.;
    starts(w.): the starting indices for each dimension of the input tensor.;
    ends(w.): the ending indices for each dimension of the input tensor.;
    strides(w.):  the stride values for each dimension, determining the step size between indices in the slicing operation.;

    4.attribute
    begin_mask: If set, the start index for that dimension is considered as 0.;
    end_mask: If set, the end index for that dimension is considered as the size of the dimension.;
    ellipsis_mask: whether allowing for the selection of all dimensions in between specified slices.;
    new_axis_mask: which dimensions should be added as new axes in the output tensor.;
    shrink_axis_mask: which dimensions should be removed from the output tensor.;
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
    1.Op Introduction
    Integrates some operations related to topk.

    2.Math formula
    ```math
            output_values, output_indices = TopK(input, K, axis, largest, sorted)
    ```

    3.activation and weight
    input(act.): input tensor.;
    kT(w.): provide a specific tensor for K values. This allows for dynamic specification of K.;

    4.attributes
    axis: the dimension of the input tensor.;
    K: how many of the largest (or smallest, depending on the largest attribute) values will be returned. defaults is -1;
    largest: whether to retrieve the largest or smallest values.;
    sorted: whether the output values should be sorted in descending order (if largest is true) or ascending order (if largest is false).;
    values_used_only: whether to return only the values of the top K elements without their corresponding indices.;
    buffer_val: the provision of a buffer tensor where the output values can be stored.;
    buffer_idx: the provision of a buffer tensor for storing the output indices of the top K elements.;
    replace_topk_indices: whether to replace the indices of the top K elements in the output with new indices.;
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
    AnyTensorOrNone:$buffer_idx,
    DefaultValuedAttr<BoolAttr, "false">:$replace_topk_indices
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
    1.Op Introduction
    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).

    2.Math formula
    ```math
            output = input[i1, i2, i3,...in] != 0
    ```

    3.activation and weight
    input(act.): input tensor.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attributes
    order: the order in which the non-zero indices should be returned.;
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
    1.Op Introduction
    Perform Deconvolution operation.

    2.Math formula
    The height and width of the output tensor can be calculated using the following formulas:
    ```math
            H_{out} = H_{in - 1} x stride[0] - 2 x pads[0] + H_k + output_padding[0]
            W_{out} = W_{in - 1} x stride[1] - 2 x pads[1] + W_k + output_padding[1]
    ```
    The output tensor is computed as:
    ```math
            output(N, C_out, H_out, W_out) = \sum(c_in) {\sum(h_k) {\sum(w_k){input(n, c_in, h_in, w_in) x filter(c_out, c_in, h_k, w_k)}}}
    ```

    3.activation and weight
    input(act.): input tensor.;
    filter(w.): the learnable weights of the convolution 2d operation.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attributes
    kernel_shape: the size of the convolution kernel (filter) as an array. ;
    strides: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    group: (optional)Number of blocked connections from input channels to output channels. Default: 1.;
    dilations: controls the spacing between the kernel points;
    output_padding: The value can be provided as a single integer or a tuple, allowing for different padding values for height and width.;
    dynweight_reorderd: whether the weights (filters) should be reordered dynamically.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    with_bias: whether to include a bias term in the convolution operation.;
    multiplier: Floating-point multiplication operations are usually converted to fixed-point multiplication operations.;
    rshift: the number of bits to right-shift the quantized values.;
    quant_mode: It determines how the output tensor should be quantized.;
    ginfo: associated with layer grouping information.;
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
    DeclareOpInterfaceMethods<TypeInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>]> {
  let summary = "3D deconvolution operator";

  let description = [{
    1.Op Introduction
    "Perform 3D deconvolution operation."

    2.Math formula
    ```math
        O(n, d, h, w) = \sum_{c=0}^{C_{\text{in}}-1} \sum_{k_d=0}^{K_d-1} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1}I\big(n, c, d' \, , h' \, , w'\big) \times F\big(c, :, k_d, k_h, k_w\big)
    ```

    3.activation and weight
    input(act.): input tensor.;
    filter(w.): the learnable weights of the convolution 2d operation.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attributes
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    strides: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.;
    group: (optional)Number of blocked connections from input channels to output channels. Default: 1.;
    dilations: controls the spacing between the kernel points;
    output_padding: The value can be provided as a single integer or a tuple, allowing for different padding values for height and width.;
    dynweight_reorderd: whether the weights (filters) should be reordered dynamically.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    with_bias: whether to include a bias term in the convolution operation.;
    multiplier: Floating-point multiplication operations are usually converted to fixed-point multiplication operations.;
    rshift: the number of bits to right-shift the quantized values.;
    quant_mode: It determines how the output tensor should be quantized.;
    ginfo: associated with layer grouping information.;
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
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Scale operator";

  let description = [{
    1.Op Introduction
    Y = X * S + B,
    where the shape of X/Y is [n, c, h, w] and the shape of S/B is [1, c, 1, 1].

    2.Math formula
    ```math
            output = input x scale + bias
    ```

    3.activation and weight
    input(act.): input tensor.;
    scale(w.): scalar;
    bias(w.): the learnable bias of the module of shape (out_channels).;
    lshift: a left shift operation applied to the dequantized data after scaling.;

    4.attributes
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    It normalizes over local input regions. The local region is defined across the channels.
    2.Math formula
    ```math
            output[i, j, k] = \frac{input[i, j, k]}{(bias + \alpha \sum_{c=\max(0,k-\text{size})}^{\min(N-1,k+\text{size})} input[i, j, c]^2)^{\beta}}
    ```

    3.activation and weight
    input(act.): input tensor.;
    table(w.): store a lookup table that may assist in optimizing the matching process.;
    mantissa(w.): store the mantissa values, which can be part of the normalization process.;

    4.attributes
    size: how many neighboring channels are considered during the normalization process.;
    alpha: a scaling factor;
    beta: a scaling factor;
    bias: A floating-point value added to the normalization denominator to prevent division by zero.;
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
    1.Op Introduction
    Perform RNN GRU operation.

    2.Math formula
    ```math
        update gate(z_t):
            z_t = Sigma(W_z · x_t + U_z ·h_(t-1) + b_z)
        reset gate(r_t):
            r_t = Sigma(W_r · x_t + U_r ·h_(t-1) + b_r)
        Candidate Activation(h_t):
            h_t = tanh(W_h · x_t + r_t \odot (U_h · h_(t-1)) + b_h)
        final output:
            output = (1 - z_t) \odot h_(t-1) + z_t \odot h_t
    ```
    where, x_t is the input at time step (t), h_(t-1) is the hidden state from the previous time step.
           W_z, W_r, W_h are the weight matrices for the input.
           U_z, U_r, U_h are the weight matrices for the hidden state.
           b_z, b_r, b_h are the bias vectors.

    3.activation and weight
    input(act.): input tensor.;
    filter(w.): the learnable weights of the convolution 2d operation.;
    recurrence(w.): the previous hidden state influences the current hidden state.;
    bias(w.): the learnable bias of the module of shape (out_channels).;
    initial_h(w.): the initial hidden state, which can be provided to start the GRU computation.;
    buffer(w.): temporary storage of intermediate results or states during the GRU computation.;
    sigmoid_table: a lookup table that contains pre-computed values for the sigmoid activation function.;
    sigmoid_slope_table: This table contains pre-computed slopes (derivatives) of the sigmoid function, enhancing the training efficiency of the LSTM.;
    tanh_table: contains pre-computed values for the hyperbolic tangent (tanh) activation function.;
    tanh_slope_table: This table contains pre-computed slopes (derivatives) of the tanh function;

    4.attributes
    hidden_size: the number of units in the GRU cell,;
    bidirectional: whether the GRU should be bidirectional;
    linear_before_reset: whether to apply a linear transformation to the input before applying the reset gate.;
    batch_first: the input and output tensors are provided in the shape (batch_size, seq_length, input_size).;
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
    1.Op Introduction
    Perform RNN LSTM operation.

    2.Math formula
    ```math
        forget gate(f_t):
            f_t = Sigma(W_f · x_t + U_f · h_(t-1) + b_f)
        input gate(i_t):
            i_t = Sigma(W_i · x_t + U_i · h_(t-1) + b_i)
        Candidate cell state(C_t):
            C_t = tanh(W_C · x_t + U_C · h_(t-1) + b_C)
        cell state update(c_t):
            c_t = f_t \odot c_(t-1) + i_t \odot C_t
        output gate(o_t):
            o_t = Sigma(W_o · x_t + U_o · h_(t-1) + b_o)
        hidden state output(h_t):
            h_t = o_t \odot tanh(c_t)
    ```

    3.activation and weight
    input(act.): input tensor.;
    filter(w.): the learnable weights of the convolution 2d operation.;
    recurrence(w.): the previous hidden state influences the current hidden state.;
    bias(w.): the learnable bias of the module of shape (out_channels).;
    initial_h(w.): the initial hidden state, which can be provided to start the LSTM computation.;
    initial_c(w.): the initial cell state, which can be provided to start the LSTM computation.;
    cont(w.): control weights or additional context that may be provided to influence the LSTM's behavior.;
    buffer(w.): temporary storage of intermediate results or states during the LSTM computation.;
    sigmoid_table: a lookup table that contains pre-computed values for the sigmoid activation function.;
    sigmoid_slope_table: This table contains pre-computed slopes (derivatives) of the sigmoid function, enhancing the training efficiency of the LSTM.;
    tanh_table: contains pre-computed values for the hyperbolic tangent (tanh) activation function.;
    tanh_slope_table: This table contains pre-computed slopes (derivatives) of the tanh function;

    4.attributes
    hidden_size: the number of units in the LSTM cell, ;
    bidirectional: A boolean indicating whether the LSTM should be bidirectional;
    batch_first: the input and output tensors are provided in the shape (batch_size, seq_length, input_size).;
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
    1.Op Introduction
    Perform opencv MatchTemplate operation.

    2.Math formula
    ```math
            R(x, y) = \sum_{i=0}^{T_w-1} \sum_{j=0}^{T_h-1} I(x+i, y+j) \cdot T(i, j)
    ```
    where:
    R(x, y) is the result of the match at position(x, y).
    I is the input image.
    T is the template image.
    T_w and T_h are the width and height of the template.

    3.activation and weight
    input(act.): input tensor(source image).;
    match(w.): the template image that will be matched against the input image.;
    table(w.): store a lookup table that may assist in optimizing the matching process.;
    mantissa_table(w.): stores a table of mantissa values used in calculations to improve precision or to handle specific numerical representations.;

    4.attributes
    mode: the method of template matching to be used (e.g., correlation, squared difference, etc.).;
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
    1.Op Introduction
    Returns a tensor with the same type as the input, with a new shape
    specified by the shape argument.

    2.Math formula
    ```math
            output[i_1, i_2, i_3,...i_k] = input[i_1 mod d_1, i_2 mod d_2, i_3 mod d_3,..., i_k mod d_k]
    ```
    where d_j represents the corresponding dimension size of the input tensor after tiling.

    3.activation and weight
    input(act.): input tensor.;
    tileT(w.): how many times to replicate the input tensor along each dimension.;
    buffer(w.): temporary storage of intermediate results or states during the computation.;

    4.attributes
    tile: the number of times to replicate the input tensor along each dimension.;
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
    1.Op Introduction
    Perform Gather operation on the given axis.

    2.Math formula
    ```math
            output = input[indices]
    ```

    3.activation and weight
    input(act.): input tensor.;
    indices(w.): the indices of the elements to be gathered from the input tensor.;
    buffer(w.): temporary storage of intermediate results or states during the computation.;

    4.attributes
    axis: the dimension of the input tensor.;
    if_neg_index: how negative indices should be handled.;
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
    1.Op Introduction
    Perform GatherElements operation on the given axis.

    2.Math formula
    ```math
            output[i_1, i_2, i_3,...i_k] = input[i_1, i_2, i_3,..., indices[i_k]]
    ```

    3.activation and weight
    input(act.): input tensor.;
    indices(w.): the indices of the elements to be gathered from the input tensor.;
    indices_coeff(w.): a scaling or modifying factor for the indices.;
    buffer(w.): temporary storage of intermediate results or states during the computation.;

    4.attributes
    axis: the dimension of the input tensor.;
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
    1.Op Introduction
    This operator is the inverse of ScatterND.

    2.Math formula
    ```math
            output_i = input[indices_i]
    ```

    3.activation and weight
    input_data(act.): input tensor.;
    indices(w.): which elements to gather from the input tensor.;

    4.attributes
    indice_dims: the number of dimensions in the indices tensor.;
    batch_dims: the number of batch dimensions in the input tensor.;
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
    1.Op Introduction
    Parametric Rectified Linear Unit is an activation function.

    2.Math formula
    ```math
            f(x) = slope * x   for x < 0
            f(x) = x           for x >= 0
    ```

    3.activation and weight
    input(act.): input tensor.;
    slope(w.): the activation function for negative input values.;

    4.attributes
    rshift: the number of bits to right-shift the quantized values.;
    rshift_pos: This attribute defines the right shift for the positive output values during quantization.;
    multiplier_pos: This parameter specifies the multiplier for the positive part of the output, used in the context of quantization.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Generic Cpu Op.

    2.Math formula
    output = GenericCpu(inputs, param)

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    cpu_op_name: the name of the CPU-specific operator to be executed.;
    param: a set of parameters.;
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
    1.Op Introduction
    Perform Interp on input.

    2.Math formula
    ```math
            H' = H x scale_h
            W' = W x scale_w
    ```

    3.activation and weight
    input(act.): input tensor.;
    shapeT(act.): an optional input tensor that specifies the desired shape for the output tensor.;
    buffer(w.): temporary storage of intermediate results or states during the computation.;

    4.attributes
    scale_h: the scaling factor for the height (number of rows) of the input tensor.;
    scale_w: the scaling factor for the width (number of columns) of the input tensor.;
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    coord_mode: whether the coordinates are normalized (ranging from 0 to 1) or absolute (based on pixel indices).;
    ginfo: associated with layer grouping information.;
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$shapeT,
    AnyTensorOrNone:$buffer,
    F64Attr:$scale_h,
    F64Attr:$scale_w,
    Tpu_ResizeModeAttr:$mode,
    Tpu_ResizeCoordModeAttr:$coord_mode,
    DefaultValuedAttr<BoolAttr, "false">:$ppl_flag,
    OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ReduceOp: Tpu_Op<"Reduce", [
  DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>]> {
  let summary = "Reduce operator";
  let description = [{
    1.Op Introduction
    Computes the mean/max/prod/sum of the input tensor's element along the provided axes.

    2.Math formula
    ```math
            1.Sum
                output[i_1, i_2, i_3,..., i_k] = \sum{j in A}input[i_1, i_2,..., j, ..., i_n]
                where ( A ) is the set of axes to reduce.
            2.Mean
                output[i_1, i_2, i_3,..., i_k] = 1 / count (\sum{j in A}input[i_1, i_2,..., j, ..., i_n])
                where count is the number of elements being summed along the axes ( A ).
            3.Max
                output[i_1, i_2, i_3,..., i_k] = max{j in A}input[i_1, i_2,..., j, ..., i_n]
            4.Product
                output[i_1, i_2, i_3,..., i_k] = \prod_{j=1}^m(input[i_1, i_2,..., i_k, j])
    ```

    3.activation and weight
    input(act.): input tensor.;
    reciprocal_mantissa_table(w.): store the mantissa values of the reciprocal.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attributes
    axes: the dimensions (axes) of the input tensor that should be squeezed (removed).;
    keepdims: whether to retain the dimensions of the input tensor in the output.
               If true, will have the same number of dimensions as the input tensor.;
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    multiplier: Floating-point multiplication operations are usually converted to fixed-point multiplication operations.;
    rshift: the number of bits to right-shift the quantized values.;
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

  let hasCanonicalizer = 1;
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
    reduce_attr_t parseParam();
  }];
}

def Tpu_ShapeReduceOp: Tpu_Op<"ShapeReduce", [
    ShapeProducer, ShapeConsumer,DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "ShapeReduce operator";

  let description = [{
    1.Op Introduction
    ShapeReduce Operation on input.

    2.Math formula
    \text{output} = \text{scale} \times \mathrm{reduce}\left(\text{input}, \text{axes}; \, \text{mode}\right)

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    axes: the dimensions (axes) of the input tensor that should be squeezed (removed).;
    keepdims: whether to retain the dimensions of the input tensor in the output.
               If true, will have the same number of dimensions as the input tensor.;
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    scale:scalar.;
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
    1.Op Introduction
    Computes the indices of the min/max/ of the input tensor's element along the provided axis.

    2.Math formula
    ```math
        maximum operation:
            output_max[i_1, i_2, i_3,..., i_k] = arg max{j}(input[i_1, i_2,..., i_k, j])
        minimum operation:
            output_min[i_1, i_2, i_3,..., i_k] = arg min{j}(input[i_1, i_2,..., i_k, j])
    ```
    where, ( j ) represents the index along the specified axis.

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    axis: the dimension of the input tensor.;
    keepdims: whether to retain the dimensions of the input tensor in the output.
               If true, will have the same number of dimensions as the input tensor.;
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    select_last_index: select the last index of the minimum or maximum value when multiple  along the specified axis.;
    use_int_input: choose if use i32/i16 as input, due to BF16 chip arch limit.;
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis,
    BoolAttr:$keepdims,
    ArgModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "true">:$select_last_index,
    DefaultValuedAttr<BoolAttr, "false">:$use_int_input
  );

  let results = (outs
    AnyRankedTensor:$indices,
    AnyTensorOrNone:$values
  );
}

def Tpu_WhereOp: Tpu_Op<"Where", [
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Where operator";
  let description = [{
    1.Op Introduction
    Return elements, either from X or Y, depending on condition.

    2.Math formula
    ```math
            output = tbrn if condition else fbrn
    ```

    3.activation and weight
    cond(act.): a tensor that serves as the condition for selecting elements from the true branch (tbrn) or the false branch (fbrn).;
    tbrn(w.): the tensor that will be selected when the condition is true.;
    fbrn(w.): the tensor that will be selected when the condition is false.;
    buffer(w.): temporary storage of intermediate results or states during the LSTM computation.;

    4.attributes
    x_is_const: the tensor for the true branch (tbrn) is a constant.;
    y_is_const: the tensor for the false branch (fbrn) is a constant.;
    x_const_val: the constant value to be used for the true branch if tbrn is not provided or is constant.;
    y_const_val: the constant value to be used for the false branch if fbrn is not provided or is constant.;
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
    1.Op Introduction
    Return elements, either from X or Const, depending on condition.

    2.Math formula
    ```math
                     brn                if inversed and cond=0
            output = brn + const_val    if inversed and cond!=0
                     brn + const_val    if !inversed and cond!=0
                     brn                if !inversed and cond=0
    ```
        If inversed is true, the operation fills the elements of brn where cond is zero with const_val, while leaving other elements unchanged.
        If inversed is false, the operation fills the elements of brn where cond is non-zero with const_val, while leaving other elements unchanged.

    3.activation and weight
    cond(act.): a tensor that serves as the condition for selecting elements from the true branch (tbrn) or the false branch (fbrn).;
    brn(w.): the input tensor from which elements will be selected based on the condition provided by the cond tensor.;

    4.attributes
    inversed: whether the mask should be inverted.;
    const_val: specifies the constant value to be added to each element of the input tensor(positive, negative, or zero).;
  }];
  let arguments = (ins
    AnyRankedTensor:$cond,
    AnyRankedTensor:$brn,
    BoolAttr:$inversed,
    F64Attr:$const_val
  );
  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizer = 1;
}

def Tpu_CompareOp: Tpu_Op<"Compare", [
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport", "assign_sec_info"]>,
  DeclareOpInterfaceMethods<DynLocalGenInterface>]> {
  let summary = "Compare operator";
  let description = [{
    1.Op Introduction
    Returns the tensor resulted from performing the compare
    operation elementwise on the input tensors A and B.

    2.Math formula
    ```math
            output[i] = 1 if lhs[i] mode rhs[i] is true
                        0 otherwise
    ```

    3.activation and weight
    lhs(act.): the first input tensor used as the left operand in the element-wise comparison.;
    rhs(act.): the second input tensor used as the right operand in the element-wise comparison.;

    4.attributes
    mode: the type of comparison to be performed between the two input tensors.
          mdoe include Equal, Not Equal, Less Than, Less Than or Equal, Greater Than and Greater Than or Equal;
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
    1.Op Introduction
    Returns the tensor resulted from performing the compare
    operation elementwise on the input tensors A and Const.

    2.Math formula
    ```math
            output[i] = 1 if input[i] mode const_val is true
                        0 otherwise
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    mode: the type of comparison to be performed between the two input tensors.
          mdoe include Equal, Not Equal, Less Than, Less Than or Equal, Greater Than and Greater Than or Equal;
    const_val: specifies the constant value to be added to each element of the input tensor(positive, negative, or zero).;
    inversed: whether the mask should be inverted.;
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
    1.Op Introduction
    layer normalization

    2.Math formula
    ```math
        1.Normalization
            mean = 1 / H \sum{j=1, H} input[j]
            var = 1 / H \sum{j=1, H} (input[j] - mean) ^ 2
        2.Layer Normalized Output
            output[i] = weight * (input[i] - mean) / sprt(var + eps) + bias
    ```

    3.activation and weight
    input(act.): input tensor;
    weight(w.): weight tensor;
    bias(w.): the learnable bias of the module of shape (out_channels).;
    // cv18xx
    table(w.): store a lookup table that may assist in optimizing the matching process.;
    mantissa_table(w.): stores a table of mantissa values used in calculations to improve precision or to handle specific numerical representations.;

    4.attributes
    normalized_shape: the shape of the input tensor that will be normalized.;
    axis: the dimension of the input tensor.;
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
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
    1.Op Introduction
    aten::index_put_
    update specific elements of an input tensor at given indices with new values.

    2.Math formula
    ```math
        if accumulate
            input[indices] += values
        else
            input[indices] = values
    ```

    3.activation and weight
    input(act.): input tensor;
    indices(w.): the indices of the elements in the input tensor that should be updated.;
    values(w.): the new values that will replace the existing values in the input tensor at the specified indices.;
    //for accumulate is True
    buffer(w.): temporary storage of intermediate results or states during the LSTM computation.;

    4.attributes
    accumulate: whether the operation should accumulate values at the specified indices or replace them.;
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
    1.Op Introduction
    instance normalization.

    2.Math formula
    ```math
        output[i] = weight * (input[i] - mean) / sprt(var + eps) + bias
    ```

    3.activation and weight
    input(act.): input tensor;
    weight(w.): weight tensor;
    bias(w.): the learnable bias of the module of shape (out_channels).;
    // cv18xx
    table(w.): store a lookup table that may assist in optimizing the matching process.;
    mantissa_table(w.): stores a table of mantissa values used in calculations to improve precision or to handle specific numerical representations.;

    4.attributes
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
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
    1.Op Introduction
    group normalization

    2.Math formula
    ```math
            mean_g = 1 / ((C / num_groups) * H * W) \sum{i=1, C/num_groups} \sum{j=1, H} \sum{k=1, W} input_{n,c,j,k}
            var_g = 1 / ((C / num_groups) * H * W) \sum{i=1, C/num_groups} \sum{j=1, H} \sum{k=1, W} (input_{n,c,j,k} - mean_g) ^ 2
            output_{n,c,j,k} = weight * (input_{n,c,j,k} - mean_g) / sqrt(var_g + eps) + bias
    ```

    3.activation and weight
    input(act.): input tensor;
    weight(w.): weight tensor;
    bias(w.): the learnable bias of the module of shape (out_channels).;
    // cv18xx
    table(w.): store a lookup table that may assist in optimizing the matching process.;
    mantissa_table(w.): stores a table of mantissa values used in calculations to improve precision or to handle specific numerical representations.;

    4.attributes
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
    num_groups: the number of groups to divide the input channels into for normalization.;
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
    1.Op Introduction
    Given an input and a flow-field grid, computes the output
    using input values and pixel locations from grid.

    2.Math formula
    ```math
            output[N, C, H', W'] = input[C, grid[N, H', W', 1], grid[N, H', W', 0]]
    ```

    3.activation and weight
    input(act.): input tensor.;
    grid(w.): The flow-field grid tensor that defines the pixel locations for sampling.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attributes
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    padding_mode: padding mode for outside grid values, Int attribute [0, 1, 2],
                                representing 'zero' | 'boundary' | 'reflection't.;
    align_corners: whether to align the corners of the input and output tensors.;
    scale: scalar.;
    mean: mean values to subtract from each channel for normalization.;
    need_permute: whether permutation of the output tensor is required;
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
    1.Op Introduction
    pixel normalization (normalize along c-axis)

    2.Math formula
    ```math
            norm_{n, i, j} = sqrt(1 /C \sum{c=1, C}input_{n, c, i, j} ^ 2) + eps
            output_{n, c, i, j} = weight * input_{n, c, i, j} / norm_{n, i, j} + bias
    ```

    3.activation and weight
    input(act.): input tensor;
    weight(w.): weight tensor;
    bias(w.): the learnable bias of the module of shape (out_channels).;
    // cv18xx
    table(w.): store a lookup table that may assist in optimizing the matching process.;
    mantissa_table(w.): stores a table of mantissa values used in calculations to improve precision or to handle specific numerical representations.;

    4.attributes

    eps: a small constant added to the denominator during normalization to prevent division by zero.;
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
    1.Op Introduction
    duplicating tensor data from an input to an output buffer.

    2.Math formula
    output = Copy(input, shape, input_stride, output_stride)

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    shape: 0: keep dim from input; -1: left dim from input.;
    input_stride: input data stride(saved as I64ArrayAttr).
    output_stride: output data stride(saved as I64ArrayAttr).
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Performs csc operation on inputs.

    2.Math formula
    ```math
            output = Csc(input, pixel_format, y_align, w_align, channel_align)
    ```
    3.activation and weight
    input(act.): input tensor;

    4.attribute
    channel_order: the order of the color channels in the output tensor.(e.g., RGB, BGR).;
    pixel_format: required, pixel format type.;
    aligned:  whether the pixel data should be aligned.;
    pixel_type: the type of pixel data.;
    y_align: width alignment of channel y.;
    w_align: width alignment of channel uv.;
    channel_align: alignment of channel.;
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
    1.Op Introduction
    Performs scale on input, y = input * scale + bias.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.


    2.Math formula
    ```math
            output = input * scale + bias
    ```
    3.activation and weight
    input(act.): input tensor;
    table(w.): store a lookup table that may assist in optimizing the matching process.;

    4.attributes
    scale: each channel scale.;
    bias: each channel bias.;
    sign: if output is signed.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Swap Channel on input.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.

    2.Math formula
    ```math
            output(h, w, c) = input(h, w, channel_order)
    ```
    3.activation and weight
    input(act.): input tensor;

    4.attributes
    channel_order: channel swap order.;
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
    1.Op Introduction
    a dimension-swapping operation based on a specified offset.

    2.Math formula
    ```math
            output = SwapDimInner(input, offset)
    ```
    3.activation and weight
    input(act.): input tensor;

    4.attributes
    offset: the position at which the input tensor is split.;

  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    I64ArrayAttr:$offset
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ScatterElementsOp: Tpu_Op<"ScatterElements", [
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]>  {
  let summary = "ScatterElements op";
  let description = [{
    1.Op Introduction
    ScatterElements takes three inputs data, updates, and indices of the same rank r >= 1 and an optional attribute axis that
    identifies an axis of data (by default, the outer-most axis, that is axis 0). The output of the operation is produced by
    creating a copy of the input data, and then updating its value to values specified by updates at specific index
    positions specified by indices. Its output shape is the same as the shape of data.

    2.Math formula
    ```math
            output = ScatterElements(input[axis], updates, indices)
    ```
    3.activation and weight
    input(act.): input tensor;
    indices(w.): Tensor of int32/int64 indices, of r >= 1 (same rank as input).
             All index values are expected to be within bounds [-s, s-1] along axis of size s.
    updates(w.): Tensor of rank r >=1 (same rank and shape as indices).
    indices_coeff(w.): a scaling or modifying factor for the indices.;
    buffer(w.): temporary storage of intermediate results or states during the LSTM computation.;

    4.attributes
    axis: the dimension of the input tensor.;
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$indices,
    AnyRankedTensor:$updates,
    AnyTensorOrNone:$indices_coeff,
    AnyTensorOrNone:$buffer,
    I64Attr:$axis,
    DefaultValuedAttr<I64Attr, "0">:$reduction,
    DefaultValuedAttr<BoolAttr, "false">:$nc_can_split
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_ScatterNDOp: Tpu_Op<"ScatterND", [
  DeclareOpInterfaceMethods<TypeInterface>]>  {
  let summary = "ScatterND operator";
  let description = [{
    1.Op Introduction
    The output of the operation is produced by creating a copy of the input data,
    and then updating its value to values specified by updates at
    specific index positions specified by indices.

    2.Math formula
    ```math
            output = ScatterND(input_data[indices], updates, reduction)
    ```

    3.activated and weight
    input_data(act.): input tensor;
    indices(w.): Tensor of rank q >= 1.;
    updates(w.): Tensor of rank q + r - indices_shape[-1] - 1.;
    buffer(w.): temporary storage of intermediate results or states during the LSTM computation.;

    4.attributes
    reduction: Type of reduction to apply: none (0 default), add(1), sub(2), max(3), min(4), mul(5).;
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
    1.Op Introduction
    RoiAlign consumes an input tensor X and region of interests
    (rois) to apply pooling across each RoI.

    2.Math formula
    ```math
        1.ROI coordinate scaling[x1, y1, x2, y2]
            x_scaled = x x spatial_scale
            y_scaled = y x spatial_scale
        2.Delineation of grid sub-areas
            bin_height = (y2_scaled - y1_scaled) / output_height
            bin_width  = (x2_scaled - x1_scaled) / output_width
        3.align_corners -> true
            x_grid = x1_scaled + (i + 0.5) x bin_width
            y_grid = y1_scaled + (j + 0.5) x bin_height
        output = RoiAlign(input, rois, output_height, output_width, sampling_ratio, spatial_scale, align_corners)
    ```

    3.activated and weight
    input(act.): input tensor(4D);
    rois: RoIs (Regions of Interest) to pool over;
          rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...].

    4.attributes
    mode: the pooling mode to be used when extracting features from the RoIs on the input feature maps.;
    output_height: the height of the output feature maps.;
    output_width: the width of the output feature maps.;
    sampling_ratio: the number of sampling points in each direction (height and width).;
    spatial_scale: a scaling factor that maps the input coordinates (RoIs) to the input feature map's scale.;
    align_corners: whether to align the corners of the input and output tensors.;
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

def Tpu_RoiExtractorOp: Tpu_Op<"RoiExtractor", [
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "RoiExtractor operator";
  let description = [{
    1.Op Introduction
    RoiExtractor consumes an input tensor X and region of interests
    (rois) to apply pooling across each RoI.

    2.Math formula
      ```math
        1.ROI coordinate scaling[x1, y1, x2, y2]
            x_scaled = x x spatial_scale
            y_scaled = y x spatial_scale
        2.Delineation of grid sub-areas
            bin_height = (y2_scaled - y1_scaled) / output_height
            bin_width  = (x2_scaled - x1_scaled) / output_width
        3.align_corners -> true
            x_grid = x1_scaled + (i + 0.5) x bin_width
            y_grid = y1_scaled + (j + 0.5) x bin_height
        output = RoiAlign(input, rois, output_height, output_width, sampling_ratio, spatial_scale, align_corners)
    ```

    3.activation and weight
    input(act.): input tensor;
    rois(act.): RoIs (Regions of Interest) to pool over;
                rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...]
    target_lvls(act.): 1-D tensor with each element denoting the index of the corresponding image in the batch.

    4.attributes
    mode: the pooling mode to be used when extracting features from the RoIs on the input feature maps.;
    num_levels: the number of feature levels.;
    output_height: the height of the output feature maps.;
    output_width: the width of the output feature maps.;
    sampling_ratio: the number of sampling points in each direction (height and width).;
    spatial_scale: a scaling factor that maps the input coordinates (RoIs) to the input feature map's scale.;
    align_corners: whether to align the corners of the input and output tensors.;
    is_static: whether the RoI extraction process has a fixed (static) configuration.;
  }];

  let arguments = (ins
    // AnyRankedTensor:$feature_0,
    // AnyRankedTensor:$feature_1,
    // AnyRankedTensor:$feature_2,
    // AnyRankedTensor:$feature_3,
    AnyRankedTensor:$rois,
    AnyRankedTensor:$target_lvls,
    Variadic<AnyRankedTensor>:$inputs,
    RoiAlignModeAttr:$mode,
    I64Attr:$num_levels,
    I64Attr:$output_height,
    I64Attr:$output_width,
    I64Attr:$sampling_ratio,
    F64ArrayAttr:$spatial_scales,
    BoolAttr:$align_corners,
    BoolAttr:$is_static
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_YoloDetectionOp : Tpu_Op<"YoloDetection"> {
  let summary = "YoloDetection operator";
  let description = [{
    1.Op Introduction
    Perform yolo detection on feature map.

    2.Math formula
    ```math
        1.Feature Map output
            raw_predictions = {(b_i, c_i, p_i) | i = 1, 2,...,N}
            b_i is the bounding box coordinates, c_i is the i-th class score, p_i is the i-th obj score.
        2.Apply Objectness Threshold
            filtered_predictions = {(b_i, c_i, p_i) | p_i >= obj_threshold}
        3.Non-Maximum Suppression(NMS)
            nms_output = NMS(filtered_predictions, nms_threshold)
        4.Top K Detections
            output = top_k(nms_output, keep_topk)
    ```

    3.activation and weight
    inputs(act.): input tensor;
    buffer(w.): temporary storage of intermediate results or states during the LSTM computation.;

    4.attributes
    net_input_h: The height of the input image.;
    net_input_w: The width of the input image;
    nms_threshold: The threshold used for Non-Maximum Suppression (NMS).;
    obj_threshold: The minimum confidence score required for an object detection to be considered valid.;
    keep_topk: The maximum number of detections to keep after applying NMS.;
    anchors: A list of anchor box dimensions.;
    version: The version of the YOLO model being used.;
    class_num: The number of classes that the YOLO model can predict.;
    num_boxes: The number of bounding boxes that the model predicts for each grid cell in the feature map.;
    agnostic_nms: whether to use class-agnostic NMS.;
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
    1.Op Introduction
    Intended for use with MultiBox detection method to generate prior.

    2.Math formula
    ```math
        1.Raw Detection Output
            raw_output = {(b_i, c_i) | i = 1, 2,...,N}
            b_i is the bounding box coordinates, c_i is the i-th confidence score.
        2.Apply Confidence threshold
            filtered_output = {(b_i, c_i) | c_i >= confidence_threshold}
        3.Non-Maximum Suppression(NMS)
            nms_output = NMS(filtered_output, nms_threshold)
        4.Top K detections
            output = top_k(nms_output, top_k)
    ```

    3.activation and weight
    inputs(act.): input tensor;
    buffer(w.): temporary storage of intermediate results or states during the LSTM computation.;

    4.attributes
    num_classes: total number of classes, including the background class.;
    background_label_id: background class, differentiate between detected objects and the background.;
    nms_threshold: The threshold used for Non-Maximum Suppression (NMS).;
    top_k: The maximum number of predictions to be considered for each image.;
    code_type: the encoding type for the bounding box coordinates.;
    keep_top_k: The number of top scoring detections to keep after applying NMS.;
    confidence_threshold: The minimum confidence score required for a detection to be considered valid.;
    share_location: whether the bounding box locations are shared across different classes.;
    variance_encoded_in_target: whether the variance for bounding box predictions is encoded in the target.;
    eta: adjusts the confidence scores during NMS.;
    onnx_nms: configuration for ONNX compatibility (default is 1).;
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
    1.Op Introduction
    Takes a tensor as input and outputs an 1D int tensor containing the shape of the input tensor.

    2.Math formula
    ```math
            output = shape(input[d1, d2,...,dn])
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Tpu_ConstantFillOp: Tpu_Op<"ConstantFill", [
    ShapeConsumer,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "constant fill operator";
  let description = [{
    1.Op Introduction
    fill the constant value

    2.Math formula
    ```math
        output = value * ones(shape(input))
    ```
    where, ones(shape(input)) generates a tensor of the same shape as the input tensor, filled with ones.

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    value: the constant value that will fill the output tensor.;
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
    1.Op Introduction
    takes data from host mem to device mem

    2.Math formula
    ```math
        output = Host2Device(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Tpu_Device2HostOp : Tpu_Op<"Device2Host", [
    ShapeProducer,
    DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "Device2Host Operation";
  let description = [{
    1.Op Introduction
    takes data from device mem to host mem

    2.Math formula
    ```math
        output = Device2Host(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Tpu_IfOp : Tpu_Op<"If", [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
              RecursiveMemoryEffects,
               NoRegionArguments]> {
  let summary = "if operation";
  let description = [{
    1.Op Introduction
    If conditional

    2.Math formula
    ```math
            output = then_branch if cond is true
                     else_branch if cond is false
    ```

    3.activation and weight
    cond(act.): which branch of execution to follow.;
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
    1.Op Introduction
     The deform gather operator for deform_conv2d.

    2.Math formula
    ```math
            output = input(offset) * use_mask * mask
    ```

    3.activation and weight
    input(act.): input tensor.;
    offset(w.): the learnable offsets of the module of shape.;
    mask(act.): place the pooled values in the output tensor.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attributes
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    strides: the stride for the cross-correlation, a single number or a tuple.;
    dilations: controls the spacing between the kernel points;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.;
    deform_group: (optional)Number of blocked connections from input channels to output channels. Default: 1.;
    use_mask: whether use mask for input tensor.;
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
    1.Op Introduction
    Custom operator

    2.Math formula
    ```math
            output = CustomFunction(inputs, name, params)
    ```

    3.activation and weight
    inputs(act.): input tensor.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attributes
    name: the name of the custom operation to be executed.;
    params: A dictionary of parameters.;
    ginfo: associated with layer grouping information.;
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
    1.Op Introduction
    Convert weight tensor to activation tensor

    2.Math formula
    ```math
            \text{output}[i] = \text{input}[i] \quad \forall i
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Tpu_Space2BatchOp: Tpu_Op<"Space2Batch"> {

  let summary = "Space2Batch operator";

  let description = [{
    1.Op Introduction
    Refer to `https://www.tensorflow.org/api_docs/python/tf/space_to_batch`

    2.Math formula
    ```math
        h_padding = h + pad_top + pad_bottom,
        w_padding = w + pad_left + pad_right,
        [n, c, h, w] => [n, c, h_padding, w_padding]
        =>[n * block_h * block_w, c, h_padding / block_h, w / block_w];
        h_padding and w_padding should satisfy:
        h_padding % block_h = 0, w_padding % block_w = 0
        The format of input or output is NCHW.
    ```

    3.activation and weight
    inputs(act.): input tensor.;
    buffer(w.): temporary storage of intermediate results or states during computation.;

    4.attributes
    block_h: The height of the blocks used to rearrange the depth into spatial dimensions.;
    block_w: The width of the blocks used to rearrange the depth into spatial dimensions.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.;
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
    1.Op Introduction
    Refer to `https://www.tensorflow.org/api_docs/python/tf/batch_to_space`

    2.Math formula
    ```math
        h_croping = h * block_h - crop_top - crop_bottom,
        w_croping = w * block_w - crop_left - crop_right,
        [n, c, h, w] => [n / (block_h * block_w), c, h * block_h, w * block_w]
        => [n / (block_h * block_w), c, h_croping, w_croping];
        The format of input or output is NCHW.
    ```

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    block_h: The height of the blocks used to rearrange the depth into spatial dimensions.;
    block_w: The width of the blocks used to rearrange the depth into spatial dimensions.;
    crops: It contains four ints with top, left, bottom, right.;
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
    1.Op Introduction
     identity operator.

     2.Math formula
    ```math
        output = Identity(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$input
  );

  let results = (outs Variadic<AnyRankedTensor>:$output);
}

def Tpu_NmsOp : Tpu_Op<"Nms"> {
  let summary = " NMS operator";
  let description = [{
    1.Op Introduction
    tpu nms

    2.Math formula
    ```math
        IOU(a, b) = Area(a \cap b) / Area(a \cup b)
    ```

    3.activation and weight
    input(act.): Variadic input tensor.;
    buffer(w.): a temporary storage area for intermediate calculations or results during the NMS process.;

    4.attributes
    center_point_box: whether the bounding boxes are defined by their center points.;
    max_output_size: the maximum number of boxes to be output after NMS.;
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
    1.Op Introduction
    A simplification of the original layer normalization (LayerNorm).
    Only normalize the last dimension of tensor.

    2.Math formula
    ```math
        output = gamma * input / sqrt(mean(input ^ 2) + epsilon)
    ```

    3.activation and weight
    input(act.): input tensor.;
    gamma(w.): scalar.;

    4.attributes
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
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
    1.Op Introduction
    Generic Looping construct, support while/do_while/for/forerver etc:

    2.Math formula
    none

    3.activation and weight
    AnyTensor;
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
    1.Op Introduction
    increase by 1 in-place

    2.Math formula
    ```math
            output = AutoIncrease(input, const_val)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    const_val: specifies the constant value to be added to each element of the input tensor(positive, negative, or zero).;
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
    1.Op Introduction
    for to alloc address, and can modify the data(with WeightOp to init it)

    2.Math formula
    ```math
            output = D2D(input, const_val)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    const_val: specifies the constant value to be added to each element of the input tensor(positive, negative, or zero).;
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
    1.Op Introduction
    Arithmetic implementation for simple data in host memory such as 'Shape, Index, Stride' etc.

    2.Math formula
    ```math
            output = ShapeArith(input, type)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    type: the kind of arithmetic operation.;
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
    1.Op Introduction
    Returns the cumulative sum of elements of input in the dimension dim.

    2.Math formula
    ```math
            output[i] = \sum{j=0, i}input[j]
    ```

    3.activation and weight
    input(act.): input tensor.;
    dim(w.): If set to 0, computed across rows, If set to 1, computed across columns.;

    4.attributes
    axis: the dimension of the input tensor.;
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
    1.Op Introduction
    layer normalization in train.

    2.Math formula
    ```math
        1.Normalization
            mean = 1 / H \sum{j=1, H} input[j]
            var = 1 / H \sum{j=1, H} (input[j] - mean) ^ 2
        2.Layer Normalized Output
            output[i] = weight * (input[i] - mean) / sprt(var + eps) + bias
    ```

    3.activation and weight
    input(act.): input tensor.;
    weight(w.): weight tensor.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attributes
    normalized_shape: the shape of the input tensor dimensions.;
    axis: the dimension of the input tensor.;
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
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
    1.Op Introduction
    layer normalization

    2.Math formula
    ```math
        1.gradient input
            grad_input = 1 / N(weight / sqrt(variance + eps) 1 / N \sum{i=1, N}grad_out)
                            - (input - mean) / N * \sum(i=1, N)weight * grad_out / sqrt(variance + eps)
        2.gradient weight
            grad_weight = \sum(i=1, N)grad_out * (input - mean) / sqrt(variance + eps)
        3.gradient bias
            grad_bias = \sum(i=1, N)grad_out
    ```

    3.activation and weight
    input(act.): input tensor.;
    grad_out(w.): the gradient of the loss with respect to the output of the layer normalization.;
    mean(w.): mean values to subtract from each channel for normalization.;
    variance(w.): adjust the predicted boxes during the training process.;
    weight(w.): weight tensor.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attributes
    normalized_shape:  the shape of the input tensor dimensions.;
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
    1.Op Introduction
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    2.Math formula
    ```math
        output = \frac{input - \mathrm{E}[input]}{ \variance + \epsilon} * \gamma + \beta
    ```

    3.activation and weight
    input(act.): input tensor;
    mean(w.): mean of input tensor in dim C;
    variance(w.): the spread of the input tensor values along the channel dimension for each mini-batch.;
    gamma(w.): scalar;
    beta(w.): scalar;
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).

    4.attribute
    epsilon;
    momentum: Momentum is a hyperparameter that controls the moving average of the mean and variance of the input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$mean,
    AnyTensor:$var,
    AnyTensorOrNone:$gamma,
    AnyTensorOrNone:$beta,
    AnyTensorOrNone:$running_status_buffer, // split h for multicore; running_mean and var
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "1e-05">:$epsilon,
    DefaultValuedAttr<F64Attr, "0.1">:$momentum
  );
  let results = (outs
    AnyTensor:$output,
    AnyTensor:$mean_out,
    AnyTensor:$saved_invstd,
    AnyTensor:$running_mean,
    AnyTensor:$running_var
  );
}


def Tpu_BatchNormBwdOp: Tpu_Op<"BatchNormBwd" ,[
  DeclareOpInterfaceMethods<LocalGenInterface, ["AllowDataSplit", "LocalGenSupport"]>
]> {
  let summary = "BatchNormalization operation";
  let description = [{
    1.Op Introduction
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    2.Math formula
    ```math
        output = \frac{input - \mathrm{E}[input]}{ \variance + \epsilon} * \gamma + \beta
    ```

    3.activation and weight
    grad_out(act.): the gradients of the output of the Batch Normalization layer with respect to the loss. ;
    input(act.): input tensor;
    weight_opt(w.): the optimized weight (or scale) parameter.;
    saved_mean(w.): the mean of the input tensor calculated during the forward pass.;
    saved_invstd(w.): the saved inverse standard deviation (or the reciprocal of the standard deviation) of the input tensor,
                      also computed during the forward pass.;
    buffer(w.): serve as a temporary storage or workspace that may be used during the computation of gradients.;
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).

    4.attribute
    epsilon;
  }];
  let arguments = (ins
    AnyTensor:$grad_out,
    AnyTensor:$input,
    AnyTensorOrNone:$weight_opt,
    AnyTensorOrNone:$saved_mean,
    AnyTensorOrNone:$saved_invstd,
    AnyTensorOrNone:$buffer,
    DefaultValuedAttr<F64Attr, "1e-05">:$epsilon
  );
  let results = (outs
    AnyTensor:$grad_in,
    AnyTensorOrNone:$weight_grad,
    AnyTensorOrNone:$bias_grad
  );
}

def Tpu_WhereBnbwdOp: Tpu_Op<"WhereBnbwdOp"> {
  let summary = "WhereBnbwd operation";
  let description = [{
    1.Op Introduction
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    2.Math formula
    ```math
        output = \frac{input - \mathrm{E}[input]}{ \sqrt{\mathrm{Var}[input] + \epsilon}} * \gamma + \beta
    ```
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).

    3.activation and weight
    where_output(act.): an optional intermediate output from a "where" operation applied during the forward pass.;
    where_grad_out(act.): the gradients computed with respect to the "where" output.;
    bnbwd_input(act.): the primary input to the batch normalization backward operation.;
    bnbwd_weight(w.): re-scale the normalized input.;
    bnbwd_bias(w.): re-centering the normalized values.;
    bnbwd_saved_mean(w.): the per-dimension mean values computed during the forward pass.;
    bnbwd_saved_invstd(w.): stores the inverse standard deviation computed during the forward pass.;
    buffer(w.): serve as a temporary storage or workspace that may be used during the computation of gradients.;

    4.attributes
    epsilon:;
    do_recompute: whether recomputed during the backward pass instead of being stored from the forward pass.;
  }];
  let arguments = (ins
    AnyTensorOrNone:$where_output,
    AnyTensor:$where_grad_out,
    AnyTensor:$bnbwd_input,
    AnyTensorOrNone:$bnbwd_weight,
    AnyTensorOrNone:$bnbwd_bias,
    AnyTensorOrNone:$bnbwd_saved_mean,
    AnyTensorOrNone:$bnbwd_saved_invstd,
    AnyTensorOrNone:$buffer,
    DefaultValuedAttr<F64Attr, "1e-05">:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$do_recompute
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
    1.Op Introduction
    layer normalization

    2.Math formula
    ```math
            output = grad_output[indices]
    ```

    3.activation and weight
    grad_output(act.): the gradient of the loss with respect to the output.;
    indices(w.): the indices of the input tokens or items.;

    4.attributes
    num_weights: the total number of embedding weights.;
    padding_idx: a index for padding.;
    scale_grad_by_freq: whether the gradient should be scaled by the inverse frequency of that token.;
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
    1.Op Introduction
    reorder Weight.

    2.Math formula
    ```math
            output = Reorder(input, reorder_mode)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    reorder_mode: rearranging the weight tensor, such as sorting, shuffling, or applying a specific permutation.;
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
    1.Op Introduction
    Integrates some operations related to softmax backward.

    2.Math formula
    ```math
            grad_input[i] = softmax(output)[i] * (grad_output[i] - \sum{j}grad_output[j] * softmax(output)[j])
    ```

    3.activation and weight
    grad_output(act.): the gradient of the loss with respect to the output.;
    output(act.): output tensor.;

    4.attributes
    dim: If set to 0, computed across rows, If set to 1, computed across columns.;
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
  let description = [{
    1.Op Introduction
    Gradient of Weight in Convolution Backward.

    2.Math formula
    ```math
            \frac{\partial L}{\partial W} = \sum_{n=1}^{N} \sum_{c=1}^{C_{in}} \sum_{h=1}^{H} \sum_{w=1}^{W} \text{input}[n, c, h, w] \cdot \text{grad\_out}[n, :, h', w']
    ```

    3.activation and weight
    input(act.): input tensor.;
    grad_output(act.): the gradient of the loss with respect to the output.;
    gradout_transpose(w.): The transposed gradient of the output tensor.;

    4.attributes
    groups: Number of blocked connections from input channels to output channels. Default: 1.;
    input_shape: The shape of the input tensor.;
    grad_out_shape: The shape of the gradient output tensor.;
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    stride: the stride for the cross-correlation, a single number or a tuple.;
    dilations: controls the spacing between the kernel points;
    padding: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    grad_bias_enable: whether to compute the gradient for the bias term as well.;
    }];
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
    1.Op Introduction
    Returns the upper or lower triangular part of input.

    2.Math formula
    ```math
            output = Triu(input, diagonal) if upper = 1
            output = Tril(input, diagonal) if upper = 0
    ```
    where, Triu() return the upper triangular part of the input tensor.
           Tril() return the lower triangular part of the input tensor.

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    upper: whether to extract the upper or lower triangular part of the input tensor.;
    diagonal: 0 refers to the main diagonal, positive values indicate diagonals above the main diagonal,
              and negative values indicate diagonals below the main diagonal.;
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
    1.Op Introduction
    Integrates some operations related to Sort.

    2.Math formula
    ```math
            output = Sort(input, axis, descending)
    ```

    3.activation and weight
    input(act.): input tensor.;
    buffer(w.): a temporary storage area for intermediate calculations or results during the NMS process.;

    4.attributes
    axis: the dimension of the input tensor.;
    descending: the order of sorting.;
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
    1.Op Introduction
    When there are multiple slices in the layer-group,
    this operation combines the store op output of each slice to output to the yield op.

    2.Math formula
    ```math
            output =  SliceMerge(input)
    ```

    3.activation and weight
    inputs(act.): input tensor.;
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MeanRstdOp: Tpu_Op<"MeanRstd">{
  let summary = "Compute Mean, Rstd, Running_mean, Running_var in batchnorm train op ";
  let description = [{
    1.Op Introduction
    computes the mean, reverse standard deviation (Rstd), running mean, and running variance.

    2.Math formula
    ```math
        mean(x),1/sqrt(var+eps),(1-momentum)*running_mean + momentum*mean,(1-momentum)*running_var + momentum*var
    ```

    3.activation and weight
    input(act.): input tensor.;
    running_mean(w.): mean during running.;
    running_var(w.): variances during running.;
    weight(w.): weight tensor.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attributes
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
    momentum: hyperparameter;
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
    1.Op Introduction
     This operation stores storeOp output data on DDR when some tentor in layer-group
     is not output to returnOp but storeAndLoad is needed.

    2.Math formula
    ```math
        output = storeOp_output
        "storeOp_output" represents the data produced by the store operation that is saved on DDR.
    ```

    3.activation and weight
    none

    4.attributes
    need_dump: whether the stored output data should be additionally dumped for debugging or analysis.;
  }];

  let arguments = (ins
    DefaultValuedAttr<BoolAttr, "false">:$need_dump
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MoveOp:Tpu_BaseOp<"Move"> {
  let summary = "local mem move op for layer group lmem alloc";
  let description = [{
    1.Op Introduction
    local mem move op for layer group lmem alloc

    2.Math formula
    ```math
        output = Move(input, move_src_add, move_dest_add, move_size, ts_id)
    ```

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    name: a human-readable identifier for the move operation.;
    move_src_add: source addresses (offsets) in the local memory.;
    move_dest_add: destination addresses (offsets) in the local memory.;
    move_size: A list of sizes corresponding to each move operation.;
    ts_id: A unique identifier (timestamp id or task id);
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
    1.Op Introduction
    group normalization

    2.Math formula
    ```math
            output = \frac{input - mean}{\sqrt{\sigma^2 + eps}} \cdot weight + bias
    ```

    3.activation and weight
    input(act.): input tensor.;
    weight(w.): weight tensor.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attributes
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
    num_groups: number of groups to divide the channels into for normalization.;
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
  let description = [{
    1.Op Introduction
    logical and operation between two variables

    2.Math formula
    ```math
            output = input1 and input2
    ```

    3.activation and weight
    inputs(act.): input tensor.;
    }];
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
  let description = [{
    1.Op Introduction
    calculate grad_input,grad_weight,grad_bias of convolution operation;

    2.Math formula
    ```math
        input(N, C_{in}, D, H, W) = \sum_{C_{out}} output(N, C_{output}, D + sD, H + sH * kH, W + sW * kW) * kernel(C_{in}, C_{out}, kD, kH, kW)
    ```
    3.activation and weight
    grad_out(act.): how the loss changes with respect to the output of the convolution operation.;
    input(act.): input tensor;
    kernel(w.): weights (filters) ;
    buffer(act.): intermediate computations or storage during the backward pass;

    4.attribute
    groups: the number of groups for grouped convolutions.;
    input_shape: the shape of the input tensor.;
    grad_out_shape: the shape of the tensor that is being backpropagated.;
    kernel_shape: an array of integers.;
    stride: an array of integers for each dimension.;
    dilations: dilation rate for the convolution, controls the spacing between kernel elements.;
    padding: an array of integers.;
    inserts: additional parameters;
    grad_input_enable: whether to compute the gradient with respect to the input tensor.;
    grad_weight_enable: whether to compute the gradient with respect to the kernel tensor.;
    grad_bias_enable: whether to compute the gradient with respect to the bias tensor.;
    }];
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
    1.Op Introduction
    MaskRCNN_RPN_get_bboxes, the sub-block with 1st NMS between RPN_head and 1st ROIAlign.

    2.Math formula
    ```math
        1.Score Filtering
            valid_indices_i = cls_scores_i > conf_threshold (for each level i)
        2.Bounding Box Adjustment
            adjusted_bboxes_i = anchors_i + bbox_preds_i * delta2bbox_std_i + delta2bbox_mean_i
        3.IoU Calculation
            iou = calculate_iou(adjusted_bboxes_i, ground_truth_boxes)
        4.NMS Application
            final_bboxes = nms(adjusted_bboxes_i[valid_indices_i], iou_threshold)
        5.final output
            output = concatenate(final_bboxes)
    ```
    3.activation and weight
    cls_scores_0: the class scores0 for each anchor;
    cls_scores_1(act.): the class scores1 for each anchor;
    cls_scores_2: the class scores2 for each anchor;
    cls_scores_3: the class scores3 for each anchor;
    cls_scores_4: the class scores4 for each anchor;
    bbox_preds_0: the bounding box0 predictions;
    bbox_preds_1: the bounding box1 predictions;
    bbox_preds_2: the bounding box2 predictions;
    bbox_preds_3: the bounding box3 predictions;
    bbox_preds_4: the bounding box4 predictions;
    max_shape: the maximum dimensions of the output bounding boxes.;
    mlvl_anchors_0: the multi-level anchors0 used for generating bounding box proposals.;
    mlvl_anchors_1: the multi-level anchors1 used for generating bounding box proposals.;
    mlvl_anchors_2: the multi-level anchors2 used for generating bounding box proposals.;
    mlvl_anchors_3: the multi-level anchors3 used for generating bounding box proposals.;
    mlvl_anchors_4: the multi-level anchors4 used for generating bounding box proposals.;
    batch_mlvl_scores: the class scores for all anchors across multiple feature levels in a batch.;
    batch_mlvl_anchors: the multi-level anchors generated for the entire batch.;
    batch_mlvl_rpn_bbox_pred: the bounding box predictions for each anchor across multiple levels in the batch.;
    batch_mlvl_proposals: the proposed bounding boxes generated for the entire batch after processing;
    batch_mlvl_ids: the ids associated with the proposals for each anchor across the batch.;
    glb_buffer_tmp_scores_stretched: the stretched class scores for processing before NMS.;
    glb_buffer_ranked_scores: The buffer stores the ranked class scores after sorting the top proposals.;
    glb_buffer_rank_inds_int32: the indices of the ranked scores in 32-bit integer format;
    glb_buffer_rank_inds_u32: stores the indices in 32-bit unsigned integer format.;
    glb_topk_inds: This buffer holds the indices of the top K proposals after the ranking process.;
    glb_buffer_gather_1: gathering specific data from the ranked scores or proposals.;
    glb_buffer_gather_2: Similar to glb_buffer_gather_1;
    glb_buffer_rpn_bbox_permuted: the permuted bounding box predictions.;
    glb_buffer_nonzero: non-zero entries from the scores or proposals.;
    result_valid_ind: the valid indices of the resulting proposals after processing.;
    glb_buffer_gather_boxes: gather the final bounding boxes from the proposals after NMS.;
    glb_buffer_gather_scores: holds the final scores associated with the gathered bounding boxes.;
    keep_3nch: maintain a specific format or structure for the gathered results(three-channel format).;
    keep_u32_1h: manage the format or structure of the output in a specific way.;
    glb_buffer_boxes: the final bounding boxes ready for output after all processing steps.;
    glb_buffer_scores: the final class scores corresponding to the output bounding boxes.;
    glb_buffer_nms: This buffer is used during the Non-Maximum Suppression process to manage overlapping proposals.;
    gather_mlvl_proposals: gather proposals from multiple levels for final processing.;
    gather_mlvl_scores: gather scores from multiple levels for the final proposal selection.;
    gather_mlvl_ids: gather ids from multiple levels to track proposals across the batch.;
    glb_buffer_result_list: the final list of results.;

    4.attribute
    delta2bbox_mean_0: the means used to normalize the bounding box0 deltas for the corresponding feature levels.;
    delta2bbox_mean_1: the means used to normalize the bounding box1 deltas for the corresponding feature levels.;
    delta2bbox_mean_2: the means used to normalize the bounding box2 deltas for the corresponding feature levels.;
    delta2bbox_mean_3: the means used to normalize the bounding box3 deltas for the corresponding feature levels.;
    delta2bbox_std_0: the standard deviations used to normalize the bounding box0 deltas for the corresponding feature levels.;
    delta2bbox_std_1: the standard deviations used to normalize the bounding box1 deltas for the corresponding feature levels.;
    delta2bbox_std_2: the standard deviations used to normalize the bounding box2 deltas for the corresponding feature levels.;
    delta2bbox_std_3: the standard deviations used to normalize the bounding box3 deltas for the corresponding feature levels.;
    delta2bbox_max_scalar_c: a scalar value;
    iou_threshold: filtering out low-quality proposals during NMS.;
    conf_threshold: a confidence score threshold.;
    MAX_LENGTH_STATIC_STRECHED: the maximum length for the output list of bounding boxes after processing.;
    NUM_INDEXES: the number of indexes.;
    NUM_CLASSES: the number of classes.;
    CHANNEL_RPN_BBOXES: the number of channels for the bounding box predictions.;
    CHANNEL_RPN_SCORES: the number of channels used for the class score predictions.;
    NMS_PRE: the number of proposals to be considered before NMS.;
    HARDWARE_FACTOR_TOPK: how many top proposals to keep.;
    NMS_MAX_LENGTH: the maximum number of boxes after NMS.;
    TOPK_ONNX_NMS: the number of top proposals when using ONNX format for NMS.;
    H_RPN_DYN_MAX: the maximum height for the dynamic RPN output.;
    W_RPN_DYN_MAX: the maximum width for the dynamic RPN output.;
    MAX_PER_IMG: the maximum number of proposals to be generated per image.;
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
    1.Op Introduction
    MaskRCNN_BBox_Pooler, the 1st ROIAlign in MaskRCNN.

    2.Math formula
    ```math
            output = ROIAlign(feature map, rois_multi_batch)
    ```

    3.activation and weight
    ptr_feat0(act.): Pointer to the feature map at level 0.;
    ptr_feat1(act.): Pointer to the feature map at level 1.;
    ptr_feat2(act.): Pointer to the feature map at level 2.;
    ptr_feat3(act.): Pointer to the feature map at level 3.;
    rois_multi_batch(w.): ROIs (Regions of Interest) for multiple batches.;

    4.attributes
    ROI_NUM_LEVELS: The number of levels in the ROI feature pyramid.;
    ROI_H: The height of the pooled ROI features.;
    ROI_W: The width of the pooled ROI features.;
    CHANNEL_ROI: The number of channels in the pooled ROI features.;
    ROI_SLICE: The number of slices or segments;
    ROI_PH: The height of the ROI in the feature map.;
    ROI_PW: The width of the ROI in the feature map.;
    ROI_LEN: The length of the ROIs being processed.;
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
    1.Op Introduction
     MaskRCNN_GetBboxB, the 2nd onnx_nms in MaskRCNN

    2.Math formula
    ```math
            decode_bbox_i = bbox_means + scale_factor x (ptr_bbox[i] x bbox_stds + [res_bbox0[i] res_bbox1[i] res_bbox[i]])
            score_i = ptr_score[i] (or a combination of res_score0,res_score1,res_score2,res_score3)
            {bboxs, lables} = NMS({decode_bbox_i}, {score_i}, nms_iou_thr)
    ```

    3.activation and weight
    ptr_rois(act.): candidate regions of interest for possible objects.;
    ptr_bbox(act.): the bounding box predictions.;
    ptr_score(act.): confidence scores associated with each proposal.;
    max_val(w.): max scores the bbox predictions.;
    scale_factor(w.): scale the decoded bounding box coordinates.;
    stds(w.): standard deviation values for each channel.;
    means(w.): mean values to subtract from each channel for normalization.;
    res_bbox(w.): bounding box prediction values.;
    res_bbox1(w.): one branch of the processed bounding box predictions.;
    res_bbox0(w.): another branch or variant of the bounding box decoding results.;
    res_score0(w.): one set of confidence scores associated with the proposals.;
    res_score1(w.): an additional set of confidence scores.;
    res_score2(w.): another set of scoring values.;
    res_score3(w.):  final selection of bounding boxes.;
    res_label2(w.): the predicted class labels.;
    result_list(w.): the final list of processed results or proposal indices after filtering and ranking.;
    keep_3nch(w.): whether retains proposals having three-channel inputs.;
    keep_u32_1h(w.): whether using a single 32-bit unsigned integer per element.;
    glb_buffer_boxes(w.): A global temporary buffer used to store bounding box.;
    glb_buffer_scores(w.): A global temporary buffer for storing scores associated with the candidate bounding boxes.;
    glb_buffer_nms(w.): This global buffer holds intermediate NMS-related information.;
    glb_buffer_nonzero(w.): stores the non-zero elements (or valid indices) from processed tensors.;
    result_valid_ind(w.): the indices of proposals.;
    glb_lables(w.): A global buffer dedicated to storing the class labels.;
    glb_lables_expand(w.): an expanded version of the global labels.;

    4.attributes
    threshold_score_eq: A threshold value used to filter out proposals with a low confidence score before applying NMS.;
    wh_ratio_log: A logarithmic scaling factor, adjust the width-to-height ratio during decoding of bounding boxes.;
    nms_iou_thr: IoU (Intersection over Union) threshold.;
    delta2bbox_means: Mean values used to decode the bounding box regression.;
    delta2bbox_stds_0: Standard deviation (first component) for scaling the decoded bbox values.;
    delta2bbox_stds_1: Standard deviation (second component) for scaling the decoded bbox values.;
    NUM_INDEXES: the number of indexes (or anchors).;
    NUM_CLASSES: The total number of object classes.;
    TOPK_ONNX_NMS: to select a fixed number of candidates.;
    NUM_CLASSES_getBboxB: Number of classes used in this bounding box decoding step.;
    MAX_NMS_LENGTH_GetBboxB: Maximum number of bounding box candidates.;
    MAX_PER_IMG: The maximum number of detections allowed per image.;
    MAX_PER_IMG_GetBboxB: maximum number of bounding boxes after the final processing.;
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
    1.Op Introduction
    MaskRCNN_Mask_Pooler, the 2st ROIAlign in MaskRCNN

    2.Math formula
    ```math
            output = \text{ROIAlign}\Bigl( \{x_i\}_{i=0}^3, \, \text{det\_bboxes\_multi\_batch}, \, \text{det\_labels\_multi\_batch}, \, \text{scale\_factor}, \, ROI\_NUM\_LEVELS, \, ROI\_H, \, ROI\_W, \, CHANNEL\_ROI, \, ROI\_SLICE, \, ROI\_PH, \, ROI\_PW, \, ROI\_LEN \Bigr)
    ```

    3.activation and weight
    x_0(act.): first level of the feature pyramid.;
    x_1(act.): second level of the feature pyramid.;
    x_2(act.): third level of the feature pyramid.;
    x_3(act.): fourth level of the feature pyramid.;
    det_bboxes_multi_batch(act.): detected bounding boxes over multiple batches.;
    det_labels_multi_batch(act.): class labels associated with the detected bounding boxes across multiple batches.;
    scale_factor(w.): scale the decoded bounding box coordinates.;
    ptr_rois_buff(w.): a temporary buffer that holds precomputed or intermediate Regions of Interest (ROIs).;
    result_filled_det_bboxes(w.): stores the processed or “filled” detection bounding boxes.;
    result_filled_det_labels(w.): corresponding class labels that have been associated with the detected bounding boxes.;
    ptr_tmp_res(w.):  temporary result buffer used internally during the pooling process.;
    ptr_rois_tmp(w.): a temporary workspace that stores intermediate ROI values.;

    4.attributes
    ROI_NUM_LEVELS: the number of feature levels (or pyramid levels) available for ROI pooling.;
    ROI_H: target height of the pooled region for each ROI.;
    ROI_W: target width of the pooled region for each ROI.;
    CHANNEL_ROI: number of channels to be kept or considered when performing ROIAlign.;
    ROI_SLICE: slicing strategy for ROIs, if the ROI needs to be segmented into sub-regions for finer pooling.;
    ROI_PH: Padding height for the ROI.;
    ROI_PW: Padding width for the ROI.;
    ROI_LEN: total length (or area) of the ROI.;
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

def tpu_MaxPoolingIndicesBwdOp: Tpu_Op<"MaxPoolingIndicesBwd", [
  DeclareOpInterfaceMethods<IndexingMapsInterface>,
  DeclareOpInterfaceMethods<TypeInterface>]> {
  let summary = "MaxPoolingIndicesBwd operator";

  let description = [{
    1.Op Introduction
    Integrates some operations related to MaxPoolingIndicesBwd.

    2.Math formula
    ```math
          grad\_input(p) = \sum_{q} \delta\big(p = indices[q]\big) \times grad\_output(q)
    ```

    3.activation and weight
    grad_output(act.): the gradient of the loss with respect to the output.;
    indices(w.): the indices of the input tokens or items.;

    4.attributes
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    strides: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    dilations: controls the spacing between the kernel points;
    input_shape: The shape of the input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$grad_output,
    AnyTensor:$indices,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    I64ArrayAttr:$input_shape
  );

  let results = (outs AnyRankedTensor:$grad_input);
}

def Tpu_CastAddOp: Tpu_Op<"CastAdd", [
  SupportFuseRelu, SupportEarlyStride]> {
  let summary = "add operator";

  let description = [{
    Cast + Add; One of original Add is casted. Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode
  );

  let results = (outs AnyRankedTensor:$output);
}

def Tpu_LayerNormCastOp: Tpu_Op<"LayerNormCast", [
  DeclareOpInterfaceMethods<TypeInterface>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>]> {
  let summary = "LayerNorm + Requant";

  let description = [{
    LayerNorm + Requant
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    SI32Attr:$axis,
    F64Attr:$eps,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode,
    DefaultValuedAttr<I64Attr, "1">:$isCastAtEnd
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_MatMulLutOp: Tpu_Op<"MatMulLut", [
    SupportFuseRelu,
    DeclareOpInterfaceMethods<TypeInterface>]>{
  let summary = "MatMul + Lut";

  let description = [{
    MatMul + Lut
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$table,
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
  let extraClassDeclaration = [{
    matmul_attr_t parseParam();
    void assign_fw_param(void *param);
  }];
}

def Tpu_FusedActiveCastOp: Tpu_Op<"FusedActiveCast",[
  InOutSameShape, SupportElementwise]>{
  let summary = "Active + Cast";
  let description = [{}];
  let arguments = (ins
    AnyRankedTensor:$input,
    Tpu_ActiveModeAttr:$mode,
    OptionalAttr<F64ArrayAttr>:$coeffs,
    DefaultValuedAttr<Tpu_RoundModeAttr, "tpu::RoundMode::HalfAwayFromZero">:$round_mode
  );
  let results = (outs AnyRankedTensor:$output);
}

def Tpu_SoftmaxCastOp: Tpu_Op<"SoftmaxCast",[
  DeclareOpInterfaceMethods<TypeInterface>,
  DeclareOpInterfaceMethods<IndexingMapsInterface>]> {
  let summary = "softmax operator";

  let description = [{
    Softmax + Requant
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
#endif // TPU_OPS
