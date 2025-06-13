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
// Defines TOP Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TOP_OPS
#define TPU_MLIR_TOP_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
//include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tpu_mlir/Interfaces/InferenceInterface.td"
include "tpu_mlir/Interfaces/FlopsInterface.td"
include "tpu_mlir/Interfaces/ShapeInterface.td"
include "tpu_mlir/Traits/Traits.td"

// =============================================================================
//
// Defines TOP Dialect.
//
//===----------------------------------------------------------------------===//

def Top_Dialect : Dialect {
  let name = "top";
  let summary = "A topdialect for the TPU_MLIR specification";
  let cppNamespace = "::tpu_mlir::top";
}

//===----------------------------------------------------------------------===//
// TOP Attributes.
//===----------------------------------------------------------------------===//

class Top_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Top_Dialect, attrName, traits> {
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
def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual", "NotEqual", "And", "Not", "Xor"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceL2","ReduceL1","ReduceSum","ReduceProd"]>;
def InterpModeAttr: AnyStrAttrOf<["nearest","linear"]>;
def InterpCoordModeAttr: AnyStrAttrOf<["align_corners", "half_pixel", "pytorch_half_pixel", "asymmetric"]>;
def PixelFormatAttr: AnyStrAttrOf<["rgb","bgr","gray","rgba","gbrg","grbg","bggr","rggb"]>;
def ChannelFormatAttr: AnyStrAttrOf<["nhwc","nchw"]>;
def PadModeAttr: AnyStrAttrOf<["normal","center"]>;
def DetectionOutputCodeTypeAttr: AnyStrAttrOf<["CORNER", "CENTER_SIZE", "CORNER_SIZE"]>;
def RoiAlignModeAttr: AnyStrAttrOf<["Avg","Max"]>;
def NonZeroOrderAttr: AnyStrAttrOf<["ColMajor","RowMajor"]>;
def StoreModeAttr: AnyStrAttrOf<["1N", "2N", "4N"]>;
def YoloVersionAttr: AnyStrAttrOf<["yolov3", "yolov3_tiny", "yolov3_spp", "yolov4", "yolov5","yolov8","yolov11"]>;
def MatchTemplateModeAttr: AnyStrAttrOf<["TM_CCOEFF_NORMED", "TM_SQDIFF"]>;
def PaddingModeAttr:AnyStrAttrOf<["constant","reflect","symmetric","edge"]>;
def AutoPadModeAttr:AnyStrAttrOf<["SAME_UPPER","SAME_LOWER","NOTSET","VALID"]>;
def EinsumModeAttr:AnyStrAttrOf<["a,b->ab", "ab,ab->a", "ab,acb->ac", "ab,cdb->acd", "abc,db->adc", "abcd,cde->abe", "abcd,acd->abc", "abcd,bed->abce", "abcd,ced->abce", "abcd,abed->abce", "abcd,abde->abce", "abcd,abce->acde", "abc,adc->abd", "abc,adc->adb", "abc,abd->acd", "abcd,cde->abce", "abc,acde->abde", "abc,abde->acde", "abc,bd->abcd", "abc,abdc,abc->abcd", "abcd,acde,abc->abce", "abc,abc->ab", "abcd,aecd->aeb", "abc,cde->abde", "abcd,aeb->aecd", "abcde,afbc->abdef", "abcde,abfge->abcdfg","abcd,aecd->abec","abcd,acde->abde","abcd,abef->acdef","abcd,abce->abde","ab,abc->ac","abcd,aecd->acbe","abcd,adbe->acbe"]>;
def RoundModeAttr: AnyStrAttrOf<["HalfAwayFromZero", "HalfUp", "HalfDown", "HalfToEven", "HalfToOdd", "HalfTowardsZero", "TowardsZero", "Up", "Down"]>;
def DequantModeAttr: AnyStrAttrOf<["Normal", "TFLite"]>;
def RequantModeAttr: AnyStrAttrOf<["TFLite_LShift", "TFLite", "MultiplierShift", "OnlyShift", "QDM", "OnlyScale"]>;
def BinaryShiftAttr: AnyStrAttrOf<["Add","Sub", "Mul"]>;
def ImageOutFormatAttr: AnyStrAttrOf<["FLOAT32", "UINT8"]>;
def Yuv2rgbFormulaAttr: AnyStrAttrOf<["_601_limited", "_601_full"]>;
def GELUModeAttr: AnyStrAttrOf<["normal", "tanh", "sigm"]>;

//===----------------------------------------------------------------------===//
// TOP Types.
//===----------------------------------------------------------------------===//

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// TOP Op Definition.
//===----------------------------------------------------------------------===//

// === BaseOp =====
class Top_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Top_Dialect, mnemonic, traits> ;

def Top_NoneOp : Top_BaseOp<"None"> {
  let summary = "none operator";

  let description = [{
    A none Op to return a NoneType.
  }];
  let results = (outs NoneType);
}

def Top_WeightOp : Top_BaseOp<"Weight"> {
  let summary = "weight operator";

  let description = [{
    If `inline_bytes` is not defined or `inline_bytes` is a null string:
      Load weight from a file. The file should be a valid .npz format file.
      This Op does not take any input, and the location captures the tensor name.
      The Output is an n-dimensional tensor whose type matches
      the tensor type in the .npz file.
    Else:
      Load weight from `inline_bytes`.
  }];

  let arguments = (ins
    OptionalAttr<F64ArrayAttr>:$scale,
    OptionalAttr<StoreModeAttr>:$store_mode,
    OptionalAttr<I64ArrayAttr>:$allow_split,
    //nnvlc param
    OptionalAttr<BoolAttr>:$do_compress,
    OptionalAttr<I64Attr>:$bias0,
    OptionalAttr<I64Attr>:$bias1,
    OptionalAttr<BoolAttr>:$is_signed,
    OptionalAttr<BoolAttr>:$zero_guard,
    OptionalAttr<StrAttr>:$inline_bytes // save weight data as bytes
  );

  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
  template<typename T>
  std::shared_ptr<std::vector<T>> read();
  std::shared_ptr<std::vector<float>> read_as_float();
  std::shared_ptr<std::vector<int32_t>> read_as_int32();
  std::shared_ptr<std::vector<uint8_t>> read_as_byte();
  std::shared_ptr<std::vector<int8_t>> read_as_f8e4m3();
  std::shared_ptr<std::vector<int8_t>> read_as_f8e5m2();
  template<typename T>
  static mlir::Value create(mlir::Operation * OwnerOp,
                            llvm::StringRef suffix,
                            const std::vector<T>& data,
                            mlir::RankedTensorType& type,
                            uint32_t store_mode = 0);
  template<typename T>
  mlir::LogicalResult update(const std::vector<T>& data, size_t count);
  mlir::Value clone_bf16(mlir::Operation * OwnerOp, std::string name = "");
  mlir::Value clone_f16(mlir::Operation * OwnerOp);
  mlir::Value clone_int(mlir::Operation *OwnerOp);
  mlir::Value clone_f8e4m3(mlir::Operation *OwnerOp, bool per_channel_scale = false, bool channel_first_dim = true);
  mlir::Value clone_f8e5m2(mlir::Operation *OwnerOp);
  mlir::Value clone(llvm::StringRef suffix);
  mlir::Value split(int begin, int end, int axis, mlir::Type to_type, std::string suffix);
  static mlir::Value create_float(mlir::Operation *OwnerOp, llvm::StringRef suffix,
                       const std::vector<float> &data, const std::vector<int64_t> &shape,
                       mlir::Type storage_type);
  }];
}

def Top_InputOp: Top_BaseOp<"Input", [
  DeclareOpInterfaceMethods<ShapeInterface>
  ]> {
  let summary = "Input operator";

  let description = [{
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    // input affects tensor's shape or not
    OptionalAttr<I64ArrayAttr>:$shape_tensor,
    // whether need preprocess
    DefaultValuedAttr<BoolAttr, "false">:$do_preprocess,
    // preprocess for input
    OptionalAttr<PixelFormatAttr>:$pixel_format,
    OptionalAttr<ChannelFormatAttr>:$channel_format,
    OptionalAttr<I64ArrayAttr>:$resize_dims,
    OptionalAttr<BoolAttr>:$keep_aspect_ratio,
    OptionalAttr<StrAttr>:$keep_ratio_mode,
    OptionalAttr<I64Attr>:$pad_value,
    OptionalAttr<PadModeAttr>:$pad_type,
    OptionalAttr<F64ArrayAttr>:$scale,
    OptionalAttr<F64ArrayAttr>:$mean,
    // for fusepreprocess
    OptionalAttr<StrAttr>:$customization_format,
    OptionalAttr<BoolAttr>:$aligned,
    // for .yuv file
    OptionalAttr<StrAttr>:$yuv_type
  );

  let results = (outs AnyTensor:$output);
}

def Top_TupleOp: Top_BaseOp<"Tuple"> {
  let summary = "Tuple operator";
  let description = [{
    1.Op Introduction
    gen by torch prim::TupleConstruct, y = (a, b)

    2.Math formula
    ```math
        output = TupleOp(input1, input2)
    ```

    3.activation and weight
    input(act.): input tensor;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
}

def Top_UnTupleOp: Top_BaseOp<"UnTuple"> {
  let summary = "UnTuple operator";
  let description = [{
    1.Op Introduction
    gen by torch prim::TupleUnpack, a, b = y

    2.Math formula
    ```math
        output1, output2 = UnTupleOp(input)
    ```

    3.activation and weight
    input(act.): input tensor;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs
    Variadic<AnyTensor>:$outputs
  );
}

// === Function Op =====
class Top_Op<string mnemonic, list<Trait> traits = []> :
    Top_BaseOp<mnemonic, !listconcat(traits,
       [DeclareOpInterfaceMethods<InferenceInterface>,
        DeclareOpInterfaceMethods<FlopsInterface>,
        DeclareOpInterfaceMethods<ShapeInterface>])>;

def Top_BatchNormOp: Top_Op<"BatchNorm", [SupportFuseRelu]> {
  let summary = "BatchNormalization operation";
  let description = [{
    1.Op Introduction
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
    as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    2.Math formula
    ```math
        output = \frac{input - \mathrm{E}[input]}{ \variance + \epsilon} * \gamma + \beta
    ```

    3.activation and weight
    input(act.): input tensor;
    mean(w.): mean of input tensor in dim C;
    variance(w.):  quantifies the spread of the input tensor values along the channel dimension (dimension C) for each mini-batch.;
    gamma(w.): scalar;
    beta(w.): scalar;
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).

    4.attribute
    epsilon;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;

  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$mean,
    AnyTensor:$variance,
    AnyTensorOrNone:$gamma,
    AnyTensorOrNone:$beta,
    DefaultValuedAttr<F64Attr, "1e-05">:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_CastOp:Top_Op<"Cast"> {
  let summary = "Cast operation";
  let description = [{
    1.Op Introduction
    quant::UniformQuantizedType cast to float type; or float type cast to quant::UniformQuantizedType

    2.Math formula
    Float(output) = Cast (input, dtype=UniformQuantizedType);
    UniformQuantizedType(output) = Cast (input, dtype=Float);

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;


  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );
  let results = (outs AnyTensor:$output);
}

def Top_DtypeCastOp:Top_Op<"DtypeCast"> {
  let summary = "Cast F32 to F16";
  let description = [{
    1.Op Introduction
    Cast F32 to F16

    2.Math formula
    FLOAT16(output) = DtypeCastOp (FLOAT32(input));

    3.activation
    input(act.): input tensor;


  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_ConcatOp: Top_Op<"Concat"> {
  let summary = "Concat operator";

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
    axis: the dimension along which the input tensors will be concated together.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    round_mode: This determines how values are rounded during the conversion from higher precision to lower precision.;

  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<SI32Attr, "1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_RequantIntOp:Top_Op<"RequantInt"> {
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
    quant_mode: the mode or method used for quantization during the requantization operation.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
    rq_axis: the axis along which the requantization operation is applied.;
    fuse_rq: whether to fuse the requantization operation with a preceding operation (such as a convolution or activation function).;

  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$multiplier,
    I64ArrayAttr:$rshift,
    RequantModeAttr:$quant_mode,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode,
    DefaultValuedAttr<SI32Attr, "1">:$rq_axis,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_rq
  );
  let results = (outs AnyTensor:$output);
}

def Top_DequantIntOp:Top_Op<"DequantInt"> {
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
    quant_mode: the mode or method used for quantization during the requantization operation.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$multiplier,
    I64ArrayAttr:$shift,
    DefaultValuedAttr<I64Attr, "0">:$lshift,
    DequantModeAttr:$quant_mode,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );
  let results = (outs AnyTensor:$output);
}

def Top_RequantFpOp:Top_Op<"RequantFp"> {
  let summary = "requant operation";
  let description = [{
    1.Op Introduction
    Requant 32/16/8 bit data to int8 or uint8 data, by scale

    2.Math formula
    int8/uint8/fp8(output) = round(float32/float16/float8(input) x scale + offset);

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    scale: Scalar;
    offset: Scalar;
    quant_mode: the mode or method used for quantization during the requantization operation.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
    first_round_mode: the rounding behavior applied to the scaled value before the offset is added.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64ArrayAttr:$scale,
    F64ArrayAttr:$offset,
    RequantModeAttr:$quant_mode,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode,
    DefaultValuedAttr<RoundModeAttr, "\"HalfUp\"">:$first_round_mode
  );
  let results = (outs AnyTensor:$output);
}

def Top_PackOp: Top_Op<"Pack"> {
  let summary = "Pack operator";

  let description = [{
    1.Op Introduction
    Pack a list of tensors in the given dimension, All tensors must have the same shape.

    2.Math formula
    output = Pack(input1, input2, input3; axis)

    3.activation and weight
    input(act.): Variadic input tensor;

    4.attribute
    axis: It specifies the dimension along which the input tensors will be packed together.;
    values_count: It indicates the number of tensors being packed.;
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    SI32Attr:$axis,
    I64Attr:$values_count
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_UnpackOp: Top_Op<"Unpack"> {
  let summary = "Unpack operator";

  let description = [{
    1.Op Introduction
    Unpack a tensor to list of tensors in the given dimension.

    2.Math formula
    output1, output2, output3 = UnPack(input; axis)

    3.activation and weight
    input(act.): Variadic input tensor;

    4.attribute
    axis: It specifies the dimension along which the input tensors will be packed together.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    SI32Attr:$axis
  );

  let results = (outs Variadic<AnyTensor>:$outputs);
  let hasCanonicalizer = 1;
}

def Top_ConvOp: Top_Op<"Conv", [SupportFuseRelu,
  DeclareOpInterfaceMethods<InferenceInterface,["backward_weight"]>]> {
  let summary = "Convolution operator";

  let description = [{
    1.Op Introduction
    The Top_ConvOp is a convolution operator designed to perform convolution operations on input tensors.
    The operation transforms an input tensor with shape ((N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})) into
    an output tensor with shape ((N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})).
    The output is computed using learnable weights (filters) and optional biases,
    following the mathematical formula for convolution, which incorporates parameters such as kernel size, strides, padding, and dilation.

    2.Math formula
    ```math
        output(N, C_{out}, H, H) = \sum_{C_{in}} input(N, C_{in}, H + sH * kH, W + sW * kW) * filter(C_{in}, C_{out}, kH, kW) + bias(C_{out}) + bisa(C_{out})
    ```
    where, kH and kW are the height and width of the filter (kernel), sH and sW are the vertical and horizontal strides.
            N is a batch size, C denotes a number of channels, H is a height of input, and W is width.
    Shape:
    - Input: (N, C_{in}, H_{in}, W_{in})
    - Output: (N, C_{out}, H_{out}, W_{out})
        ```math
        H_{output} = {H_{in} + {padding}[0] + {padding}[2] - {dilation}[0] x ({kernel_size}[0] - 1) - 1} / {stride}[0] + 1
        ```
        ```math
        W_{output} = {W_{in} + {padding}[1] + {padding}[3] - {dilation}[1] x ({kernel_size}[1] - 1) - 1} / {stride}[1] + 1
        ```

    3.activation and weight
    input(act.): Variadic input tensor;
    filter(w.): the learnable weights of the convolution 2d operation.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attribute
    kernel_shape: the size of the convolution kernel (filter) as an array. ;
    stride: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    groups: (optional)Number of blocked connections from input channels to output channels. Default: 1.;
    dilation: controls the spacing between the kernel points;
    inserts: additional parameters that may be used for specific optimizations or configurations.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    dynweight_reorderd: whether the weights (filters) should be reordered dynamically.;
    weight_is_coeff: whether the weights should be treated as coefficients.;
    do_winograd: whether to use the Winograd algorithm for convolution,
                 which can speed up the computation by reducing the number of multiplications needed.;
    auto_pad: It can be set to different modes (e.g., SAME, VALID) to automatically calculate the necessary padding based on
              the input size,kernel size, and stride.;
    in_int4_scale: This attribute defines the scaling factor for input tensors represented in 4-bit integer format (INT4).;
    in_int4_zp: The in_int4_zp attribute specifies the zero-point for the input tensors in INT4 format.;
    out_int8_scale: This attribute defines the scaling factor for output tensors represented in 8-bit integer format (INT8).;
    out_int8_zp: The out_int8_zp attribute specifies the zero-point for the output tensors in INT8 format.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads, // top,left,bottom,right
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<I64ArrayAttr>:$inserts,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$dynweight_reorderd,
    DefaultValuedAttr<I64Attr, "1">:$weight_is_coeff,
    OptionalAttr<BoolAttr>:$do_winograd,
    OptionalAttr<AutoPadModeAttr>:$auto_pad, // only used in shape infer
    OptionalAttr<F64Attr>:$in_int4_scale,
    OptionalAttr<F64Attr>:$in_int4_zp,
    OptionalAttr<F64Attr>:$out_int8_scale,
    OptionalAttr<F64Attr>:$out_int8_zp
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    conv_attr_t parseParam();
    conv_attr_t dynparseParam();
  }];
}

def Top_CorrelationOp:Top_Op<"Correlation"> {
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
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<I64Attr, "0">:$max_disp,
    DefaultValuedAttr<I64Attr, "1">:$num_groups
  );

  let results = (outs AnyTensor:$output);
}

class Top_PoolOp <string mnemonic> : Top_Op<mnemonic,[SupportFuseRelu]> {
  let summary = "pool operator";

  let description = [{
    1.Op Introduction
    This performs an  pooling over the given input tensor. A sliding
    window of size given by <kernel size> is passed over the input tensor.

    2.Math formula
    ```math
        {output}(N_i, C_j, p, q) = pool(input(N_i, C_j, start_h, start_w), kernel_size)
    ```
    start_h and start_w are the starting indices for the pooling window in the height and width dimensions.
    pool can be any pooling function (max pooling, average pooling) applied over the region defined by the kernel size.
    N is a batch size, C denotes a number of channels,
    Shape:
    - Input: (N, C_{in}, H_{in}, W_{in})
    - Output: (N, C_{out}, H_{out}, W_{out})
        ```math
        H_{output} = {H_{in} + {padding}[0] + {padding}[2] - {kernel_shape}[0]} / {stride}[0] + 1
        ```
        ```math
        W_{output} = {W_{in} + {padding}[1] + {padding}[3] - {kernel_shape}[1]} / {stride}[1] + 1
        ```

    3.activation and weight
    input(act.): Variadic input tensor;

    4.attribute
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    stride: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    ceil_mode: whether to use ceiling or floor when calculating the output size.;
    auto_pad: It can be set to different modes (e.g., SAME, VALID) to automatically calculate the necessary padding based on
              the input size,kernel size, and stride.;
    is_adaptive: whether the pooling operation is adaptive.
                 If true, adjusts the kernel size based on the input size to produce a specified output size.
    keepdims: whether to retain the dimensions of the input tensor in the output.
               If true, will have the same number of dimensions as the input tensor.;
    pad_value: whether to retain the dimensions of the input tensor in the output.
                If true, will have the same number of dimensions as the input tensor.;
    count_include_pad: whether to include the padded values in the pooling count.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
    first_round_mode: the rounding behavior applied to the scaled value before the offset is added.;

  }];

  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    OptionalAttr<BoolAttr>:$ceil_mode,
    OptionalAttr<AutoPadModeAttr>:$auto_pad, // only used in shape infer
    DefaultValuedAttr<BoolAttr, "false">:$is_adaptive, //when input_shape % output_shape != 0, indicating true
    DefaultValuedAttr<BoolAttr, "true">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$pad_value,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$first_round_mode
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    pool_attr_t parseParam();
  }];
}

def Top_AvgPoolOp:Top_PoolOp<"AvgPool">;
def Top_MaxPoolOp:Top_PoolOp<"MaxPool">;

def Top_AdaptiveAvgPoolOp:Top_PoolOp<"AdaptiveAvgPool"> {
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$output_size
  );

  let results = (outs AnyTensor:$output);
}

def Top_MaxPoolWithMaskOp:Top_PoolOp<"MaxPoolWithMask"> {
  let results = (outs
    AnyTensor:$output,
    AnyTensor:$mask);
}

def Top_PoolMaskOp: Top_Op<"PoolMask"> {
  let summary = "pool mask operator";

  let description = [{
    1.Op Introduction
    pooling mask on input

    2.Math formula
    ```math
        {output}(N_i, C_j, p, q) = scale x pool(input(N_i, C_j, start_h, start_w), kernel_size)
    ```
    start_h and start_w are the starting indices for the pooling window in the height and width dimensions.
    pool can be any pooling function (max pooling, average pooling) applied over the region defined by the kernel size.
    N is a batch size, C denotes a number of channels,
    Shape:
    - Input: (N, C_{in}, H_{in}, W_{in})
    - Output: (N, C_{out}, H_{out}, W_{out})
        ```math
        H_{output} = {H_{in} + {padding}[0] + {padding}[2] - {kernel_shape}[0]} / {stride}[0] + 1
        ```
        ```math
        W_{output} = {W_{in} + {padding}[1] + {padding}[3] - {kernel_shape}[1]} / {stride}[1] + 1
        ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    scale: a scaling factor is applied to the output of the pooling operation, can adjust the intensity or magnitude of the output mask.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$scale
  );

  let results = (outs AnyTensor:$output);
}

def Top_Depth2SpaceOp: Top_Op<"Depth2Space"> {

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
    ins AnyTensor:$input,
    I64Attr:$block_h,
    I64Attr:$block_w,
    BoolAttr:$is_CRD,
    BoolAttr:$is_inversed,
    DefaultValuedAttr<BoolAttr, "true">:$in_is_NCHW,
    DefaultValuedAttr<BoolAttr, "true">:$out_is_NCHW,
    DefaultValuedAttr<BoolAttr, "false">:$swap_cr
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_AddOp: Top_Op<"Add", [SupportFuseRelu, SupportConstant, ScalarProducer, ScalarConsumer]> {
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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;

  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SubOp: Top_Op<"Sub", [SupportFuseRelu, SupportConstant, ScalarProducer, ScalarConsumer]> {
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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
}

def Top_MulOp: Top_Op<"Mul", [SupportFuseRelu, SupportConstant, ScalarProducer, ScalarConsumer]> {
  let summary = "Mul operator";

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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_MinOp: Top_Op<"Min", [SupportConstant]> {
  let summary = "min operator";

  let description = [{
    1.Op Introduction
    Element-wise min of each of the input tensors. All inputs and outputs must have the same data type.

    2.Math formula
    ```math
        output = min(input1,input2, ..., inputN)
    ```
    Where input1, input2, ..., inputN are the input tensors.

    3.activation and weight
    input(act.): input tensor;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_MaxOp: Top_Op<"Max", [SupportConstant]> {
  let summary = "max operator";

  let description = [{
    1.Op Introduction
    Element-wise max of each of the input tensors. All inputs and outputs must have the same data type.

    2.Math formula
    ```math
        output = max(input1,input2, ..., inputN)
    ```
    Where input1, input2, ..., inputN are the input tensors.

    3.activation and weight
    input(act.): input tensor;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_AddConstOp: Top_Op<"AddConst", [SupportFuseRelu, SupportPermuteMove, ScalarProducer, ScalarConsumer]> {
  let summary = "Add Const operator";

  let description = [{
    1.Op Introduction
    The AddConst operator is designed to perform an element-wise addition of a constant value to an input tensor.

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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
}

def Top_SubConstOp: Top_Op<"SubConst", [SupportFuseRelu, ScalarProducer, ScalarConsumer]> {
  let summary = "Sub Const operator";

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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
}

def Top_MulConstOp: Top_Op<"MulConst", [SupportFuseRelu, SupportPermuteMove, ScalarProducer, ScalarConsumer]> {
  let summary = "Mul Const operator";

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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_MinConstOp: Top_Op<"MinConst", [ScalarProducer, ScalarConsumer]> {
  let summary = "Min Const operator";

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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
}

def Top_MaxConstOp: Top_Op<"MaxConst", [ScalarProducer, ScalarConsumer]> {
  let summary = "Max Const operator";

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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
}

def Top_DivConstOp: Top_Op<"DivConst", [ScalarProducer, ScalarConsumer]> {
  let summary = "Div Const operator";

  let description = [{
    1.Op Introduction
    The DivConst operator is designed to perform element-wise division of an input tensor by a constant value.

    2.Math formula
    ```math
        output = input/const_val or const_val/input
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    const_val: the constant value to be added to each element of the input tensor(positive, negative, or zero).;
    is_reverse: whether the subtraction operation is performed in reverse order.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$const_val,
    // I64Attr:$const_val
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
  // let hasCanonicalizer = 1;
}

def Tpu_BinaryShiftOp: Top_Op<"BinaryShift"> {
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
  }];

  let arguments = (
    ins AnyTensor:$input1,
    AnyTensor:$input2,
    BinaryShiftAttr:$mode,
    SI32Attr:$shift,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "true">:$saturation,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_RopeOp: Top_Op<"Rope"> {
  let summary = "Rope operator";

  let description = [{
    1.Op Introduction
    The Rope operator is a specialized tensor operation designed for efficient computations involving multiple input tensors.

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
  }];

  let arguments = (
    ins AnyTensor:$input1,
    AnyTensor:$input2,
    AnyTensor:$input3,
    DefaultValuedAttr<BoolAttr, "false">:$is_permute_optimize,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$mul1_round_mode,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$mul2_round_mode,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$add_round_mode,
    DefaultValuedAttr<SI32Attr, "0">:$mul1_shift,
    DefaultValuedAttr<SI32Attr, "0">:$mul2_shift,
    DefaultValuedAttr<SI32Attr, "0">:$add_shift,
    DefaultValuedAttr<BoolAttr, "true">:$mul1_saturation,
    DefaultValuedAttr<BoolAttr, "true">:$mul2_saturation,
    DefaultValuedAttr<BoolAttr, "true">:$add_saturation
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_BinaryConstShiftOp: Top_Op<"BinaryConstShift"> {
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
    round_mode: It determines how values are rounded during the conversion from higher precision to lower precision.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    SI32Attr:$scale,
    BinaryShiftAttr:$mode,
    SI32Attr:$shift,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "true">:$saturation,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );

  let results = (outs AnyTensor:$output);
}

def Top_NormalizeOp: Top_Op<"Normalize"> {
  let summary = "Normalize operator";

  let description = [{
    1.Op Introduction
    Normalizes an array across batch and spatial dimensions.

    2.Math formula
    ```math
        output = ((input - Mu) / (Sigma + Epsilon)) x scale
    ```
    where, Mu is the mean of the input values calculated across the specified dimensions.

    3.activation and weight
    input(act.): input tensor;
    scale(w.): the scale weight tensor.;

    4.attribute
    across_spatial: determines whether the normalization is performed across the spatial dimensions of the input tensor.;
    channel_shared: indicates whether the scaling weight tensor should be shared across channels.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$scale,
    DefaultValuedAttr<BoolAttr, "false">:$across_spatial,
    DefaultValuedAttr<BoolAttr, "true">:$channel_shared
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ReciprocalOp: Top_Op<"Reciprocal", [SupportFuseRelu]> {
  let summary = "Constant scalar divide tensor operator";

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
  }];

  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<F64Attr, "1.0">: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_MatMulOp: Top_Op<"MatMul", [SupportFuseRelu]> {
  let summary = "matmul operator";

  let description = [{
    1.Op Introduction
    The MatMul operator performs two-dimensional matrix multiplication between two input tensors.

    2.Math formula
    ```math
        output = input x right + bias
    ```

    3.activation and weight
    input(act.): the first input tensor;
    right(act.): the second input tensor;
    bias(w.):  an optional tensor can be added to the result of the matrix multiplication. ;

    4.attribute
    right_transpose: whether to transpose the right input tensor before performing the multiplication.;
    left_transpose: whether to transpose the input tensor before performing the multiplication.;
    output_transpose: whether to transpose the output tensor after the multiplication.;
    hdim_is_batch: whether the first dimension of the input tensor represents the batch size.;
    keep_dims: whether to keep the dimensions of the output tensor the same as the input tensors.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    weight_bits: the bit-width for quantizing the weights.;
    in_int4_scale: the scaling factor for input tensors that are quantized to 4 bits.;
    in_int4_zp: the zero point for input tensors that are quantized to 4 bits.;
    out_int8_scale:the scaling factor for the output tensor when quantized to 8 bits.;
    out_int8_zp: This optional attribute specifies the zero point for the output tensor that is quantized to 8 bits.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$right,
    AnyTensorOrNone:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$right_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$left_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$output_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$hdim_is_batch,
    DefaultValuedAttr<BoolAttr, "true">:$keep_dims,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<I64Attr>:$weight_bits,
    OptionalAttr<F64Attr>:$in_int4_scale,
    OptionalAttr<F64Attr>:$in_int4_zp,
    OptionalAttr<F64Attr>:$out_int8_scale,
    OptionalAttr<F64Attr>:$out_int8_zp
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    matmul_attr_t parseParam();
    matmul_attr_t dynparseParam();
  }];
}

def Top_A16MatMulOp: Top_Op<"A16MatMul"> {
  let summary = "w8a16 / w4a16 matmul operator";

  let description = [{
    The special matrix multiplication designed for LLM Linear Layer.
    Weight is saved in int8 with f16 per-channel quant scale.

    y_f16 = x_f16 x (quantized_w.to(f16) * scale_f16)
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$scale,
    AnyTensor:$zp,
    AnyTensorOrNone:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$right_transpose,
    DefaultValuedAttr<I64Attr, "128">:$q_group_size,
    DefaultValuedAttr<I64Attr, "8">:$weight_bits
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    a16matmul_attr_t parseParam();
  }];
}

def Top_AttentionOp: Top_Op<"Attention"> {
  let summary = "Attention operator";

  let description = [{
    1.Op Introduction
    Performs a multi head attention block. https://en.wikipedia.org/wiki/Attention_(machine_learning)
    This block has Q_w, K_w,V_w, O_w and mask;
    This operator utilizes query, key, and value weights (denoted as (Q_w), (K_w), and (V_w), respectively) to compute attention scores and
    generate output representations.

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


    4.attribute
    scale: a scaling factor applied to the attention scores before they are passed through the softmax function.;
    head: the number of attention heads to use in the multi-head attention mechanism.;
    dim: the size of the input features or the size of the query, key, and value vectors.;
    scale_param: adjust the scaling factor for the attention scores, It allows for flexibility in tuning the attention mechanism.;
    zp_param: the zero-point parameters for quantization,;
    has_bias: whether the attention mechanism includes bias terms in its computations.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$keys,
    AnyTensor:$values,
    AnyTensor:$queries_weight,
    AnyTensorOrNone:$queries_bias,
    AnyTensor:$keys_weight,
    AnyTensorOrNone:$keys_bias,
    AnyTensor:$values_weight,
    AnyTensorOrNone:$values_bias,
    AnyTensor:$out_weight,
    AnyTensorOrNone:$out_bias,
    AnyTensorOrNone:$mask,
    F64Attr:$scale,
    I64Attr:$head,
    DefaultValuedAttr<I64Attr, "0">:$dim,
    DefaultValuedAttr<F64ArrayAttr, "{1.0}">:$scale_param,
    DefaultValuedAttr<I64ArrayAttr, "{0}">:$zp_param,
    DefaultValuedAttr<I64Attr, "0">:$has_bias
  );

  let results = (outs AnyTensor:$output);
}

def Top_EinsumOp: Top_Op<"Einsum", [SupportFuseRelu]> {
  let summary = "Einsum operator";

  let description = [{
    1.Op Introduction
    # https://pytorch.org/docs/1.13/generated/torch.einsum.html?highlight=einsum#torch.einsum
    The Einsum operator implements Einstein summation notation, which provides a concise way to specify tensor operations.

    2.Math formula
    ```math
        \mathrm{Output}=\sum_{i\in I}A_iB_j\ldots
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    mode: determines how the input tensors will be combined based on the specified subscripts. ;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    EinsumModeAttr:$mode
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_PadOp:Top_Op<"Pad"> {
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

    4.attribute
    paddings: defines how much padding to add before and after the contents of the input tensor for each dimension. ;
    val: the value to be used for padding the input tensor. ;
    mode: the padding mode include constant(Pads with a constant value); reflect(Pads with a reflection of the tensor values); replicate(Pads by replicating the edge values of the tensor).;
  }];
  let arguments = (ins
    AnyTensor:$input,
    Optional<AnyTensorOrNone>:$paddingsT,
    I64ArrayAttr:$paddings,
    DefaultValuedAttr<F64Attr, "0.0">:$val,
    PaddingModeAttr:$mode
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_PermuteOp: Top_Op<"Permute"> {

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

    4.attribute
    order: An array of integers specifying the permutation order of the input tensor's dimensions.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64ArrayAttr:$order
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    permute_attr_t parseParam();
  }];
}

def Top_TransposeOp: Top_Op<"Transpose"> {

  let summary = "Transpose operator";

  let description = [{
    1.Op Introduction
    Transpose on input.

    2.Math formula
    ```math
        output(dim1, dim0) = TransposeOp(input(dim0, dim1))
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    dim0: the first dimension of input tensor.;
    dim1: the second dimension of input tensor.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    SI32Attr:$dim0,
    SI32Attr:$dim1
  );
  let results = (outs AnyTensor:$output);
}

def Top_ShuffleChannelOp: Top_Op<"ShuffleChannel"> {
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
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64Attr:$group
  );

  let results = (outs AnyTensor:$output);
}

def Top_ReluOp: Top_Op<"Relu", [SupportPermuteMove]> {
  let summary = "Relu operator";

  let description = [{
    1.Op Introduction
    ReLU with a scalar maximum value. if limit is zero, do not use upper limit.

    2.Math formula
    ```math
        output = ReluOp(input) -> (0, 1)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def Top_ListOp: Top_Op<"List"> {
  let summary = "List operator";
  let description = [{
    1.Op Introduction
    gen by torch prim::ListConstruct, y = [a, b]
    output shape is [1]

    2.Math formula
    ```math
        output = [input1, input2, ..., inputN]
    ```

    3.activation and weight
    input(act.): Variadic input tensor.;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ReshapeOp:Top_Op<"Reshape"> {
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
    shapeT(act.): an optional input tensor that specifies the desired shape for the output tensor.;

    4.attribute
    shape: 0: keep dim from input; -1: left dim from input.;
    flatten_start_dim: the starting dimension from which to begin flattening the input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    Optional<AnyTensor>:$shapeT,
    OptionalAttr<I64ArrayAttr>:$shape,
    DefaultValuedAttr<I64Attr, "-1">:$flatten_start_dim
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Top_ViewOp:Top_Op<"View"> {
  let summary = "View operation";
  let description = [{
    1.Op Introduction
    gen by torch aten::view, the view operation allows for changing the shape of the tensor without altering its data.
    0: keep dim from input
    -1: left dim from input

    2.Math formula
    ```math
        output = ViewOp(input, shape)
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

def Top_FlattenOp:Top_Op<"Flatten"> {
  let summary = "Flatten operation";
  let description = [{
    1.Op Introduction
    gen by torch aten::flatten or onnx, the flatten operation collapses the specified dimensions of the tensor into a single dimension,
    effectively reducing the number of dimensions of the tensor.

    2.Math formula
    ```math
        output = FlattenOp(input, start_dim, end_dim)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    start_dim: the first dimension to flatten.;
    end_dim: the last dimension to flatten.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<I64Attr, "0">:$start_dim,
    DefaultValuedAttr<I64Attr, "-1">:$end_dim
  );
  let results = (outs AnyTensor:$output);
}

def Top_FAttentionOp: Top_Op<"FAttention"> {
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

def Top_ReverseOp:Top_Op<"Reverse"> {
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
    AnyTensor:$input,
    I64Attr:$axis
  );
  let results = (outs AnyTensor:$output);
}

def Top_SigmoidOp : Top_Op<"Sigmoid", [SupportPermuteMove]> {
  let summary = " Exp operator,  scale * Sigmoid + bias";
  let description = [{
    1.Op Introduction
    Y = scale * Sigmoid(x) + bias
    if log --> Y = Log(scale * Sigmoid(x) + bias)

    2.Math formula
    ```math
        output = scale * Sigmoid(input) + bias

    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    scale: a scaling factor applied to the attention scores before they are passed through the softmax function.;
    bias: added to the result of the matrix multiplication. ;
    log: whether the output should be computed using the logarithm of the scaled sigmoid function. ;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<F64Attr, "1">:$scale,
    DefaultValuedAttr<F64Attr, "0">:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$log,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );

  let results = (outs AnyTensor:$output);
}

def Top_SignOp : Top_Op<"Sign"> {
  let summary = " Sign Operator";
  let description = [{
    1.Op Introduction
    Calculate the sign of the given input tensor element-wise.
    If input > 0, output 1. if input < 0, output -1. if input == 0,
    output 0.

    2.Math formula
    ```math
        output = SignOp(input) -> 0/1
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (
    ins AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_SoftsignOp: Top_Op<"Softsign"> {
  let summary = " Softsign Operator";
  let description = [{
    1.Op Introduction
    The Softsign operation is an activation function that provides a smooth approximation of the sign function.

    2.Math formula
    ```math
        output = input / (1 + |input|)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (
    ins AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_SizeOp: Top_Op<"Size"> {
  let summary = "Size operator";
  let description = [{
    1.Op Introduction
    gen by torch aten::size. The Size operation retrieves the size (or shape) of the given input tensor.

    2.Math formula
    ```math
        output = SizeOp(input) ->[dim0, dim1, dim2..., dimN]
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    axis: the dimension of reverse;
  }];

  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<SI32Attr>:$axis
  );

  let results = (outs AnyTensor:$output);
}

def Top_ArangeOp: Top_Op<"Arange"> {
  let summary = "Arange operator";
  let description = [{
    1.Op Introduction
    gen by torch aten::arange.

    2.Math formula
    ```math
        output = [x | x = start + n * step, n is an integer, and start ≤ x < end]
    ```

    3.activation and weight
    start(act.): The starting value of the sequence. This can be a tensor or None. If None, it defaults to 0.;
    end(act.): The exclusive upper limit of the sequence.;
    step(act.): The increment between each value in the sequence. This can be a tensor or None. If None, it defaults to 1.;
  }];

  let arguments = (ins
    AnyTensorOrNone:$start,
    AnyTensor:$end,
    AnyTensorOrNone:$step
  );

  let results = (outs AnyTensor:$output);
}

def Top_RandnLikeOp: Top_Op<"RandnLike"> {
  let summary = "randn_like operator, y = randn_like(x)";
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

  let results = (outs
    AnyTensor:$output
  );
}

def Top_RangeOp: Top_Op<"Range"> {
  let summary = "Range operator";
  let description = [{
    1.Op Introduction
    onnx range op.
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
    AnyTensorOrNone:$start,
    AnyTensor:$limit,
    AnyTensorOrNone:$delta
  );

  let results = (outs AnyTensor:$output);
}

def Top_ConstantFillOp: Top_Op<"ConstantFill"> {
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
}

def Top_SiLUOp : Top_Op<"SiLU"> {
  let summary = " SiLU operator,  y = x * Sigmoid(x)";
  let description = [{
    1.Op Introduction
    An activation function.
    Smooth nonlinear transformation is provided to avoid the gradient disappearance problem of ReLU.;

    2.Math formula
    ```math
        output = input * Sigmoid(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (
    ins AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_GELUOp : Top_Op<"GELU"> {
  let summary = " GELU operator,  0.5x * (1.0 + tf.erf(x / tf.sqrt(2.0)))";
  let description = [{
    1.Op Introduction
    An activation function based on Gaussian error function.;

    2.Math formula
    ```math
        Y = 0.5*input * (1.0 + tf.erf(input / tf.sqrt(2.0)))
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
    approx_mode: include three mode.
                  normal: This mode uses the exact mathematical form of the GELU function;
                  tanh: This mode uses tanh to speed up computation;
                  sigm: This mode uses sigmoid to speed up computation;

  }];
  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode,
    DefaultValuedAttr<GELUModeAttr, "\"normal\"">:$approx_mode
  );

  let results = (outs AnyTensor:$output);
}

def Top_SplitOp: Top_Op<"Split"> {
  let summary = "Split operator";

  let description = [{
    1.Op Introduction
    Split input tensor into a list of tensors.

    2.Math formula
    ```math
        output = input[i * split_size: (i + 1) * split_size] for i = 0, 1, ... num - 1
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attribute
    axis: the dimension of split;
    num: the number of equal parts to split the input tensor into along the specified axis.;
    split_size: the exact sizes of each split along the specified axis.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    SI32Attr:$axis,
    I64Attr:$num,
    OptionalAttr<I64ArrayAttr>:$split_size
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
  let hasCanonicalizer = 1;
}

def Top_SliceOp: Top_Op<"Slice"> {
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

    4.attribute
    offset: An array of the starting indices for slicing along each axis.;
    steps: An array of the step sizes for slicing along each axis.;
    ends: An array of the ending indices for slicing along each axis.;
    axes: An array of the axes along which to perform the slicing operation.;
    hasparamConvert_axes: whether parameter conversion is needed for the specified axes.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensorOrNone:$offsetT,
    AnyTensorOrNone:$endsT,
    AnyTensorOrNone:$stepsT,
    I64ArrayAttr:$offset,
    I64ArrayAttr:$steps,
    I64ArrayAttr:$ends,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$axes,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$hasparamConvert_axes
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    void paramConvert();
  }];
}

def Top_StridedSliceOp: Top_Op<"StridedSlice"> {
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
    AnyTensor:$input,
    AnyTensor:$starts,
    AnyTensor:$ends,
    AnyTensor:$strides,
    I64Attr:$begin_mask,
    I64Attr:$end_mask,
    I64Attr:$ellipsis_mask,
    I64Attr:$new_axis_mask,
    I64Attr:$shrink_axis_mask
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SliceAxisOp: Top_Op<"SliceAxis"> {
  let summary = "Slice operator on one axis";

  let description = [{
    1.Op Introduction
    Slice Operation on input.

    2.Math formula
    ```math
            output[i] = input[starts[j] + i * strides[j]]
    ```

    3.activation and weight
    input(act.): input tensor.;
    axis(w.): the dimension of the input tensor.;
    start(w.): the ending indices for slicing along each axis.;
    step: the step sizes for slicing along each axis.;
    end: the ending indices for slicing along each axis.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$axis,
    AnyTensor:$start,
    AnyTensorOrNone:$step,
    AnyTensor:$end
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SoftmaxOp:Top_Op<"Softmax"> {
  let summary = "Softmax operation";
  let description = [{
    1.Op Introduction
    Integrates some operations related to softmax.

    2.Math formula
    ```math
            \text{output}[i] = \frac{e^{\text{input}[i]}}{\sum_{j} e^{\text{input}[j]}}
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    axis: the dimension of the input tensor.;
    log: when set to true, indicates that the output should be computed in log space.;
    beta: scaling factor.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    SI32Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$log,
    DefaultValuedAttr<F64Attr, "1.0">:$beta,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );
  let results = (outs AnyTensor:$output);
}

def Top_SoftplusOp:Top_Op<"Softplus"> {
  let summary = "Softplus operation";
  let description = [{
    1.Op Introduction
    a smooth approximation of the ReLU (Rectified Linear Unit) activation function.

    2.Math formula
    ```math
            output = ln(exp(input) + 1)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_FloorOp:Top_Op<"Floor"> {
  let summary = "Floor operation";
  let description = [{
    1.Op Introduction
    the Floor function rounds down each element of the input tensor to the nearest integer less than or equal to that element.

    2.Math formula
    ```math
            output = floor(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_TopKOp:Top_Op<"TopK"> {
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

    4.attributes
    axis: the dimension of the input tensor.;
    K: how many of the largest (or smallest, depending on the largest attribute) values will be returned. defaults is -1;
    largest: whether to retrieve the largest or smallest values.;
    sorted: whether the output values should be sorted in descending order (if largest is true) or ascending order (if largest is false).;
    kT: provide a specific tensor for K values. This allows for dynamic specification of K.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    DefaultValuedAttr<I64Attr, "-1">:$K,
    DefaultValuedAttr<BoolAttr, "true">:$largest,
    DefaultValuedAttr<BoolAttr, "true">:$sorted,
    Optional<AnyTensor>:$kT,
    DefaultValuedAttr<BoolAttr, "false">:$replace_topk_indices
  );
  let results = (outs
    AnyTensorOrNone:$values,
    AnyTensorOrNone:$indices
  );
  let hasCanonicalizer=1;
}

def Top_TriluOp:Top_Op<"Trilu"> {
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

def Top_NonZeroOp:Top_Op<"NonZero"> {
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

    4.attributes
    order: the order in which the non-zero indices should be returned.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    NonZeroOrderAttr:$order
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Top_LeakyReluOp : Top_Op<"LeakyRelu"> {
  let summary = "LeakyRelu operation";
  let description = [{
    1.Op Introduction
    LeakyRelu takes input data (Tensor<T>) and an argument alpha,
    and produces one output data (Tensor<T>)
    where the function f(x) = alpha * x for x < 0, f(x) = x for x >= 0,
    is applied to the data tensor elementwise.

    2.Math formula
    ```math
            output = alpha * input, if input < 0
            output = input, if input >= 0
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    alpha: a scalar factor.;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$alpha,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );
  let results = (outs AnyTensor:$output);
}

def Top_UpsampleOp : Top_Op<"Upsample", [SupportFuseRelu]> {
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
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$scale_h,
    I64Attr:$scale_w,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let hasCanonicalizer=1;
  let results = (outs AnyTensor:$output);
}

def Top_MaxUnpoolOp : Top_Op<"MaxUnpool"> {
  let summary = "MaxUnpool operation";
  let description = [{
    1.Op Introduction
    Perform  MaxUnpool on input.

    2.Math formula
    ```math
            output[i, j] = Upsample(input[i / scale_h, j / scale_w])
    ```

    3.activation and weight
    input(act.): input tensor.;
    mask(act.): the positions of the maximum values that were retained during the max pooling operation.;

    4.attributes
    scale_h: the scaling factor for the height (number of rows) of the input tensor.;
    scale_w: the scaling factor for the width (number of columns) of the input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$mask,
    I64Attr:$scale_h,
    I64Attr:$scale_w
  );
  let results = (outs AnyTensor:$output);
}

def Top_LogOp: Top_Op<"Log"> {
  let summary = "Log operator";

  let description = [{
    1.Op Introduction
    Calculates the natural log of the given input tensor, element-wise.

    2.Math formula
    ```math
            output = ln(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_LogBOp: Top_Op<"LogB"> {
  let summary = "LogB operator";

  let description = [{
    1.Op Introduction
    Calculates the log of the given input tensor to the base B, element-wise.

    2.Math formula
    ```math
            output = ln(input) / ln(B)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$base
  );

  let results = (outs AnyTensor:$output);
}

def Top_LRNOp: Top_Op<"LRN"> {
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

    4.attributes
    size: how many neighboring channels are considered during the normalization process.;
    alpha: a scaling factor;
    beta: a scaling factor;
    bias: A floating-point value added to the normalization denominator to prevent division by zero.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$size,
    DefaultValuedAttr<F64Attr, "0.0001">:$alpha,
    DefaultValuedAttr<F64Attr, "0.75">:$beta,
    DefaultValuedAttr<F64Attr, "1.0">:$bias
  );

  let results = (outs AnyTensor:$output);
}

def Top_ExpOp: Top_Op<"Exp"> {
  let summary = "Exp operator";

  let description = [{
    1.Op Introduction
    Calculates the exponent of the given input tensor, element-wise.

    2.Math formula
    ```math
            output = exp(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_ExpandOp: Top_Op<"Expand"> {
  let summary = "Expand operator";

  let description = [{
    1.Op Introduction
    Broadcast the input tensor following the given shape and the broadcast rule.

    2.Math formula
    ```math
            output[i1, i2, i3,...in] = input[j1, j2, j3,...jn]
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    shape: An array of the target shape to which the input tensor will be expanded.;
    shapeT: An optional tensor that can be used to specify the shape dynamically.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$shape,
    Optional<AnyTensor>:$shapeT
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_CosOp: Top_Op<"Cos"> {
  let summary = "Cos operator";

  let description = [{
    1.Op Introduction
    Calculates the Cos of the given input tensor, element-wise.

    2.Math formula
    ```math
            output = cos(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_CoshOp: Top_Op<"Cosh"> {
  let summary = "Cosh operator";

  let description = [{
    1.Op Introduction
    Calculates the Cosh of the given input tensor, element-wise.

    2.Math formula
    ```math
            output = cosh(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_SinOp: Top_Op<"Sin"> {
  let summary = "Sin operator";

  let description = [{
    1.Op Introduction
    Calculates the Sin of the given input tensor, element-wise.

    2.Math formula
    ```math
            output = sin(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_SinhOp: Top_Op<"Sinh"> {
  let summary = "Sinh operator";

  let description = [{
    1.Op Introduction
    Calculates the Sinh of the given input tensor, element-wise.

    2.Math formula
    ```math
            output = sinh(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_ArctanhOp: Top_Op<"Arctanh">{
  let summary = "Arctanh operator";

  let description = [{
    1.Op Introduction
    Calculates the Arctanh of the given input tensor.

    2.Math formula
    ```math
            output = Arctanh(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

// def Top_ArctanOp: Top_Op<"Arctan">{
//   let summary = "Arctan operator";

//   let description = [{
//     Calculates the Arctan of the given input tensor.
//   }];

//   let arguments = (ins
//     AnyTensor:$input
//   );

//   let results = (outs AnyTensor:$output);
// }

def Top_ArccosOp: Top_Op<"Arccos">{
  let summary = "Arccos operator";

  let description = [{
    1.Op Introduction
    Calculates the Arccos of the given input tensor.

    2.Math formula
    ```math
            output = Arccos(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_TanOp: Top_Op<"Tan"> {
  let summary = "Tan operator";

  let description = [{
    1.Op Introduction
    Calculates the tan of the given input tensor, element-wise.

    2.Math formula
    ```math
            output = Tan(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_TanhOp: Top_Op<"Tanh"> {
  let summary = "Tanh operator";

  let description = [{
    1.Op Introduction
    Calculates the tanh of the given input tensor, element-wise.

    2.Math formula
    ```math
            output = Tan(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );

  let results = (outs AnyTensor:$output);
}

def Top_MishOp: Top_Op<"Mish"> {
  let summary = "Mish operator";

  let description = [{
    1.Op Introduction
    Calculates the mish of the given input tensor, element-wise.

    2.Math formula
    ```math
            output[i] = input[i] x tanh(softplus(input[i]))
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_DivOp: Top_Op<"Div", [ScalarProducer, ScalarConsumer]> {
  let summary = "Div operator";

  let description = [{
    1.Op Introduction
    Performs element-wise binary division.

    2.Math formula
    ```math
            output[i] = \frac{inputs[i]}{divisor}
            if is_reverse == True;
            output[i] = \frac{divisor}{inputs[i]}
    ```

    3.activation and weight
    inputs(act.): input tensor.;

    4.attributes
    is_reverse: whether the subtraction operation is performed in reverse order.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SqueezeOp: Top_Op<"Squeeze", [ScalarProducer]> {
  let summary = "Squeeze operator";

  let description = [{
    1.Op Introduction
    The operator squeeze the input shapes by given axis.

    2.Math formula
    ```math
            output = squeeze(input, axes)
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    axes: the dimensions (axes) of the input tensor that should be squeezed (removed).;
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_UnsqueezeOp: Top_Op<"Unsqueeze", [ScalarConsumer]> {
  let summary = "Unsqueeze operator";

  let description = [{
    1.Op Introduction
    The operator unsqueeze the input shapes by given axis.

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
    AnyTensor:$input,
    I64ArrayAttr:$axes
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}


def Top_ClipOp: Top_Op<"Clip"> {
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
  }];

  let arguments = (ins
    AnyTensor:$inputs,
    F64Attr:$min,
    F64Attr:$max
  );

  let results = (outs AnyTensor:$output);
}


def Top_Yuv2rgbFormulaOp: Top_Op<"Yuv2rgbFormula"> {
  let summary = "Yuv2rgb formula operator";

  let description = [{
    1.Op Introduction
    Yuv2rgb formula Operator.

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
    ImageOutFormatAttr:$image_format,
    Yuv2rgbFormulaAttr:$formula_mode,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );

  let results = (outs AnyTensor:$output);
}

def Top_DeconvOp: Top_Op<"Deconv", [SupportFuseRelu]> {
  let summary = "Deconvolution operator";

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
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<I64ArrayAttr>:$output_padding,
    DefaultValuedAttr<BoolAttr, "false">:$dynweight_reorderd,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    deconv_attr_t parseParam();
    deconv_attr_t dynparseParam();
  }];
}

def Top_ScaleOp: Top_Op<"Scale", [SupportFuseRelu]> {
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

    4.attributes
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$scale,
    AnyTensor:$bias,

    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_GRUOp: Top_Op<"GRU"> {
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

    4.attributes
    hidden_size: the number of units in the GRU cell,;
    bidirectional: whether the GRU should be bidirectional;
    linear_before_reset: whether to apply a linear transformation to the input before applying the reset gate.;
    batch_first: the input and output tensors are provided in the shape (batch_size, seq_length, input_size).;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$recurrence,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$initial_h,
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

def Top_LSTMOp: Top_Op<"LSTM"> {
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

    4.attributes
    hidden_size: the number of units in the LSTM cell, ;
    bidirectional: A boolean indicating whether the LSTM should be bidirectional;
    batch_first: the input and output tensors are provided in the shape (batch_size, seq_length, input_size).;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$recurrence,
    AnyTensorOrNone:$bias,
    AnyTensorOrNone:$initial_h,
    AnyTensorOrNone:$initial_c,
    AnyTensorOrNone:$cont,
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

def Top_NmsOp : Top_Op<"Nms"> {
  let summary = " NMS operator";
  let description = [{
    1.Op Introduction
    onnx nms
    used to eliminate redundant overlapping bounding boxes by selecting only the most relevant ones based on their confidence scores.

    2.Math formula
    ```math
        IOU(a, b) = Area(a \cap b) / Area(a \cup b)
    ```

    3.activation and weight
    input(act.): Variadic input tensor.;

    4.attributes
    center_point_box: whether the bounding boxes are defined by their center points.;
    max_output_size: the maximum number of boxes to be output after NMS.;
  }];
  let arguments = (ins
    Variadic<AnyTensor>: $inputs,
    I64Attr: $center_point_box,
    I64Attr: $max_output_size
  );

  let results = (outs AnyTensor:$output);
}

def Top_MatchTemplateOp: Top_Op<"MatchTemplate"> {
  let summary = "opencv MatchTemplate operator";

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
    input(act.): input tensor.;
    match(w.): the template image that will be matched against the input image.;

    4.attributes
    mode: the method of template matching to be used (e.g., correlation, squared difference, etc.).;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$match,
    MatchTemplateModeAttr:$mode
  );

  let results = (outs AnyTensor:$output);
}

def Top_GatherOp: Top_Op<"Gather", [ScalarProducer]> {
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
    indices(w.): the indices of the elements to be gathered from the input tensor. ;

    4.attributes
    keepdims: whether to retain the dimensions of the input tensor in the output.
               If true, will have the same number of dimensions as the input tensor.;
    axis: the dimension of the input tensor.;
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,
    DefaultValuedAttr<BoolAttr, "true">:$keepdims,
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_GatherElementsOp: Top_Op<"GatherElements"> {
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
    indices(w.): the indices of the elements to be gathered from the input tensor. ;

    4.attributes
    axis: the dimension of the input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,

    DefaultValuedAttr<I64Attr, "2">:$axis
  );

  let results = (outs AnyTensor:$output);
}

def Top_TileOp: Top_Op<"Tile"> {
  let summary = "Tile operator";
  let description = [{
    1.Op Introduction
    Perform Tile operation on the given tensor.

    2.Math formula
    ```math
            output[i_1, i_2, i_3,...i_k] = input[i_1 mod d_1, i_2 mod d_2, i_3 mod d_3,..., i_k mod d_k]
    ```
    where d_j represents the corresponding dimension size of the input tensor after tiling.

    3.activation and weight
    input(act.): input tensor.;
    tileT(w.): how many times to replicate the input tensor along each dimension.;

    4.attributes
    tile: the number of times to replicate the input tensor along each dimension.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    Optional<AnyTensor>:$tileT,
    OptionalAttr<I64ArrayAttr>:$tile
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_RepeatOp: Top_Op<"Repeat"> {
  let summary = "Repeat operator";
  let description = [{
    1.Op Introduction
    Perform aten::repeat operation on the given tensor.

    2.Math formula
    ```math
            output[i_1, i_2, i_3,...i_k] = input[i_1 / r_1, i_2 / r_2, i_3 / r_3,..., i_k mod r_k]
    ```
    where r_j represents the corresponding value from the repeats tensor for dimension ( j ).

    3.activation and weight
    input(act.): input tensor.;
    repeats(w.): the number of times to repeat the input tensor along each dimension.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$repeats
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_AbsOp : Top_Op<"Abs"> {
  let summary = " Abs operator";
  let description = [{
    1.Op Introduction
    computes the absolute value of each element in the input tensor.

    2.Math formula
    ```math
            output = abs(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_ModOp : Top_Op<"Mod"> {
  let summary = " Mod operator";
  let description = [{
    1.Op Introduction
    a mathematical operation that calculates the remainder of the division of two numbers (or tensors) element-wise.

    2.Math formula
    ```math
            f(x, y) = Mod(x, y)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
}

def Top_PReluOp : Top_Op<"PRelu"> {
  let summary = "PRelu operator";
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
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$slope
  );
  let hasCanonicalizer = 1;
  let results = (outs AnyTensor:$output);
}

def Top_InterpOp : Top_Op<"Interp"> {
  let summary = "Interp operation";
  let description = [{
    1.Op Introduction
    Perform linear upsample on input.

    2.Math formula
    ```math
            H' = H x scale_h
            W' = W x scale_w
    ```

    3.activation and weight
    input(act.): input tensor.;
    target_shape(w.): the desired shape of the output tensor after interpolation.;

    4.attributes
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    coord_mode: whether the coordinates are normalized (ranging from 0 to 1) or absolute (based on pixel indices).;
    scale_h: the scaling factor for the height (number of rows) of the input tensor.;
    scale_w: the scaling factor for the width (number of columns) of the input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$target_shape,
    InterpModeAttr:$mode,
    InterpCoordModeAttr:$coord_mode,
    DefaultValuedAttr<F64Attr, "-1.0">:$scale_h,
    DefaultValuedAttr<F64Attr, "-1.0">:$scale_w
  );
  let hasCanonicalizer=1;
  let results = (outs AnyTensor:$output);
}

def Top_MeshGridOp : Top_Op<"MeshGrid"> {
  let summary = "MeshGrid operation";
  let description = [{
    1.Op Introduction
    torch mesh grid operation

    2.Math formula
    ```math
            X[i] = x[i mod m] for i = 0, 1, ..., m · n-1
            Y[i] = y[j // m] for j = 0, 1, ..., m · n-1
    ```
    where X, Y will have shape (m, n);

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    is_reverse: whether the subtraction operation is performed in reverse order.;
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    BoolAttr:$is_reverse
  );

  let results = (outs Variadic<AnyTensor>:$outputs);
  let hasCanonicalizer = 1;
}

def GridSamplerPadModeAttr: AnyStrAttrOf<["zeros","border","reflection"]>;
def Top_GridSamplerOp : Top_Op<"GridSampler"> {
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

    4.attributes
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    padding_mode: padding mode for outside grid values, Int attribute [0, 1, 2],
                                representing 'zero' | 'boundary' | 'reflection't.;
    align_corners: whether to align the corners of the input and output tensors.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$grid,
    I64Attr:$mode,
    I64Attr:$padding_mode,
    BoolAttr:$align_corners
  );

  let results = (outs AnyTensor:$output);
}

def Top_ReduceOp : Top_Op<"Reduce", [ScalarProducer]>{
  let summary = "Reduce operation";
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
    grid(w.): The flow-field grid tensor that defines the pixel locations for sampling.;

    4.attributes
    axes: the dimensions (axes) of the input tensor that should be squeezed (removed).;
    keepdims: whether to retain the dimensions of the input tensor in the output.
               If true, will have the same number of dimensions as the input tensor.;
    mode: the type of binary operation to be performed, addition, subtraction, or other types of binary operations.;
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$axes,
    BoolAttr:$keepdims,
    ReduceModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ArgOp : Top_Op<"Arg"> {
  let summary = "Arg operation";
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
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    BoolAttr:$keepdims,
    ArgModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "true">:$select_last_index
  );
  let results = (outs
    AnyTensor:$indices,
    AnyTensorOrNone:$values
  );
  let hasCanonicalizer = 1;
}

def Top_PowOp : Top_Op<"Pow"> {
  let summary = "Pow operation";
  let description = [{
    1.Op Introduction
    computes the element-wise power of an input tensor raised to a specified exponent.

    2.Math formula
    ```math
            output = input ^ n
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    exponent: the power to which each element of the input tensor will be raised.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $exponent
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_Pow2Op : Top_Op<"Pow2"> {
  let summary = "Pow2 operation";
  let description = [{
    1.Op Introduction
    computes the result of raising a constant value ( n ) to the power of each element in the input tensor.

    2.Math formula
    ```math
            output = n ^ input
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    const_val: specifies the constant value to be added to each element of the input tensor(positive, negative, or zero).;
  }];
  let arguments = (ins
    F64Attr: $const_val,
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}

def Top_Pow3Op : Top_Op<"Pow3"> {
  let summary = "Pow3 operation";
  let description = [{
    1.Op Introduction
    computes the element-wise power of two input tensors.

    2.Math formula
    ```math
            output = input1 ^ input2
    ```

    3.activation and weight
    inputs(act.): input tensor.;
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );
  let results = (outs AnyTensor:$output);
}

def Top_SqrtOp : Top_Op<"Sqrt"> {
  let summary = "Sqrt operation";
  let description = [{
    1.Op Introduction
    Computes the square root of the input tensor's element.

    2.Math formula
    ```math
            output = Sqrt(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_WhereOp : Top_Op<"Where"> {
  let summary = "Where operation";
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

    4.attributes
    x_is_const: the tensor for the true branch (tbrn) is a constant.;
    y_is_const: the tensor for the false branch (fbrn) is a constant.;
    x_const_val: the constant value to be used for the true branch if tbrn is not provided or is constant.;
    y_const_val: the constant value to be used for the false branch if fbrn is not provided or is constant.;
  }];
  let arguments = (ins
    AnyTensor:$cond,
    AnyTensorOrNone:$tbrn,
    AnyTensorOrNone:$fbrn,
    DefaultValuedAttr<BoolAttr, "false">:$x_is_const,
    DefaultValuedAttr<BoolAttr, "false">:$y_is_const,
    DefaultValuedAttr<F64Attr, "0.0">:$x_const_val,
    DefaultValuedAttr<F64Attr, "0.0">:$y_const_val
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_MaskedFillOp : Top_Op<"MaskedFill"> {
  let summary = "MaskedFill operation";
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
    AnyTensor:$cond,
    AnyTensor:$brn,
    BoolAttr:$inversed,
    F64Attr:$const_val
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_CompareOp : Top_Op<"Compare", [SupportConstant]> {
  let summary = "Compare operation";
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
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    CompareModeAttr:$mode
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_CompareConstOp : Top_Op<"CompareConst", [ScalarProducer, ScalarConsumer]> {
  let summary = "CompareConst operation";
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
    is_scalar: whether the addition operation is performed with scalar values or tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    CompareModeAttr:$mode,
    F64Attr:$const_val,
    BoolAttr:$inversed,
    DefaultValuedAttr<BoolAttr, "false">:$is_scalar
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ErfOp : Top_Op<"Erf"> {
  let summary = "Erf operation";
  let description = [{
    1.Op Introduction
    Computes the error function of the given input tensor element-wise.

    2.Math formula
    ```math
            output = Erf(input)
    ```

    3.activation and weight
    input(act.): input tensor;
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_HardSigmoidOp : Top_Op<"HardSigmoid"> {
  let summary = "HardSigmoid operation";
  let description = [{
    1.Op Introduction
    a piecewise linear function to the input tensor element-wise.
    hardsigmoid(x; alpha, beta) := min(max(alpha*x + beta, 0), 1).

    2.Math formula
    ```math
            output[i] = min(max(alpha * input[i] + beta, 0), 1)
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    alpha: scalar;
    beta: scalar;
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$alpha,
    F64Attr:$beta
  );
  let results = (outs AnyTensor:$output);
}

def Top_HardSwishOp : Top_Op<"HardSwish"> {
  let summary = "HardSwish operation";
  let description = [{
    1.Op Introduction
    hardswish(x) := x * hardsigmoid(x; 1/6, 0.5)

    2.Math formula
    ```math
            output[i] = input[i] * min(max(1/6 * input[i] + 0.5, 0), 1)
    ```

    3.activation and weight
    input(act.): input tensor;
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_SwishOp : Top_Op<"Swish"> {
  let summary = "Swish operation";
  let description = [{
    1.Op Introduction
    hardswish(x) := x * sigmoid(x * beta)

    2.Math formula
    ```math
            output[i] = input[i] * sigmoid(input[i] * beta)
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    beta: scalar;
    round_mode: how values are rounded during the conversion from higher precision to lower precision.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$beta,
    DefaultValuedAttr<RoundModeAttr, "\"HalfAwayFromZero\"">:$round_mode
  );
  let results = (outs AnyTensor:$output);
}

def Top_PriorBoxOp : Top_Op<"PriorBox"> {
  let summary = "PriorBox operation";
  let description = [{
    1.Op Introduction
    Intended for use with MultiBox detection method to generate prior.

    2.Math formula
    ```math
            output[i] = input(x, y, w, h) i in [0, num_priors]
    ```

    3.activation and weight
    inputs(act.): input tensor;

    4.attributes
    min_size: the minimum size of the prior boxes.;
    max_size: the maximum size of the prior boxes.;
    aspect_ratios: A list of aspect ratios for the generated prior boxes.;
    variance: adjust the predicted boxes during the training process.;
    clip: whether the prior boxes should be clipped to the image boundaries.;
    step_h: The vertical step size for generating prior boxes.;
    step_w: The horizontal step size for generating prior boxes.;
    img_h: The height of the input image.;
    img_w: The width of the input image.;
    offset: A value used to offset the prior boxes from their calculated positions.;
    num_priors: The number of prior boxes to be generated for each location in the feature map.;
    use_default_aspect_ratio: whether to use a default aspect ratio (1.0);
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    F64ArrayAttr:$min_size,
    F64ArrayAttr:$max_size,
    F64ArrayAttr:$aspect_ratios,
    F64ArrayAttr:$variance,
    DefaultValuedAttr<BoolAttr, "true">:$clip,
    F64Attr:$step_h,
    F64Attr:$step_w,
    I64Attr:$img_h,
    I64Attr:$img_w,
    DefaultValuedAttr<F64Attr, "0.5">:$offset,
    I64Attr:$num_priors,
    DefaultValuedAttr<BoolAttr, "true">:$use_default_aspect_ratio
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_DetectionOutputOp : Top_Op<"DetectionOutput"> {
  let summary = "DetectionOutput operation";
  let description = [{
    1.Op Introduction
    takes the predicted bounding boxes, class scores, and other relevant information to produce the final detection results.

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
    DefaultValuedAttr<F64Attr, "1">:$eta
  );
  let results = (outs AnyTensor:$output);
}

def Top_YoloDetectionOp : Top_Op<"YoloDetection"> {
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
    Variadic<AnyTensor>:$inputs,
    I64Attr:$net_input_h,
    I64Attr:$net_input_w,
    F64Attr:$nms_threshold,
    F64Attr:$obj_threshold,
    I64Attr:$keep_topk,
    F64ArrayAttr:$anchors,
    YoloVersionAttr:$version,
    DefaultValuedAttr<I64Attr, "80">:$class_num,
    DefaultValuedAttr<I64Attr, "3">:$num_boxes,
    DefaultValuedAttr<BoolAttr, "false">:$agnostic_nms
  );

  let results = (outs AnyTensor:$output);
}

def Top_QuantizeLinearOp : Top_Op<"QuantizeLinear"> {
  let summary = "Linear quantize operation";
  let description = [{
    1.Op Introduction
    QuantizeLinear(x) := saturate ((x / y_scale) + y_zero_point)

    2.Math formula
    ```math
        output[i] = saturate((input[i] / y_scale[j]) + y_zero_point[j])
    ```
    i indexes the elements of the input tensor, and j corresponds to the specific dimension or channel.

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    y_scale: Each element to a specific output channel or dimension, determining how much to scale the input values.;
    y_zero_point: an array of zero points used in the quantization process.;
    axis: the dimension of the input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64ArrayAttr:$y_scale,
    I32ArrayAttr:$y_zero_point,
    DefaultValuedAttr<I64Attr, "1">:$axis
  );

  let results = (outs AnyTensor:$output);
}

def Top_LayerNormOp : Top_Op<"LayerNorm"> {
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

    4.attributes
    normalized_shape: the shape of the input tensor that will be normalized.;
    axis: the dimension of the input tensor.;
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$normalized_shape,
    SI32Attr:$axis,
    F64Attr:$eps
  );
  let results = (outs
    AnyTensor:$output
  );
  let hasCanonicalizer = 1;
}

def Top_IndexPutOp : Top_Op<"IndexPut"> {
  let summary = "Index_put_ operation";
  let description = [{
    1.Op Introduction
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

    4.attributes
    accumulate: whether the operation should accumulate values at the specified indices or replace them.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,
    AnyTensor:$values,
    DefaultValuedAttr<BoolAttr, "false">:$accumulate
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Top_InstanceNormOp : Top_Op<"InstanceNorm"> {
  let summary = "Instance Norm operation";
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

    4.attributes
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    F64Attr:$eps
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Top_GroupNormOp : Top_Op<"GroupNorm"> {
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

    4.attributes
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
    num_groups: the number of groups to divide the input channels into for normalization.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    I64Attr:$num_groups,
    F64Attr:$eps
  );
  let results = (outs
    AnyTensor:$output
  );
  let hasCanonicalizer = 1;
}

def Top_PixelNormOp : Top_Op<"PixelNorm"> {
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

    4.attributes
    eps: a small constant added to the denominator during normalization to prevent division by zero.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    F64Attr:$eps
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Top_ProposalOp: Top_Op<"Proposal"> {
  let summary = "Proposal operator";

  let description = [{
    1.Op Introduction
    generate candidate bounding boxes primarily in object detection tasks.

    2.Math formula
    ```math
        1.anchor box generation and regression
            a_i = (x + \delta(x), y + \delta(y), w * e ^ \delta(w), e ^ \delta(h))
        2.obj score
            s_i = sigmoid(output_ModelPredict(a_i))
        3.Final select output
            output = NMS(s_i)
    ```
    3.activation and weight
    input(act.): input tensor;

    4.attributes
    net_input_h: net input height.;
    net_input_w: net input width.;
    feat_stride: anchor box stride size.;
    anchor_base_size: anchor box base size.;
    rpn_obj_threshold: obj threshold.;
    rpn_nms_threshold: nms threshold for generate proposal boxes.;
    rpn_nms_post_top_n: keep num boxes after nms.;
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    I64Attr:$net_input_h,
    I64Attr:$net_input_w,
    I64Attr:$feat_stride,
    I64Attr:$anchor_base_size,
    F64Attr:$rpn_obj_threshold,
    F64Attr:$rpn_nms_threshold,
    I64Attr:$rpn_nms_post_top_n
  );

  let results = (outs AnyTensor:$output);
}

def Top_ROIPoolingOp : Top_Op<"ROIPooling">  {

  let summary = "ROIPooling operator";

  let description = [{
    1.Op Introduction
    Max pooling on ROI.

    2.Math formula
    ```math
            output(pooled_h, pooled_w) = max_pooling(input(h, w) in ROI(spatial_scale))
    ```
    3.activation and weight
    input(act.): input tensor;

    4.attribute
    pooled_h: pooled output height.;
    pooled_w: pooled output width.;
    spatial_scale: adjust the ROI coordinates.;
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    I64Attr:$pooled_h,
    I64Attr:$pooled_w,
    F64Attr:$spatial_scale
  );

  let results = (outs AnyTensor:$output);
}


def Top_DequantizeLinearOp : Top_Op<"DequantizeLinear"> {
  let summary = "Linear dequantize operation";
  let description = [{
    1.Op Introduction
    DequantizeLinear(x) := (x - x_zero_point) * x_scale

    2.Math formula
    ```math
            output = (input - x_zero_point) * x_scale
    ```
    3.activation and weight
    input(act.): input tensor;

    4.attribute
    x_scale: convert the quantized values back original;
    x_zero_point: adjust the zero point of quantized values, represents4 that corresponds to the original value of zero.;
    axis: the dimension of the input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64ArrayAttr:$x_scale,
    I32ArrayAttr:$x_zero_point,
    DefaultValuedAttr<I64Attr, "1">:$axis
  );

  let results = (outs AnyTensor:$output);
}

def Top_FrcnDetectionOp : Top_Op<"FrcnDetection"> {
  let summary = "Faster rcnn detection operator";

  let description = [{
    1.Op Introduction
    outputs the detected classes and their corresponding bounding boxes,
    filtering results based on specified thresholds.

    2.Math formula
    ```math
            output = FrcnDetection(inputs, class_num, obj_threshold, nms_threshold, keep_topk)
    ```
    3.activation and weight
    inputs(act.): input tensor;

    4.attributes
    class_num: detection class num.;
    obj_threshold: object threshold.;
    nms_threshold: nms threshold.;
    keep_topk: keep top k.;
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    I64Attr:$class_num,
    F64Attr:$obj_threshold,
    F64Attr:$nms_threshold,
    I64Attr:$keep_topk
  );

  let results = (outs AnyTensor:$output);
}

def Top_CopyOp: Top_Op<"Copy"> {
  let summary = "Copy operator";

  let arguments = (
    ins AnyTensor:$input,
    I64ArrayAttr:$shape,
    I64ArrayAttr:$input_stride,
    I64ArrayAttr:$output_stride
  );

  let results = (outs AnyTensor:$output);
}

def Top_CscOp: Top_Op<"Csc"> {
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
    pixel_format: required, pixel format type.;
    y_align: width alignment of channel y.;
    w_align: width alignment of channel uv.;
    channel_align: alignment of channel.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$pixel_format,
    BoolAttr:$aligned,
    I64Attr:$y_align,
    I64Attr:$w_align,
    I64Attr:$channel_align
  );

  let results = (outs AnyTensor:$output);
}

def Top_LutOp: Top_Op<"Lut">{
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
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$table
  );

  let results = (outs AnyTensor:$output);
}

def Top_ScaleLutOp: Top_Op<"ScaleLut"> {
  let summary = "Scale by lut operator";

  let description = [{
    1.Op Introduction
    Performs scale on input, y = input * scale + bias.

    2.Math formula
    ```math
            output = input * scale + bias
    ```
    3.activation and weight
    input(act.): input tensor;

    4.attributes
    scale: each channel scale.;
    bias: each channel bias.;
    sign: if output is signed.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    F64ArrayAttr:$scale,
    F64ArrayAttr:$bias,
    DefaultValuedAttr<BoolAttr, "true">:$sign
  );

  let results = (outs AnyTensor:$output);
}

def Top_SwapChannelOp: Top_Op<"SwapChannel"> {
  let summary = "swap channel operator, normally RGB <=> BGR";

  let description = [{
    1.Op Introduction
    Swap Channel on input.

    2.Math formula
    ```math
            output(h, w, c) = input(h, w, channel_order)
    ```
    3.activation and weight
    input(act.): input tensor;

    4.attributes
    channel_order: channel swap order.;
    quant: a QuantParam struct attributes.;
    name: name for calibration, comparing, or debug.;
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64ArrayAttr:$channel_order
  );

  let results = (outs AnyTensor:$output);
}

def Top_SwapDimInnerOp: Top_Op<"SwapDimInner"> {
  let summary = "if offset is not 0, split there and swap first part and second part of it";

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
    ins AnyTensor:$input,
    I64ArrayAttr:$offset
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ScatterElementsOp: Top_Op<"ScatterElements"> {
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

    4.attributes
    axis: the dimension of the input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,
    AnyTensor:$updates,
    I64Attr:$axis,
    DefaultValuedAttr<I64Attr, "0">:$reduction,
    DefaultValuedAttr<BoolAttr, "false">:$nc_can_split
  );

  let results = (outs AnyTensor:$output);
}

def Top_ScatterNDOp: Top_Op<"ScatterND"> {
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

    3.activation and weight
    input_data(act.): input tensor;
    indices(w.): Tensor of rank q >= 1.;
    updates(w.): Tensor of rank q + r - indices_shape[-1] - 1.;

    4.attributes
    reduction: Type of reduction to apply: none (0 default), add(1), sub(2), max(3), min(4), mul(5).;
  }];

  let arguments = (ins
    AnyTensor:$input_data,
    AnyTensor:$indices,
    AnyTensor:$updates,
    DefaultValuedAttr<I32Attr, "0">:$reduction
  );

  let results = (outs AnyTensor:$output);
}

def Top_RoiAlignOp: Top_Op<"RoiAlign"> {
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

    3.activation and weight
    input(act.): input tensor(4D);
    rois(w.): RoIs (Regions of Interest) to pool over.;
          rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...].;

    4.attributes
    mode: the type of comparison to be performed between the two input tensors.
          mdoe include Equal, Not Equal, Less Than, Less Than or Equal, Greater Than and Greater Than or Equal;
    output_height: the height of the output feature maps.;
    output_width: the width of the output feature maps.;
    sampling_ratio: the number of sampling points in each direction (height and width).;
    spatial_scale: a scaling factor that maps the input coordinates (RoIs) to the input feature map's scale.;
    align_corners: whether to align the corners of the input and output tensors.;
    batch_indices: 1-D tensor with each element denoting the index of the corresponding image in the batch.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$rois,
    RoiAlignModeAttr:$mode,
    I64Attr:$output_height,
    I64Attr:$output_width,
    I64Attr:$sampling_ratio,
    F64Attr:$spatial_scale,
    BoolAttr:$align_corners
  );

  let results = (outs AnyTensor:$output);
}

def Top_RoiExtractorOp: Top_Op<"RoiExtractor"> {
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
        output_i = RoiAlign(input, rois_i, output_height, output_width, sampling_ratio, spatial_scale, align_corners)
    ```

    3.activation and weight
    inputs(act.): input tensor;
    rois(w.): RoIs (Regions of Interest) to pool over.;
          rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...].;
    target_lvls(w.): 1-D tensor with each element denoting the index of the corresponding image in the batch.;

    4.attributes
    mode: the type of comparison to be performed between the two input tensors.
          mdoe include Equal, Not Equal, Less Than, Less Than or Equal, Greater Than and Greater Than or Equal;
    num_levels: The number of levels in the feature pyramid.;
    output_height: the height of the output feature maps.;
    output_width: the width of the output feature maps.;
    sampling_ratio: the number of sampling points in each direction (height and width).;
    spatial_scale: a scaling factor that maps the input coordinates (RoIs) to the input feature map's scale.;
    align_corners: whether to align the corners of the input and output tensors.;
    is_static: whether the operation has a static shape.;
  }];

  let arguments = (ins
    // AnyTensor:$feature_0,
    // AnyTensor:$feature_1,
    // AnyTensor:$feature_2,
    // AnyTensor:$feature_3,
    AnyTensor:$rois,
    AnyTensor:$target_lvls,
    Variadic<AnyTensor>:$inputs,
    RoiAlignModeAttr:$mode,
    I64Attr:$num_levels,
    I64Attr:$output_height,
    I64Attr:$output_width,
    I64Attr:$sampling_ratio,
    F64ArrayAttr:$spatial_scales,
    BoolAttr:$align_corners,
    BoolAttr:$is_static
  );

  let results = (outs AnyTensor:$output);
}

def Top_PreprocessOp: Top_Op<"Preprocess"> {
  let summary = "FusePreprcess, it's just a placeholder op.";
  let description = [{
    1.Op Introduction
    It may be divided to permute + slice + scale/scale_lut ops.

    2.Math formula
    ```math
            output = Scale(Slice(Permute(input,channel_order), resize_dims), scale) - mean.
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    quant_mode: the mode or method used for quantization during the requantization operation.;
    customization_format: custom format for the input data.;
    channel_order: The order of color channels in the input tensor.;
    resize_dims: resize the input tensor' dimensions.;
    scale: each channel scale.;
    mean: mean values to subtract from each channel for normalization.;
    sign: if output is signed.;
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
  let results = (outs AnyTensor:$output);
}

def Top_MeanStdScaleOp: Top_Op<"MeanStdScale"> {
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

    4.attributes
    quant_mode: the mode or method used for quantization during the requantization operation.;
    customization_format: custom format for the input data.;
    channel_order: The order of color channels in the input tensor.;
    scale: each channel scale.;
    mean: mean values to subtract from each channel for normalization.;
    sign: if output is signed.;
    std: standard deviation values for each channel.;
    zero_points: zero point values for each channel.;
    resize_dims: resize the input tensor' dimensions.;
    rounding_mode: The rounding method to use during quantization.;
  }];
  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$quant_mode,
    StrAttr:$customization_format,
    StrAttr:$channel_order,
    DefaultValuedAttr<BoolAttr, "true">:$sign,
    F64ArrayAttr:$scale,
    F64ArrayAttr:$std,
    F64ArrayAttr:$mean,
    F64ArrayAttr:$zero_points,
    I64ArrayAttr:$resize_dims,
    StrAttr:$rounding_mode
  );
  let results = (outs AnyTensor:$output);
}

def Top_DepackRawOp: Top_Op<"DepackRaw"> {
  let summary = "Postprocess for raw image.";
  let description = [{
    1.Op Introduction
    postprocess raw image from 4 channels to mixed pixel pattern,
    remove padding first if padded before,
    then for each channel depack to 2 * 2 image block.

    2.Math formula
    ```math
            output = Depack(RemovePadding(input, padding_h, padding_w), channel_order)
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    padding_h: The height of the padding.;
    padding_w: The width of the padding.;
    white_level: The maximum intensity value for white in the image data.;
    black_level: The minimum intensity value for black in the image data.;
    channel_order: The order of color channels in the input tensor.;
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

def Top_Mmap2RgbmapOp: Top_Op<"Mmap2Rgbmap"> {
  let summary = "isp mmap2rgbmap.";
  let description = [{
    isp mmap2rgbmap.
  }];
  let arguments = (
    ins AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_RetinaFaceDetectionOp : Top_Op<"RetinaFaceDetection"> {
  let summary = "RetinaFaceDetection operator";
  let description = [{
    1.Op Introduction
    Perform retinaface detection on feature map

    2.Math formula
    ```math
            output = NMS(Filter(Detect(inputs), confidence_threshold), nms_threshold, keep_topk)
    ```

    3.activation and weight
    inputs(act.): input tensor;

    4.attributes
    nms_threshold: nms threshold.;
    confidence_threshold: classification confidence threshold.;
    keep_topk: after nms, keep bbox num.;
  }];
  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    F64Attr:$nms_threshold,
    F64Attr:$confidence_threshold,
    I64Attr:$keep_topk
  );
  let results = (outs AnyTensor:$output);
}

def Top_EluOp : Top_Op<"Elu"> {
  let summary = "Elu operation";
  let description = [{
    1.Op Introduction
    Elu takes input data (Tensor<T>) and an argument alpha,
    and produces one output data (Tensor<T>)
    where the function f(x) = alpha * (e^x - 1) for x <= 0, f(x) = x for x > 0,
    is applied to the data tensor elementwise.

    2.Math formula
    ```math
            output[i] = alpha * (e ^ input - 1) if input[i] < 0 && input[i] == 0
                        input[i]
    ```

    3.activation and weight
    input(act.): input tensor;

    4.attributes
    alpha: scalar.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$alpha
  );
  let results = (outs AnyTensor:$output);
}

def Top_YieldOp : Top_BaseOp<"Yield", [Terminator, HasParent<"IfOp, LoopOp">]> {
  let summary = "Yield operation";
  let description = [{
    1.Op Introduction
    The `top.Yield` operation represents a return operation within an subgraph.
    The operation takes variable number of operands and produces no results.
    This operation is not part of the standard and was added to assist tpu-mlr.

    2.Math formula
    ```math
            output = Yield(operands)
    ```
  }];

  let arguments = (ins Variadic<AnyTensor>:$operands);
}

def Top_IfOp : Top_Op<"If"> {
  let summary = "if operation";
  let hasVerifier = 1;
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
  let results = (outs Variadic<AnyTensor>:$output);
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

def Top_LoopOp : Top_Op<"Loop"> {
  let summary = "Loop operation";
  let description = [{
  Generic Looping construct. This loop has multiple termination conditions:

  1. Trip count. Iteration count specified at runtime. Set by
     specifying the input M. Optional. Set to empty string to omit.
     Note that a static trip count (specified at graph construction time) can be
     specified by passing in a constant node for input M.
  2. Loop termination condition. This is an input to the op that determines
     whether to run the first iteration and also a loop-carried dependency for
     the body graph. The body graph must yield a value for the condition variable,
     whether this input is provided or not.

  This table summarizes the operating modes of this operator with equivalent
  C-style code:

      Operator inputs defined as (max_trip_count, condition_var).

      input (\"\", \"\"):
          for (int i=0; ; ++i) {
            cond = ... // Note this value is ignored, but is required in the body
          }

      input (\"\", cond) // Note this is analogous to a while loop
          bool cond = ...;
          for (int i=0; cond; ++i) {
            cond = ...;
          }

      input (\"\", 1) // Note this is analogous to a do-while loop
          bool cond = true
          for (int i=0; cond; ++i) {
            cond = ...;
          }

      input (trip_count, \"\") // Note this is analogous to a for loop
          int trip_count = ...
          for (int i=0; i < trip_count; ++i) {
            cond = ...; // ignored
          }

      input (trip_count, cond)
          int trip_count = ...;
          bool cond = ...;
          for (int i=0; i < trip_count && cond; ++i) {
            cond = ...;
          }


  *Sample usage - cond as well as trip count*

      graph predict-net {
        %a = Constant[value = <Scalar Tensor [3]>]()
        %b = Constant[value = <Scalar Tensor [6]>]()
        %keepgoing = Constant[value = <Scalar Tensor [1]>]()
        %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
        %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
        return
      }

      graph body-net (
        %i[INT32, scalar]           // iteration number
        %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
        %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
      ) {
        %my_local = Add(%a, %b_in)
        %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
        %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
        %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
        return %keepgoing_out, %b_out, %user_defined_val
      }

  *Sample equivalent C code*

      {
        /* User-defined code (enclosing scope) */
        int a = 3, b = 6;
        bool keepgoing = true; // Analogous to input cond
        /* End user-defined code */

        /* Implicitly-defined code */
        const int max_trip_count = 10; // Analogous to input M
        int user_defined_vals[]; // Imagine this is resizable
        /* End implicitly-defined code */
        /* initialize loop-carried variables and scan-output variables */
        bool keepgoing_out = keepgoing
        int b_out = b

        for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
          /* Implicitly-defined code: bind actual parameter values
             to formal parameter variables of loop-body */
          bool keepgoing_in = keepgoing_out;
          bool b_in = b_out;

          /* User-defined code (loop body) */
          int my_local = a + b_in; // Reading value \"a\" from the enclosing scope is fine
          b_out = a - b_in;
          keepgoing_out = my_local > b_out;
          user_defined_val = b_in + b_in; // b_in and b_out are different variables
          /* End user-defined code */

          /* Implicitly defined-code */
          user_defined_vals[i] = user_defined_val // accumulate scan-output values
        }
        // int t = my_local; // Can't do this. my_local is not accessible here.

        // The values below are bound to the output variables of the loop and therefore accessible
        // b_out; user_defined_vals; keepgoing_out;
      }

  There are several things of note in this code snippet:

  1. Values from the enclosing scope (i.e. variable \"a\" here) are in scope and can
     be referenced in the inputs of the loop.
  2. Any values computed in the loop body that needs to be used in a subsequent
     iteration or after the loop are modelled using a pair of variables in the loop-body,
     consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
     These are referred to as loop-carried dependences. The loop operation node
     supplies the input value of the input variable for the first iteration, and
     returns the output value of the output variable produced by the final
     iteration.
  3. Scan_output variables are used to implicitly concatenate values computed across
     all the iterations. In the above example, the value of user_defined_val computed
     over all iterations are concatenated and returned as the value of user_defined_vals
     after the loop.
  4. Values created in the body cannot be accessed in the enclosing scope,
     except using the mechanism described above.

  Note that the semantics of this op support \"diagonal\" or \"wavefront\" execution.
  (See Step 3 here for an example:
  https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
  Frontends should emit multi-layer RNNs as a series of While operators (with
  time being the inner looping dimension), with each successive layer consuming
  the scan_outputs from the previous layer, possibly going through several
  point-wise operators (e.g. dropout, residual connections, linear layer).

  The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
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

def Top_ShapeOp : Top_Op<"Shape"> {
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

    4.attributes
    start(w.): the ending indices for slicing along each axis.;
    step: the step sizes for slicing along each axis.;
    end: the ending indices for slicing along each axis.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<I64Attr>:$start,
    OptionalAttr<I64Attr>:$end,
    OptionalAttr<I64Attr>:$step
  );
  let results = (outs AnyTensor:$output);
}

def Top_GatherNDOp: Top_Op<"GatherND"> {
  let summary = "GatherND operator";
  let description = [{
    1.Op Introduction
    This operator is the inverse of ScatterND.

    2.Math formula
    ```math
            output_i = input[indices_i]
    ```

    3.activation and weight
    input(act.): input tensor.;
    indices(w.): which elements to gather from the input tensor.;

    4.attributes
    indice_dims: the number of dimensions in the indices tensor.;
    batch_dims: the number of batch dimensions in the input tensor.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,
    OptionalAttr<I64Attr>:$indice_dims,
    DefaultValuedAttr<I64Attr, "0">:$batch_dims
  );

  let results = (outs AnyTensor:$output);
}

def Top_DeformConv2DOp: Top_Op<"DeformConv2D", [SupportFuseRelu]> {
  let summary = "Deformable Convolution Operator";

  let description = [{
    1.Op Introduction
    In the simplest case, the output value of the layer with input size
    $$(N, C_{\text{in}}, H, W)$$ and output $$(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$$
    can be precisely described as:

    2.Math formula
        - Input: $$(N, C_{in}, H_{in}, W_{in})$$
        - Output: $$(N, C_{out}, H_{out}, W_{out})$$ where
            weight (Tensor): the learnable weights of the module of shape
            $$(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},
            \text{kernel\_size[0]}, \text{kernel\_size[1]})$$

            offset (Tensor): the learnable offsets of the module of shape
            $$(\text{N}, \times{\text{2}}{\text{offset\_groups}{\text{kernel\_size[0]}}{\text{kernel\_size[1]}}},
            \text{H_{\text{out}}}, \text{W_{\text{out}}})$$

            mask (Tensor): the learnable masks of the module of shape
            $$(\text{N}, \times{\text{offset\_groups}{\text{kernel\_size[0]}}{\text{kernel\_size[1]}}},
            \text{H_{\text{out}}}, \text{W_{\text{out}}})$$

            bias (Tensor optional): the learnable bias of the module of shape (out_channels).

          ```math
              H_{out} = \left\lfloor\frac{H_{in}  + \text{padding}[0] + \text{padding}[2] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          ```
          ```math
              W_{out} = \left\lfloor\frac{W_{in}  + \text{padding}[1] + \text{padding}[3] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
          ```
    3.activation and weight
    input(act.): input tensor.;
    filter(w.): the learnable weights of the convolution 2d operation.;
    offset(w.): the learnable offsets of the module of shape.;
    mask(w.): the learnable masks of the module of shape.;
    bias(w.): the learnable bias of the module of shape (out_channels).;

    4.attributes
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    strides: the stride for the cross-correlation, a single number or a tuple.;
    pads: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    group: (optional)Number of blocked connections from input channels to output channels. Default: 1.;
    deform_group: (optional)Number of blocked connections from input channels to output channels. Default: 1.;
    use_mask: whether use mask for input tensor.;
    dilations: controls the spacing between the kernel points;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$offset,
    AnyTensorOrNone:$mask,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads, // top,left,bottom,right
    DefaultValuedAttr<I64Attr, "1">:$group,
    DefaultValuedAttr<I64Attr, "1">:$deform_group,
    DefaultValuedAttr<BoolAttr, "false">:$use_mask,
    OptionalAttr<I64ArrayAttr>:$dilations,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 0;
  let extraClassDeclaration = [{
    deform_conv2d_attr_t parseParam();
  }];
}


def Top_CustomOp: Top_Op<"Custom"> {
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

    4.attributes
    name: the name of the custom operation to be executed.;
    params: A dictionary of parameters.;
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    StrAttr:$name,
    DictArrayAttr:$params
    );

  let results = (outs Variadic<AnyTensor>:$outputs);
}
def Top_ScaleDotProductAttentionOp: Top_Op<"ScaleDotProductAttention"> {
  let summary = "ScaleDotProductAttention operator (pytorch)";

  let description = [{
    1.Op Introduction
    Scale_Dot_Product_Attention Operation for pytorch.

    2.Math formula
    ```math
            output = (softmax(Q * K^T) / sqrt(d_k) + mask) * V
    ```

    3.activation and weight
    query(act.): queries input tensor.;
    key(act.): keys input tensor.;
    value(act.): values input tensor.;
    mask(w.): the learnable masks of the module of shape.;

    4.attributes
    dropout_p: the dropout probability for attention weights.;
    is_causal: whether the attention should be causal.;
    scale: the scaling factor.;
  }];

  let arguments = (ins
    AnyTensor:$query,
    AnyTensor:$key,
    AnyTensor:$value,
    AnyTensorOrNone:$mask,
    DefaultValuedAttr<F64Attr, "0.0">:$dropout_p,
    DefaultValuedAttr<BoolAttr, "false">:$is_causal,
    DefaultValuedAttr<F64Attr, "0.0">:$scale
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}


def Top_CeilOp : Top_Op<"Ceil"> {
  let summary = " Ceil operator";
  let description = [{
    1.Op Introduction
    y = ceil(x)
    rounds each element of the input tensor up to the nearest integer.

     2.Math formula
     ```math
            output = ceil(input)
     ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_RMSNormOp : Top_Op<"RMSNorm"> {
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

def Top_RemainderOp: Top_Op<"Remainder"> {
  let summary = " Remainder operator";
  let description = [{
    1.Op Introduction
    computes the element-wise remainder of division between two tensors.

    2.Math formula
    ```math
            quo = x / y;
            floor_quo = floor(quo);
            output = torch.remainder(x, y) = x - y * floor_quo.
    ```

    3.activation and weight
    inputs(act.): input tensor.;
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
}

def Top_CumSumOp: Top_Op<"CumSum"> {
  let summary = " CumSum operator";
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

def Top_RoundOp : Top_Op<"Round"> {
  let summary = " Round operator";
  let description = [{
    1.Op Introduction
    Round takes one input Tensor and rounds the values, element-wise,
    meaning it finds the nearest integer for each value. In case of halfs,
    the rule is to round them to the nearest even integer.
    If input x is integral, +0, -0, NaN, or infinite, x itself is returned.
    The output tensor has the same shape and type as the input.

    2.Math formula
    ```math
            output = Round(input)
    ```

    3.activation and weight
    input(act.): input tensor.;
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs
    AnyTensor:$output);
}

def Top_BatchNormTrainOp: Top_Op<"BatchNormTrain", [SupportFuseRelu]> {
  let summary = "BatchNormalization train operation";
  let description = [{
    1.Op Introduction
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    2.Math formula
    ```math
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    ```
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).

    3.activation and weight
    input(act.): input tensor.;
    mean(w.): mean values to subtract from each channel for normalization.;
    variance(w.): adjust the predicted boxes during the training process.;

    4.attributes
    gamma;
    beta;
    epsilon;
    momentum: hyperparameter;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.;
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$mean,
    AnyTensor:$var,
    AnyTensorOrNone:$gamma,
    AnyTensorOrNone:$beta,
    DefaultValuedAttr<F64Attr, "1e-05">:$epsilon,
    DefaultValuedAttr<F64Attr, "0.1">:$momentum,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let results = (outs
    AnyTensor:$output,
    AnyTensor:$mean_out,
    AnyTensor:$saved_invstd,
    AnyTensor:$running_mean,
    AnyTensor:$running_var
  );
}

def Top_BatchNormBwdOp: Top_Op<"BatchNormBwd"> {
  let summary = "BatchNormalization backward operation";
  let description = [{
    1.Op Introduction
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    2.Math formula
    ```math
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    ```
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).

    3.activation and weight
    input(act.): input tensor.;
    grad_out(w.): the gradient of the loss with respect to the output of the layer normalization.;

    4.attributes
    weight_opt: the optimal scaling for the normalized output.;
    saved_mean: the mean of the input tensor calculated.;
    saved_invstd: the inverse standard deviation of the input tensor calculated.;
    epsilon;
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


def Top_LayerNormTrainOp : Top_Op<"LayerNormTrain"> {
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
    AnyTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$normalized_shape,
    SI32Attr:$axis,
    F64Attr:$eps
  );
  let results = (outs
    AnyTensor:$output,
    AnyTensor:$mean,
    AnyTensor:$variance
  );
}

def Top_LayerNormBwdOp : Top_Op<"LayerNormBwd"> {
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

def Top_EmbDenseBwdOp : Top_Op<"EmbDenseBwd"> {
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
  }];
  let arguments = (ins
    AnyTensor:$grad_output,
    AnyTensor:$indices,
    SI32Attr:$num_weights
  );
  let results = (outs
    AnyTensor:$output
  );
}

def Top_WeightReorderOp: Top_Op<"WeightReorder"> {
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


def Top_SoftmaxBwdOp: Top_Op<"SoftmaxBwd"> {
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

def Top_ConvBwdWeightOp:Top_Op<"ConvBwd_Weight">{
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
def Top_VarianceOp:Top_Op<"Variance">{
  let summary = "Compute Variance operator";
  let description = [{
    1.Op Introduction
    variance

    2.Math formula
    ```math
            output = \frac{1}{N} \sum_{i=1}^{N} (input_i - mean)^2
    ```

    3.activation and weight
    input(act.): input tensor.;

    4.attributes
    reduce_list: A list of dimensions along which to compute the variance.;
    correction: correction factor.;
    keep_dims: whether to keep the dimensions of the output tensor the same as the input tensors.;
    }];
  let arguments = (ins
  AnyTensor:$input,
  I64ArrayAttr:$reduce_list,
  F64Attr:$correction,
  DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs
  AnyTensor:$output
  );
}

def Top_RsqrtOp:Top_Op<"Rsqrt">{
  let summary = "Rsqrt Operator";
  let description = [{
    1.Op Introduction
    Reverse square root.

    2.Math formula
    ```math
            output = \frac{1}{\sqrt{input}}
    ```

    3.activation and weight
    input(act.): input tensor.;
    }];
  let arguments = (ins
    AnyTensor:$input);
  let results = (outs
    AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SortOp:Top_Op<"Sort"> {
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

    4.attributes
    axis: the dimension of the input tensor.;
    descending: the order of sorting.;
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    DefaultValuedAttr<BoolAttr, "true">:$descending
  );
  let results = (outs
    AnyTensorOrNone:$values,
    AnyTensor:$indices
  );
}
def Top_MeanRstdOp: Top_Op<"MeanRstd">{
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

def Top_GroupNormTrainOp : Top_Op<"GroupNormTrain"> {
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
def Top_LogicalAndOp :Top_Op<"LogicalAnd">{
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
def Top_ConvbwdOp :Top_Op<"Convbwd">{
  let summary = "convolution backward";
  let description = [{
    1.Op Introduction
    calculate grad_input,grad_weight,grad_bias of convolution operation.

    2.Math formula
    ```math
        \text{grad\_input} = \sum_{k=0}^{K-1} \text{grad\_out} \ast \text{kernel}_{k}
        where \( K \) is the number of output channels and \( \ast \) denotes the convolution operation.

        \text{grad\_weight} = \sum_{n=0}^{N-1} \text{input}_{n} \ast \text{grad\_out}
        where \( N \) is the number of input channels.

        \text{grad\_bias} = \sum_{i=0}^{M-1} \text{grad\_out}_{i}
        where \( M \) is the number of output channels.
    ```

    3.activation and weight
    input(act.): input tensor.;
    grad_out(w.): the gradient of the loss with respect to the output of the layer normalization.;
    kernel(w.): convolution kernel (filter) tensor.;

    4.attributes
    groups: Number of blocked connections from input channels to output channels. Default: 1.;
    input_shape: The shape of the input tensor.;
    grad_out_shape: The shape of the gradient output tensor.;
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    stride: the stride for the cross-correlation, a single number or a tuple.;
    dilations: controls the spacing between the kernel points;
    padding: the amount of padding applied to the input. It contains four ints with top, left, bottom, right.
    inserts: selectively enabling or disabling the calculation of gradients;
    grad_input_enable: whether to compute the gradient for the input tensor.;
    grad_weight_enable: whether to compute the gradient for the weight tensor (kernel).;
    grad_bias_enable: whether to compute the gradient for the bias term as well.;
    }];
  let arguments = (ins
    AnyTensor:$grad_out,
    AnyTensor:$input,
    AnyTensor:$kernel,
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

def Top_MaskRCNNRPNGetBboxesOp: Top_Op<"MaskRCNN_RPNGetBboxes"> {
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
    cls_scores_1: the class scores1 for each anchor;
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

def Top_MaskRCNNBboxPoolerOp: Top_Op<"MaskRCNN_BboxPooler"> {
  let summary = "BBox_Pooler gen by PPL";

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

def Top_MaskRCNNGetBboxBOp: Top_Op<"MaskRCNN_GetBboxB">{
  let summary = "MaskRCNN GetBboxB gen by PPL";
  let description = [{
    1.Op Introduction
    Mask rcnn consist of three part: backbone to get proposals, bbox head to get bbox pred, mask_head to get mask pred, GetBboxB is final part in bbox head

    2.Math formula
    ```math
            output = \text{NMS}\Bigl( \text{Decode}\Bigl( \text{ptr\_rois},\, \text{ptr\_bbox},\, \text{max\_val},\, \text{scale\_factor},\, \text{delta2bbox\_means},\, \text{delta2bbox\_stds} \Bigr),\, \text{ptr\_score},\, \text{threshold\_score\_eq},\, \text{nms\_iou\_thr} \Bigr)
    ```

    3.activation and weight
    ptr_rois(act.): candidate regions of interest for possible objects.;
    ptr_bbox(act.): the bounding box predictions.;
    ptr_score(act.): confidence scores associated with each proposal.;
    max_val(w.): max scores the bbox predictions.;
    scale_factor(w.): scale the decoded bounding box coordinates.;

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

def Top_MaskRCNNMaskPoolerOp: Top_Op<"MaskRCNN_MaskPooler"> {
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

def Top_MaxPoolingIndicesBwdOp: Top_Op<"MaxPoolingIndicesBwd"> {
  let summary = "MaxPoolingIndicesBwd operator";

  let description = [{
    1.Op Introduction
    MaxPoolingIndicesBwd operator

    2.Math formula
    ```math
        output = grad_output[indices]
    ```

    3.activation and weight
    grad_output(act.): the gradient of the loss with respect to the output.;
    indices(w.): the indices of the input tokens or items.;

    4.attributes
    kernel_shape: the size of the convolution kernel (filter) as an array.;
    stride: the stride for the cross-correlation, a single number or a tuple.;
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

  let results = (outs AnyTensor:$grad_input);
}

#endif // Top_OPS
