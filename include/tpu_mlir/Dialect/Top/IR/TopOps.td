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
def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual", "NotEqual", "And", "Not"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceL2","ReduceL1","ReduceSum","ReduceProd"]>;
def InterpModeAttr: AnyStrAttrOf<["nearest","linear"]>;
def InterpCoordModeAttr: AnyStrAttrOf<["align_corners", "half_pixel", "pytorch_half_pixel"]>;
def PixelFormatAttr: AnyStrAttrOf<["rgb","bgr","gray","rgba"]>;
def ChannelFormatAttr: AnyStrAttrOf<["nhwc","nchw"]>;
def PadModeAttr: AnyStrAttrOf<["normal","center"]>;
def DetectionOutputCodeTypeAttr: AnyStrAttrOf<["CORNER", "CENTER_SIZE", "CORNER_SIZE"]>;
def RoiAlignModeAttr: AnyStrAttrOf<["Avg","Max"]>;
def NonZeroOrderAttr: AnyStrAttrOf<["ColMajor","RowMajor"]>;
def StoreModeAttr: AnyStrAttrOf<["1N", "2N", "4N"]>;

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
  let summary = "load weight operator";

  let description = [{
    Load weight from a file. The file should be a valid .npz format file.
    This Op does not take any input, and the location captures the tensor name.
    The Output is an n-dimensional tensor whose type matches
    the tensor type in the .npz file.
  }];

  let arguments = (ins
    OptionalAttr<F64ArrayAttr>:$scale,
    OptionalAttr<BoolAttr>:$do_compress,
    OptionalAttr<StoreModeAttr>:$store_mode
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
  template<typename T>
  std::shared_ptr<std::vector<T>> read();
  std::shared_ptr<std::vector<float>> read_as_float();
  std::shared_ptr<std::vector<int32_t>> read_as_int32();
  std::shared_ptr<std::vector<uint8_t>> read_as_byte();
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
  }];
}

def Top_InputOp: Top_BaseOp<"Input"> {
  let summary = "Input operator";

  let description = [{
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    // preprocess for input
    OptionalAttr<PixelFormatAttr>:$pixel_format,
    OptionalAttr<ChannelFormatAttr>:$channel_format,
    OptionalAttr<I64ArrayAttr>:$resize_dims,
    OptionalAttr<BoolAttr>:$keep_aspect_ratio,
    OptionalAttr<I64Attr>:$pad_value,
    OptionalAttr<PadModeAttr>:$pad_type,
    OptionalAttr<F64ArrayAttr>:$scale,
    OptionalAttr<F64ArrayAttr>:$mean,
    //for cv18xx fusepreprocess
    OptionalAttr<StrAttr>:$customization_format,
    OptionalAttr<BoolAttr>:$aligned
  );

  let results = (outs AnyTensor:$output);
}

def Top_TupleOp: Top_BaseOp<"Tuple"> {
  let summary = "Tuple operator";
  let description = [{
    gen by torch prim::TupleConstruct, y = (a, b)
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
}

def Top_UnTupleOp: Top_BaseOp<"UnTuple"> {
  let summary = "UnTuple operator";
  let description = [{
    gen by torch prim::TupleUnpack, a, b = y
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
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_CastOp:Top_Op<"Cast"> {
  let summary = "Cast operation";
  let description = [{
    quant::UniformQuantizedType cast to float type; or float type cast to quant::UniformQuantizedType
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_ConcatOp: Top_Op<"Concat", [SupportFuseRelu]> {
  let summary = "Concat operator";

  let description = [{
  Concatenates the given sequence of seq tensors in the given dimension.
  All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<SI32Attr, "1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_PackOp: Top_Op<"Pack"> {
  let summary = "Pack operator";

  let description = [{
  Pack a list of tensors in the given dimension.
  All tensors must have the same shape.
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
  Unpack a tensor to list of tensors in the given dimension.
  }];

  let arguments = (
    ins AnyTensor:$input,
    SI32Attr:$axis
  );

  let results = (outs Variadic<AnyTensor>:$outputs);
  let hasCanonicalizer = 1;
}

def Top_ConvOp: Top_Op<"Conv", [SupportFuseRelu]> {
  let summary = "Convolution operator";

  let description = [{
    In the simplest case, the output value of the layer with input size
    $$(N, C_{\text{in}}, H, W)$$ and output $$(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$$
    can be precisely described as:

    ```math
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
    ```


    where $$\star$$ is the valid 2D cross-correlation operator,
    $$N$$ is a batch size, $$C$$ denotes a number of channels,
    $$H$$ is a height of input planes in pixels, and $$W$$ is
    width in pixels.

    weight (Tensor): the learnable weights of the module of shape
    $$(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},
    \text{kernel\_size[0]}, \text{kernel\_size[1]})$$

    bias (Tensor optional): the learnable bias of the module of shape (out_channels).

    - **stride**: controls the stride for the cross-correlation, a single
      number or a tuple.

    - **padding**: controls the amount of padding applied to the input. It
      contains four ints with top, left, bottom, right respectively.

    - **dilation**: controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this
      [Link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
      has a nice visualization of what **dilation** does.

    - **groups**: (optional): Number of blocked connections from input
            channels to output channels. Default: 1


    Shape:
        - Input: $$(N, C_{in}, H_{in}, W_{in})$$
        - Output: $$(N, C_{out}, H_{out}, W_{out})$$ where

          ```math
              H_{out} = \left\lfloor\frac{H_{in}  + \text{padding}[0] + \text{padding}[2] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          ```
          ```math
              W_{out} = \left\lfloor\frac{W_{in}  + \text{padding}[1] + \text{padding}[3] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
          ```
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
    OptionalAttr<F64Attr>:$in_int4_scale,
    OptionalAttr<F64Attr>:$in_int4_zp,
    OptionalAttr<F64Attr>:$out_int8_scale,
    OptionalAttr<F64Attr>:$out_int8_zp
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    conv_attr_t parseParam();
  }];
}

class Top_PoolOp <string mnemonic> : Top_Op<mnemonic,[SupportFuseRelu]> {
  let summary = "pool operator";

  let description = [{
    This performs an  pooling over the given input tensor. A sliding
    window of size given by <kernel size> is passed over the input tensor.
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    OptionalAttr<BoolAttr>:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "true">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$pad_value,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
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
    pooling mask on input
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
    Refer to `https://github.com/onnx/onnx/blob/main/docs/Operators.md#depthtospace`
    [n, c, h, w] => [n, c / (block_h * block_w), h * block_h, w * block_w];
    if inversed, [n, c, h, w] => [n, c * block_h * block_w, h / block_h, w / block_w];

    if DCR(depth-column-row), channel ordered by block_h * block_w * c;
    else CRD(column-row-depth), channel ordered by c * block_h * block_w;

    The format of input or output is NCHW or NHWC.
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

def Top_AddOp: Top_Op<"Add", [SupportFuseRelu]> {
  let summary = "add operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SubOp: Top_Op<"Sub", [SupportFuseRelu]> {
  let summary = "sub operator";

  let description = [{
    Elementwise subtraction of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff
  );

  let results = (outs AnyTensor:$output);
}

def Top_MulOp: Top_Op<"Mul", [SupportFuseRelu]> {
  let summary = "Mul operator";

  let description = [{
    Elementwise multiplication of input1 and input2. input1 and input2 are tensors.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_MinOp: Top_Op<"Min"> {
  let summary = "min operator";

  let description = [{
    Element-wise min of each of the input tensors. All inputs and outputs must have the same data type.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
}

def Top_MaxOp: Top_Op<"Max"> {
  let summary = "max operator";

  let description = [{
    Element-wise max of each of the input tensors. All inputs and outputs must have the same data type.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
}

def Top_AddConstOp: Top_Op<"AddConst", [SupportFuseRelu, SupportPermuteMove]> {
  let summary = "Add Const operator";

  let description = [{
    Y = X + const_val
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
}

def Top_SubConstOp: Top_Op<"SubConst", [SupportFuseRelu]> {
  let summary = "Sub Const operator";

  let description = [{
    Y = X - const_val or const_val - X
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
}

def Top_MulConstOp: Top_Op<"MulConst", [SupportFuseRelu]> {
  let summary = "Mul Const operator";

  let description = [{
    Y = X * const_val
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_NormalizeOp: Top_Op<"Normalize"> {
  let summary = "Normalize operator";

  let description = [{
    Normalizes an array across batch and spatial dimensions.

    Inputs:
      `input`           : required, the input activation tensor.
      `scale`           : required, the scale weight tensor. even channel_shared is true, extend to tensor.

    Attributes:
      `across_spatial`  : required, normalize cross channel or not.
      `channel_shared`  : required, scale cross channel or not.

    Result:
      `output`          : result tensor.
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
    Y = const_val / X
  }];

  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<F64Attr, "1.0">: $const_val,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
}

def Top_MatMulOp: Top_Op<"MatMul", [SupportFuseRelu]> {
  let summary = "matmul operator";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.
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
    OptionalAttr<F64Attr>:$in_int4_scale,
    OptionalAttr<F64Attr>:$in_int4_zp,
    OptionalAttr<F64Attr>:$out_int8_scale,
    OptionalAttr<F64Attr>:$out_int8_zp
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    matmul_attr_t parseParam();
  }];
}

def Top_AttentionOp: Top_Op<"Attention"> {
  let summary = "Attention operator";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.
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
    AnyTensorOrNone:$musk,
    F64Attr:$scale,
    I64Attr:$head,
    DefaultValuedAttr<I64Attr, "0">:$has_bias
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_PadOp:Top_Op<"Pad"> {
  let summary = "Pad operation";
  let description = [{
    This operation pads a tensor according to the paddings you specify.
    paddings is an integer tensor with shape [2, n], where n is the rank of tensor.
    For each dimension D of input, paddings[0, D] indicates how many values to add
    before the contents of tensor in that dimension, and paddings[1, D] indicates
    how many values to add after the contents of tensor in that dimension.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$paddings,
    DefaultValuedAttr<F64Attr, "0.0">:$val,
    DefaultValuedAttr<I64Attr, "0">:$mode
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_PermuteOp: Top_Op<"Permute"> {

  let summary = "Permute operator";

  let description = [{
      Perform permute on input.
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
      Transpose on input.
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
      Perform ShuffleChannel on input.
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
     ReLU with a scalar maximum value. if limit is zero, do not use upper limit.
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
    gen by torch prim::ListConstruct, y = [a, b]
    output shape is [1]
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs AnyTensor:$output);
}

def Top_ReshapeOp:Top_Op<"Reshape"> {
  let summary = "Reshape operation";
  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
    0: keep dim from input
    -1: left dim from input
  }];
  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<I64ArrayAttr>:$shape
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ViewOp:Top_Op<"View"> {
  let summary = "View operation";
  let description = [{
    gen by torch aten::view
    0: keep dim from input
    -1: left dim from input
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
    gen by torch aten::flatten
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<I64Attr, "0">:$start_dim,
    DefaultValuedAttr<I64Attr, "-1">:$end_dim
  );
  let results = (outs AnyTensor:$output);
}

def Top_ReverseOp:Top_Op<"Reverse"> {
  let summary = "Reverse operation";
  let description = [{
    Reverse on input.
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
     Y = scale * Sigmoid(x) + bias
     if log --> Y = Log(scale * Sigmoid(x) + bias)
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<F64Attr, "1">:$scale,
    DefaultValuedAttr<F64Attr, "0">:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$log
  );

  let results = (outs AnyTensor:$output);
}

def Top_SoftsignOp: Top_Op<"Softsign"> {
  let summary = " Softsign Operator";
  let description = [{
     Y = x / (1 + |x|)
  }];
  let arguments = (
    ins AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_SizeOp: Top_Op<"Size"> {
  let summary = "Size operator";
  let description = [{
    gen by torch aten::size
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
    gen by torch aten::arange
  }];

  let arguments = (ins
    AnyTensorOrNone:$start,
    AnyTensor:$end,
    AnyTensorOrNone:$step
  );

  let results = (outs AnyTensor:$output);
}

def Top_ConstantFillOp: Top_Op<"ConstantFill"> {
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
}

def Top_SiLUOp : Top_Op<"SiLU"> {
  let summary = " SiLU operator,  y = x * Sigmoid(x)";
  let description = [{
     Y = x * Sigmoid(x)
  }];
  let arguments = (
    ins AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_GELUOp : Top_Op<"GELU"> {
  let summary = " GELU operator,  0.5x * (1.0 + tf.erf(x / tf.sqrt(2.0)))";
  let description = [{
     Y = 0.5x * (1.0 + tf.erf(x / tf.sqrt(2.0)))
  }];
  let arguments = (
    ins AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_SplitOp: Top_Op<"Split"> {
  let summary = "Split operator";

  let description = [{
    Split input tensor into a list of tensors.
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
    Slice Operation on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64ArrayAttr:$offset,
    I64ArrayAttr:$steps
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_StridedSliceOp: Top_Op<"StridedSlice"> {
  let summary = "Strided Slice operator";

  let description = [{
    Strided Slice Operation on input.
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
    Slice Operation on input.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$axis,
    AnyTensor:$start,
    AnyTensor:$step,
    AnyTensor:$end
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SoftmaxOp:Top_Op<"Softmax"> {
  let summary = "Softmax operation";
  let description = [{
    Integrates some operations related to softmax.
  }];
  let arguments = (ins
    AnyTensor:$input,
    SI32Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$log,
    DefaultValuedAttr<F64Attr, "1.0">:$beta
  );
  let results = (outs AnyTensor:$output);
}

def Top_SoftplusOp:Top_Op<"Softplus"> {
  let summary = "Softplus operation";
  let description = [{
    y = ln(exp(x) + 1)
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_FloorOp:Top_Op<"Floor"> {
  let summary = "Floor operation";
  let description = [{
    y = floor(x)
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_TopKOp:Top_Op<"TopK"> {
  let summary = "TopK operation";
  let description = [{
    Integrates some operations related to topk.
  }];
  let arguments = (ins
    AnyTensor:$input,
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

def Top_NonZeroOp:Top_Op<"NonZero"> {
  let summary = "NonZero operation";
  let description = [{
    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
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
    LeakyRelu takes input data (Tensor<T>) and an argument alpha,
    and produces one output data (Tensor<T>)
    where the function f(x) = alpha * x for x < 0, f(x) = x for x >= 0,
    is applied to the data tensor elementwise.
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$alpha
  );
  let results = (outs AnyTensor:$output);
}

def Top_UpsampleOp : Top_Op<"Upsample", [SupportFuseRelu]> {
  let summary = "Upsample operation";
  let description = [{
    Perform nearest upsample on input.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$scale_h,
    I64Attr:$scale_w,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let results = (outs AnyTensor:$output);
}

def Top_MaxUnpoolOp : Top_Op<"MaxUnpool"> {
  let summary = "MaxUnpool operation";
  let description = [{
    Perform  MaxUnpool on input.
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
    Calculates the natural log of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_LRNOp: Top_Op<"LRN"> {
  let summary = "Local Response Normalization";

  let description = [{
    It normalizes over local input regions. The local region is defined across the channels.
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
    Calculates the exponent of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_CosOp: Top_Op<"Cos"> {
  let summary = "Cos operator";

  let description = [{
    Calculates the Cos of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_CoshOp: Top_Op<"Cosh"> {
  let summary = "Cosh operator";

  let description = [{
    Calculates the Cosh of the giventanh input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_SinOp: Top_Op<"Sin"> {
  let summary = "Sin operator";

  let description = [{
    Calculates the Sin of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_SinhOp: Top_Op<"Sinh"> {
  let summary = "Sinh operator";

  let description = [{
    Calculates the Sinh of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_TanOp: Top_Op<"Tan"> {
  let summary = "Tan operator";

  let description = [{
    Calculates the tan of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_TanhOp: Top_Op<"Tanh"> {
  let summary = "Tanh operator";

  let description = [{
    Calculates the tanh of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_MishOp: Top_Op<"Mish"> {
  let summary = "Mish operator";

  let description = [{
    Calculates the mish of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_DivOp: Top_Op<"Div"> {
  let summary = "Div operator";

  let description = [{
    Performs element-wise binary division.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SqueezeOp: Top_Op<"Squeeze"> {
  let summary = "Squeeze operator";

  let description = [{
    The operator squeeze the input shapes by given axis.
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$axes
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_UnsqueezeOp: Top_Op<"Unsqueeze"> {
  let summary = "Unsqueeze operator";

  let description = [{
    The operator unsqueeze the input shapes by given axis.
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
    The operator limits the given input to a certain range.
  }];

  let arguments = (ins
    AnyTensor:$inputs,
    F64Attr:$min,
    F64Attr:$max
  );

  let results = (outs AnyTensor:$output);
}

def Top_DeconvOp: Top_Op<"Deconv", [SupportFuseRelu]> {
  let summary = "Deconvolution operator";

  let description = [{
    Perform Deconvolution operation.
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
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    deconv_attr_t parseParam();
  }];
}

def Top_ScaleOp: Top_Op<"Scale", [SupportFuseRelu]> {
  let summary = "Scale operator";

  let description = [{
    Y = X * S + B,
    where the shape of X/Y is [n, c, h, w] and the shape of S/B is [1, c, 1, 1].
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
    Perform RNN GRU operation.
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
    Perform RNN LSTM operation.
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
      onnx nms
  }];
  let arguments = (ins
    Variadic<AnyTensor>: $inputs,
    I64Attr: $center_point_box,
    I64Attr: $max_output_size
  );

  let results = (outs AnyTensor:$output);
}

def Top_GatherOp: Top_Op<"Gather"> {
  let summary = "Gather operator";
  let description = [{
    Perform Gather operation on the given axis.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,

    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_GatherElementsOp: Top_Op<"GatherElements"> {
  let summary = "GatherElements operator";
  let description = [{
    Perform GatherElements operation on the given axis.
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
    Perform Tile operation on the given tensor.
  }];

  let arguments = (ins
    AnyTensor:$input,
    SI32Attr:$axis,
    I64Attr:$tile
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_RepeatOp: Top_Op<"Repeate"> {
  let summary = "Repeat operator";
  let description = [{
    Perform aten::repeat operation on the given tensor
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
     y = abs(x)
  }];
  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_PReluOp : Top_Op<"PRelu"> {
  let summary = "PRelu operator";
  let description = [{
     f(x) = slope * x   for x < 0
     f(x) = x           for x >= 0
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
     Perform linear upsample on input
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$target_shape,
    InterpModeAttr:$mode,
    InterpCoordModeAttr:$coord_mode,
    DefaultValuedAttr<F64Attr, "-1.0">:$scale_h,
    DefaultValuedAttr<F64Attr, "-1.0">:$scale_w
  );

  let results = (outs AnyTensor:$output);
}

def Top_MeshGridOp : Top_Op<"MeshGrid"> {
  let summary = "MeshGrid operation";
  let description = [{
     torch mesh grid operation
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
    BoolAttr:$align_corners
  );

  let results = (outs AnyTensor:$output);
}

def Top_ReduceOp : Top_Op<"Reduce"> {
  let summary = "Reduce operation";
  let description = [{
    Computes the mean/max/prod/sum of the input tensor's element along the provided axes.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$axes,
    BoolAttr:$keepdims,
    ReduceModeAttr:$mode
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ArgOp : Top_Op<"Arg"> {
  let summary = "Arg operation";
  let description = [{
    Computes the indices of the min/max/ of the input tensor's element along the provided axis.
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
    output = input ^ n
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $exponent
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SqrtOp : Top_Op<"Sqrt"> {
  let summary = "Sqrt operation";
  let description = [{
    Computes the square root of the input tensor's element.
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_WhereOp : Top_Op<"Where"> {
  let summary = "Where operation";
  let description = [{
    Return elements, either from X or Y, depending on condition.
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
}

def Top_MaskedFillOp : Top_Op<"MaskedFill"> {
  let summary = "MaskedFill operation";
  let description = [{
    Return elements, either from X or Const, depending on condition.
  }];
  let arguments = (ins
    AnyTensor:$cond,
    AnyTensor:$brn,
    BoolAttr:$inversed,
    F64Attr:$const_val
  );
  let results = (outs AnyTensor:$output);
}

def Top_CompareOp : Top_Op<"Compare"> {
  let summary = "Compare operation";
  let description = [{
    Returns the tensor resulted from performing the compare
    operation elementwise on the input tensors A and B
  }];
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    CompareModeAttr:$mode
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_CompareConstOp : Top_Op<"CompareConst"> {
  let summary = "CompareConst operation";
  let description = [{
    Returns the tensor resulted from performing the compare
    operation elementwise on the input tensors A and Const
  }];
  let arguments = (ins
    AnyTensor:$input,
    CompareModeAttr:$mode,
    F64Attr:$const_val,
    BoolAttr:$inversed
  );
  let results = (outs AnyTensor:$output);
}

def Top_ErfOp : Top_Op<"Erf"> {
  let summary = "Erf operation";
  let description = [{
    Computes the error function of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_HardSigmoidOp : Top_Op<"HardSigmoid"> {
  let summary = "HardSigmoid operation";
  let description = [{
    hardsigmoid(x; alpha, beta) := min(max(alpha*x + beta, 0), 1)
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
    hardswish(x) := x * hardsigmoid(x; 1/6, 0.5)
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_PriorBoxOp : Top_Op<"PriorBox"> {
  let summary = "PriorBox operation";
  let description = [{
    Intended for use with MultiBox detection method to generate prior.
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
    DefaultValuedAttr<F64Attr, "1">:$eta
  );
  let results = (outs AnyTensor:$output);
}

def Top_YoloDetectionOp : Top_Op<"YoloDetection"> {
  let summary = "YoloDetection operator";
  let description = [{
    Perform yolo detection on feature map
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
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
    DefaultValuedAttr<ConfinedAttr<I64ArrayAttr, [ArrayCount<3>]>, "{0,0,0}">:$scale,
    DefaultValuedAttr<ConfinedAttr<I64ArrayAttr, [ArrayCount<9>]>, "{0,0,0,0,0,0,0,0,0}">:$mask
  );

  let results = (outs AnyTensor:$output);
}

def Top_QuantizeLinearOp : Top_Op<"QuantizeLinear"> {
  let summary = "Linear quantize operation";
  let description = [{
    QuantizeLinear(x) := saturate ((x / y_scale) + y_zero_point)
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
    layer normalization
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
    AnyTensorOrNone:$mean,
    AnyTensorOrNone:$rstd
  );
  let hasCanonicalizer = 1;
}

def Top_InstanceNormOp : Top_Op<"InstanceNorm"> {
  let summary = "Instance Norm operation";
  let description = [{
    instance normalization
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
    AnyTensor:$output
  );
  let hasCanonicalizer = 1;
}

def Top_PixelNormOp : Top_Op<"PixelNorm"> {
  let summary = "PixelNorm operation";
  let description = [{
    pixel normalization (normalize along c-axis)
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
    Inputs:
      `inputs`                : required, the input activation tensor.

    Attributes:
      `net_input_h`           : required, net input height
      `net_input_w`           : required, net input width
      `feat_stride`           : required, anchor box stride size
      `anchor_base_size`      : required, anchor box base size
      `rpn_obj_threshold`     : required, obj threshold
      `rpn_nms_threshold`     : required, nms threshold for generate proposal boxes
      `rpn_nms_post_top_n`    : required, keep num boxes after nms

    Result:
      `output`                : result tensor.
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
    Max pooling on ROI.

    Inputs:
      `inputs`          : required, the input activation tensor.

    Attributes:
      `pooled_h`        : required, pooled output height.
      `pooled_w`        : required, pooled output width
      `spatial_scale`   : required, spatial_scale

    Result:
      `output`          : result tensor.
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
    DequantizeLinear(x) := (x - x_zero_point) * x_scale
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

    Inputs:
      `inputs`          : required, input tensors.

    Attributes:
      `class_num`       : required, detection class num.
      `obj_threshold`   : required, object threshold.
      `nms_threshold`   : required, nms threshold.
      `keep_topk`       : required, keep top k.

    Result:
      `output`          : result tensor.
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
    Performs csc operation on inputs.

    Inputs:
      `input`          : required, input tensors.

    Attributes:
      `pixel_format`    : required, pixel format type.
      `y_align`         : width alignment of channel y.
      `w_align`         : width alignment of channel uv.
      `channel_align`   : alignment of channel.

    Result:
      `output`          : result tensor.
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

def Top_ScaleLutOp: Top_Op<"ScaleLut"> {
  let summary = "Scale by lut operator";

  let description = [{
    Performs scale on input, y = input * scale + bias.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `scale`   : each channel scale
      `bias`    : each channel bias
      `sign`    : if output is signed


    Result:
      `output`          : result tensor.

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
    Swap Channel on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `channel_order`   : required, channel swap order
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

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
    AnyTensor:$input,
    AnyTensor:$indices,
    AnyTensor:$updates,
    I64Attr:$axis
  );

  let results = (outs AnyTensor:$output);
}

def Top_ScatterNDOp: Top_Op<"ScatterND"> {
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

def Top_PreprocessOp: Top_Op<"Preprocess"> {
  let summary = "FusePreprcess, it's just a placeholder op.";
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
  let results = (outs AnyTensor:$output);
}

def Top_RetinaFaceDetectionOp : Top_Op<"RetinaFaceDetection"> {
  let summary = "RetinaFaceDetection operator";
  let description = [{
    Perform retinaface detection on feature map

    Inputs:
      `inputs`                  : required, input tensors
      `nms_threshold`           : required, nms threshold
      `confidence_threshold`    : required, classification confidence threshold
      `keep_topk`               : required, after nms, keep bbox num

    Result:
      `output`          : result tensor.
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
    Elu takes input data (Tensor<T>) and an argument alpha,
    and produces one output data (Tensor<T>)
    where the function f(x) = alpha * (e^x - 1) for x <= 0, f(x) = x for x > 0,
    is applied to the data tensor elementwise.
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$alpha
  );
  let results = (outs AnyTensor:$output);
}

def Top_YieldOp : Top_BaseOp<"Yield", [Terminator, HasParent<"IfOp">]> {
  let summary = "Yield operation";
  let description = [{
    The `top.Yield` operation represents a return operation within an subgraph.
    The operation takes variable number of operands and produces no results.

    This operation is not part of the standard and was added to assist tpu-mlr.
  }];

  let arguments = (ins Variadic<AnyTensor>:$operands);
}

def Top_IfOp : Top_Op<"If"> {
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

def Top_LoopOp : Top_Op<"Loop"> {
  let summary = "Loop operation";
  let description = [{
  Generic Looping construct. This loop has multiple termination conditions:

  1) Trip count. Iteration count specified at runtime. Set by
     specifying the input M. Optional. Set to empty string to omit.
     Note that a static trip count (specified at graph construction time) can be
     specified by passing in a constant node for input M.
  2) Loop termination condition. This is an input to the op that determines
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

  1) Values from the enclosing scope (i.e. variable \"a\" here) are in scope and can
     be referenced in the inputs of the loop.
  2) Any values computed in the loop body that needs to be used in a subsequent
     iteration or after the loop are modelled using a pair of variables in the loop-body,
     consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
     These are referred to as loop-carried dependences. The loop operation node
     supplies the input value of the input variable for the first iteration, and
     returns the output value of the output variable produced by the final
     iteration.
  3) Scan_output variables are used to implicitly concatenate values computed across
     all the iterations. In the above example, the value of user_defined_val computed
     over all iterations are concatenated and returned as the value of user_defined_vals
     after the loop.
  4) Values created in the body cannot be accessed in the enclosing scope,
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
  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$M,
    AnyTypeOf<[AnyTensor, NoneType]>:$cond,
    Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$v_final_and_scan_outputs);
  let regions = (region SizedRegion<1>:$body);
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
    Takes a tensor as input and outputs an 1D int tensor containing the shape
     of the input tensor.
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Top_GatherNDOp: Top_Op<"GatherND"> {
  let summary = "GatherND operator";
  let description = [{
    This operator is the inverse of ScatterND.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,
    DefaultValuedAttr<I64Attr, "0">:$batch_dims
  );

  let results = (outs AnyTensor:$output);
}

def Top_DeformConv2DOp: Top_Op<"DeformConv2D", [SupportFuseRelu]> {
  let summary = "Deformable Convolution Operator";

  let description = [{
    In the simplest case, the output value of the layer with input size
    $$(N, C_{\text{in}}, H, W)$$ and output $$(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$$
    can be precisely described as:

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

    - **stride**: controls the stride for the cross-correlation, a single
      number or a tuple.

    - **padding**: controls the amount of padding applied to the input. It
      contains four ints with top, left, bottom, right respectively.

    - **dilation**: controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this
      [Link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
      has a nice visualization of what **dilation** does.

    - **groups**: (optional): Number of blocked connections from input
            channels to output channels. Default: 1

    - **deform_groups**: (optional): Number of blocked connections from input
            channels to output channels. Default: 1


    Shape:
        - Input: $$(N, C_{in}, H_{in}, W_{in})$$
        - Output: $$(N, C_{out}, H_{out}, W_{out})$$ where

          ```math
              H_{out} = \left\lfloor\frac{H_{in}  + \text{padding}[0] + \text{padding}[2] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          ```
          ```math
              W_{out} = \left\lfloor\frac{W_{in}  + \text{padding}[1] + \text{padding}[3] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
          ```
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

#endif // Top_OPS
