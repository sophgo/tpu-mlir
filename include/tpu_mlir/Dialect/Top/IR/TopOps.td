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
  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
  let extraClassDeclaration = [{
  void loadWeightFile(llvm::StringRef file) {
    wFile = std::make_unique<mlir::TensorFile>(file, false);
  }
  std::unique_ptr<mlir::TensorFile> wFile;
  }];
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

def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceL2","ReduceL1","ReduceSum"]>;
def InterpModeAttr: AnyStrAttrOf<["nearest","linear"]>;
def InterpCoordModeAttr: AnyStrAttrOf<["align_corners", "half_pixel", "pytorch_half_pixel"]>;
def PixelFormatAttr: AnyStrAttrOf<["rgb","bgr","gray","rgba"]>;
def ChannelFormatAttr: AnyStrAttrOf<["nhwc","nchw"]>;
def PadModeAttr: AnyStrAttrOf<["normal","center"]>;
def DetectionOutputCodeTypeAttr: AnyStrAttrOf<["CORNER", "CENTER_SIZE", "CORNER_SIZE"]>;

//===----------------------------------------------------------------------===//
// TOP Types.
//===----------------------------------------------------------------------===//

def AnyTenor: AnyTypeOf<[AnyRankedTensor]>;
def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// TOP Op Definition.
//===----------------------------------------------------------------------===//

class Top_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Top_Dialect, mnemonic, !listconcat(traits,[NoSideEffect])> ;

class Top_Op<string mnemonic, list<Trait> traits = []> :
    Top_BaseOp<mnemonic, !listconcat(traits,
       [DeclareOpInterfaceMethods<InferenceInterface>,
        DeclareOpInterfaceMethods<FlopsInterface>])> ;

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
    OptionalAttr<F64ArrayAttr>:$scale
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
  template<typename T>
  mlir::LogicalResult update(const std::vector<T>& data, size_t count);
  template<typename T>
  std::shared_ptr<std::vector<T>> read();
  std::shared_ptr<std::vector<float>> read_as_float();
  std::shared_ptr<std::vector<uint8_t>> read_as_byte();
  template<typename T>
  static mlir::Value create(mlir::Operation * OwnerOp,
                            llvm::StringRef suffix,
                            const std::vector<T>& data,
                            mlir::RankedTensorType& type);
  mlir::Value clone_bf16(mlir::Operation * OwnerOp);
  mlir::Value clone_f16(mlir::Operation * OwnerOp);
  }];
}

def Top_InputOp: Top_BaseOp<"Input"> {
  let summary = "Input operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    // preprocess for input
    OptionalAttr<PixelFormatAttr>:$pixel_format,
    OptionalAttr<ChannelFormatAttr>:$channel_format,
    OptionalAttr<I64ArrayAttr>:$resize_dims,
    OptionalAttr<BoolAttr>:$keep_aspect_ratio,
    OptionalAttr<I64Attr>:$pad_value,
    OptionalAttr<PadModeAttr>:$pad_type,
    OptionalAttr<F64ArrayAttr>:$scale,
    OptionalAttr<F64ArrayAttr>:$mean
  );

  let results = (outs AnyTensor:$output);
}

def Top_BatchNormOp: Top_Op<"BatchNorm", [SupportFuseRelu, InOutSameShape]> {
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

def Top_CastOp:Top_Op<"Cast",[InOutSameShape]> {
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
    DefaultValuedAttr<I64Attr, "1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_PackOp: Top_Op<"Pack", [SupportFuseRelu]> {
  let summary = "Pack operator";

  let description = [{
  Pack a list of tensors in the given dimension.
  All tensors must have the same shape.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    I64Attr:$axis,
    I64Attr:$values_count,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_UnpackOp: Top_Op<"Unpack", [SupportFuseRelu]> {
  let summary = "Unpack operator";

  let description = [{
  Unpack a tensor to list of tensors in the given dimension.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64Attr:$axis,
    I64Attr:$num,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs Variadic<AnyTensor>:$outputs);
  // let hasCanonicalizer = 1;
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
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64Attr:$block_h,
    I64Attr:$block_w,
    BoolAttr:$is_CRD,
    BoolAttr:$is_inversed
  );

  let results = (outs AnyTensor:$output);
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
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift
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

def Top_AddConstOp: Top_Op<"AddConst", [SupportFuseRelu, InOutSameShape]> {
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

def Top_MulConstOp: Top_Op<"MulConst", [SupportFuseRelu, InOutSameShape]> {
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

def Top_NormalizeOp: Top_Op<"Normalize", [InOutSameShape]> {
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

def Top_ReciprocalOp: Top_Op<"Reciprocal", [SupportFuseRelu, InOutSameShape]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64Attr>:$in_int4_scale,
    OptionalAttr<F64Attr>:$in_int4_zp,
    OptionalAttr<F64Attr>:$out_int8_scale,
    OptionalAttr<F64Attr>:$out_int8_zp
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    matmul_attr_t parseParam();
  }];
}

def Top_PadOp:Top_Op<"Pad"> {
  let summary = "Pad operation";
  let description = [{
    This operation pads a tensor according to the paddings you specify.
    paddings is an integer tensor with shape [n, 2], where n is the rank of tensor.
    For each dimension D of input, paddings[D, 0] indicates how many values to add
    before the contents of tensor in that dimension, and paddings[D, 1] indicates
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

def Top_ReluOp: Top_Op<"Relu",[InOutSameShape]> {
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

def Top_ReshapeOp:Top_Op<"Reshape"> {
  let summary = "Reshape operation";
  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ReverseOp:Top_Op<"Reverse", [InOutSameShape]> {
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

def Top_SigmoidOp : Top_Op<"Sigmoid", [InOutSameShape]> {
  let summary = " Exp operator,  scale * Sigmoid + bias";
  let description = [{
     Y = scale * Sigmoid(x) + bias
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<F64Attr, "1">:$scale,
    DefaultValuedAttr<F64Attr, "0">:$bias
  );

  let results = (outs AnyTensor:$output);
}

def Top_SiLUOp : Top_Op<"SiLU", [InOutSameShape]> {
  let summary = " SiLU operator,  y = x * Sigmoid(x)";
  let description = [{
     Y = x * Sigmoid(x)
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
    I64Attr:$axis,
    I64Attr:$num
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
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
}

def Top_SoftmaxOp:Top_Op<"Softmax",[InOutSameShape]> {
  let summary = "Softmax operation";
  let description = [{
    Integrates some operations related to softmax.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$log,
    DefaultValuedAttr<F64Attr, "1.0">:$beta
  );
  let results = (outs AnyTensor:$output);
}

def Top_LeakyReluOp : Top_Op<"LeakyRelu", [InOutSameShape]> {
  let summary = "LeakyRelu operation";
  let description = [{
    LeakyRelu takes input data (Tensor<T>) and an argument alpha,
    and produces one output data (Tensor<T>)
    where the function f(x) = alpha * x for x < 0, f(x) = x for x >= 0,
    is applied to the data tensor elementwise.
  }];
  let arguments = (ins
    AnyTenor:$input,
    F64Attr:$alpha
  );
  let results = (outs AnyTenor:$output);
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

def Top_LogOp: Top_Op<"Log", [InOutSameShape]> {
  let summary = "Log operator";

  let description = [{
    Calculates the natural log of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_LRNOp: Top_Op<"LRN", [InOutSameShape]> {
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

def Top_ExpOp: Top_Op<"Exp", [InOutSameShape]> {
  let summary = "Exp operator";

  let description = [{
    Calculates the exponent of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_TanhOp: Top_Op<"Tanh", [InOutSameShape]> {
  let summary = "Tanh operator";

  let description = [{
    Calculates the tanh of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_MishOp: Top_Op<"Mish", [InOutSameShape]> {
  let summary = "Mish operator";

  let description = [{
    Calculates the mish of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_DivOp: Top_Op<"Div", [InOutSameShape]> {
  let summary = "Div operator";

  let description = [{
    Performs element-wise binary division.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<SI32Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift
  );

  let results = (outs AnyTensor:$output);
  // let hasCanonicalizer = 1;
}

def Top_SqueezeOp: Top_Op<"Squeeze"> {
  let summary = "Squeeze operator";

  let description = [{
    The operator squeeze the input shapes by given axis.
  }];

  let arguments = (ins
    AnyTensor:$inputs,
    I64ArrayAttr:$axes
  );

  let results = (outs AnyTensor:$output);
}

def Top_ClipOp: Top_Op<"Clip", [InOutSameShape]> {
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
    OptionalAttr<I64ArrayAttr>:$inserts,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    deconv_attr_t parseParam();
  }];
}

def Top_ScaleOp: Top_Op<"Scale", [SupportFuseRelu, InOutSameShape]> {
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

  let results = (outs AnyTenor:$output);
  let hasCanonicalizer = 1;
}

def Top_TileOp: Top_Op<"Tile"> {
  let summary = "Tile operator";
  let description = [{
    Perform Tile operation on the given tensor.
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    I64Attr:$tile
  );

  let results = (outs AnyTenor:$output);
  let hasCanonicalizer = 1;
}

def Top_AbsOp : Top_Op<"Abs", [InOutSameShape]> {
  let summary = " Abs operator";
  let description = [{
     y = abs(x)
  }];
  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_PReluOp : Top_Op<"PRelu", [InOutSameDim]> {
  let summary = "PRelu operator";
  let description = [{
     f(x) = slope * x   for x < 0
     f(x) = x           for x >= 0
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$slope
  );

  let results = (outs AnyTensor:$output);
}

def Top_InterpOp : Top_Op<"Interp"> {
  let summary = "Interp operation";
  let description = [{
     Perform linear upsample on input
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$scale_h,
    F64Attr:$scale_w,
    InterpModeAttr:$mode,
    InterpCoordModeAttr:$coord_mode
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
    I64Attr:$keepdims,
    ReduceModeAttr:$mode
  );
  let results = (outs AnyTensor:$output);
}

def Top_PowOp : Top_Op<"Pow", [InOutSameShape]> {
  let summary = "Pow operation";
  let description = [{
    output = input ^ n
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr: $exponent
  );
  let results = (outs AnyTensor:$output);
}

def Top_SqrtOp : Top_Op<"Sqrt", [InOutSameShape]> {
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
    AnyTensor:$tbrn,
    AnyTensor:$fbrn
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
    I64Attr:$background_label_id,
    F64Attr:$nms_threshold,
    I64Attr:$top_k,
    DetectionOutputCodeTypeAttr:$code_type,
    I64Attr:$keep_top_k,
    F64Attr:$confidence_threshold,
    DefaultValuedAttr<BoolAttr, "true">:$share_location
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
    DefaultValuedAttr<StrAttr, "">:$anchors
  );

  let results = (outs AnyTensor:$output);
}

def Top_QuantizeLinearOp : Top_Op<"QuantizeLinear", [InOutSameShape]> {
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
    AnyTensor:$weight,
    AnyTensorOrNone:$bias,
    I64Attr:$axis,
    F64Attr:$eps
  );
  let results = (outs
  	AnyTensor:$output,
  	AnyTensorOrNone:$mean,
  	AnyTensorOrNone:$rstd
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


def Top_DequantizeLinearOp : Top_Op<"DequantizeLinear", [InOutSameShape]> {
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
#endif // Top_OPS
