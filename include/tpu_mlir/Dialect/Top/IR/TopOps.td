//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// =============================================================================
//
// Defines TOP Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TOP_OPS
#define TOP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tpu_mlir/Dialect/Top/IR/TopAttr.td"
include "tpu_mlir/Interfaces/InferenceInterface.td"
include "tpu_mlir/Interfaces/LoweringInterface.td"
include "tpu_mlir/Interfaces/FlopsInterface.td"
include "tpu_mlir/Traits/Traits.td"


def AnyTenor: AnyTypeOf<[AnyRankedTensor]>;
def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;

class Top_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Top_Dialect, mnemonic, !listconcat(traits,[NoSideEffect])> ;

class Top_Op<string mnemonic, list<Trait> traits = []> :
    Top_BaseOp<mnemonic, !listconcat(traits,
       [HasCommonAttributes,
        DeclareOpInterfaceMethods<InferenceInterface>,
        DeclareOpInterfaceMethods<LoweringInterface>,
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
    This Op does not take any input, and the attribute captures the file name.
    The Output is an n-dimensional tensor whose type matches
    the tensor type in the .npz file.
  }];

  let arguments = (ins
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
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
    StrAttr:$name,
    OptionalAttr<Top_PreprocessParamAttr>:$preprocess
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
    AnyTensor:$gamma,
    AnyTensor:$beta,
    AnyTensor:$mean,
    AnyTensor:$variance,
    DefaultValuedAttr<F64Attr, "1e-05">:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
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
    AnyTensor:$input,
    StrAttr:$name
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(void *param);
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(void *param);
  }];
  let hasCanonicalizer = 1;
}

def Top_AvgPoolOp:Top_PoolOp<"AvgPool">;
def Top_MaxPoolOp:Top_PoolOp<"MaxPool">;

def Top_AddOp: Top_Op<"Add", [SupportFuseRelu, InOutSameShape]> {
  let summary = "add operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff,
    StrAttr:$name
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
    else CRD(column-row-depth), chennel ordered by c * block_h * block_w;
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64Attr:$block_h,
    I64Attr:$block_w,
    BoolAttr:$is_CRD,
    BoolAttr:$is_inversed,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Top_MulOp: Top_Op<"Mul", [SupportFuseRelu, InOutSameShape]> {
  let summary = "Mul operator";

  let description = [{
    Elementwise mul of input1 and input2. input1 and input2 are tensors.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<I64Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
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
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(
      int64_t &batch, int64_t &M, int64_t &K, int64_t &N, bool &with_bias, bool &do_relu,
      double &relu_limit);
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
    StrAttr:$name
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
    I64ArrayAttr:$order,
    StrAttr:$name
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
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
    AnyTensor:$input,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SigmoidOp : Top_Op<"Sigmoid", [InOutSameShape]> {
  let summary = " Exp operator,  scale * Sigmoid + bias";
  let description = [{
     Y = scale * Sigmoid(x) + bias
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<F64Attr, "1">:$scale,
    DefaultValuedAttr<F64Attr, "0">:$bias,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Top_SiLUOp : Top_Op<"SiLU", [InOutSameShape]> {
  let summary = " SiLU operator,  y = x * Sigmoid(x)";
  let description = [{
     Y = x * Sigmoid(x)
  }];
  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Top_SliceOp: Top_Op<"Slice"> {
  let summary = "Slice operator";

  let description = [{
    Slice Operation on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64ArrayAttr:$offset,
    I64ArrayAttr:$steps,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_SoftmaxOp:Top_Op<"Softmax",[InOutSameShape]> {
  let summary = "Softmax operation";
  let description = [{
    The operator computes the normalized exponential values for the given input.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    OptionalAttr<F64Attr>:$beta,
    StrAttr:$name
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
    F64Attr:$alpha,
    StrAttr:$name
  );
  let results = (outs AnyTenor:$output);
}

def Top_DropoutOp : Top_Op<"Dropout", [InOutSameShape]> {
  let summary = "Dropout operation";
  let description = [{
    During evaluation or test, DropoutOp is just like an IdentityOp,
    transferring input to output without any changes.
    Argument `ratio` is only used during training.
  }];
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$ratio,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

def Top_LogOp: Top_Op<"Log", [InOutSameShape]> {
  let summary = "Log operator";

  let description = [{
    Calculates the natural log of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$name
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    // quant param
    DefaultValuedAttr<I64Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    StrAttr:$name
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
    I64ArrayAttr:$axes,
    StrAttr:$name
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
    F64Attr:$max,
    StrAttr:$name
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(void *param);
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
    DefaultValuedAttr<F64Attr, "0.0">:$relu_limit,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

#endif // Top_OPS


