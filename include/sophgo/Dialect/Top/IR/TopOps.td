// =============================================================================
//
// Defines TOP Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TOP_OPS
#define TOP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "sophgo/Interfaces/InferenceInterface.td"
include "sophgo/Interfaces/QuantizeInterface.td"

def Top_Dialect : Dialect {
  let name = "top";
  let summary = "A topdialect for the SOPHGO specification";
  let cppNamespace = "::sophgo::top";
  let extraClassDeclaration = [{
  void loadWeightFile(llvm::StringRef file) {
    wFile = std::make_unique<mlir::TensorFile>(file, false);
  }
  std::unique_ptr<mlir::TensorFile> wFile;
  }];
}

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

class Top_Op<string mnemonic, list<Trait> traits = []> :
    Op<Top_Dialect, mnemonic, traits> ;

def Top_NoneOp : Top_Op<"None", [NoSideEffect]> {
  let summary = "none operator";

  let description = [{
    A none Op to return a NoneType.
  }];
  let results = (outs NoneType);
}

def Top_WeightOp : Top_Op<"Weight", [NoSideEffect]> {
  let summary = "load weight operator";

  let description = [{
    Load weight from a file. The file should be a valid .npz format file.
    This Op does not take any input, and the attribute captures the file name.
    The Output is an n-dimensional tensor whose type matches
    the tensor type in the .npz file.
  }];

  let arguments = (ins
    StrAttr:$name,
    DefaultValuedAttr<I64Attr, "0">:$addr
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
  }];
}

def Top_InputOp: Top_Op<"Input",[NoSideEffect]> {
  let summary = "Input operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I64Attr, "0">:$addr,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Top_BatchNormOp: Top_Op<"BatchNorm",[NoSideEffect]> {
  let summary = "BatchNormalization operation";
  let description = [{
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    ```math
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        y = \frac{input - mean}{variance + epsilon} * scale + bias
    ```

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
    of size C (where C is the input channel size).
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$scale,
    AnyTensor:$mean,
    AnyTensor:$variance,
    AnyTensorOrNone:$bias,
    DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Top_ConvOp: Top_Op<"Conv",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>,
  DeclareOpInterfaceMethods<QuantizeInterface>]> {
  let summary = "convolution operator";

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
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(int64_t &n, int64_t &ic, int64_t &ih, int64_t &iw, int64_t &oc,
                    int64_t &oh, int64_t &ow, int64_t &g, int64_t &kh, int64_t &kw, int64_t &ins_h,
                    int64_t &ins_w, int64_t &sh, int64_t &sw, int64_t &pt, int64_t &pb, int64_t &pl,
                    int64_t &pr, int64_t &dh, int64_t &dw, bool &is_dw, bool &with_bias, bool &do_relu);
  }];
}

class Top_PoolOp <string mnemonic> : Top_Op<mnemonic,
  [NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>,
  DeclareOpInterfaceMethods<QuantizeInterface>]> {
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
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(
      int64_t &n, int64_t &c, int64_t &ih, int64_t &iw, int64_t &oh, int64_t &ow,
      int64_t &kh, int64_t &kw, int64_t &sh, int64_t &sw, int64_t &pt, int64_t &pb,
      int64_t &pl, int64_t &pr, int64_t &pad_value,
      bool &is_global, bool &count_include_pad);
  }];
}

def Top_AvgPoolOp:Top_PoolOp<"AvgPool">;
def Top_MaxPoolOp:Top_PoolOp<"MaxPool">;

def Top_AddOp: Top_Op<"Add",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface>,
  DeclareOpInterfaceMethods<QuantizeInterface>]> {
  let summary = "add operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    OptionalAttr<F64ArrayAttr>:$coeff,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Top_MatMulOp: Top_Op<"MatMul",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>,
  DeclareOpInterfaceMethods<QuantizeInterface>]> {
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
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(
      int64_t &batch, int64_t &M, int64_t &K, int64_t &N, bool &with_bias);
  }];
}

def Top_ReluOp: Top_Op<"Relu",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface>,
  DeclareOpInterfaceMethods<QuantizeInterface>]> {
  let summary = "Relu operator";

  let description = [{
     ReLU with a scalar maximum value.
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def Top_ReshapeOp:Top_Op<"Reshape",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface>,
  DeclareOpInterfaceMethods<QuantizeInterface>]> {
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

def Top_SoftmaxOp:Top_Op<"Softmax",[NoSideEffect]> {
  let summary = "Softmax operation";
  let description = [{
    The operator computes the normalized exponential values for the given input.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

def Top_PadOp:Top_Op<"Pad",[NoSideEffect]> {
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

#endif // Top_OPS
