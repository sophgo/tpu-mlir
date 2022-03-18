// =============================================================================
//
// Defines TPU Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_OPS
#define TPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

def Tpu_Dialect : Dialect {
  let name = "tpu";
  let summary = "A tpu dialect for the SOPHGO AI chips";
  let cppNamespace = "::sophgo::tpu";
}

class Tpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tpu_Dialect, mnemonic, traits> ;

def Tpu_ConvOp: Tpu_Op<"Conv",[NoSideEffect]> {
  let summary = "convolution operator";

  let description = [{
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
    //quant param
    DefaultValuedAttr<I64Attr, "0">:$rshift,
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

class Tpu_PoolOp <string mnemonic> : Tpu_Op<mnemonic,
  [NoSideEffect]> {
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

def Tpu_AvgPoolOp:Tpu_PoolOp<"AvgPool">;
def Tpu_MaxPoolOp:Tpu_PoolOp<"MaxPool">;

def Tpu_AddOp: Tpu_Op<"Add",[NoSideEffect]> {
  let summary = "add operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    OptionalAttr<F64ArrayAttr>:$coeff,
    // quant param
    I64ArrayAttr:$rshifts,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_MatMulOp: Tpu_Op<"MatMul",[NoSideEffect]> {
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

def Tpu_ReluOp: Tpu_Op<"Relu",[NoSideEffect]> {
  let summary = "Relu operator";

  let description = [{
     ReLU with a scalar maximum value.
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_ReshapeOp:Tpu_Op<"Reshape",[NoSideEffect]> {
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
}

def Tpu_CastOp:Tpu_Op<"Cast",[NoSideEffect]> {
  let summary = "Cast operation";
  let description = [{
  }];
  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

#endif // TPU_OPS
