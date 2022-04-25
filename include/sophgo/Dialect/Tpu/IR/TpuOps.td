// =============================================================================
//
// Defines TPU Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_TPU_OPS
#define SOPHGO_TPU_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "sophgo/Dialect/Tpu/IR/TpuAttr.td"
include "sophgo/Interfaces/LayerGroupInterface.td"
include "sophgo/Interfaces/WeightReorderInterface.td"
include "sophgo/Interfaces/CodegenInterface.td"
include "sophgo/Interfaces/InferenceInterface.td"
include "sophgo/Traits/Traits.td"

// ----------
// type
// ----------

def AnyTenor: AnyTypeOf<[AnyRankedTensor]>;
def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;

// ----------
// op definition
// ----------
class Tpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tpu_Dialect, mnemonic, !listconcat(traits,
       [NoSideEffect, HasCommonAttributes,
       DeclareOpInterfaceMethods<CodegenInterface>,
       DeclareOpInterfaceMethods<InferenceInterface>])> ;

def Tpu_ConvOp: Tpu_Op<"Conv",[
    SupportFuseRelu,
    DeclareOpInterfaceMethods<WeightReorderInterface>,
    DeclareOpInterfaceMethods<LayerGroupInterface, ["BackwardH"]>]> {
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
    //new param
    BoolAttr:$with_bias,
    OptionalAttr<I64ArrayAttr>:$rshift,
    OptionalAttr<I64ArrayAttr>:$multiplier,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
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
  [SupportFuseRelu,
   DeclareOpInterfaceMethods<LayerGroupInterface, ["Verify","BackwardH"]>]> {
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
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(
      int64_t &n, int64_t &c, int64_t &ih, int64_t &iw, int64_t &oh, int64_t &ow,
      int64_t &kh, int64_t &kw, int64_t &sh, int64_t &sw, int64_t &pt, int64_t &pb,
      int64_t &pl, int64_t &pr, int64_t &pad_value, bool &do_relu,
      bool &is_global, bool &count_include_pad);
  }];
}

def Tpu_AvgPoolOp:Tpu_PoolOp<"AvgPool">;
def Tpu_MaxPoolOp:Tpu_PoolOp<"MaxPool">;

def Tpu_AddOp: Tpu_Op<"Add", [SupportFuseRelu,
  DeclareOpInterfaceMethods<LayerGroupInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$rectified_bias,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

// def Tpu_ConcatOp: Tpu_Op<"Concat", [SupportFuseRelu,
//   DeclareOpInterfaceMethods<LayerGroupInterface>]> {
//   let summary = "concat operator";

//   let description = [{
//   }];

//   let arguments = (ins
//     Variadic<AnyTensor>:$inputs,
//     DefaultValuedAttr<I32Attr, "1">:$axis,
//     OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
//     StrAttr:$name
//   );

//   let results = (outs AnyTensor:$output);
// }

def Tpu_MatMulOp: Tpu_Op<"MatMul", [SupportFuseRelu]> {
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
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    DefaultValuedAttr<I64Attr, "0">:$multiplier,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(
      int64_t &batch, int64_t &M, int64_t &K, int64_t &N, bool &with_bias, bool &do_relu);
  }];
}

def Tpu_ReluOp: Tpu_Op<"Relu",
  [DeclareOpInterfaceMethods<LayerGroupInterface>]>{
  let summary = "Relu operator";

  let description = [{
     ReLU with a scalar maximum value.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_ReshapeOp:Tpu_Op<"Reshape"> {
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

def Tpu_CastOp:Tpu_Op<"Cast",
  [DeclareOpInterfaceMethods<LayerGroupInterface>]> {
  let summary = "Cast operation";
  let description = [{
  }];
  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_RequantOp:Tpu_Op<"Requant",
  [DeclareOpInterfaceMethods<LayerGroupInterface>]> {
  let summary = "requant operation";
  let description = [{
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$quant,
    OptionalAttr<I64Attr>:$multiplier,
    OptionalAttr<I64Attr>:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

#endif // TPU_OPS
