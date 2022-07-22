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
// Defines TPU Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TPU_OPS
#define TPU_MLIR_TPU_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.td"
include "tpu_mlir/Interfaces/LocalGenInterface.td"
include "tpu_mlir/Interfaces/WeightReorderInterface.td"
include "tpu_mlir/Interfaces/GlobalGenInterface.td"
include "tpu_mlir/Interfaces/InferenceInterface.td"
include "tpu_mlir/Traits/Traits.td"

// ----------
// type
// ----------

def AnyTenor: AnyTypeOf<[AnyRankedTensor]>;
def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;

// ----------
// op definition
// ----------
class Tpu_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Tpu_Dialect, mnemonic, !listconcat(traits,[NoSideEffect, TpuTypeRestrict])> ;

class Tpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tpu_Dialect, mnemonic, !listconcat(traits,
       [NoSideEffect, HasCommonAttributes, TpuTypeRestrict,
       DeclareOpInterfaceMethods<GlobalGenInterface>,
       DeclareOpInterfaceMethods<InferenceInterface>])> ;

def Tpu_ConvOp: Tpu_Op<"Conv",[
    SupportFuseRelu,
    DeclareOpInterfaceMethods<WeightReorderInterface>,
    DeclareOpInterfaceMethods<LocalGenInterface, ["BackwardH"]>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$coeff_merged,
    DefaultValuedAttr<I64Attr, "0">:$use_3ic_optimize,
    OptionalAttr<I64ArrayAttr>:$multiplier,
    OptionalAttr<I64ArrayAttr>:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
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
   DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport","BackwardH"]>]> {
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
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
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

def Tpu_MaxPoolOp:Tpu_PoolOp<"MaxPool">;
def Tpu_AvgPoolOp:Tpu_PoolOp<"AvgPool"> {
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    DefaultValuedAttr<I64Attr, "0">:$pad_value,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<I64Attr, "0">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    StrAttr:$name
  );
}

def Tpu_AddOp: Tpu_Op<"Add", [SupportFuseRelu, InOutSameShape,
  DeclareOpInterfaceMethods<LocalGenInterface>]> {
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
    OptionalAttr<I64ArrayAttr>:$multipliers,
    OptionalAttr<I64ArrayAttr>:$rshifts,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_MulOp: Tpu_Op<"Mul", [SupportFuseRelu, InOutSameShape,
  DeclareOpInterfaceMethods<LocalGenInterface>]> {
  let summary = "mul operator";

  let description = [{
    Elementwise mul of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    // quant param
    DefaultValuedAttr<I64Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

// def Tpu_ConcatOp: Tpu_Op<"Concat", [SupportFuseRelu,
//   DeclareOpInterfaceMethods<LocalGenInterface>]> {
//   let summary = "concat operator";

//   let description = [{
//   }];

//   let arguments = (ins
//     Variadic<AnyTensor>:$inputs,
//     DefaultValuedAttr<I32Attr, "1">:$axis,
//     OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
//     StrAttr:$name
//   );

//   let results = (outs AnyTensor:$output);
// }

def Tpu_Depth2SpaceOp: Tpu_Op<"Depth2Space"> {

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

def Tpu_LutOp: Tpu_Op<"Lut",
  [DeclareOpInterfaceMethods<LocalGenInterface>, InOutSameShape]>{
  let summary = "Lut operator";

  let description = [{
    lookup table in index [0-255], y[i] = table(x[i])
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$table,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

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
    DefaultValuedAttr<I64Attr, "1">:$multiplier,
    DefaultValuedAttr<I64Attr, "0">:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let extraClassDeclaration = [{
    void parseParam(
      int64_t &batch, int64_t &M, int64_t &K, int64_t &N, bool &with_bias, bool &do_relu);
  }];
}

def Tpu_ReluOp: Tpu_Op<"Relu",
  [DeclareOpInterfaceMethods<LocalGenInterface>, InOutSameShape]>{
  let summary = "Relu operator";

  let description = [{
     ReLU with a scalar maximum value.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
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
  [DeclareOpInterfaceMethods<LocalGenInterface>, InOutSameShape]> {
  let summary = "Cast operation";
  let description = [{
  }];
  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def Tpu_LoadOp:Tpu_Op<"Load",
  [DeclareOpInterfaceMethods<LocalGenInterface>, InOutSameShape]> {
  let summary = "Load operation";
  let description = [{
    load input or weight from gmem to lmem;
    if do_bcast, [1,1,1,w] will load to [1,npu,1,w]
  }];
  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$do_bcast,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_StoreOp:Tpu_Op<"Store",
  [DeclareOpInterfaceMethods<LocalGenInterface>, InOutSameShape]> {
  let summary = "Store operation";
  let description = [{
  }];
  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_RequantOp:Tpu_Op<"Requant",
  [DeclareOpInterfaceMethods<LocalGenInterface>, InOutSameShape]> {
  let summary = "requant operation";
  let description = [{
    Requant int32 or f32 data to int8 or uint8 data
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$quant,
    OptionalAttr<I64Attr>:$multiplier,
    OptionalAttr<I64Attr>:$rshift,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_GroupOp:Tpu_BaseOp<"Group"> {
  let summary = "Group operation";
  let description = [{
    Make ops in one group to inferece by local mem
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    I64Attr:$nsecs,
    I64Attr:$hsecs
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let extraClassDeclaration = [{
    mlir::Operation* getRefOp(Value v);
  }];
}

def Tpu_YieldOp : Tpu_BaseOp<"Yield", [NoSideEffect,
    Terminator, HasParent<"GroupOp">]> {
  let summary = "Yield values to parent operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Tpu_SiLUOp : Tpu_Op<"SiLU",
    [DeclareOpInterfaceMethods<LocalGenInterface>, InOutSameShape]> {
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

def Tpu_SoftmaxOp: Tpu_Op<"Softmax", [InOutSameShape]> {
  let summary = "softmax operator";

  let description = [{
    The softmax function, also known as softargmax or normalized exponential
    function, is a generalization of the logistic function to multiple dimensions.
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    // quant param
    OptionalAttr<I64Attr>:$multiplier,
    OptionalAttr<I64Attr>:$rshift,
    OptionalAttr<I64ArrayAttr>:$table,
    OptionalAttr<Tpu_LayerGroupAttr>:$layer_group,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tpu_LeakyReluOp: Tpu_Op<"LeakyRelu", [
    DeclareOpInterfaceMethods<LocalGenInterface>,
    InOutSameShape]> {
  let summary = "leakyrelu operation";
  let description = [{
    The LeakyRelu operation multiples alpha with negative values, and the others keep changeless
  }];

  let arguments = (ins
    AnyTenor:$input,
    OptionalAttr<F64Attr>:$alpha,
    // quantize param
    OptionalAttr<I64Attr>:$multiplier,
    OptionalAttr<I64Attr>:$rshift,
    StrAttr:$name,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info
  );
  let results = (outs AnyTenor:$output);
}

def Tpu_ConcatOp:Tpu_Op<"Concat"> {
  let summary = "Concatate operation";
  let description = [{
  Concatenates the given sequence of seq tensors in the given dimension.
  All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  }];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    I64Attr:$axis,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

def Tpu_PermuteOp: Tpu_Op<"Permute"> {

  let summary = "Permute operator";

  let description = [{
      Perform permute on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64ArrayAttr:$order,
    StrAttr:$name,
    OptionalAttr<Tpu_LayerGroupAttr>:$group_info
  );

  let results = (outs AnyTensor:$output);
}

#endif // TPU_OPS
