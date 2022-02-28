// =============================================================================
//
// Defines TOPS Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TOPS_OPS
#define TOPS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "sophgo/Interfaces/InferenceInterface.td"

def Tops_Dialect : Dialect {
  let name = "tops";
  let summary = "A tops dialect for the SOPHGO specification";
  let cppNamespace = "::mlir::tops";
}

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

class Tops_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tops_Dialect, mnemonic, traits> ;

def Tops_NoneOp : Tops_Op<"None", [NoSideEffect]> {
  let summary = "none operator";

  let description = [{
    A none Op to return a NoneType.
  }];
  let results = (outs NoneType);
}

def Tops_WeightOp : Tops_Op<"Weight", [NoSideEffect]> {
  let summary = "load weight operator";

  let description = [{
  }];

  let arguments = (
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tops_InputOp: Tops_Op<"Input",[NoSideEffect]> {
  let summary = "Input operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tops_BatchNormOp: Tops_Op<"BatchNorm",[NoSideEffect]> {
  let summary = "BatchNormalization operation";
  let description = [{
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

def Tops_ConvOp: Tops_Op<"Conv",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>]> {
  let summary = "conv operator";

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

class Tops_PoolOp <string mnemonic> : Tops_Op<mnemonic,
  [NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>]> {
  let summary = "pool operator";

  let description = [{
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

def Tops_AvgPoolOp:Tops_PoolOp<"AvgPool">;
def Tops_MaxPoolOp:Tops_PoolOp<"MaxPool">;

def Tops_AddOp: Tops_Op<"Add",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface>]> {
  let summary = "add operator";

  let description = [{
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    OptionalAttr<F32ArrayAttr>:$coeff,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tops_MatMulOp: Tops_Op<"MatMul",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>]> {
  let summary = "matmul operator";

  let description = [{
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
      int64_t &batch, int64_t &M, int64_t &K, int64_t &N);
  }];
}

def Tops_ReluOp: Tops_Op<"Relu",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface>]> {
  let summary = "Relu operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def Tops_ReshapeOp:Tops_Op<"Reshape",[NoSideEffect,
  DeclareOpInterfaceMethods<InferenceInterface>]> {
  let summary = "TOPS Reshape operation";
  let description = [{
  }];
  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}
#endif // Tops_OPS
