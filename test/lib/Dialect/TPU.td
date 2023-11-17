//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_OPS
#define TPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// define tpu dialect
def TPU_Dialect : Dialect {
  let name = "tpu";
  let cppNamespace = "::tpu";
  let summary = "TPU instruction dialect.";
}

class TPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<TPU_Dialect, mnemonic, traits>;


// %R2.3333 = tpu.usemem : memref<43x22x44xf32, address=23ef22, layout=[2,3,4]>
// def MemRefOp : TPU_Op<"memref",
//     [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
//   let results = (outs AnyMemRef:$result);
// }

// %R2.dd.dd, %dmaId23 = tpu.dma.tensor %G0e34f, %tiuId32 {
//   mask = 5fea339e,
// } (memref<23x32xi64, strided=[23,44]>) -> (memref<45x89x77xi32>)
def SubViewOp : TPU_Op<"subview", [Pure]> {
  let arguments = (ins AnyMemRef:$source,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source ``
    $static_offsets
    $static_sizes
    $static_strides
    attr-dict `:` type($source) `to` type($result)
  }];
}


// %R2.dd.dd, %dmaId23 = tpu.dma.tensor %G0e34f, %tiuId32 {
//   mask = 5fea339e,
// } (memref<23x32xi64, strided=[23,44]>) -> (memref<45x89x77xi32>)
def GDMAOp : TPU_Op<"dma.transfer"> {
  let arguments = (ins
    AnyMemRef:$source,
    I64:$dep_id
  );
  let results = (outs AnyType:$target, I64:$id);
}

// %R2.dd.dd, %TID23 = tpu.matmul2.generic %R2.333.2323, %c2.3232, %DID120 {
//   partial_sum = true,
// } (memref<23x32xi64, strided=[23,44]>, f32) -> (memref<45x89x77xi32>)
def MatMulOp : TPU_Op<"matmul2"> {
  let arguments = (ins
    AnyMemRef:$left,
    AnyMemRef:$right,
    AnyType:$result
);
}

#endif // TPU_OPS
