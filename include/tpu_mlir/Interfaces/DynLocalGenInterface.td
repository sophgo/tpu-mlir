//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_DYN_LOCAL_CODEGEN_INTERFACE
#define TPU_MLIR_DYN_LOCAL_CODEGEN_INTERFACE

include "mlir/IR/OpBase.td"

def DynLocalGenInterface : OpInterface<"DynLocalGenInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"dyn_codegen_local_cv18xx",
        /*args=*/(ins "void *":$buffer),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return 0;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int32_t",
        /*methodName=*/"dyn_codegen_local_bm1684",
        /*args=*/(ins "void *":$ir_layer_info)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"dyn_codegen_local_bm1684x",
        /*args=*/(ins "void *":$buffer)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"DynBackwardH",
        /*args=*/(ins "int64_t&":$in_idx, "int64_t&":$in_slice, "int64_t":$out_idx, "int64_t":$out_slice),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          in_idx = out_idx;
          in_slice = out_slice;
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"DynBackwardKh",
        /*args=*/(ins  "int64_t&":$in_kh, "int64_t":$out_kh),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          in_kh = out_kh;
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"DynBackwardStrideH",
        /*args=*/(ins  "int64_t&":$in_stride_h, "int64_t":$out_stride_h),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          in_stride_h = out_stride_h;
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"DynBackwardUpPadH",
        /*args=*/(ins  "int64_t&":$in_up_pad_h, "int64_t":$out_up_pad_h),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          in_up_pad_h = out_up_pad_h;
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"DynBackwardDownPadH",
        /*args=*/(ins  "int64_t&":$in_down_pad_h, "int64_t":$out_down_pad_h),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          in_down_pad_h = out_down_pad_h;
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"Dyn_BackwardN",
        /*args=*/(ins "int64_t&":$in_idx, "int64_t&":$in_slice, "int64_t":$out_idx, "int64_t":$out_slice),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          in_idx = out_idx;
          in_slice = out_slice;
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::int64_t",
        /*methodName=*/"DynForwardHeight",
        /*args=*/(ins "int64_t":$in_height),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return in_height;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"tpu_mlir::group_info_t",
        /*methodName=*/"DynGetGroupInfo",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return DynLocalGenInterface::DynGetGroupInfo(this->getOperation(), n_step, h_step);
        }]
      >
  ];
  let extraClassDeclaration = [{
  static constexpr llvm::StringRef kLayerGroupAttrName = "ginfo";
  static tpu_mlir::group_info_t DynGetGroupInfo(mlir::Operation * op, int64_t n_step = 0, int64_t h_step = 0);
  static tpu_mlir::group_info_t DynGetGroupInfo(mlir::Value, int64_t n_step = 0, int64_t h_step = 0);
  }];
}
#endif // TPU_MLIR_DYN_LOCAL_CODEGEN_INTERFACE
