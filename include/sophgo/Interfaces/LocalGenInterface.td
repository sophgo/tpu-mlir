//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_LOCAL_CODEGEN_INTERFACE
#define SOPHGO_LOCAL_CODEGEN_INTERFACE

include "mlir/IR/OpBase.td"

def LocalGenInterface : OpInterface<"LocalGenInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"LocalGenSupport",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local_int8_bm1684",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local_int8_bm1686",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local_float_bm1686",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step),
        [{}],
        [{
          auto module = Module::getModuleOp($_op);
          auto chip = Module::getChip(module);
          if (chip == Module::Chip::BM1684) {
            $_op.codegen_local_int8_bm1684(n_step, h_step);
          } else if (chip == Module::Chip::BM1686) {
            if (Module::isFloatType($_op.getResult())) {
              $_op.codegen_local_float_bm1686(n_step, h_step);
            } else {
              $_op.codegen_local_int8_bm1686(n_step, h_step);
            }
          }
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"BackwardH",
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
        /*methodName=*/"BackwardN",
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
        /*retType=*/"sophgo::group_info_t",
        /*methodName=*/"getGroupInfo",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return LocalGenInterface::getGroupInfo(this->getOperation(), n_step, h_step);
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"getBufferSize_bm1684",
        /*args=*/(ins "int64_t":$out_n, "int64_t":$out_c, "int64_t":$out_h, "int64_t":$out_w, "int64_t":$out_lmem_bytes)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"getBufferSize_bm1686",
        /*args=*/(ins "int64_t":$out_n, "int64_t":$out_c, "int64_t":$out_h, "int64_t":$out_w, "int64_t":$out_lmem_bytes)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"getBufferSize",
        /*args=*/(ins "int64_t":$out_n, "int64_t":$out_c, "int64_t":$out_h, "int64_t":$out_w, "int64_t":$out_lmem_bytes),
        [{}],
        [{
          auto module = Module::getModuleOp($_op);
          auto chip = Module::getChip(module);
          if (chip == Module::Chip::BM1684) {
            return $_op.getBufferSize_bm1684(out_n, out_c, out_h, out_w, out_lmem_bytes);
          } else if (chip == Module::Chip::BM1686) {
            return $_op.getBufferSize_bm1686(out_n, out_c, out_h, out_w, out_lmem_bytes);
          }
          return 0;
        }]
      >,
  ];
  let extraClassDeclaration = [{
  static constexpr llvm::StringRef kLayerGroupAttrName = "group_info";
  static void fixSlice(int64_t &in_idx, int64_t &in_slice, int64_t in_length);
  static sophgo::group_info_t getGroupInfo(mlir::Operation * op, int64_t n_step = 0, int64_t h_step = 0);
  static sophgo::group_info_t getGroupInfo(mlir::Value, int64_t n_step = 0, int64_t h_step = 0);
  }];
}
#endif // SOPHGO_LOCAL_CODEGEN_INTERFACE
