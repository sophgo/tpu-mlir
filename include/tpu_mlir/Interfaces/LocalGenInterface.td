//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_LOCAL_CODEGEN_INTERFACE
#define TPU_MLIR_LOCAL_CODEGEN_INTERFACE

include "mlir/IR/OpBase.td"

def LocalGenInterface : OpInterface<"LocalGenInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
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
        /*methodName=*/"assign_sec_info",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step, "local_sec_info_t &":$sec_info),
        [{}],
        [{
            memset(&sec_info, 0, sizeof(local_sec_info_t));
            int64_t n, c, h, w, on, oc, oh, ow;
            auto input = $_op->getOperand(0);
            auto output = $_op->getResult(0);
            module::getNCHW(input, n, c, h, w);
            module::getNCHW(output, on, oc, oh, ow);
            auto gi = getGroupInfo(n_step, h_step);
            auto in_gi = LocalGenInterface::getGroupInfo(input, n_step, h_step);
            sec_info.n_slice = in_gi.n_slice;
            sec_info.d_slice = 1;
            sec_info.h_slice = in_gi.h_slice;
            sec_info.h_idx = in_gi.h_idx;
            sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
            sec_info.w_slice = w;
            sec_info.out_n_slice = gi.n_slice;
            sec_info.out_h_idx = gi.h_idx;
            sec_info.out_h_slice = gi.h_slice;
            sec_info.out_w_slice = ow;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local_bm1684",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step, "local_sec_info_t &":$sec_info)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local_bm1684x",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step, "local_sec_info_t &":$sec_info)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local_bm168x",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step, "local_sec_info_t &":$sec_info),
        [{}],
        [{
          if (module::isBM1684XFamily()) {
            $_op.codegen_local_bm1684x(n_step, h_step, sec_info);
          } else if (module::isBM1684Family()) {
            $_op.codegen_local_bm1684(n_step, h_step, sec_info);
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
        /*retType=*/"void",
        /*methodName=*/"codegen_local_cv18xx",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step, "int64_t":$layer_id)
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
        /*retType=*/"tpu_mlir::group_info_t",
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
        /*args=*/(ins "int64_t":$in_lmem_bytes, "int64_t":$out_lmem_bytes,
                      "int64_t":$in_nslice, "int64_t":$in_hslice,
                      "int64_t":$out_nslice, "int64_t":$out_hslice)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"getBufferSize_bm1684x",
        /*args=*/(ins "int64_t":$in_lmem_bytes, "int64_t":$out_lmem_bytes,
                      "int64_t":$in_nslice, "int64_t":$in_hslice,
                      "int64_t":$out_nslice, "int64_t":$out_hslice)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"getBufferSize_cv18xx",
        /*args=*/(ins "int64_t":$in_lmem_bytes, "int64_t":$out_lmem_bytes,
                      "int64_t":$in_nslice, "int64_t":$in_hslice,
                      "int64_t":$out_nslice, "int64_t":$out_hslice)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"getBufferSize",
        /*args=*/(ins "int64_t":$in_lmem_bytes, "int64_t":$out_lmem_bytes,
                      "int64_t":$in_nslice, "int64_t":$in_hslice,
                      "int64_t":$out_nslice, "int64_t":$out_hslice),
        [{}],
        [{
          if (module::isBM1684Family()) {
            return $_op.getBufferSize_bm1684(in_lmem_bytes, out_lmem_bytes,
                                             in_nslice, in_hslice, out_nslice, out_hslice);
          } else if (module::isBM1684XFamily()) {
            return $_op.getBufferSize_bm1684x(in_lmem_bytes, out_lmem_bytes,
                                              in_nslice, in_hslice, out_nslice, out_hslice);
          } else if (module::isCV18xx()) {
            return $_op.getBufferSize_cv18xx(in_lmem_bytes, out_lmem_bytes,
                                              in_nslice, in_hslice, out_nslice, out_hslice);
          }
          return 0;
        }]
      >,
  ];
  let extraClassDeclaration = [{
  static constexpr llvm::StringRef kLayerGroupAttrName = "ginfo";
  static void fixSlice(int64_t &in_idx, int64_t &in_slice, int64_t in_length, bool last = false);
  static tpu_mlir::group_info_t getGroupInfo(mlir::Operation * op, int64_t n_step = 0, int64_t h_step = 0);
  static tpu_mlir::group_info_t getGroupInfo(mlir::Value, int64_t n_step = 0, int64_t h_step = 0);
  }];
}
#endif // TPU_MLIR_LOCAL_CODEGEN_INTERFACE
