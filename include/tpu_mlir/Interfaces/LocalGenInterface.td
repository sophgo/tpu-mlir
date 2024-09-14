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
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"AllowDataSplit",
        /*args=*/(ins "int64_t":$axis, "group_type_t":$group_type),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"assign_sec_info",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$c_step, "int64_t":$h_step, "int64_t":$d_step, "int64_t":$w_step, "group_type_t":$group_type, "local_sec_info_t &":$sec_info),
        [{}],
        [{
            memset(&sec_info, 0, sizeof(local_sec_info_t));
            sec_info.group_type = group_type;
            int64_t n, c, d, h, w, on, oc, od, oh, ow;
            auto input = $_op->getOperand(0);
            auto output = $_op->getResult(0);
            module::getNCDHW(input, n, c, d, h, w, group_type);
            module::getNCDHW(output, on, oc, od, oh, ow, group_type);
            auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
            auto in_gi = LocalGenInterface::getGroupInfo(input, n_step, h_step, d_step, w_step, c_step);
            sec_info.n_slice = in_gi.n_slice;
            sec_info.d_slice = in_gi.d_slice;
            sec_info.h_slice = in_gi.h_slice;
            sec_info.w_slice = in_gi.w_slice;
            sec_info.c_slice = gi.c_slice;
            sec_info.n_idx = in_gi.n_idx;
            sec_info.d_idx = in_gi.d_idx;
            sec_info.h_idx = in_gi.h_idx;
            sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
            sec_info.w_idx = in_gi.w_idx;
            sec_info.is_w_split = !(in_gi.w_idx == 0 && in_gi.w_slice == w);
            sec_info.c_idx = gi.c_idx;
            sec_info.is_c_split = !(in_gi.c_idx == 0 && in_gi.c_slice == c);
            sec_info.out_n_slice = gi.n_slice;
            sec_info.out_h_idx = gi.h_idx;
            sec_info.out_h_slice = gi.h_slice;
            sec_info.out_w_idx = gi.w_idx;
            sec_info.out_w_slice = gi.w_slice;
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
        /*args=*/(ins "int64_t":$n_step, "int64_t":$c_step, "int64_t":$h_step, "int64_t":$d_step, "int64_t":$w_step, "group_type_t":$group_type, "local_sec_info_t &":$sec_info)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local_bm168x",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$c_step, "int64_t":$h_step, "int64_t":$d_step, "int64_t":$w_step, "group_type_t":$group_type, "local_sec_info_t &":$sec_info),
        [{}],
        [{
          if (module::isBM1684XFamily()
              || module::isBM1690Family()) {
            $_op.codegen_local_bm1684x(n_step, c_step, h_step, d_step, w_step, group_type, sec_info);
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
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step, "int64_t":$d_step, "int64_t":$w_step, "group_type_t":$group_type, "local_sec_info_t &":$sec_info, "int64_t":$layer_id)
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
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"BackwardD",
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
        /*methodName=*/"BackwardW",
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
        /*methodName=*/"BackwardC",
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
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step, "int64_t":$d_step, "int64_t":$w_step, "int64_t":$c_step),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return LocalGenInterface::getGroupInfo(this->getOperation(), n_step, h_step, d_step, w_step, c_step);
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
                      "int64_t":$in_nslice, "int64_t":$in_cslice, "int64_t":$in_hslice, "int64_t":$in_dslice, "int64_t":$in_wslice,
                      "int64_t":$out_nslice, "int64_t":$out_cslice, "int64_t":$out_hslice, "int64_t":$out_dslice, "int64_t":$out_wslice,
                      "group_type_t":$group_type)
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
                      "int64_t":$in_nslice, "int64_t":$in_cslice, "int64_t":$in_hslice, "int64_t":$in_dslice, "int64_t":$in_wslice,
                      "int64_t":$out_nslice, "int64_t":$out_cslice, "int64_t":$out_hslice, "int64_t":$out_dslice, "int64_t":$out_wslice,
                      "group_type_t":$group_type),
        [{}],
        [{
          if (module::isBM1684Family()) {
            return $_op.getBufferSize_bm1684(in_lmem_bytes, out_lmem_bytes,
                                             in_nslice, in_hslice, out_nslice, out_hslice);
          } else if (module::isBM1684XFamily()
                     || module::isBM1690Family()) {
            return $_op.getBufferSize_bm1684x(in_lmem_bytes, out_lmem_bytes,
                                              in_nslice, in_cslice, in_hslice, in_dslice, in_wslice,
                                              out_nslice, out_cslice, out_hslice, out_dslice, out_wslice,
                                              group_type);
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
  static tpu_mlir::group_info_t getGroupInfo(mlir::Operation * op, int64_t n_step = 0, int64_t h_step = 0, int64_t d_step = 0, int64_t w_step = 0, int64_t c_step = 0, mlir::Operation *user_op = nullptr);
  static tpu_mlir::group_info_t getGroupInfo(mlir::Value, int64_t n_step = 0, int64_t h_step = 0, int64_t d_step = 0, int64_t w_step = 0, int64_t c_step = 0, mlir::Operation *user_op = nullptr);
  }];
}
#endif // TPU_MLIR_LOCAL_CODEGEN_INTERFACE
