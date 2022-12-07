//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LayerNormOp::codegen_global_bm1684x() {
  const bool have_bias = !bias().getType().isa<NoneType>();
  const bool need_mean = !mean().getType().isa<NoneType>();
  const bool need_rstd = !rstd().getType().isa<NoneType>();
  const auto input_shape = Module::getShape(input());
  layer_norm_global_param_t param = {0};
  param.input_addr = Module::getAddress(input());
  param.weight_addr = Module::getAddress(weight());
  param.bias_addr = have_bias ? Module::getAddress(bias()) : UINT64_MAX;
  param.output_addr = Module::getAddress(output());
  param.mean_addr = Module::getAddress(mean());
  param.rstd_addr = Module::getAddress(rstd());
  param.dims = input_shape.size();
  for (int i = 0; i < param.dims; ++i) {
    param.shape[i] = (int)input_shape[i];
  }
  param.axis = (int)axis();
  param.eps = eps().convertToDouble();
  param.affine = have_bias ? 3 : 1;
  param.need_mean = need_mean;
  param.need_rstd = need_rstd;
  param.dtype = DTYPE_FP32;
  BM168x::call_global_func("backend_api_layer_norm_global", &param, sizeof(param));
}

// // =========================================
// // LocalGenInterface
// // =========================================

// int64_t tpu::LayerNormOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
//                                                 int64_t out_lmem_bytes,
//                                                 int64_t in_nslice, int64_t in_hslice,
//                                                 int64_t out_nslice,
//                                                 int64_t out_hslice) {
//   // TODO: supports group-3d case
//   int64_t n, c, h, w;
//   Module::getNCHW(input(), n, c, h, w);
//   int num = in_nslice; // num = depth * nslice
//   int in_wslice = 1;
//   int c_per_npu = ceiling_func(c, BM1684X::NPU_NUM);
//   const int EU_NUM = BM1684X::EU_BYTES / 4;
//   int buffer1_size = sizeof(float) * num * c_per_npu * EU_NUM;
//   int tensor_size = sizeof(float) * num * c_per_npu *
//                     align_up((int)in_hslice * in_wslice, EU_NUM);
//   int64_t buffer_size = buffer1_size + tensor_size;
//   const bool need_mean = !mean().getType().isa<NoneType>();
//   const bool need_rstd = !rstd().getType().isa<NoneType>();
//   if (!need_mean) buffer_size += buffer1_size;
//   if (!need_rstd) buffer_size += buffer1_size;
//   return buffer_size;
// }

// void tpu::LayerNormOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
//   const bool have_bias = !bias().getType().isa<NoneType>();
//   const bool need_mean = !mean().getType().isa<NoneType>();
//   const bool need_rstd = !rstd().getType().isa<NoneType>();
//   // TODO: support group-3d case
//   assert(!need_mean && !need_rstd);
//   int64_t n, c, h, w;
//   Module::getNCHW(input(), n, c, h, w);
//   auto gi = getGroupInfo(n_step, h_step);
//   auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
//   layer_norm_local_param_t param = {0};
//   param.input_addr = (uint32_t)in_gi.out_addr;
//   param.weight_addr = Module::getAddress(weight());
//   param.bias_addr = have_bias ? Module::getAddress(bias()): UINT64_MAX;
//   param.output_addr = (uint32_t)gi.out_addr;
//   param.buffer_addr = (uint32_t)gi.buffer_addr;
//   // NOTE: split mean and rstd if needed
//   // param.mean_addr = ;
//   // param.rstd_addr = ;
//   param.input_n = gi.n_slice;
//   param.input_c = c;
//   param.input_h = gi.h_slice;
//   param.input_w = w;
//   param.depth = 1;
//   param.eps = eps().convertToDouble();
//   param.affine = have_bias ? 3 : 1;
//   param.need_mean = need_mean;
//   param.need_rstd = need_rstd;
//   param.dtype = DTYPE_FP32;
//   BM168x::call_local_func("backend_api_layer_norm_local", &param, sizeof(param));
// }
