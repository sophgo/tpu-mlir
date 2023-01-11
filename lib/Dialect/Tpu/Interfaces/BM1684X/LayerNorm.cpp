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
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct layer_norm_common_spec {
  int axis;
  float eps;
  int affine;
  int need_mean;
  int need_rstd;
} layer_norm_common_spec_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LayerNormOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  layer_norm_common_spec_t param = {0};
  const bool have_bias = !getBias().getType().isa<NoneType>();
  const bool need_mean = !getMean().getType().isa<NoneType>();
  const bool need_rstd = !getRstd().getType().isa<NoneType>();
  param.axis = (int)getAxis();
  param.eps = getEps().convertToDouble();
  param.affine = have_bias ? 3 : 1;
  param.need_mean = need_mean;
  param.need_rstd = need_rstd;
  BM168x::call_global_func("backend_api_layer_norm_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// // =========================================
// // LocalGenInterface
// // =========================================

// int64_t tpu::LayerNormOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
//                                                 int64_t out_lmem_bytes,
//                                                 int64_t in_nslice, int64_t
//                                                 in_hslice, int64_t
//                                                 out_nslice, int64_t
//                                                 out_hslice) {
//   // TODO: supports group-3d case
//   int64_t n, c, h, w;
//   module::getNCHW(getInput(), n, c, h, w);
//   int num = in_nslice; // num = depth * nslice
//   int in_wslice = 1;
//   int c_per_npu = ceiling_func(c, BM1684X::NPU_NUM);
//   const int EU_NUM = BM1684X::EU_BYTES / 4;
//   int buffer1_size = sizeof(float) * num * c_per_npu * EU_NUM;
//   int tensor_size = sizeof(float) * num * c_per_npu *
//                     align_up((int)in_hslice * in_wslice, EU_NUM);
//   int64_t buffer_size = buffer1_size + tensor_size;
//   const bool need_mean = !getMean().getType().isa<NoneType>();
//   const bool need_rstd = !getRstd().getType().isa<NoneType>();
//   if (!need_mean) buffer_size += buffer1_size;
//   if (!need_rstd) buffer_size += buffer1_size;
//   return buffer_size;
// }

// void tpu::LayerNormOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step)
// {
//   const bool have_bias = !getBias().getType().isa<NoneType>();
//   const bool need_mean = !getMean().getType().isa<NoneType>();
//   const bool need_rstd = !getRstd().getType().isa<NoneType>();
//   // TODO: support group-3d case
//   assert(!need_mean && !need_rstd);
//   int64_t n, c, h, w;
//   module::getNCHW(getInput(), n, c, h, w);
//   auto gi = getGroupInfo(n_step, h_step);
//   auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
//   layer_norm_local_param_t param = {0};
//   param.input_addr = (uint32_t)in_gi.out_addr;
//   param.weight_addr = module::getAddress(getWeight());
//   param.bias_addr = have_bias ? module::getAddress(getBias()): UINT64_MAX;
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
//   param.eps = getEps().convertToDouble();
//   param.affine = have_bias ? 3 : 1;
//   param.need_mean = need_mean;
//   param.need_rstd = need_rstd;
//   param.dtype = DTYPE_FP32;
//   BM168x::call_local_func("backend_api_layer_norm_local", &param,
//   sizeof(param));
// }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LayerNormOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}
