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

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::SubConstOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  module::getNCHW(output(), n, c, h, w);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto input_type = module::getStorageType(input());
  constbinary_global_spec_t param = {0};
  param.common.binary_type = BINARY_SUB;
  param.common.if_relu = do_relu();
  param.common.relu_upper_limit = relu_limit().convertToDouble();
  param.common.B_const_val = const_val().convertToDouble();
  param.common.inversed = is_reverse();
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  if (module::isUniformQuantized(input())) {
    param.common.B_dtype = DTYPE_INT32;
    param.common.scale_A = multiplier();
    param.common.rshift_A = rshift();
  } else {
    param.common.B_dtype =
        input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }
  BM168x::call_global_func("backend_api_constbinary_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

static bool is_sign(DATA_TYPE_T dtype) {
  return !(dtype == DTYPE_UINT8 || dtype == DTYPE_UINT16 ||
           dtype == DTYPE_UINT32);
}

int64_t tpu::SubConstOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  auto dtype_A = BM168x::getDataType(input());
  if (dtype_A == DTYPE_INT8 || dtype_A == DTYPE_UINT8) {
    buffer_size = in_lmem_bytes * 2;
  }
  return buffer_size;
}

void tpu::SubConstOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                      void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(input(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  sec_info->n_slice = in_gi.n_slice;
  sec_info->d_slice = 1;
  sec_info->h_slice = in_gi.h_slice;
  sec_info->h_idx = in_gi.h_idx;
  sec_info->is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info->w_slice = w;
  sec_info->out_n_slice = gi.n_slice;
  sec_info->out_h_idx = gi.h_idx;
  sec_info->out_h_slice = gi.h_slice;
  sec_info->out_w_slice = w;
}

void tpu::SubConstOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                            void *sec_info_) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto input_type = module::getStorageType(input());

  constbinary_local_spec_t param = {0};
  param.common.binary_type = BINARY_SUB;
  param.common.if_relu = do_relu();
  param.common.relu_upper_limit = relu_limit().convertToDouble();
  param.common.B_const_val = const_val().convertToDouble();
  param.common.inversed = is_reverse();
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  if (module::isUniformQuantized(input())) {
    param.common.B_dtype = DTYPE_INT32;
    param.common.scale_A = multiplier();
    param.common.rshift_A = rshift();
  } else {
    param.common.B_dtype =
        input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }

  BM168x::call_local_func("backend_api_constbinary_local", &param,
                          sizeof(param), sec_info_, input_spec->data(),
                          output_spec->data());
}
