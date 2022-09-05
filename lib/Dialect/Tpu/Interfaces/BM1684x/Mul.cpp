//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Binary_param.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
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

// use for eltbinary or bcbinary
typedef struct binary_common_spec {
  int32_t binary_type;
  int32_t if_relu;
  float relu_limit;
  int32_t scale_A;
  int32_t scale_B;
  int32_t rshift_A;
  int32_t rshift_B;
} binary_common_spec_t;

typedef struct binary_local_spec {
  binary_common_spec_t common;
  uint32_t buffer_addr;
} binary_local_spec_t;

typedef struct binary_local_param {
  binary_local_spec_t spec;
  int32_t A_is_coeff;
  int32_t B_is_coeff;
} binary_local_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::MulOp::codegen_global_bm1684x() {
  bcbinary_common_spec_t param{0};
  param.binary_type = BINARY_MUL;
  param.if_relu = do_relu();
  param.relu_upper_limit = relu_limit().convertToDouble();
  param.rshift_A = rshift();
  param.rshift_B = 0;
  param.scale_A = multiplier();
  param.scale_B = 1;
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  BM1684x::instance().call_global_func("backend_api_bcbinary_global", &param,
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

int64_t tpu::MulOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  int64_t buffer_size = 0;
  auto dtype_A = BM168x::getDataType(inputs()[0]);
  auto dtype_B = BM168x::getDataType(inputs()[1]);
  auto dtype_O = BM168x::getDataType(output());
  if (dtype_A == DTYPE_INT8 || dtype_A == DTYPE_UINT8) {
    if (multiplier() != 1 || rshift() != 0) {
      buffer_size = in_lmem_bytes * 2;
    }
  } else if ((sizeof(dtype_A) > sizeof(dtype_O)) &&
             (is_sign(dtype_A) || is_sign(dtype_B)) && (!is_sign(dtype_O))) {
    buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::MulOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  auto in0_gi = LocalGenInterface::getGroupInfo(inputs()[0], n_step, h_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(inputs()[1], n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  uint32_t input_offset[] = {(uint32_t)in0_gi.out_addr,
                             (uint32_t)in1_gi.out_addr};
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto out_type = Module::getStorageType(output());
  auto in_type = Module::getStorageType(inputs()[0]);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  local_sec_info_t sec_info{0};
  sec_info.n_slice = gi.n_slice;
  sec_info.h_slice = in0_gi.h_slice;
  sec_info.w_slice = w;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.is_h_split = !(gi.h_idx == 0 && gi.h_slice == h);
  sec_info.h_idx = in0_gi.h_idx;

  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.is_w_split = false;
  sec_info.out_w_slice = w;

  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BINARY_MUL;
  param.spec.common.if_relu = do_relu();
  param.spec.common.relu_upper_limit = relu_limit().convertToDouble();
  param.spec.common.rshift_A = rshift();
  param.spec.common.rshift_B = 0;
  param.spec.common.scale_A = multiplier();
  param.spec.common.scale_B = 1;
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;
  BM1684x::instance().call_local_func("backend_api_bcbinary_local", &param,
                                      sizeof(param), &sec_info,
                                      input_spec->data(), output_spec->data());
}
