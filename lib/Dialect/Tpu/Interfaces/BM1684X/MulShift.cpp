//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  unsigned int buffer_addr; // only used for local layer
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int scale_val;
  int rshift_num;
  DATA_TYPE_T input_dtype;
  DATA_TYPE_T scale_dtype;
  DATA_TYPE_T output_dtype;
  ROUND_MODE_T round_mode;
} mulshift_param_t;

#ifdef __cplusplus
}
#endif

void tpu::MulShiftOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto op = getOperation();
  if (Quant::isUniformQuantized(input())) {
    auto in_qtype = Quant::getUniformQuantizedType(input());
    auto out_qtype = Quant::getUniformQuantizedType(output());
    auto in_zp = in_qtype.getZeroPoint();
    auto out_zp = out_qtype.getZeroPoint();
    if (in_zp != 0 || out_zp != 0) {
      requant_int_param_t param = {0};
      param.input_addr = Module::getAddress(input());
      param.output_addr = Module::getAddress(output());
      param.n = (int)n;
      param.c = (int)c;
      param.h = (int)h;
      param.w = (int)w;
      param.mul_value = multiplier();
      param.shift_value = -rshift();
      param.offset_value = out_zp;
      param.zx_value = in_zp;
      param.mode = 2;
      param.input_dtype = BM168x::getDataType(input());
      param.output_dtype = BM168x::getDataType(output());
      BM168x::call_global_func("backend_api_requant_int_global",
                                           &param, sizeof(param));
      return;
    }
  }
  mulshift_param_t param = {0};
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  param.input_n = n;
  param.input_c = c;
  param.input_h = h;
  param.input_w = w;
  param.scale_val = multiplier();
  param.rshift_num = rshift();
  param.input_dtype = BM168x::getDataType(input());
  param.scale_dtype = param.scale_val < 0 ? DTYPE_INT8 : DTYPE_UINT8;
  param.output_dtype = BM168x::getDataType(output());
  param.round_mode = ROUND_UP;
  BM168x::call_global_func("backend_api_mulshift_global", &param,
                                       sizeof(param));
}

int64_t tpu::MulShiftOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  auto in_sType = Module::getStorageType(input());
  auto out_sType = Module::getStorageType(output());
  if (Quant::isUniformQuantized(input())) {
    auto in_qType = Quant::getUniformQuantizedType(input());
    auto out_qType = Quant::getUniformQuantizedType(output());
    if (in_qType.getZeroPoint() != 0 || out_qType.getZeroPoint() != 0) {
      return 2 * in_lmem_bytes;
    }
  }
  if (in_sType.isUnsignedInteger(8) == false &&
      out_sType.isUnsignedInteger(8)) {
    return 2 * in_lmem_bytes;
  }
  return 0;
}

void tpu::MulShiftOp::codegen_local_bm1684x(int64_t n_step,
                                                 int64_t h_step) {
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto op = getOperation();
  if (Quant::isUniformQuantized(input())) {
    auto in_qtype = Quant::getUniformQuantizedType(input());
    auto out_qtype = Quant::getUniformQuantizedType(output());
    auto in_zp = in_qtype.getZeroPoint();
    auto out_zp = out_qtype.getZeroPoint();
    if (in_zp != 0 || out_zp != 0) {
      requant_int_param_t param = {0};
      param.input_addr = (uint32_t)in_gi.out_addr;
      param.output_addr = (uint32_t)gi.out_addr;
      param.buffer_local_addr = (uint32_t)gi.buffer_addr;
      param.n = gi.n_slice;
      param.c = c;
      param.h = gi.h_slice;
      param.w = w;
      param.mul_value = multiplier();
      param.shift_value = -rshift();
      param.offset_value = out_zp;
      param.zx_value = in_zp;
      param.input_dtype = BM168x::getDataType(input());
      param.output_dtype = BM168x::getDataType(output());
      param.mode = 2;
      BM168x::call_local_func("backend_api_requant_int_local",
                                          &param, sizeof(param));
      return;
    }
  }
  mulshift_param_t param = {0};
  param.input_addr = in_gi.out_addr;
  param.output_addr = gi.out_addr;
  param.buffer_addr = gi.buffer_addr;
  param.input_n = in_gi.n_slice;
  param.input_c = c;
  param.input_h = in_gi.h_slice;
  param.input_w = w;
  param.scale_val = multiplier();
  param.rshift_num = rshift();
  param.input_dtype = BM168x::getDataType(input());
  param.scale_dtype = param.scale_val < 0 ? DTYPE_INT8 : DTYPE_UINT8;
  param.output_dtype = BM168x::getDataType(output());
  param.round_mode = ROUND_UP;
  BM168x::call_local_func("backend_api_mulshift_local", &param,
                                      sizeof(param));
}
