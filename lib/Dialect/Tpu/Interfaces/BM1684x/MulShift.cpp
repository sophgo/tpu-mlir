//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
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

void tpu::MulShiftOp::codegen_global_int8_bm1684x() {
  mulshift_param_t param = {0};
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  param.input_n = n;
  param.input_c = c;
  param.input_h = h;
  param.input_w = w;
  param.scale_val = multiplier();
  param.rshift_num = rshift();
  param.input_dtype = BM168x::getDataType(input());
  param.scale_dtype = DTYPE_UINT8; // default
  param.output_dtype = BM168x::getDataType(output());
  param.round_mode = ROUND_UP;
  BM1684x::instance().call_global_func("backend_api_mulshift_global", &param,
                                       sizeof(param));
}

void tpu::MulShiftOp::codegen_global_float_bm1684x() {
  llvm_unreachable("not support now");
}

int64_t tpu::MulShiftOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  auto in_sType = Module::getStorageType(input());
  auto out_sType = Module::getStorageType(output());
  if (in_sType.isUnsignedInteger(8) == false &&
      out_sType.isUnsignedInteger(8)) {
    return 2 * in_lmem_bytes;
  }
  return 0;
}

void tpu::MulShiftOp::codegen_local_int8_bm1684x(int64_t n_step,
                                                 int64_t h_step) {
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
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
  param.scale_dtype = DTYPE_UINT8; // default
  param.output_dtype = BM168x::getDataType(output());
  param.round_mode = ROUND_UP;
  BM1684x::instance().call_local_func("backend_api_mulshift_local", &param,
                                      sizeof(param));
}

void tpu::MulShiftOp::codegen_local_float_bm1684x(int64_t n_step,
                                                  int64_t h_step) {
  llvm_unreachable("not support now");
}
