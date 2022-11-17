//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t requant_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
  int round_mode;
} requant_fp_param_t;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantFpOp::codegen_global_bm1684x() {
  requant_fp_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  auto oqtype = Quant::getUniformQuantizedType(output());
  param.scale_value = scaleAttr().getValueAsDouble();
  param.offset_value = oqtype.getZeroPoint();
  param.mode = static_cast<int>(quant_mode()) / 2;
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  param.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
  auto op = getOperation();
  BM168x::instance(Module::getChip(op))->call_global_func("backend_api_requant_float_global", &param,
                                       sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantFpOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  if (quant_mode() != RequantMode::Normal) {
    buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::RequantFpOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  requant_fp_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = gi.n_slice;
  param.c = c;
  param.h = gi.h_slice;
  param.w = w;

  auto oqtype = Quant::getUniformQuantizedType(output());
  param.scale_value = scaleAttr().getValueAsDouble();
  param.offset_value = oqtype.getZeroPoint();
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  param.mode = static_cast<int>(quant_mode()) / 2;
  param.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
  auto op = getOperation();
  BM168x::instance(Module::getChip(op))->call_local_func("backend_api_requant_float_local", &param,
                                      sizeof(param));
}
