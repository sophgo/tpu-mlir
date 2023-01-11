//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int axis;
  int axis_num;
  int has_bias;
  int if_relu;
  float relu_upper_limit;
  int scale_sign;
  int bias_sign;
  int merge_weight_bias;
  int round_mode;
  int version;
} scale_global_spec_t;

typedef struct {
  /*common param*/
  int if_relu;
  float relu_upper_limit;
  int is_scale_coeff;
  int is_bias_coeff;
  int input_num;
  int merge_weight_bias;

  /*param for float*/
  int scale_shape[4];

  /*param for fixed*/
  unsigned int buffer_local_addr;
  int is_shift_coeff;
  int round_mode;
  int version;
  int bias_dtype;
} scale_local_spec_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::ScaleOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  scale_global_spec_t p = {0};
  p.axis = 1;
  p.axis_num = 1;
  p.has_bias = true;
  p.if_relu = getDoRelu();
  p.relu_upper_limit = getReluLimit().convertToDouble();
  p.merge_weight_bias = 0;
  p.round_mode = ROUND_UP;
  if (module::isUniformQuantized(getInput())) {
    p.scale_sign = !module::getStorageType(getScale()).isUnsignedInteger();
    p.bias_sign = !module::getStorageType(getBias()).isUnsignedInteger();
    p.version = 10;
    BM168x::call_global_func("backend_api_scale_global", &p, sizeof(p),
                             input_spec->data(), output_spec->data());
  } else {
    BM168x::call_global_func("backend_api_scale_global", &p, sizeof(p),
                             input_spec->data(), output_spec->data());
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ScaleOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  } else if (out_type.isBF16() || out_type.isF16()) {
    return out_lmem_bytes;
  }
  return 0;
}

void tpu::ScaleOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                   void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
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

void tpu::ScaleOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                         void *sec_info_) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  auto gi = getGroupInfo(n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  scale_local_spec_t p{0};

  p.if_relu = getDoRelu();
  p.relu_upper_limit = getReluLimitAttr().getValueAsDouble();
  p.is_scale_coeff = isa_and_nonnull<tpu::LoadOp>(getScale().getDefiningOp());
  p.is_bias_coeff = isa_and_nonnull<tpu::LoadOp>(getBias().getDefiningOp());
  p.input_num = input_spec->size();
  p.merge_weight_bias = 0;
  if (module::isUniformQuantized(getInput())) {
    p.buffer_local_addr = gi.buffer_addr;
    p.is_shift_coeff = 1;
    p.round_mode = ROUND_UP;
    p.version = 10; // 1684x:10 1684:0
    p.bias_dtype = BM168x::getDataType(getBias());
  } else {
    for (int i = 0; i < 4; ++i) {
      p.scale_shape[i] = 1;
    }
    auto shape = module::getShape(getScale());
    for (auto v : llvm::enumerate(shape)) {
      p.scale_shape[v.index()] = v.value();
    }
  }
  BM168x::call_local_func("backend_api_scale_local", &p, sizeof(p), sec_info_,
                          input_spec->data(), output_spec->data());
}

//dynamic codegen
int64_t tpu::ScaleOp::dyn_codegen_local_bm1684x(void *buffer) {
return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ScaleOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}
