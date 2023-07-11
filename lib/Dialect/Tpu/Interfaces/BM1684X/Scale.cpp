//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

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
    p.scale_sign = module::isSign(getScale());
    p.bias_sign = module::isSign(getBias());
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
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  } else if (out_type.isBF16() || out_type.isF16()) {
    return out_lmem_bytes;
  }
  return 0;
}

void tpu::ScaleOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                         int64_t h_step, int64_t d_step,
                                         int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);

  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
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
  BM168x::call_local_func("backend_api_scale_local", &p, sizeof(p), &sec_info,
                          input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::ScaleOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(scale_local_spec_t);
  group_type_t group_type = GROUP_UNSUPPORT;
  auto op = getOperation();
  if (auto gOp = dyn_cast<GroupOp>(op->getParentOp())) {
    group_type = static_cast<group_type_t>(gOp.getGroupType());
  }
  assert(group_type < GROUP_UNSUPPORT);
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
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
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ScaleOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(scale_global_spec_t);
  scale_global_spec_t p = {0};
  p.axis = 1;
  p.axis_num = 1;
  p.has_bias = true;
  p.if_relu = getDoRelu();
  p.relu_upper_limit = getReluLimit().convertToDouble();
  p.merge_weight_bias = 0;
  p.round_mode = ROUND_UP;
  if (module::isUniformQuantized(getInput())) {
    p.scale_sign = module::isSign(getScale());
    p.bias_sign = module::isSign(getBias());
    p.version = 10;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

int64_t tpu::ScaleOp::get_fw_type_bm1684x() { return FW_BMNET_SCALE; }
