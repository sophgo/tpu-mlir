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

void tpu::MulOp::codegen_global_bm1684x() {
  bcbinary_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_type = BINARY_MUL;
  spec.if_relu = getDoRelu();
  spec.relu_upper_limit = getReluLimit().convertToDouble();
  spec.rshift_A = getRshift();
  spec.rshift_B = 0;
  spec.scale_A = getMultiplier();
  spec.scale_B = 1;
  auto scales = module::getF64Array(getOutF8Scales(), 1, 1.0);
  spec.f8_scale_A = scales->at(0);
  spec.f8_scale_B = 1.0;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (module::isUniformQuantized(getInputs()[0])) {
    spec.izp_A = module::getUniformQuantizedType(getInputs()[0]).getZeroPoint();
  }
  if (module::isUniformQuantized(getInputs()[1])) {
    spec.izp_B = module::getUniformQuantizedType(getInputs()[1]).getZeroPoint();
  }
  if (module::isUniformQuantized(getOutput())) {
    spec.ozp = module::getUniformQuantizedType(getOutput()).getZeroPoint();
  }
  BM168x::call_global_func("backend_api_bcbinary_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

static bool is_sign(DATA_TYPE_T dtype) {
  return !(dtype == DTYPE_UINT8 || dtype == DTYPE_UINT16 ||
           dtype == DTYPE_UINT32);
}

int64_t tpu::MulOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t buffer_size = 0;
  auto dtype_A = BM168x::getDataType(getInputs()[0]);
  auto dtype_B = BM168x::getDataType(getInputs()[1]);
  auto dtype_O = BM168x::getDataType(getOutput());
  if (dtype_A == DTYPE_INT8 || dtype_A == DTYPE_UINT8) {
    buffer_size = out_lmem_bytes * 2;
    // aligned with backend
    if (getMultiplier() != 1 || getRshift() != 0) {
      buffer_size += out_lmem_bytes * 2;
    }
  } else if (dtype_A == DTYPE_F8E4M3) {
    // calc method keep the same as add/sub at backend
    if (dtype_B == DTYPE_F8E4M3) {
      buffer_size = 3 * out_lmem_bytes * sizeof(int16_t);
    } else if (dtype_B == DTYPE_FP16) {
      buffer_size = 2 * out_lmem_bytes * sizeof(int16_t);
    }
  } else if ((BM168x::getFmtBytes(dtype_A) > BM168x::getFmtBytes(dtype_O)) &&
             (is_sign(dtype_A) || is_sign(dtype_B)) && (!is_sign(dtype_O))) {
    buffer_size = out_lmem_bytes;
  }
  return buffer_size;
}

void tpu::MulOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                       int64_t h_step, int64_t d_step,
                                       int64_t w_step, group_type_t group_type,
                                       local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BINARY_MUL;
  param.spec.common.if_relu = getDoRelu();
  param.spec.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.spec.common.rshift_A = getRshift();
  param.spec.common.rshift_B = 0;
  param.spec.common.scale_A = getMultiplier();
  param.spec.common.scale_B = 1;
  auto scales = module::getF64Array(getOutF8Scales(), 1, 1.0);
  param.spec.common.f8_scale_A = scales->at(0);
  param.spec.common.f8_scale_B = 1.0;

  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;
  if (module::isUniformQuantized(getInputs()[0])) {
    param.spec.common.izp_A =
        module::getUniformQuantizedType(getInputs()[0]).getZeroPoint();
  }
  if (module::isUniformQuantized(getInputs()[1])) {
    param.spec.common.izp_B =
        module::getUniformQuantizedType(getInputs()[1]).getZeroPoint();
  }
  if (module::isUniformQuantized(getOutput())) {
    param.spec.common.ozp =
        module::getUniformQuantizedType(getOutput()).getZeroPoint();
  }

  BM168x::call_local_func("backend_api_bcbinary_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::MulOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(bcbinary_local_param_t);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BINARY_MUL;
  param.spec.common.if_relu = getDoRelu();
  param.spec.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.spec.common.rshift_A = getRshift();
  param.spec.common.rshift_B = 0;
  param.spec.common.scale_A = getMultiplier();
  param.spec.common.scale_B = 1;
  auto scales = module::getF64Array(getOutF8Scales(), 1, 1.0);
  param.spec.common.f8_scale_A = scales->at(0);
  param.spec.common.f8_scale_B = 1.0;
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;
  if (module::isUniformQuantized(getInputs()[0])) {
    param.spec.common.izp_A =
        module::getUniformQuantizedType(getInputs()[0]).getZeroPoint();
  }
  if (module::isUniformQuantized(getInputs()[1])) {
    param.spec.common.izp_B =
        module::getUniformQuantizedType(getInputs()[1]).getZeroPoint();
  }
  if (module::isUniformQuantized(getOutput())) {
    param.spec.common.ozp =
        module::getUniformQuantizedType(getOutput()).getZeroPoint();
  }

  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MulOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(bcbinary_global_param_t);
  bcbinary_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_type = BINARY_MUL;
  spec.if_relu = getDoRelu();
  spec.relu_upper_limit = getReluLimit().convertToDouble();
  spec.rshift_A = getRshift();
  spec.rshift_B = 0;
  spec.scale_A = getMultiplier();
  spec.scale_B = 1;
  auto scales = module::getF64Array(getOutF8Scales(), 1, 1.0);
  spec.f8_scale_A = scales->at(0);
  spec.f8_scale_B = 1.0;
  if (module::isUniformQuantized(getInputs()[0])) {
    spec.izp_A = module::getUniformQuantizedType(getInputs()[0]).getZeroPoint();
  }
  if (module::isUniformQuantized(getInputs()[1])) {
    spec.izp_B = module::getUniformQuantizedType(getInputs()[1]).getZeroPoint();
  }
  if (module::isUniformQuantized(getOutput())) {
    spec.ozp = module::getUniformQuantizedType(getOutput()).getZeroPoint();
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::MulOp::get_fw_type_bm1684x() { return FW_BMNET_BROADCAST_BINARY; }
