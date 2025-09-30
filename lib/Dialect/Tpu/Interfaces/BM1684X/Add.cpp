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
void tpu::AddOp::codegen_global_bm1684x() {
  std::vector<int64_t> multi_v;
  std::vector<int64_t> rshift_v;
  std::vector<float> f8_scales;

  bcbinary_global_param_t param{0};
  auto &spec = param.spec;

  if (module::isUniformQuantized(getInputs()[0], getOutput()) ||
      (module::isUniformQuantized(getInputs()[1], getOutput()))) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
    if (module::isUniformQuantized(getInputs()[0])) {
      spec.izp_A =
          module::getUniformQuantizedType(getInputs()[0]).getZeroPoint();
    }
    if (module::isUniformQuantized(getInputs()[1])) {
      spec.izp_B =
          module::getUniformQuantizedType(getInputs()[1]).getZeroPoint();
    }
    spec.ozp = module::getUniformQuantizedType(getOutput()).getZeroPoint();
  } else if (module::getStorageType(getOutput()).isFloat8E4M3FN() ||
             module::getStorageType(getOutput()).isFloat8E5M2()) {
    auto scales = module::getF64Array(getF8Scales(), 2, 1.0);
    for (auto scale : *scales)
      f8_scales.push_back((float)scale);
  }

  spec.binary_type = BINARY_ADD;
  spec.if_relu = getDoRelu();
  spec.relu_upper_limit = getReluLimit().convertToDouble();
  spec.rshift_A = rshift_v.empty() ? 0 : rshift_v[0];
  spec.rshift_B = rshift_v.empty() ? 0 : rshift_v[1];
  spec.scale_A = multi_v.empty() ? 1 : multi_v[0];
  spec.scale_B = multi_v.empty() ? 1 : multi_v[1];
  spec.f8_scale_A = f8_scales.empty() ? 1 : f8_scales[0];
  spec.f8_scale_B = f8_scales.empty() ? 1 : f8_scales[1];

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_bcbinary_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::AddOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto in0_type = module::getStorageType(getInputs()[0]);
  auto in1_type = module::getStorageType(getInputs()[1]);
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  } else if (in0_type.isFloat8E4M3FN() && in1_type.isFloat8E4M3FN()) {
    return 3 * out_lmem_bytes * sizeof(int16_t);
  } else if (in0_type.isFloat8E4M3FN() && in1_type.isF16()) {
    return 2 * out_lmem_bytes * sizeof(int16_t);
  }
  return 0;
}

void tpu::AddOp::codegen_local_bm1684x_kernel(
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos, local_sec_info_t &sec_info,
    std::shared_ptr<std::vector<tensor_spec_t>> input_spec,
    std::shared_ptr<std::vector<tensor_spec_t>> output_spec) {
  auto gi = out_group_infos[0];
  bcbinary_local_param_t param = {0};

  std::vector<int64_t> multi_v;
  std::vector<int64_t> rshift_v;
  std::vector<float> f8_scales;
  if (module::isUniformQuantized(getInputs()[0], getOutput()) ||
      (module::isUniformQuantized(getInputs()[1], getOutput()))) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
    if (module::isUniformQuantized(getInputs()[0])) {
      param.spec.common.izp_A =
          module::getUniformQuantizedType(getInputs()[0]).getZeroPoint();
    }
    if (module::isUniformQuantized(getInputs()[1])) {
      param.spec.common.izp_B =
          module::getUniformQuantizedType(getInputs()[1]).getZeroPoint();
    }
    param.spec.common.ozp =
        module::getUniformQuantizedType(getOutput()).getZeroPoint();
  } else if (module::getStorageType(getOutput()).isFloat8E4M3FN() ||
             module::getStorageType(getOutput()).isFloat8E5M2()) {
    auto scales = module::getF64Array(getF8Scales(), 2, 1.0);
    for (auto scale : *scales)
      f8_scales.push_back((float)scale);
  }

  param.spec.common.binary_type = BINARY_ADD;
  param.spec.common.if_relu = getDoRelu();
  param.spec.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.spec.common.rshift_A = rshift_v.empty() ? 0 : rshift_v[0];
  param.spec.common.rshift_B = rshift_v.empty() ? 0 : rshift_v[1];
  param.spec.common.scale_A = multi_v.empty() ? 1 : multi_v[0];
  param.spec.common.scale_B = multi_v.empty() ? 1 : multi_v[1];
  param.spec.common.f8_scale_A = f8_scales.empty() ? 1 : f8_scales[0];
  param.spec.common.f8_scale_B = f8_scales.empty() ? 1 : f8_scales[1];
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;
  BM168x::call_local_func("backend_api_bcbinary_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::AddOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(bcbinary_local_param_t);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  std::vector<int64_t> multi_v;
  std::vector<int64_t> rshift_v;
  std::vector<float> f8_scales;

  bcbinary_local_param_t param;
  memset(&param, 0, sizeof(param));
  if (module::isUniformQuantized(getInputs()[0], getOutput()) ||
      (module::isWeight(getInputs()[0]) &&
       module::isUniformQuantized(getInputs()[1], getOutput()))) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
    if (module::isUniformQuantized(getInputs()[0])) {
      param.spec.common.izp_A =
          module::getUniformQuantizedType(getInputs()[0]).getZeroPoint();
    }
    if (module::isUniformQuantized(getInputs()[1])) {
      param.spec.common.izp_B =
          module::getUniformQuantizedType(getInputs()[1]).getZeroPoint();
    }
    param.spec.common.ozp =
        module::getUniformQuantizedType(getOutput()).getZeroPoint();
  } else if (module::getStorageType(getOutput()).isFloat8E4M3FN() ||
             module::getStorageType(getOutput()).isFloat8E5M2()) {
    auto scales = module::getF64Array(getF8Scales(), 2, 1.0);
    for (auto scale : *scales)
      f8_scales.push_back((float)scale);
  }

  param.spec.common.binary_type = BINARY_ADD;
  param.spec.common.if_relu = getDoRelu();
  param.spec.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.spec.common.rshift_A = rshift_v.empty() ? 0 : rshift_v[0];
  param.spec.common.rshift_B = rshift_v.empty() ? 0 : rshift_v[1];
  param.spec.common.scale_A = multi_v.empty() ? 1 : multi_v[0];
  param.spec.common.scale_B = multi_v.empty() ? 1 : multi_v[1];
  param.spec.common.f8_scale_A = f8_scales.empty() ? 1 : f8_scales[0];
  param.spec.common.f8_scale_B = f8_scales.empty() ? 1 : f8_scales[1];

  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::AddOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(bcbinary_global_param_t);
  }
  bcbinary_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_type = BINARY_ADD;
  spec.if_relu = getDoRelu();
  spec.relu_upper_limit = getReluLimit().convertToDouble();
  auto m_v = module::getI64Array(getMultipliers(), 2, 1);
  auto r_v = module::getI64Array(getRshifts(), 2, 0);
  auto f8_scales = module::getF64Array(getF8Scales(), 2, 1.0);
  spec.rshift_A = r_v->at(0);
  spec.rshift_B = r_v->at(1);
  spec.scale_A = m_v->at(0);
  spec.scale_B = m_v->at(1);

  if (module::isUniformQuantized(getInputs()[0])) {
    spec.izp_A = module::getUniformQuantizedType(getInputs()[0]).getZeroPoint();
  }
  if (module::isUniformQuantized(getInputs()[1])) {
    spec.izp_B = module::getUniformQuantizedType(getInputs()[1]).getZeroPoint();
  }
  if (module::isUniformQuantized(getOutput())) {
    spec.ozp = module::getUniformQuantizedType(getOutput()).getZeroPoint();
  }
  spec.f8_scale_A = f8_scales->at(0);
  spec.f8_scale_B = f8_scales->at(1);
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::AddOp::get_fw_type_bm1684x() { return FW_BMNET_BROADCAST_BINARY; }
