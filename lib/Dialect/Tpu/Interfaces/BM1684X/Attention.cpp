//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

// void get_param(tpu::AttentionOp op, attention_common_spec_t spec) {
//   spec.hasbias = module::isNone(op.getQueriesBias()) ? 0 : 1;
//   spec.hasbias &= module::isNone(op.getKeysBias()) ? 0 : 0x01<<1;
//   spec.hasbias &= module::isNone(op.getValuesBias()) ? 0 : 0x01<<2;
//   spec.hasbias &= module::isNone(op.getOutBias()) ? 0 : 0x01<<3;
//   spec.head = op.getHead();
//   spec.scale = op.getScale().convertToDouble();
//   spec.hasmask = !module::isNone(op.getMask());
// }

// =========================================
// GlobalGenInterface
// =========================================
void tpu::AttentionOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  attention_global_spec_t param = {0};
  auto &common = param.common;
  // get_param(op, common);
  common.hasbias = getHasBias();
  common.head = getHead();
  common.dim = getDim();
  common.weight_reshape = true;
  common.scale = getScale().convertToDouble();
  common.hasmask = !module::isNone(getMask());
  common.input_num =
      module::isNone(getKeys()) ? 1 : (module::isNone(getValues()) ? 2 : 3);
  auto quant_param = module::getI64Array(getQuantParam());
  for (int i = 0; i < quant_param->size(); ++i) {
    common.quant_param[i] = quant_param->at(i);
  }

  BM168x::call_global_func("backend_api_attention_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::AttentionOp::dyn_codegen_global_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::AttentionOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t batch, M_q, N_q, W;
  module::getNCHW(getInput(), batch, M_q, N_q, W, group_type);
  M_q = in_cslice;
  int64_t M_k =
      module::isNone(getKeys()) ? M_q : module::getShape(getKeys())[1];
  int64_t d = getDim();
  int64_t M_v = M_k;
  auto out_type = module::getStorageType(getOutput());

  int64_t buffer_size = 0;
  int c_per_npu = ceiling_func(M_q, BM168x::NPU_NUM);
  if (out_type.isInteger(32)) {
    auto eu_num_i32 = BM168x::eu_num(sizeof(int));
    auto eu_num_i8 = BM168x::eu_num(sizeof(int8_t));
    int64_t d_size = align_up(d, eu_num_i8) * sizeof(int8_t);

    int64_t k_buffer_size = d_size * ceiling_func(M_k, BM168x::NPU_NUM);
    int64_t v_buffer_size = d_size * ceiling_func(M_v, BM168x::NPU_NUM);
    int64_t mat0_size = c_per_npu * align_up(M_k, eu_num_i8) * sizeof(int8_t);
    int64_t mat1_size = c_per_npu * d_size;
    int64_t softmax_buffer_size =
        c_per_npu * sizeof(int32_t) *
        (align_up(1, eu_num_i32) + align_up(M_k, eu_num_i32));
    int64_t max_m = std::max(M_q, M_k);
    int64_t mat_buffer_size0 =
        ceiling_func(max_m, BM168x::NPU_NUM) * align_up(d, eu_num_i32);
    int64_t mat_buffer_size1 =
        ceiling_func(M_q, BM168x::NPU_NUM) * align_up(M_k, eu_num_i32);
    int64_t mat_buffer_size = std::max(mat_buffer_size0, mat_buffer_size1);
    int64_t const_size =
        align_up(mat0_size + mat1_size + k_buffer_size + v_buffer_size,
                 BM168x::LMEM_BANK_BYTES);
    int64_t imm_buffer_size =
        align_up(std::max(mat_buffer_size, softmax_buffer_size),
                 BM168x::LMEM_BANK_BYTES);
    return std::min(imm_buffer_size + const_size,
                    BM168x::LMEM_BANK_BYTES * (BM168x::LMEM_BANKS / 3));
  }
  auto eu_num_f32 = BM168x::eu_num(sizeof(float));
  auto eu_num_f16 = BM168x::eu_num(sizeof(short));
  int64_t softmax_buffer_size = c_per_npu * eu_num_f32 * sizeof(float);
  // 32 coeff and 192 table
  softmax_buffer_size += align_up((int64_t)32, eu_num_f32) * sizeof(float);
  softmax_buffer_size += align_up((int64_t)192, eu_num_f32) * sizeof(float);
  softmax_buffer_size +=
      c_per_npu * align_up(M_k, eu_num_f32) * sizeof(float) * 4;

  int64_t d_size = 0;
  if (out_type.isBF16() || out_type.isF16()) {
    d_size = align_up(d, eu_num_f16) * sizeof(short);
  } else {
    d_size = align_up(d, eu_num_f32) * sizeof(float);
  }
  int64_t buffer_in_size = d_size * (ceiling_func(M_v, BM168x::NPU_NUM) +
                                     ceiling_func(M_k, BM168x::NPU_NUM));

  buffer_size =
      std::min(softmax_buffer_size + buffer_in_size, BM168x::LMEM_BYTES / 3);

  return buffer_size;
}

void tpu::AttentionOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                             int64_t h_step, int64_t d_step,
                                             int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  const auto &gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  attention_local_spec_t param = {0};
  auto &common = param.common;
  param.buffer_addr = gi.buffer_addr;
  // get_param(op, common);
  common.hasbias = getHasBias();
  common.head = getHead();
  common.dim = getDim();
  common.weight_reshape = true;
  common.scale = getScale().convertToDouble();
  common.hasmask = !module::isNone(getMask());
  common.input_num =
      module::isNone(getKeys()) ? 1 : (module::isNone(getValues()) ? 2 : 3);
  auto quant_param = module::getI64Array(getQuantParam());
  for (int i = 0; i < quant_param->size(); ++i) {
    common.quant_param[i] = quant_param->at(i);
  }

  BM168x::call_local_func("backend_api_attention_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

int64_t tpu::AttentionOp::dyn_codegen_local_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::AttentionOp::get_fw_type_bm1684x() { return -1; }
