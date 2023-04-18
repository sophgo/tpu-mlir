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
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684x;

int64_t data_copy(top::WeightOp weight, bool is_f16, std::shared_ptr<std::vector<short>> &data_i16, int64_t offset) {
  auto data_fp32 = weight.read<float>();
  auto count = data_fp32->size();
  for (uint32_t i = 0; i < count; i++) {
    data_i16->at(offset + i) = is_f16 ? f32_to_f16(data_fp32->at(i))
                                      : f32_to_bf16(data_fp32->at(i));
  }
  return offset + count;
}

static LogicalResult
canonicalize_transformer_operand_weight(tpu::TransformerOp op, PatternRewriter &rewriter) {
  auto stype = module::getStorageType(op.getInput());
  bool is_f16 = stype.isF16();
  auto none_op = module::getNoneOp(op);

  auto q_shape = module::getShape(op.getQueriesWeight());
  auto k_shape = module::getShape(op.getKeysWeight());
  auto o_shape = module::getShape(op.getOutWeight());
  auto q_w = op.getQueriesWeight().getDefiningOp<top::WeightOp>();
  auto k_w = op.getKeysWeight().getDefiningOp<top::WeightOp>();
  auto v_w = op.getValuesWeight().getDefiningOp<top::WeightOp>();
  int64_t bias_len = module::isNone(op.getQueriesBias()) ? 0 : 1;
  bias_len += module::isNone(op.getKeysBias()) ? 0 : 1;
  bias_len += module::isNone(op.getValuesBias()) ? 0 : 1;
  int64_t weight_h = (q_shape[0] + k_shape[0] * 2 + bias_len);
  auto data_i16 = std::make_shared<std::vector<short>>(weight_h * q_shape[1]);

  int offset = data_copy(q_w, is_f16, data_i16, 0);
  offset = data_copy(k_w, is_f16, data_i16, offset);
  offset = data_copy(v_w, is_f16, data_i16, offset);
  if (module::isNone(op.getQueriesBias())) {
    auto q_b = op.getQueriesBias().getDefiningOp<top::WeightOp>();
    offset = data_copy(q_b, is_f16, data_i16, offset);
  }
  if (module::isNone(op.getKeysBias())) {
    auto k_b = op.getKeysBias().getDefiningOp<top::WeightOp>();
    offset = data_copy(k_b, is_f16, data_i16, offset);
  }
  if (module::isNone(op.getValuesBias())) {
    auto v_b = op.getValuesBias().getDefiningOp<top::WeightOp>();
    offset = data_copy(v_b, is_f16, data_i16, offset);
  }
  std::vector<int64_t> weight_shape = {weight_h, q_shape[1]};
  auto new_type = RankedTensorType::get(weight_shape, stype);
  auto new_op =
          top::WeightOp::create(op, "filter_reorderd", *data_i16, new_type);
  op->setOperand(3, new_op);
  op->setOperand(4, none_op);
  op->setOperand(5, none_op);
  op->setOperand(6, none_op);
  op->setOperand(7, none_op);
  op->setOperand(8, none_op);

  if (module::isNone(op.getOutBias())) {
    auto o_data_i16 = std::make_shared<std::vector<short>>((o_shape[0] + 1) * o_shape[1]);
    auto o_w = op.getOutWeight().getDefiningOp<top::WeightOp>();
    offset = data_copy(o_w, is_f16, data_i16, 0);
    auto o_b = op.getOutBias().getDefiningOp<top::WeightOp>();
    offset = data_copy(o_b, is_f16, data_i16, offset);
    std::vector<int64_t> weight_shape = {(o_shape[0] + 1), o_shape[1]};
    auto new_type = RankedTensorType::get(weight_shape, stype);
    auto new_op =
            top::WeightOp::create(op, "filter_o_reorderd", *data_i16, new_type);
    op->setOperand(9, new_op);
    op->setOperand(10, none_op);
  }

  return success();
}

template <>
LogicalResult WeightReorder<tpu::TransformerOp, BFloat16Type>::matchAndRewrite(
    tpu::TransformerOp op, PatternRewriter &rewriter) const {
  return canonicalize_transformer_operand_weight(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::TransformerOp, Float16Type>::matchAndRewrite(
    tpu::TransformerOp op, PatternRewriter &rewriter) const {
  return canonicalize_transformer_operand_weight(op, rewriter);
}

// void get_param(tpu::TransformerOp op, attention_common_spec_t spec) {
//   spec.hasbias = module::isNone(op.getQueriesBias()) ? 0 : 1;
//   spec.hasbias &= module::isNone(op.getKeysBias()) ? 0 : 0x01<<1;
//   spec.hasbias &= module::isNone(op.getValuesBias()) ? 0 : 0x01<<2;
//   spec.hasbias &= module::isNone(op.getOutBias()) ? 0 : 0x01<<3;
//   spec.head = op.getHead();
//   spec.scale = op.getScale().convertToDouble();
//   spec.hasmusk = !module::isNone(op.getMusk());
// }

// =========================================
// GlobalGenInterface
// =========================================
void tpu::TransformerOp::codegen_global_bm1684x() {
  // llvm_unreachable("Not Implemented");
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  attention_global_spec_t param = {0};
  auto &common = param.common;
  // get_param(op, common);
  common.hasbias = module::isNone(getQueriesBias()) ? 0 : 1;
  common.hasbias &= module::isNone(getKeysBias()) ? 0 : 0x01<<1;
  common.hasbias &= module::isNone(getValuesBias()) ? 0 : 0x01<<2;
  common.hasbias &= module::isNone(getOutBias()) ? 0 : 0x01<<3;
  common.head = getHead();
  common.scale = getScale().convertToDouble();
  common.hasmusk = !module::isNone(getMusk());

  BM168x::call_global_func("backend_api_attention_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::TransformerOp::dyn_codegen_global_bm1684x(void *buffer) {
  llvm_unreachable("Not Implemented");
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::TransformerOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  int64_t batch, M_q, N_q, W;
  module::getNCHW(getInput(), batch, M_q, N_q, W, group_type);
  int64_t M_k = module::getShape(getKeys())[1];
  auto head = getHead();
  int64_t d = N_q / head;
  int64_t M_v = module::getShape(getValues())[1];
  auto out_type = module::getStorageType(getOutput());

  int64_t buffer_size = 0;
  int c_per_npu = ceiling_func(M_q, BM168x::NPU_NUM);
  auto eu_num_f32 = BM168x::eu_num(sizeof(float));
  auto eu_num_f16 = BM168x::eu_num(sizeof(short));
  int64_t softmax_buffer_size = c_per_npu * eu_num_f32 * sizeof(float);
  // 32 coeff and 192 table
  softmax_buffer_size += align_up((int64_t)32, eu_num_f32) * sizeof(float);
  softmax_buffer_size += align_up((int64_t)192, eu_num_f32) * sizeof(float);
  softmax_buffer_size += c_per_npu * align_up(M_k, eu_num_f32) * sizeof(float) * 2;

  int64_t d_size = 0;
  if (out_type.isBF16() || out_type.isF16()) {
    d_size = align_up(d, eu_num_f16) * sizeof(short);
  } else {
    d_size = align_up(d, eu_num_f32) * sizeof(float);
  }
  int64_t buffer_in_size = d_size * (c_per_npu + ceiling_func(M_k, BM168x::NPU_NUM));
  int64_t mat1_buffer_size = d_size * (c_per_npu + ceiling_func(M_v, BM168x::NPU_NUM));

  buffer_size = std::max(std::max(softmax_buffer_size, buffer_in_size), mat1_buffer_size);
  buffer_size += c_per_npu * align_up(M_k, eu_num_f32) * sizeof(float);

  return buffer_size;
}

void tpu::TransformerOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                               group_type_t group_type,
                                               local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  const auto &gi = getGroupInfo(n_step, h_step, d_step, w_step);

  attention_local_spec_t param = {0};
  auto &common = param.common;
  param.buffer_addr = gi.buffer_addr;
  // get_param(op, common);
  common.hasbias = module::isNone(getQueriesBias()) ? 0 : 1;
  common.hasbias &= module::isNone(getKeysBias()) ? 0 : 0x01<<1;
  common.hasbias &= module::isNone(getValuesBias()) ? 0 : 0x01<<2;
  common.hasbias &= module::isNone(getOutBias()) ? 0 : 0x01<<3;
  common.head = getHead();
  common.scale = getScale().convertToDouble();
  common.hasmusk = !module::isNone(getMusk());

  BM168x::call_local_func("backend_api_attention_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

int64_t tpu::TransformerOp::dyn_codegen_local_bm1684x(void *buffer) {
  llvm_unreachable("Not Implemented");
}

int64_t tpu::TransformerOp::get_fw_type_bm1684x() {
  return -1;
}
