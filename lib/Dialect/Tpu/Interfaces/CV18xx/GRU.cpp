//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/WeightReorder.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"



using namespace tpu_mlir::backend;
using namespace tpu_mlir::cv18xx;

// ======================================
// WeightReorderInterface
// ======================================

template <typename T> static void transpose_row_col(T *data, int row, int col) {
  std::vector<T> w_t(row * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      w_t[j * row + i] = data[i * col + j];
    }
  }
  std::copy(w_t.begin(), w_t.end(), data);
}

Value lowerWeight(Operation *op, Value w_value) {
  std::vector<int64_t> w_shape;
  int64_t elem_num = 0;
  module::getShapeVec(w_value, w_shape);
  elem_num = module::getNumElements(w_value);

  auto w_op = dyn_cast<top::WeightOp>(w_value.getDefiningOp());
  auto w_data_bf16 = w_op.read<uint16_t>();
  auto w_name = module::getName(w_value).str();
  // transpose weight
  assert(w_shape.size() == 3);
  int64_t num_dir = w_shape[0];
  int64_t hidden_size = w_shape[2];
  assert(w_shape[1] % hidden_size == 0);
  int gate_num = w_shape[1] / hidden_size;
  uint16_t *p_data = w_data_bf16->data();
  for (int64_t i = 0; i < gate_num * num_dir; ++i) {
    transpose_row_col(p_data, hidden_size, hidden_size);
    p_data += hidden_size * hidden_size;
  }
  auto ctx = w_op.getContext();
  auto lowered_weight_type =
      RankedTensorType::get(w_shape, FloatType::getBF16(ctx));
  auto lowered_weight_operand = top::WeightOp::create(
      op, w_name + "_transpose_bf16", *w_data_bf16, lowered_weight_type);
  return lowered_weight_operand;
}
// for bf16 bias
static void
transposeBiasFp32(const std::shared_ptr<std::vector<float>> &bias_f32,
                  std::vector<uint32_t> &bias_u32) {
  // Split into high/low part
  std::vector<uint16_t> bias_fp32_high;
  std::vector<uint16_t> bias_fp32_low;
  float *biasFloatPtr = bias_f32->data();
  int size = bias_f32->size();
  for (int i = 0; i < size; ++i) {
    unsigned short *temp_short_ptr =
        reinterpret_cast<unsigned short *>(biasFloatPtr + i);
    bias_fp32_high.push_back(temp_short_ptr[1]);
    bias_fp32_low.push_back(temp_short_ptr[0]);
  }
  std::vector<uint16_t> bias_reshape_fp32;
  bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_high.begin(),
                           bias_fp32_high.end());
  bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(),
                           bias_fp32_low.end());
  // then copy into uint32_t
  assert(bias_u32.size() == bias_f32->size());
  memcpy(bias_u32.data(), bias_reshape_fp32.data(), size * sizeof(uint32_t));
}

Value lowerBias(Operation *op, Value b_value) {
  std::vector<int64_t> b_shape;
  int64_t elem_num = 0;
  module::getShapeVec(b_value, b_shape);
  elem_num = module::getNumElements(b_value);

  auto b_op = dyn_cast<top::WeightOp>(b_value.getDefiningOp());
  auto b_data_f32 = b_op.read<float>();
  auto b_name = module::getName(b_value).str();
  std::vector<uint32_t> b_data_uint32(elem_num);
  transposeBiasFp32(b_data_f32, b_data_uint32);

  auto ctx = b_op.getContext();
  auto lowered_bias_type =
      RankedTensorType::get(b_shape, IntegerType::get(ctx, 32));
  auto lowered_bias_operand = top::WeightOp::create(
      op, b_name + "_lower_uint32", b_data_uint32, lowered_bias_type);
  return lowered_bias_operand;
}

// common for weight
template <>
LogicalResult WeightReorder<tpu::GRUOp, BFloat16Type>::matchAndRewrite(
    tpu::GRUOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.input()).isBF16())
    return failure();
  auto lower_recurrence = lowerWeight(op.getOperation(), op.recurrence());
  auto lower_bias = lowerBias(op.getOperation(), op.bias());
  op.setOperand(2, lower_recurrence);
  op.setOperand(3, lower_bias);
  return success();
}

template <>
LogicalResult WeightReorder<tpu::GRUOp, int8_t>::matchAndRewrite(
    tpu::GRUOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.input()).isBF16())
    return failure();
  auto lower_recurrence = lowerWeight(op.getOperation(), op.recurrence());
  auto lower_bias = lowerBias(op.getOperation(), op.bias());
  op.setOperand(2, lower_recurrence);
  op.setOperand(3, lower_bias);
  return success();
}

// =========================================
// GlobalGenInterface
// =========================================
void tpu::GRUOp::codegen_global_cv18xx(int64_t layer_id) {
  bool only_last = false;
  int64_t seq_len, batch_size, input_size, garbage;
  int64_t seq_len2, num_dir, batch_size2, hidden_size;
  auto in_shape = module::getShape(input());
  module::getNCHW(in_shape, seq_len, batch_size, input_size, garbage);
  Value output;
  if (!getResults()[0].getType().isa<mlir::NoneType>()) {
    output = getResults()[0];
  } else {
    output = getResults()[1];
  }
  auto out_shape = module::getShape(output);
  if (out_shape.size() == 4) {
    module::getNCHW(out_shape, seq_len2, num_dir, batch_size2, hidden_size);
    assert(seq_len == seq_len2);
  } else {
    module::getNCHW(out_shape, num_dir, batch_size2, hidden_size, garbage);
    only_last = true;
  }

  assert(batch_size == batch_size2);
  assert(input_size == num_dir * 3 * hidden_size);

  bool with_bias = !bias().getType().isa<mlir::NoneType>();
  bool with_h0 = !initial_h().getType().isa<mlir::NoneType>();
  gaddr_t ga_input = module::getAddress(input());
  gaddr_t ga_output = module::getAddress(output);
  gaddr_t ga_bias = with_bias ? module::getAddress(bias()) : 0;
  gaddr_t ga_initial_h = with_h0 ? module::getAddress(initial_h()) : 0;
  gaddr_t ga_recurrence = module::getAddress(recurrence());
  gaddr_t ga_sigmoid_table = module::getAddress(sigmoid_table());
  gaddr_t ga_sigmoid_slope = module::getAddress(sigmoid_slope_table());
  gaddr_t ga_tanh_table = module::getAddress(tanh_table());
  gaddr_t ga_tanh_slope = module::getAddress(tanh_slope_table());

  bool is_linear_before_reset = linear_before_reset();
  bool is_bidirectional = bidirectional();

  if (module::isUniformQuantized(output)) {
    llvm_unreachable("Not supported now");
  } else {
    cvi_backend_tg_bf16_gru_kernel(
        layer_id, ga_input, ga_recurrence, ga_bias, ga_initial_h,
        ga_sigmoid_table, ga_sigmoid_slope, ga_tanh_table, ga_tanh_slope,
        ga_output, seq_len, num_dir, batch_size, hidden_size, with_bias,
        with_h0, is_linear_before_reset, is_bidirectional, only_last);
  }
}
