//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"

using namespace cv18xx;

// ======================================
// WeightReorderInterface
// ======================================

template <typename T>
static void transpose_row_col(T *data, int row, int col) {
  std::vector<T> w_t(row * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      w_t[j * row + i] = data[i * col + j];
    }
  }
  std::copy(w_t.begin(), w_t.end(), data);
}

Value lowerWeight(Operation *op, Value w_value) {
  int64_t elem_num = 0;
  auto w_shape = module::getShape(w_value);
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
  auto b_shape = module::getShape(b_value);
  auto elem_num = module::getNumElements(b_value);

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
LogicalResult WeightReorder<tpu::GRUOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::GRUOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getInput()).isBF16())
    return failure();
  auto lower_recurrence = lowerWeight(op.getOperation(), op.getRecurrence());
  auto lower_bias = lowerBias(op.getOperation(), op.getBias());
  op.setOperand(2, lower_recurrence);
  op.setOperand(3, lower_bias);
  return success();
}

template <>
LogicalResult WeightReorder<tpu::GRUOp, int8_t>::matchAndRewriteImpl(
    tpu::GRUOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getInput()).isBF16())
    return failure();
  auto lower_recurrence = lowerWeight(op.getOperation(), op.getRecurrence());
  auto lower_bias = lowerBias(op.getOperation(), op.getBias());
  op.setOperand(2, lower_recurrence);
  op.setOperand(3, lower_bias);
  return success();
}
