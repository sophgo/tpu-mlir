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

template <typename T>
static void transposeRecurrence(std::vector<T> &w,
                                const std::vector<int64_t> &shape) {
  assert(shape.size() == 3);
  int64_t num_dir = shape[0];
  int64_t hidden_size = shape[2];
  assert(shape[1] % hidden_size == 0);
  int gate_num = shape[1] / hidden_size;
  T *p_w = w.data();
  for (int i = 0; i < gate_num * num_dir; i++) {
    transpose_row_col(p_w, hidden_size, hidden_size);
    p_w += hidden_size * hidden_size;
  }
}

static void transposeBiasFp32(std::vector<float> &bias_f32,
                              std::vector<uint32_t> &bias_u32) {
  // Split into high/low part
  std::vector<uint16_t> bias_fp32_high;
  std::vector<uint16_t> bias_fp32_low;
  float *biasFloatPtr = bias_f32.data();
  int size = bias_f32.size();
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
  assert(bias_u32.size() == bias_f32.size());
  memcpy(bias_u32.data(), bias_reshape_fp32.data(), size * sizeof(uint32_t));
}

// ======================================
// WeightReorderInterface
// ======================================
template <>
LogicalResult WeightReorder<tpu::LSTMOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::LSTMOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getInput()).isBF16())
    return failure();
  auto recc_shape = module::getShape(op.getRecurrence());
  auto reccOp = op.getRecurrence().getDefiningOp<top::WeightOp>();
  auto recc_u16 = reccOp.read<uint16_t>();
  transposeRecurrence(*recc_u16, recc_shape);
  reccOp.update(*recc_u16, recc_u16->size());
  if (auto biasOp = op.getBias().getDefiningOp<top::WeightOp>()) {
    auto bias_data = biasOp.read<float_t>();
    std::vector<uint32_t> bias_u32(bias_data->size());
    transposeBiasFp32(*bias_data, bias_u32);
    biasOp.update(bias_u32, bias_u32.size());
    auto new_bias_type = RankedTensorType::get(module::getShape(op.getBias()),
                                               rewriter.getIntegerType(32));
    op.getBias().setType(new_bias_type);
  }
  return success();
}
