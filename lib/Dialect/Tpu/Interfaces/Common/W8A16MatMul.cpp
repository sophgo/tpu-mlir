//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

// function to compute per-channel scale
std::shared_ptr<std::vector<float>>
computePerChannelScale(const std::shared_ptr<std::vector<float>> &weight_data,
                       int row, int col, bool sign) {
  auto per_channel_scale = std::make_shared<std::vector<float>>(row);

  for (int c = 0; c < row; c++) {
    float *p_weight = weight_data->data() + c * col;
    auto w_max = findMaxabs(p_weight, col);
    per_channel_scale->at(c) = w_max / (sign ? 127 : 255);
  }
  return per_channel_scale;
}

LogicalResult tpu::W8A16MatMulOp::init(InferenceParameter &p) {
  // (MxK), (KxN) matmul
  auto matmul = new MatMul();
  auto in_shape = getInput().getType().cast<RankedTensorType>().getShape();
  int64_t M = 1;
  for (int i = 0; i < in_shape.size() - 1; i++) {
    M *= in_shape[i];
  }
  auto weight_value = getWeight();
  auto weight_shape =
      weight_value.getType().cast<RankedTensorType>().getShape();

  auto w_transpose = getWTranspose();
  int K = w_transpose ? weight_shape[1] : weight_shape[0];
  int N = w_transpose ? weight_shape[0] : weight_shape[1];

  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[3], p.outputs[0], 1, 1, M, K,
                N, false, -1.0, 0, 0, w_transpose, false, false, false);
  p.handle = (void *)matmul;

  return success();
}

void tpu::W8A16MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::W8A16MatMulOp::inference(InferenceParameter &p) {
  // dequant weight back to f16
  auto weight = p.inputs[1];
  auto scale = p.inputs[2];
  auto weight_value = getWeight();
  auto weight_shape =
      weight_value.getType().cast<RankedTensorType>().getShape();
  int K = weight_shape[0];
  int N = weight_shape[1];

  for (int i = 0; i < K; i++) {
    auto offset = i * N;
    for (int j = 0; j < N; j++) {
      weight[offset + j] = weight[offset + j] * scale[i];
    }
  }
  // hand over the rest work to onednn matmul
  if (p.handle == nullptr) {
    return failure();
  }
  auto matmul = (MatMul *)p.handle;

  matmul->run();
  auto num_elem = module::getNumElements(getOutput());
  F16(p.outputs[0], p.outputs[0], num_elem);
  return success();
}

LogicalResult tpu::W8A16MatMulOp::canonicalize(W8A16MatMulOp op,
                                               PatternRewriter &rewriter) {
  auto input_value = op.getInput();
  if (!module::getElementType(input_value).isF16()) {
    llvm_unreachable("input of W8A16MatMul has to be F16");
  }
  auto weight_op = op.getWeight().getDefiningOp<top::WeightOp>();
  if (module::getElementType(weight_op.getOutput()).isInteger(8)) {
    return failure();
  }
  auto bias_value = op.getBias();
  auto bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());

  // bias is saved the same as input/ output dtype in BM1684x
  if (bias_op && module::isBM1684X()) {
    op.setOperand(3, bias_op.clone_f16(op));
  }

  // 1. compute per-channel scale for quantize
  auto weight_f32_data = weight_op.read<float>();
  auto weight_shape = weight_op.getType().cast<RankedTensorType>().getShape();
  int64_t row = weight_shape[0];
  int64_t col = weight_shape[1];

  // R_transpose used to optimize mm2 backend branch. Maybe used in the future.
  // auto in_shape =
  // input_value.getType().cast<RankedTensorType>().getShape(); auto in_ele_num
  // = module::getNumElements(input_value); auto in_col_num = in_shape.back();
  // if (row > col && in_ele_num > in_col_num) {
  //   // transpose the weight data
  //   auto trans_weight =
  //       std::make_shared<std::vector<float>>(weight_f32_data->size());
  //   for (int i = 0; i < col; ++i) {
  //     for (int j = 0; j < row; ++j) {
  //       (*trans_weight)[i * row + j] = (*weight_f32_data)[j * col + i];
  //     }
  //   }
  //   std::swap(row, col);
  //   op.setWTranspose(true);
  // }

  float global_min, global_max;
  findMinMax(weight_f32_data->data(), row * col, &global_min, &global_max);
  auto sign = !(global_min > 0);

  auto scale_f32_data = computePerChannelScale(weight_f32_data, row, col, sign);
  // 2. quantize weight to i8
  auto weight_i8_data =
      std::make_shared<std::vector<int8_t>>(weight_f32_data->size());
  auto weight_u8_data =
      std::make_shared<std::vector<uint8_t>>(weight_f32_data->size());
  for (auto c = 0; c < row; c++) {
    auto offset = c * col;
#pragma omp parallel for schedule(static, omp_schedule(col))
    for (auto i = 0; i < col; i++) {
      if (sign) {
        weight_i8_data->at(offset + i) =
            to_int8(weight_f32_data->at(offset + i) / scale_f32_data->at(c));
      } else {
        weight_u8_data->at(offset + i) =
            to_uint8(weight_f32_data->at(offset + i) / scale_f32_data->at(c));
      }
    }
  }
  // 3. create f16 weightOp for scale
  rewriter.setInsertionPoint(op);
  auto scale_type = RankedTensorType::get({row}, rewriter.getF32Type());
  auto scale_value =
      top::WeightOp::create(op, "scale", *scale_f32_data, scale_type);
  op.setOperand(2, scale_value.getDefiningOp<top::WeightOp>().clone_f16(op));

  // 4. create i8 weightOp
  auto new_weight_type =
      RankedTensorType::get({row, col}, rewriter.getIntegerType(8, sign));
  if (sign) {
    auto new_weight_value =
        top::WeightOp::create(op, "i8", *weight_i8_data, new_weight_type);
    op.setOperand(1, new_weight_value);
  } else {
    auto new_weight_value =
        top::WeightOp::create(op, "u8", *weight_u8_data, new_weight_type);
    op.setOperand(1, new_weight_value);
  }

  return success();
};
