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

std::shared_ptr<std::vector<float>>
computePerChannelScale(const std::shared_ptr<std::vector<float>> &weight_data,
                       int row, int col, bool sign, int bitwidth) {
  auto per_channel_scale = std::make_shared<std::vector<float>>(row);

  for (int c = 0; c < row; c++) {
    float *p_weight = weight_data->data() + c * col;
    auto w_max = findMaxabs(p_weight, col);
    auto pow_value = sign ? bitwidth - 1 : bitwidth;
    per_channel_scale->at(c) = w_max / (std::pow(2, pow_value) - 1);
  }
  return per_channel_scale;
}

LogicalResult tpu::A16MatMulOp::init(InferenceParameter &p) {
  // (MxK), (KxN) matmul
  if (getWeightBits() == 8) {
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

    matmul->setup(p.inputs[0], p.inputs[1], p.inputs[3], p.outputs[0], 1, 1, M,
                  K, N, false, -1.0, 0, 0, w_transpose, false, false, false);
    p.handle = (void *)matmul;
  }
  return success();
}

void tpu::A16MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::A16MatMulOp::inference(InferenceParameter &p) {
  // dequant weight back to f16/ bf16
  auto scale = p.inputs[2];
  auto sign = getSign();
  auto weight_value = getWeight();
  auto weight_shape =
      weight_value.getType().cast<RankedTensorType>().getShape();
  int K = weight_shape[0];
  int N = weight_shape[1];
  auto weight = p.inputs[1];
  if (getWeightBits() == 4) {
    N *= 2;
    auto in_shape = getInput().getType().cast<RankedTensorType>().getShape();
    int64_t M = 1;
    for (int i = 0; i < in_shape.size() - 1; i++) {
      M *= in_shape[i];
    }
    auto new_weight = std::vector<float>(K * N, 0);
    for (int i = 0; i < K; i++) {
      auto offset = i * N;
      for (int j = 0; j < N; j++) {
        new_weight[offset + j] =
            ((int(weight[(offset + j) / 2]) & 0x0F) - (sign ? 8 : 0)) *
            scale[i];
        j++;
        new_weight[offset + j] =
            ((int(weight[(offset + j) / 2]) >> 4) - (sign ? 8 : 0)) * scale[i];
      }
    }
    auto matmul = new MatMul();
    matmul->setup(p.inputs[0], new_weight.data(), p.inputs[3], p.outputs[0], 1,
                  1, M, K, N, false, -1.0, 0, 0, false, false, false, false);
    matmul->run();
    delete matmul;
  } else {
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
  }

  auto num_elem = module::getNumElements(getOutput());
  if (module::isF16Modes()) {
    F16(p.outputs[0], p.outputs[0], num_elem);
  } else {
    BF16(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

LogicalResult tpu::A16MatMulOp::canonicalize(A16MatMulOp op,
                                                   PatternRewriter &rewriter) {
  auto input_value = op.getInput();
  auto ele_type = module::getElementType(input_value);
  if (!ele_type.isF16() && !ele_type.isBF16()) {
    llvm_unreachable("input of A16MatMul has to be F16 or BF16");
  }
  auto weight_op = op.getWeight().getDefiningOp<top::WeightOp>();
  if (module::getElementType(weight_op.getOutput()).isInteger(8)) {
    return failure();
  }
  auto bias_value = op.getBias();
  auto bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());

  // bias is saved the same as input/ output dtype in BM1684x
  if (bias_op && module::isBM1684X()) {
    op.setOperand(3, ele_type.isF16() ? bias_op.clone_f16(op) : bias_op.clone_bf16(op));
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
  op.setSign(sign);

  auto bitwidth = op.getWeightBits();

  auto scale_f32_data =
      computePerChannelScale(weight_f32_data, row, col, sign, bitwidth);

  // 2. quantize weight to low bit integer
  auto weight_size = weight_f32_data->size();
  if (bitwidth == 4) {
    assert(weight_size % 2 == 0);
    weight_size /= 2;
  }
  auto int_weight_data = std::make_shared<std::vector<int8_t>>(weight_size, 0);
  auto uint_weight_data =
      std::make_shared<std::vector<uint8_t>>(weight_size, 0);
  for (auto c = 0; c < row; c++) {
    int offset = c * col;
    int int_offset = offset / 2;
    for (auto i = 0; i < col; i++) {
      auto tmp_value = weight_f32_data->at(offset + i) / scale_f32_data->at(c);

      if (bitwidth == 8) {
        if (sign) {
          int_weight_data->at(offset + i) = to_int8(tmp_value);
        } else {
          uint_weight_data->at(offset + i) = to_uint8(tmp_value);
        }
      } else {
        int index = i / 2;
        if (i % 2) {
          if (sign) {
            uint_weight_data->at(int_offset + index) |= to_uint4(tmp_value + 8)
                                                        << 4;
          } else {
            uint_weight_data->at(int_offset + index) |= to_uint4(tmp_value)
                                                        << 4;
          }
        } else {
          if (sign) {
            uint_weight_data->at(int_offset + index) = to_uint4(tmp_value + 8);
          } else {
            uint_weight_data->at(int_offset + index) = to_uint4(tmp_value);
          }
        }
      }
    }
  }
  // 3. create f16 weightOp for scale
  rewriter.setInsertionPoint(op);
  auto scale_type = RankedTensorType::get({row}, rewriter.getF32Type());
  auto f32_scale_op =
      top::WeightOp::create(op, "scale", *scale_f32_data, scale_type).getDefiningOp<top::WeightOp>();
  auto half_scale_value = ele_type.isF16() ? f32_scale_op.clone_f16(op) : f32_scale_op.clone_bf16(op);
  op.setOperand(2, half_scale_value);

  // 4. create i8 weightOp
  auto new_weight_type =
      RankedTensorType::get({row, bitwidth == 8 ? col : col / 2},
                            rewriter.getIntegerType(8, sign && bitwidth == 8));
  if (sign && bitwidth == 8) {
    auto new_weight_value =
        top::WeightOp::create(op, "int", *int_weight_data, new_weight_type);
    op.setOperand(1, new_weight_value);
  } else {
    auto new_weight_value =
        top::WeightOp::create(op, "uint", *uint_weight_data, new_weight_type);
    op.setOperand(1, new_weight_value);
  }

  return success();
};
