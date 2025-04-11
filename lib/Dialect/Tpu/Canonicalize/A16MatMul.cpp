//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {

/***
 * The function for computing per-channel scale and zero point.
 * Only int4 will do asymmetric quantization.
 ***/
void computePerChannelParam(
    const std::shared_ptr<std::vector<float>> &weight_data, int row, int col,
    bool sign, int bitwidth, std::shared_ptr<std::vector<float>> &scale,
    std::shared_ptr<std::vector<uint8_t>> &zp) {

  if (bitwidth != 4) {
    for (auto c = 0; c < row; c++) {
      float *p_weight = weight_data->data() + c * col;
      auto w_max = findMaxabs(p_weight, col);
      auto pow_value = sign ? bitwidth - 1 : bitwidth;
      scale->at(c) = w_max / (std::pow(2, pow_value) - 1);
    }
  } else {
    int max_int = std::pow(2, bitwidth) - 1;
    int min_int = 0;
    float max_val = 0;
    float min_val = 0;
    for (auto c = 0; c < row; c++) {
      float *p_weight = weight_data->data() + c * col;
      findMinMax(p_weight, col, &min_val, &max_val);
      scale->at(c) = std::max(max_val - min_val, (float)1e-5) / max_int;
      zp->at(c) = (uint8_t)std::clamp(-(int)std::round(min_val / scale->at(c)),
                                      min_int, max_int);
    }
  }
}

void computePerChannelParam(
    const std::shared_ptr<std::vector<float>> &weight_data, int row, int col,
    bool sign, int bitwidth, std::shared_ptr<std::vector<float>> &scale,
    std::shared_ptr<std::vector<float>> &zp) {

  if (bitwidth != 4) {
    for (auto c = 0; c < row; c++) {
      float *p_weight = weight_data->data() + c * col;
      auto w_max = findMaxabs(p_weight, col);
      auto pow_value = sign ? bitwidth - 1 : bitwidth;
      scale->at(c) = w_max / (std::pow(2, pow_value) - 1);
    }
  } else {
    int max_int = std::pow(2, bitwidth) - 1;
    int min_int = 0;
    float max_val = 0;
    float min_val = 0;
    for (auto c = 0; c < row; c++) {
      float *p_weight = weight_data->data() + c * col;
      findMinMax(p_weight, col, &min_val, &max_val);
      scale->at(c) = std::max(max_val - min_val, (float)1e-5) / max_int;
      zp->at(c) = std::clamp(-(int)std::round(min_val / scale->at(c)), min_int,
                             max_int);
    }
  }
}
/***
 *  The function for per-group int4/int8 asymmetric/symmetric quantization
 ***/
void computePerGroupParam(
    const std::shared_ptr<std::vector<float>> &weight_data, int row, int col,
    int bitwidth, int q_group_size, bool q_symmetric,
    std::shared_ptr<std::vector<float>> &scale,
    std::shared_ptr<std::vector<uint8_t>> &zp) {
  assert(col % q_group_size == 0 && "invalid q_group_size");
  int num_groups = row * col / q_group_size;
  int max_int = bitwidth == 4 ? std::pow(2, 4) - 1 : std::pow(2, 8) - 1;
  int min_int = 0;
  float max_val = 0;
  float min_val = 0;

  for (int i = 0; i < num_groups; i++) {
    float *p_weight = weight_data->data() + i * q_group_size;
    findMinMax(p_weight, q_group_size, &min_val, &max_val);
    auto abs_max = findMaxabs(p_weight, q_group_size);
    if (q_symmetric) {
      scale->at(i) = 2 * std::abs(abs_max) / max_int;
    } else {
      scale->at(i) =
          std::max(max_val - min_val, (float)1e-5) / (max_int - min_int);
      zp->at(i) = (uint8_t)std::clamp(-(int)std::round(min_val / scale->at(i)),
                                      min_int, max_int);
    }
  }
}

void computePerGroupParam(
    const std::shared_ptr<std::vector<float>> &weight_data, int row, int col,
    int bitwidth, int q_group_size, bool q_symmetric,
    std::shared_ptr<std::vector<float>> &scale,
    std::shared_ptr<std::vector<float>> &zp) {
  assert(col % q_group_size == 0 && "invalid q_group_size");
  int num_groups = row * col / q_group_size;
  int max_int = bitwidth == 4 ? std::pow(2, 4) - 1 : std::pow(2, 8) - 1;
  int min_int = 0;
  float max_val = 0;
  float min_val = 0;

  for (int i = 0; i < num_groups; i++) {
    float *p_weight = weight_data->data() + i * q_group_size;
    findMinMax(p_weight, q_group_size, &min_val, &max_val);
    auto abs_max = findMaxabs(p_weight, q_group_size);
    if (q_symmetric) {
      scale->at(i) = 2 * std::abs(abs_max) / max_int;
    } else {
      scale->at(i) =
          std::max(max_val - min_val, (float)1e-5) / (max_int - min_int);
      zp->at(i) = std::clamp(-(int)std::round(min_val / scale->at(i)), min_int,
                             max_int);
    }
  }
}

/***
 *  The function for quantizing weight data
 *  Output: inplace-changed int_weight_data, uint_weight_data
 ***/
void weightQuantization(
    int bitwidth, bool sign, int row, int col, int q_group_size,
    std::shared_ptr<std::vector<float>> &weight_f32_data,
    std::shared_ptr<std::vector<float>> &scale,
    std::shared_ptr<std::vector<uint8_t>> &zp,
    std::shared_ptr<std::vector<int8_t>> &int_weight_data,
    std::shared_ptr<std::vector<uint8_t>> &uint_weight_data) {
  if (!q_group_size) {
    for (auto c = 0; c < row; c++) {
      int offset = c * col;
      int int4_offset = offset / 2;
      for (auto i = 0; i < col; i++) {
        auto tmp_value =
            std::round(weight_f32_data->at(offset + i) / scale->at(c)) +
            zp->at(c);

        if (bitwidth == 8) {
          if (sign) {
            int_weight_data->at(offset + i) = to_int8(tmp_value);
          } else {
            uint_weight_data->at(offset + i) = to_uint8(tmp_value);
          }
        } else {
          int index = i / 2;
          if (i % 2) {
            uint_weight_data->at(int4_offset + index) |= to_uint4(tmp_value)
                                                         << 4;
          } else {
            uint_weight_data->at(int4_offset + index) = to_uint4(tmp_value);
          }
        }
      }
    }
  } else {
    for (auto i = 0; i < row * col; i++) {
      int quant_idx = i / q_group_size;
      auto tmp_value =
          std::round(weight_f32_data->at(i) / scale->at(quant_idx)) +
          zp->at(quant_idx);
      if (bitwidth == 8) {
        uint_weight_data->at(i) = to_uint8(tmp_value);
      } else {
        int real_weight_idx = i / 2;
        if (i % 2) {
          uint_weight_data->at(real_weight_idx) |= to_uint4(tmp_value) << 4;
        } else {
          uint_weight_data->at(real_weight_idx) = to_uint4(tmp_value);
        }
      }
    }
  }
}

void weightQuantization(
    int bitwidth, bool sign, int row, int col, int q_group_size,
    std::shared_ptr<std::vector<float>> &weight_f32_data,
    std::shared_ptr<std::vector<float>> &scale,
    std::shared_ptr<std::vector<float>> &zp,
    std::shared_ptr<std::vector<int8_t>> &int_weight_data,
    std::shared_ptr<std::vector<uint8_t>> &uint_weight_data) {
  if (!q_group_size) {
    for (auto c = 0; c < row; c++) {
      int offset = c * col;
      int int4_offset = offset / 2;
      for (auto i = 0; i < col; i++) {
        auto tmp_value = std::round(
            (weight_f32_data->at(offset + i) + zp->at(c)) / scale->at(c));

        if (bitwidth == 8) {
          if (sign) {
            int_weight_data->at(offset + i) = to_int8(tmp_value);
          } else {
            uint_weight_data->at(offset + i) = to_uint8(tmp_value);
          }
        } else {
          int index = i / 2;
          if (i % 2) {
            uint_weight_data->at(int4_offset + index) |= to_uint4(tmp_value)
                                                         << 4;
          } else {
            uint_weight_data->at(int4_offset + index) = to_uint4(tmp_value);
          }
        }
      }
    }
  } else {
    for (auto i = 0; i < row * col; i++) {
      int quant_idx = i / q_group_size;
      auto tmp_value =
          std::round(weight_f32_data->at(i) / scale->at(quant_idx)) +
          zp->at(quant_idx);
      if (bitwidth == 8) {
        uint_weight_data->at(i) = to_uint8(tmp_value);
      } else {
        int real_weight_idx = i / 2;
        if (i % 2) {
          uint_weight_data->at(real_weight_idx) |= to_uint4(tmp_value) << 4;
        } else {
          uint_weight_data->at(real_weight_idx) = to_uint4(tmp_value);
        }
      }
    }
  }
}

struct A16MatMulAdjust : public OpRewriterPatternEx<tpu::A16MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  A16MatMulAdjust(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::A16MatMulOp>(context, "A16MatMulAdjust") {}

  LogicalResult matchAndRewriteImpl(tpu::A16MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getScale().getDefiningOp<top::WeightOp>()) {
      return failure();
    }
    auto input_value = op.getInput();
    auto ele_type = module::getElementType(input_value);
    if (!ele_type.isF16() && !ele_type.isBF16() &&
        !module::isCalibratedType(input_value)) {
      llvm_unreachable("input of A16MatMul has to be F16 or BF16");
    }
    bool use_dq2 = false;
    if (module::isMARS3() || module::isSGTPUV8()) {
      use_dq2 = (module::getQuantGroupSize() >= 32) &&
                (module::getQuantGroupSize() % 32 == 0) &&
                (op.getWeightBits() == 4);
      if (use_dq2) {
        assert(ele_type.isBF16());
      }
    }
    auto weight_op = op.getWeight().getDefiningOp<top::WeightOp>();
    if (module::getElementType(weight_op.getOutput()).isInteger(8)) {
      return failure();
    }
    auto bias_value = op.getBias();
    auto bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());

    if (bias_op) {
      op.setOperand(4, ele_type.isF16() ? bias_op.clone_f16(op)
                                        : bias_op.clone_bf16(op));
    }

    // 1. compute per-channel scale for quantize
    auto weight_f32_data = weight_op.read<float>();
    auto weight_shape = weight_op.getType().cast<RankedTensorType>().getShape();
    int64_t row = weight_shape[0];
    int64_t col = weight_shape[1];

    // transpose the weight data
    auto trans_weight =
        std::make_shared<std::vector<float>>(weight_f32_data->size());
    for (int i = 0; i < col; ++i) {
      for (int j = 0; j < row; ++j) {
        (*trans_weight)[i * row + j] = (*weight_f32_data)[j * col + i];
      }
    }
    std::swap(row, col);
    op.setWTranspose(true);

    float global_min, global_max;
    findMinMax(trans_weight->data(), row * col, &global_min, &global_max);
    auto sign = !(global_min > 0);
    op.setSign(sign);

    auto bitwidth = op.getWeightBits();

    int q_group_size =
        (bitwidth == 4 || bitwidth == 8) ? module::getQuantGroupSize() : 0;
    bool q_symmetric = q_group_size != 0 ? module::isQuantSymmetric() : false;
    int64_t quant_param_size = !q_group_size ? row : (row * col / q_group_size);

    int zp_value = bitwidth == 4 ? 8 : 128;

    auto scale = std::make_shared<std::vector<float>>(quant_param_size, 0);
    auto zp_int8 = std::make_shared<std::vector<uint8_t>>(
        quant_param_size, q_symmetric ? zp_value : 0);
    auto zp_fp = std::make_shared<std::vector<float>>(
        quant_param_size, q_symmetric ? zp_value : 0);

    if (!q_group_size) {
      if (module::isSG2380())
        computePerChannelParam(trans_weight, row, col, sign, bitwidth, scale,
                               zp_fp);
      else
        computePerChannelParam(trans_weight, row, col, sign, bitwidth, scale,
                               zp_int8);
    } else {
      op.setQGroupSize(q_group_size);
      if (module::isSG2380() || use_dq2)
        computePerGroupParam(trans_weight, row, col, bitwidth, q_group_size,
                             q_symmetric, scale, zp_fp);
      else
        computePerGroupParam(trans_weight, row, col, bitwidth, q_group_size,
                             q_symmetric, scale, zp_int8);
    }

    // 2. quantize weight to low bit integer
    auto weight_size = trans_weight->size();
    if (bitwidth == 4) {
      assert(weight_size % 2 == 0);
      weight_size /= 2;
    }
    auto int_weight_data =
        std::make_shared<std::vector<int8_t>>(weight_size, 0);
    auto uint_weight_data =
        std::make_shared<std::vector<uint8_t>>(weight_size, 0);

    if (module::isSG2380() || use_dq2)
      weightQuantization(bitwidth, sign, row, col, q_group_size, trans_weight,
                         scale, zp_fp, int_weight_data, uint_weight_data);
    else
      weightQuantization(bitwidth, sign, row, col, q_group_size, trans_weight,
                         scale, zp_int8, int_weight_data, uint_weight_data);

    // 3. create f16 weightOp for scale/zp
    rewriter.setInsertionPoint(op);
    std::vector<int64_t> quant_param_shape = {row, quant_param_size / row};
    auto scale_type =
        RankedTensorType::get(quant_param_shape, rewriter.getF32Type());
    auto f32_scale_op = top::WeightOp::create(op, "scale", *scale, scale_type)
                            .getDefiningOp<top::WeightOp>();
    auto half_scale_value = ele_type.isF16() ? f32_scale_op.clone_f16(op)
                                             : f32_scale_op.clone_bf16(op);
    op.setOperand(2, half_scale_value);
    if (module::isSG2380() || use_dq2) {
      auto zp_type =
          RankedTensorType::get(quant_param_shape, rewriter.getF32Type());
      auto fp_zp_op = top::WeightOp::create(op, "zp", *zp_fp, zp_type)
                          .getDefiningOp<top::WeightOp>();
      auto half_zp_value =
          ele_type.isF16() ? fp_zp_op.clone_f16(op) : fp_zp_op.clone_bf16(op);
      op.setOperand(3, half_zp_value);
    } else {
      auto zp_type = RankedTensorType::get(quant_param_shape,
                                           rewriter.getIntegerType(8, false));
      auto zp_op = top::WeightOp::create(op, "zp", *zp_int8, zp_type)
                       .getDefiningOp<top::WeightOp>();
      op.setOperand(3, zp_op.getOutput());
    }

    // 4. create i8 weightOp
    auto new_weight_type = RankedTensorType::get(
        {row, bitwidth == 8 ? col : col / 2},
        rewriter.getIntegerType(8, !q_group_size && sign && bitwidth == 8));
    if (!q_group_size && sign && bitwidth == 8) {
      auto new_weight_value =
          top::WeightOp::create(op, "int", *int_weight_data, new_weight_type);
      op.setOperand(1, new_weight_value);
    } else {
      auto new_weight_value =
          top::WeightOp::create(op, "uint", *uint_weight_data, new_weight_type);
      op.setOperand(1, new_weight_value);
    }

    return success();
  }
};

void tpu::A16MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<A16MatMulAdjust>(context);
}

} // namespace tpu
} // namespace tpu_mlir
