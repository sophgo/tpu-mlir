//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "ConvUtils.h"

using namespace bm1684x;

static LogicalResult
canonicalize_matmul_operand_shape(tpu::MatMulOp op, PatternRewriter &rewriter) {
  auto left_shape = module::getShape(op.getInput());
  auto right_shape = module::getShape(op.getRight());
  auto res = failure();
  // modify right shape from (K, N) to (1, K, N) if left is (B, M, K)
  if (module::isWeight(op.getRight())) {
    if (left_shape.size() == 3 && right_shape.size() == 2) {
      std::vector<int64_t> new_shape{1, right_shape[0], right_shape[1]};
      auto newType = RankedTensorType::get(
          new_shape, module::getElementType(op.getRight()));
      op.getRight().setType(newType);
    }
    res = success();
  }
  // if left_transpose, bias_shape = (1, X, 1, 1)
  // if !left_transpose, bias_shape = (1, 1, 1, X)
  if (module::isWeight(op.getBias())) {
    auto bias_shape = module::getShape(op.getBias());
    if (bias_shape.size() == 1) {
      std::vector<int64_t> new_shape(4, 1);
      new_shape[(op.getLeftTranspose() ? 1 : 3)] = bias_shape[0];
      auto newType = RankedTensorType::get(
          new_shape, module::getElementType(op.getBias()));
      op.getBias().setType(newType);
    }
    res = success();
  }

  return res;
}

template <>
LogicalResult WeightReorder<tpu::MatMulOp, int8_t>::matchAndRewriteImpl(
    tpu::MatMulOp op, PatternRewriter &rewriter) const {
  // if (!module::getStorageType(op.getBias()).isInteger(32))
  //   return failure();
  auto p = op.parseParam();

  // bias merge input zp
  if (p.input_zp == 0) {
    auto in_stype = module::getStorageType(op.getInput());
    bool isINT4MM = in_stype.isInteger(4);
    if (isINT4MM) {
      // filter
      auto filterOp = dyn_cast<top::WeightOp>(op.getRight().getDefiningOp());

      auto filter_i8 = filterOp.read<int8_t>();
      std::vector<int64_t> coeff_shape; // = {1, p.K, 1, p.N};
      auto shape = module::getShape(op.getRight());
      for (int i = 0; i < shape.size(); ++i) {
        coeff_shape.push_back(shape[i]);
      }

      tpu::compact_coeff_for_int4(filter_i8, coeff_shape, false);
      bool sign = true;
      // auto stype = module::getStorageType(op.getRight());
      // auto new_type = RankedTensorType::get(coeff_shape, stype);
      auto new_type =
          RankedTensorType::get(coeff_shape, rewriter.getIntegerType(4, sign));
      auto new_op =
          top::WeightOp::create(op, "filter_reorderd", *filter_i8, new_type);
      op->setOperand(1, new_op);
      op.getRight().setType(new_type);
    } else {
      return failure();
    }
  }
  i32_array_t bias_quant;
  if (module::isWeight(op.getBias())) {
    bias_quant =
        cast<top::WeightOp>(op.getBias().getDefiningOp()).read<int32_t>();
    for (size_t i = 0; i < p.N; ++i) {
      bias_quant->data()[i] += p.input_zp * p.right_zp * p.K;
    }
  } else {
    i32_array_t bias_quant;
    if (module::isWeight(op.getBias())) {
      bias_quant =
          cast<top::WeightOp>(op.getBias().getDefiningOp()).read<int32_t>();
      for (size_t i = 0; i < p.N; ++i) {
        bias_quant->data()[i] += p.input_zp * p.right_zp * p.K;
      }
    } else {
      bias_quant = i32_array_t(new std::vector<int32_t>(p.N, 0));
      for (size_t i = 0; i < p.N; ++i) {
        bias_quant->data()[i] += p.input_zp * p.right_zp * p.K;
      }
      int64_t left_num_dims = module::getShape(op.getInput()).size();
      std::vector<int64_t> bias_shape(left_num_dims, 1);
      bias_shape[left_num_dims - 1] = p.N;
      auto new_type = RankedTensorType::get(bias_shape, rewriter.getI32Type());
      auto new_op =
          top::WeightOp::create(op, "bias_merge_izp", *bias_quant, new_type);
      op->setOperand(2, new_op);
    }
  }
  canonicalize_matmul_operand_shape(op, rewriter);
  return success();
}

template <>
LogicalResult WeightReorder<tpu::MatMulOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::MatMulOp op, PatternRewriter &rewriter) const {
  return canonicalize_matmul_operand_shape(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::MatMulOp, Float16Type>::matchAndRewriteImpl(
    tpu::MatMulOp op, PatternRewriter &rewriter) const {
  return canonicalize_matmul_operand_shape(op, rewriter);
}
