//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Distribute/Distribute.h"

// ======================================
// pattern MatMulSliceMerge
// e.g. ChatGlm2
// ======================================

namespace tpu_mlir {
namespace tpu {

// e.g. [12, 16, 18] => [12, 16, 9]
static bool isHalfSlice(tpu::SliceOp op) {
  auto offset = module::getI64Array(op.getOffset());
  auto steps = module::getI64Array(op.getSteps());
  auto in_shape = module::getShape(op.getInput());
  auto out_shape = module::getShape(op.getOutput());
  for (int i = 0; i < in_shape.size(); i++) {
    if (steps->at(i) != 1) {
      return false;
    }
    auto o = offset->at(i);
    if (i == in_shape.size() - 1) {
      if ((o != 0 && o != in_shape[i] / 2) || out_shape[i] != in_shape[i] / 2) {
        return false;
      }
    } else if (o != 0 || in_shape[i] != out_shape[i]) {
      return false;
    }
  }
  return true;
}

LogicalResult
MatMulSliceMerge::matchAndRewrite(tpu::MatMulOp op,
                                  PatternRewriter &rewriter) const {
  if (!isLargeMatMul(op) || module::isOpInDistribution(op)) {
    return failure();
  }
  std::vector<Operation *> users(op->user_begin(), op->user_end());
  if (users.size() != 2) {
    return failure();
  }
  Operation *res_op = nullptr;
  for (auto user : users) {
    auto slice = dyn_cast<tpu::SliceOp>(user);
    if (!slice || !slice->hasOneUse() || !isHalfSlice(slice)) {
      return failure();
    }
    auto next = *slice->user_begin();
    while (next != nullptr) {
      if (isBinaryOp(next)) {
        if (res_op == nullptr) {
          res_op = next;
          continue;
        } else if (next != res_op) {
          return failure();
        }
        break;
      } else if (false == next->hasOneUse() ||
                 !next->hasTrait<trait::SupportElementwise>()) {
        return failure();
      }
      next = *next->user_begin();
    }
  }
  if (!res_op->hasOneUse()) {
    return failure();
  }
  auto next = *res_op->user_begin();
  while (next != nullptr) {
    if (isLargeMatMul(next)) {
      break;
    }
    if (false == next->hasOneUse() ||
        !next->hasTrait<trait::SupportElementwise>()) {
      return failure();
    }
  }
  // Bingo !!
  distribute(rewriter, op, next, tpu::DistributionPattern::MatMulSliceMerge);
  return success();
}

template <>
void DoDistribution<MatMulSliceMerge>(PatternRewriter &rewriter,
                                      tpu::DistributionBeginOp op,
                                      int64_t num_devices) {
  auto next_op = *op->user_begin();
  auto mm0 = cast<tpu::MatMulOp>(next_op);
  auto filterOp = mm0.getRight().getDefiningOp<top::WeightOp>();
  auto filterShape = module::getShape(filterOp.getOutput());
  auto outputShape = module::getShape(mm0.getOutput());
  auto attrs = op->getAttrs();
  auto has_bias = !module::isNone(mm0.getBias());
  auto num_dims = filterShape.size();
  auto N = filterShape[num_dims - 1];
  auto N_half = N / 2;
  auto slice_n = ceiling_func(N_half, num_devices);
  std::vector<Operation *> slices(mm0->user_begin(), mm0->user_end());
  auto slice0Op = cast<tpu::SliceOp>(slices[0]);
  auto offset = module::getI64Array(slice0Op.getOffset());
  if (offset->back() != 0) {
    std::swap(slices[0], slices[1]);
  }
  std::vector<Value> end_operands;
  Operation *end_op = nullptr;
  for (int i = 0; i < num_devices; i++) {
    std::vector<Value> res_operands;
    auto offset = i * slice_n;
    auto length = std::min(slice_n, N_half - offset);
    auto suffix = std::to_string(i);
    // slice one half
    for (int half = 0; half < 2; half++) {
      auto offset_half = offset + half * N_half;
      auto suffix_half = suffix + "_" + std::to_string(half);
      auto newFilter0 = module::opSliceAxis(mm0.getRight(), num_dims - 1,
                                            offset_half, length);
      std::vector<Value> operands;
      operands.push_back(mm0.getInput());
      operands.push_back(newFilter0);
      if (has_bias) {
        auto new_bias = module::opSliceAxis(mm0.getBias(), num_dims - 1,
                                            offset_half, length);
        operands.push_back(new_bias);
      } else {
        operands.push_back(mm0.getBias());
      }
      auto new_loc = module::getLocLike(mm0.getOutput(), suffix_half);
      std::vector<int64_t> new_shape = outputShape;
      new_shape[new_shape.size() - 1] = length;
      auto new_type = module::getTypeLike(mm0.getOutput(), new_shape);
      rewriter.setInsertionPointAfter(mm0);
      auto new_mm0 =
          rewriter.create<tpu::MatMulOp>(new_loc, new_type, operands, attrs);
      Value cur_output = new_mm0.getOutput();
      next_op = *slices[half]->user_begin();
      while (!isBinaryOp(next_op)) {
        auto new_op = cloneOp(rewriter, next_op, new_shape, suffix_half);
        new_op->setOperand(0, cur_output);
        cur_output = new_op->getResult(0);
        next_op = *next_op->user_begin();
      }
      res_operands.push_back(cur_output);
    }
    // res_op: add/mul
    auto new_shape = module::getShape(res_operands[0]);
    auto new_op = cloneOp(rewriter, next_op, new_shape, suffix);
    new_op->setOperands(res_operands);
    Value cur_output = new_op->getResult(0);
    next_op = *next_op->user_begin();
    // matmul op
    while (!isa<tpu::MatMulOp>(next_op)) {
      new_op = cloneOp(rewriter, next_op, new_shape, suffix);
      new_op->setOperand(0, cur_output);
      cur_output = new_op->getResult(0);
      next_op = *next_op->user_begin();
    }
    auto mm1 = cast<tpu::MatMulOp>(next_op);
    auto new_loc = module::getLocLike(next_op, suffix);
    std::vector<Value> operands;
    operands.push_back(cur_output);
    auto newFilter1 =
        module::opSliceAxis(mm1.getRight(), num_dims - 2, offset, length);
    operands.push_back(newFilter1);
    if (module::isNone(mm1.getBias())) {
      operands.push_back(mm1.getBias());
    } else {
      auto bias = mm1.getBias().getDefiningOp<top::WeightOp>();
      operands.push_back(bias.clone(suffix));
    }
    rewriter.setInsertionPointAfter(next_op);
    auto new_mm1 = rewriter.create<tpu::MatMulOp>(
        new_loc, mm1.getOutput().getType(), operands, mm1->getAttrs());
    end_operands.push_back(new_mm1.getOutput());
    if (i == 0) {
      end_op = *next_op->user_begin();
    } else {
      assert(end_op == *next_op->user_begin());
    }
  }
  assert(isa<tpu::DistributionEndOp>(end_op));
  end_op->setOperands(end_operands);
  eraseForward(rewriter, mm0);
}

} // namespace tpu
} // namespace tpu_mlir
