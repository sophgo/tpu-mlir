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
// pattern MatMulSliceMerge2
// e.g. Llama2
// ======================================

namespace tpu_mlir {
namespace tpu {

// in -> Acitve -> out
// in -> Cast -> Active -> Cast -> out
Value isCastActive(Value in) {
  auto active_op = in.getDefiningOp();
  if (dyn_cast<tpu::ActiveOp>(active_op)) {
    return active_op->getOperand(0);
  } else if (auto cast_out_op = dyn_cast<tpu::CastOp>(active_op)) {
    active_op = dyn_cast<tpu::ActiveOp>(cast_out_op.getInput().getDefiningOp());
    if (!active_op) {
      return nullptr;
    }
    auto cast_in_op =
        dyn_cast<tpu::CastOp>(active_op->getOperand(0).getDefiningOp());
    if (cast_in_op) {
      return cast_in_op.getInput();
    }
  }
  return nullptr;
}

LogicalResult
MatMulSliceMerge2::matchAndRewrite(tpu::MatMulOp op,
                                   PatternRewriter &rewriter) const {
  if (!isLargeMatMul(op) || module::isOpInDistribution(op)) {
    return failure();
  }
  if (op->hasOneUse() == false) {
    return failure();
  }
  // MulOp
  auto mul_down_op = dyn_cast<tpu::MulOp>(op.getOperand(0).getDefiningOp());
  if (!mul_down_op) {
    return failure();
  }
  // ActiveOp
  Value active_value = isCastActive(mul_down_op.getOperand(0));
  if (!active_value) {
    return failure();
  }
  // MatMulOp
  auto mul_right_op =
      dyn_cast<tpu::MatMulOp>(mul_down_op.getOperand(1).getDefiningOp());
  if (!mul_right_op) {
    return failure();
  }
  // MatMulOp
  auto matmul_op = dyn_cast<tpu::MatMulOp>(active_value.getDefiningOp());
  if (!matmul_op) {
    return failure();
  }
  // RMSNormOp or CastOp
  auto top_op = matmul_op.getOperand(0).getDefiningOp();
  if (isa<tpu::CastOp>(top_op)) {
    top_op = dyn_cast<tpu::CastOp>(top_op);
  } else if (isa<tpu::RMSNormOp>(top_op)) {
    top_op = dyn_cast<tpu::RMSNormOp>(top_op);
  } else {
    return failure();
  }

  // Bingo !!
  distributeAfter(rewriter, top_op, op, tpu::DistributionPattern::MatMulSliceMerge2);
  return success();
}

template <>
void splitByDevices<MatMulSliceMerge2>(PatternRewriter &rewriter,
                                       tpu::DistributionBeginOp op,
                                       int64_t num_devices) {
  // 1. Define params
  // MatMul params
  std::vector<Operation *> mm(op->user_begin(), op->user_end());
  auto mm_left = cast<tpu::MatMulOp>(mm[0]);
  auto mm_right = cast<tpu::MatMulOp>(mm[1]);
  auto topShape = module::getShape(op->getResult(0));

  // Swap MatMul->Cast->Active->Cast for MatMul
  if (isa<tpu::MulOp>(*mm_left->user_begin())) {
    std::swap(mm_left, mm_right);
  }

  auto leftFilterOp = mm_left.getRight().getDefiningOp<top::WeightOp>();
  auto filterShape = module::getShape(leftFilterOp.getOutput());
  auto outputShape = module::getShape(mm_left.getOutput());
  auto num_dims = filterShape.size();
  auto N = filterShape[num_dims - 1];
  auto slice_n = ceiling_func(N, num_devices);
  std::vector<Value> end_operands;
  Operation *end_op = nullptr;

  std::vector<Value> operands;
  bool biasAdd = false;
  Value biasValue;
  for (int i = 0; i < num_devices; i++) {
    auto offset = i * slice_n;
    auto length = std::min((i + 1) * slice_n, N) - offset;
    auto suffix = std::to_string(i);
    auto newShape = {outputShape[0], outputShape[1], length};

    // 2. Create Left MatMul
    auto newLeftFilterOp =
        module::opSliceAxis(mm_left.getRight(), num_dims - 1, offset, length);
    operands.clear();
    operands.push_back(op->getResult(0));
    operands.push_back(newLeftFilterOp);
    operands.push_back(mm_left.getBias());

    rewriter.setInsertionPointAfter(mm_left);
    auto new_mm_left = rewriter.create<tpu::MatMulOp>(
        module::getLocLike(mm_left.getOutput(), "mm_left_" + suffix),
        module::getTypeLike(mm_left.getOutput(), newShape), operands,
        mm_left->getAttrs());

    // 3. Create Right MatMul
    auto newRightFilterOp =
        module::opSliceAxis(mm_right.getRight(), num_dims - 1, offset, length);
    operands.clear();
    operands.push_back(op->getResult(0));
    operands.push_back(newRightFilterOp);
    operands.push_back(mm_right.getBias());

    rewriter.setInsertionPointAfter(mm_right);
    auto new_mm_right = rewriter.create<tpu::MatMulOp>(
        module::getLocLike(mm_right.getOutput(), "mm_right_" + suffix),
        module::getTypeLike(mm_right.getOutput(), newShape), operands,
        mm_right->getAttrs());

    // 4. Create Left Active
    operands.clear();
    Value cur_output = new_mm_left.getOutput();
    auto next_op = *mm_left->user_begin();
    while (isa<tpu::CastOp, tpu::ActiveOp>(next_op)) {
      auto new_op = cloneOp(rewriter, next_op, newShape, suffix);
      new_op->setOperand(0, cur_output);
      cur_output = new_op->getResult(0);
      next_op = *next_op->user_begin();
    }
    operands.push_back(cur_output);
    operands.push_back(new_mm_right->getResult(0));

    rewriter.setInsertionPointAfter(new_mm_right);
    auto mul_op = rewriter.create<tpu::MulOp>(
        module::getLocLike(next_op->getResult(0), "mul_" + suffix),
        module::getTypeLike(next_op->getResult(0), newShape), operands, next_op->getAttrs());

    // 5. Create Final MatMul
    next_op = *next_op->user_begin();
    auto newFinalFilterOp = module::opSliceAxis(next_op->getOperand(1),
                                                num_dims - 2, offset, length);
    operands.clear();
    operands.push_back(mul_op->getResult(0));
    operands.push_back(newFinalFilterOp);

    auto bias = next_op->getOperand(2);
    if (module::isNone(bias)) {
      operands.push_back(bias);
    } else if (module::isWeight(bias)) {
      auto bias_weight = bias.getDefiningOp<top::WeightOp>();
      operands.push_back(bias_weight.clone(suffix));
    } else {
      operands.push_back(module::getNoneOp(op));
      biasAdd = true;
      biasValue = bias;
    }

    rewriter.setInsertionPointAfter(next_op);
    auto new_mm_final = rewriter.create<tpu::MatMulOp>(
        module::getLocLike(next_op->getResult(0), "mm_final_" + suffix),
        module::getTypeLike(next_op->getResult(0), topShape), operands, next_op->getAttrs());
    end_operands.push_back(new_mm_final.getOutput());
    end_op = *next_op->user_begin();
  }

  // 6. Erase
  assert(isa<tpu::DistributionEndOp>(end_op));
  end_op->setOperands(end_operands);
  if (biasAdd) {
    auto dis_op = cast<tpu::DistributionEndOp>(end_op);
    auto dis_out = dis_op.getOutput();
    rewriter.setInsertionPointAfter(end_op);
    auto new_loc = module::getLocLike(dis_out, "add");
    auto add_op = rewriter.create<tpu::AddOp>(new_loc, dis_out.getType(),
                                              ValueRange{dis_out, biasValue});
    dis_out.replaceAllUsesExcept(add_op.getOutput(), add_op);
  }
  eraseForward(rewriter, mm[0]);
  eraseForward(rewriter, mm[1]);
}

} // namespace tpu
} // namespace tpu_mlir
