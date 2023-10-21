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

template <typename MatMulTy>
LogicalResult
MatMulSliceMerge2<MatMulTy>::matchAndRewrite(MatMulTy op,
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
  // MatMulOp / A16MatMulOp
  auto mul_right_op =
      dyn_cast<MatMulTy>(mul_down_op.getOperand(1).getDefiningOp());
  if (!mul_right_op) {
    return failure();
  }
  // MatMulOp
  auto matmul_op = dyn_cast<MatMulTy>(active_value.getDefiningOp());
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
  distributeAfter(rewriter, top_op, op,
                  tpu::DistributionPattern::MatMulSliceMerge2);
  return success();
}

template LogicalResult MatMulSliceMerge2<tpu::MatMulOp>::matchAndRewrite(
    tpu::MatMulOp op, PatternRewriter &rewriter) const;

template LogicalResult MatMulSliceMerge2<tpu::A16MatMulOp>::matchAndRewrite(
    tpu::A16MatMulOp op, PatternRewriter &rewriter) const;

template <typename MatMulTy>
void sliceMerge2Split(MatMulTy mm_left, PatternRewriter &rewriter,
                      tpu::DistributionBeginOp op, int64_t num_devices) {
  // 1. Define params
  // MatMul params
  if (!mm_left) {
    return;
  }
  std::vector<Operation *> mm(op->user_begin(), op->user_end());
  auto mm_right = cast<MatMulTy>(mm[1]);
  auto topShape = module::getShape(op->getResult(0));

  // Swap MatMul->Cast->Active->Cast for MatMul
  if (isa<tpu::MulOp>(*mm_left->user_begin())) {
    std::swap(mm_left, mm_right);
  }

  auto leftFilterOp =
      mm_left.getOperand(1).template getDefiningOp<top::WeightOp>();
  auto filterShape = module::getShape(leftFilterOp.getOutput());
  auto outputShape = module::getShape(mm_left.getOutput());
  auto num_dims = filterShape.size();
  auto N = filterShape[num_dims - 1];
  auto slice_n = ceiling_func(N, num_devices);
  std::vector<Value> end_operands;
  Operation *end_op = nullptr;
  auto a16_mm_left = dyn_cast<tpu::A16MatMulOp>(mm_left.getOperation());
  int weight_bits = a16_mm_left ? a16_mm_left.getWeightBits() : 16;

  std::vector<Value> operands;
  bool biasAdd = false;
  Value biasValue;
  for (int i = 0; i < num_devices; i++) {
    auto offset = i * slice_n;
    auto length = std::min((i + 1) * slice_n, N) - offset;
    auto suffix = std::to_string(i);
    auto newShape = {outputShape[0], outputShape[1],
                     (weight_bits == 4 ? 2 : 1) * length};

    // 2. Create Left MatMul
    auto newLeftFilterOp = module::opSliceAxis(mm_left.getOperand(1),
                                               num_dims - 1, offset, length);
    operands.clear();
    operands.push_back(op->getResult(0));
    operands.push_back(newLeftFilterOp);
    if (a16_mm_left) {
      auto scale_op =
          mm_left->getOperand(2).template getDefiningOp<top::WeightOp>();
      operands.push_back(scale_op.clone(suffix));
    }
    operands.push_back(mm_left.getBias());

    rewriter.setInsertionPointAfter(mm_left);
    auto new_mm_left = rewriter.create<MatMulTy>(
        module::getLocLike(mm_left.getOutput(), "mm_left_" + suffix),
        module::getTypeLike(mm_left.getOutput(), newShape), operands,
        mm_left->getAttrs());

    // 3. Create Right MatMul
    auto newRightFilterOp = module::opSliceAxis(mm_right.getOperand(1),
                                                num_dims - 1, offset, length);
    operands.clear();
    operands.push_back(op->getResult(0));
    operands.push_back(newRightFilterOp);
    if (a16_mm_left) {
      auto scale_op =
          mm_right->getOperand(2).template getDefiningOp<top::WeightOp>();
      operands.push_back(scale_op.clone(suffix));
    }
    operands.push_back(mm_right.getBias());

    rewriter.setInsertionPointAfter(mm_right);
    auto new_mm_right = rewriter.create<MatMulTy>(
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
        module::getTypeLike(next_op->getResult(0), newShape), operands,
        next_op->getAttrs());

    // 5. Create Final MatMul
    next_op = *next_op->user_begin();
    auto newFinalFilterOp =
        module::opSliceAxis(next_op->getOperand(1), num_dims - 2,
                            (weight_bits == 4 ? 2 : 1) * offset,
                            (weight_bits == 4 ? 2 : 1) * length);
    operands.clear();
    operands.push_back(mul_op->getResult(0));
    operands.push_back(newFinalFilterOp);
    if (a16_mm_left) {
      auto new_scale = module::opSliceAxis(next_op->getOperand(2), 0,
                                           (weight_bits == 4 ? 2 : 1) * offset,
                                           (weight_bits == 4 ? 2 : 1) * length);
      operands.push_back(new_scale);
    }
    auto bias = next_op->getOperand(a16_mm_left ? 3 : 2);
    if (module::isNone(bias)) {
      operands.push_back(bias);
    } else if (module::isWeight(bias)) {
      auto bias_weight = bias.template getDefiningOp<top::WeightOp>();
      operands.push_back(bias_weight.clone(suffix));
    } else {
      operands.push_back(module::getNoneOp(op));
      biasAdd = true;
      biasValue = bias;
    }

    rewriter.setInsertionPointAfter(next_op);
    auto new_mm_final = rewriter.create<MatMulTy>(
        module::getLocLike(next_op->getResult(0), "mm_final_" + suffix),
        module::getTypeLike(next_op->getResult(0), topShape), operands,
        next_op->getAttrs());
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

template void sliceMerge2Split(tpu::MatMulOp mm_left, PatternRewriter &rewriter,
                               tpu::DistributionBeginOp op,
                               int64_t num_devices);

template void sliceMerge2Split(tpu::A16MatMulOp mm_left,
                               PatternRewriter &rewriter,
                               tpu::DistributionBeginOp op,
                               int64_t num_devices);

} // namespace tpu
} // namespace tpu_mlir
