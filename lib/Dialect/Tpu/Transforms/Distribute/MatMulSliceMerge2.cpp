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
  auto mul_right_op = mul_down_op.getOperand(1).getDefiningOp();
  auto right_mm_op = dyn_cast<tpu::MatMulOp>(mul_right_op);
  auto right_a16mm_op = dyn_cast<tpu::A16MatMulOp>(mul_right_op);
  if (!right_mm_op && !right_a16mm_op) {
    return failure();
  }
  // MatMulOp
  auto mul_left_op = active_value.getDefiningOp();
  auto left_mm_op = dyn_cast<tpu::MatMulOp>(mul_left_op);
  auto left_a16mm_op = dyn_cast<tpu::A16MatMulOp>(mul_left_op);
  if (!left_mm_op && !left_a16mm_op) {
    return failure();
  }
  // RMSNormOp or CastOp
  auto top_op = left_mm_op ? left_mm_op.getOperand(0).getDefiningOp()
                           : left_a16mm_op.getOperand(0).getDefiningOp();
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

void sliceMerge2Split(PatternRewriter &rewriter, tpu::DistributionBeginOp op,
                      int64_t num_devices) {
  // 1. Define params
  // MatMul params
  std::vector<Operation *> mm(op->user_begin(), op->user_end());

  // Swap MatMul->Cast->Active->Cast for MatMul
  if (isa<tpu::MulOp>(*mm[0]->user_begin())) {
    std::swap(mm[0], mm[1]);
  }
  auto a16mm_left = dyn_cast<tpu::A16MatMulOp>(mm[0]);
  auto left_wbits = a16mm_left ? a16mm_left.getWeightBits() : 16;

  auto a16mm_right = dyn_cast<tpu::A16MatMulOp>(mm[1]);
  auto right_wbits = a16mm_right ? a16mm_right.getWeightBits() : 16;

  auto topShape = module::getShape(op->getResult(0));

  auto leftFilterOp =
      mm[0]->getOperand(1).template getDefiningOp<top::WeightOp>();
  auto rightFilterOp =
      mm[1]->getOperand(1).template getDefiningOp<top::WeightOp>();
  auto leftFilterShape = module::getShape(leftFilterOp.getOutput());
  auto rightFiltershape = module::getShape(rightFilterOp.getOutput());
  auto outputShape = module::getShape(mm[0]->getResult(0));
  auto num_dims = leftFilterShape.size();
  // [MxK], [KxN] matmul
  auto left_N = leftFilterShape[num_dims - 1];
  auto left_slice_n = ceiling_func(left_N, num_devices);
  auto right_N = rightFiltershape[num_dims - 1];
  auto right_slice_n = ceiling_func(right_N, num_devices);

  std::vector<Value> end_operands;
  Operation *end_op = nullptr;

  std::vector<Value> operands;
  bool biasAdd = false;
  Value biasValue;
  for (int i = 0; i < num_devices; i++) {
    auto left_offset = i * left_slice_n;
    auto left_length = std::min((i + 1) * left_slice_n, left_N) - left_offset;
    auto right_offset = i * right_slice_n;
    auto right_length =
        std::min((i + 1) * right_slice_n, right_N) - right_offset;
    auto suffix = std::to_string(i);

    // 2. Create Left MatMul
    auto newLeftFilterOp = module::opSliceAxis(
        mm[0]->getOperand(1), num_dims - 1, left_offset, left_length);
    operands.clear();
    operands.push_back(op->getResult(0));
    operands.push_back(newLeftFilterOp);
    if (a16mm_left) {
      auto scale_op =
          mm[0]->getOperand(2).template getDefiningOp<top::WeightOp>();
      operands.push_back(scale_op.clone(suffix));
    }
    operands.push_back(mm[0]->getOperand(a16mm_left ? 3 : 2));

    rewriter.setInsertionPointAfter(mm[0]);
    auto new_left_loc =
        module::getLocLike(mm[0]->getResult(0), "mm_left_" + suffix);
    auto new_left_shape = {outputShape[0], outputShape[1],
                           (left_wbits == 4 ? 2 : 1) * left_length};
    auto new_left_type =
        module::getTypeLike(mm[0]->getResult(0), new_left_shape);

    auto new_mm_left =
        a16mm_left
            ? rewriter.create<tpu::A16MatMulOp>(new_left_loc, new_left_type,
                                                operands, mm[0]->getAttrs())
            : rewriter.create<tpu::MatMulOp>(new_left_loc, new_left_type,
                                             operands, mm[0]->getAttrs());

    // 3. Create Right MatMul
    auto newRightFilterOp = module::opSliceAxis(
        mm[1]->getOperand(1), num_dims - 1, right_offset, right_length);
    operands.clear();
    operands.push_back(op->getResult(0));
    operands.push_back(newRightFilterOp);
    if (a16mm_right) {
      auto scale_op =
          mm[1]->getOperand(2).template getDefiningOp<top::WeightOp>();
      operands.push_back(scale_op.clone(suffix));
    }
    operands.push_back(mm[1]->getOperand(a16mm_right ? 3 : 2));

    rewriter.setInsertionPointAfter(mm[1]);

    auto new_right_loc =
        module::getLocLike(mm[0]->getResult(0), "mm_right_" + suffix);
    auto new_right_shape = {outputShape[0], outputShape[1],
                            (right_wbits == 4 ? 2 : 1) * right_length};
    auto new_right_type =
        module::getTypeLike(mm[0]->getResult(0), new_right_shape);

    auto new_mm_right =
        a16mm_right
            ? rewriter.create<tpu::A16MatMulOp>(new_right_loc, new_right_type,
                                                operands, mm[1]->getAttrs())
            : rewriter.create<tpu::MatMulOp>(new_right_loc, new_right_type,
                                             operands, mm[1]->getAttrs());

    // 4. Create Left Active
    operands.clear();
    Value cur_output = new_mm_left->getResult(0);
    auto next_op = *mm[0]->user_begin();
    while (isa<tpu::CastOp, tpu::ActiveOp>(next_op)) {
      auto new_op = cloneOp(rewriter, next_op, new_left_shape, suffix);
      new_op->setOperand(0, cur_output);
      cur_output = new_op->getResult(0);
      next_op = *next_op->user_begin();
    }
    operands.push_back(cur_output);
    operands.push_back(new_mm_right->getResult(0));

    rewriter.setInsertionPointAfter(new_mm_right);
    auto mul_op = rewriter.create<tpu::MulOp>(
        module::getLocLike(next_op->getResult(0), "mul_" + suffix),
        module::getTypeLike(next_op->getResult(0), new_left_shape), operands,
        next_op->getAttrs());

    // 5. Create Final MatMul
    next_op = *next_op->user_begin();
    auto final_a16mm = dyn_cast<tpu::A16MatMulOp>(next_op);
    auto newFinalFilterOp =
        module::opSliceAxis(next_op->getOperand(1), num_dims - 2,
                            (left_wbits == 4 ? 2 : 1) * left_offset,
                            (left_wbits == 4 ? 2 : 1) * left_length);
    operands.clear();
    operands.push_back(mul_op->getResult(0));
    operands.push_back(newFinalFilterOp);
    if (final_a16mm) {
      auto new_scale = module::opSliceAxis(
          next_op->getOperand(2), 0, (left_wbits == 4 ? 2 : 1) * left_offset,
          (left_wbits == 4 ? 2 : 1) * left_length);
      operands.push_back(new_scale);
    }
    auto bias = next_op->getOperand(final_a16mm ? 3 : 2);
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
    auto new_final_loc =
        module::getLocLike(next_op->getResult(0), "mm_final_" + suffix);
    auto new_final_type = module::getTypeLike(next_op->getResult(0), topShape);
    auto new_final_mm =
        final_a16mm
            ? rewriter.create<tpu::A16MatMulOp>(new_final_loc, new_final_type,
                                                operands, next_op->getAttrs())
            : rewriter.create<tpu::MatMulOp>(new_final_loc, new_final_type,
                                             operands, next_op->getAttrs());
    end_operands.push_back(new_final_mm->getResult(0));
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
