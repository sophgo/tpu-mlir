//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Distribute/Distribute.h"

namespace tpu_mlir {
namespace tpu {

// MatMul + Elementwise * n + TopK
template <typename MatMulTy>
LogicalResult
MatMulTopK<MatMulTy>::matchAndRewrite(MatMulTy op,
                                      PatternRewriter &rewriter) const {
  if (!isLargeMatMul(op) || module::isOpInDistribution(op) ||
      !op->hasOneUse()) {
    return failure();
  }
  auto next_op = *op->user_begin();
  while (next_op->template hasTrait<trait::SupportElementwise>() &&
         next_op->hasOneUse()) {
    next_op = *next_op->user_begin();
  }
  auto topk = dyn_cast<tpu::TopKOp>(next_op);
  if (!topk || topk.getK() != 1 || !topk.getValues().use_empty()) {
    return failure();
  }
  // Bingo !!
  distribute(rewriter, op, next_op, tpu::DistributionPattern::MatMulTopK);
  return success();
}

template LogicalResult
MatMulTopK<tpu::MatMulOp>::matchAndRewrite(tpu::MatMulOp op,
                                           PatternRewriter &rewriter) const;

template LogicalResult
MatMulTopK<tpu::A16MatMulOp>::matchAndRewrite(tpu::A16MatMulOp op,
                                              PatternRewriter &rewriter) const;

template <typename MatMulTy>
void topKSplit(MatMulTy mm, PatternRewriter &rewriter,
               tpu::DistributionBeginOp op, int64_t num_devices) {
  if (!mm) {
    return;
  }
  auto next_op = *op->user_begin();
  // auto mm = cast<tpu::MatMulOp>(next_op);
  auto filterOp = mm.getOperand(1).template getDefiningOp<top::WeightOp>();
  auto filterShape = module::getShape(filterOp.getOutput());
  auto outputShape = module::getShape(mm.getOutput());
  auto mm_attrs = mm->getAttrs();
  auto has_bias = !module::isNone(mm.getBias());
  auto num_dims = filterShape.size();
  auto N = filterShape[num_dims - 1];
  auto slice_n = ceiling_func(N, num_devices);
  auto a16_mm = dyn_cast<tpu::A16MatMulOp>(mm.getOperation());
  auto weight_bits = a16_mm ? a16_mm.getWeightBits() : 16;
  Operation *end_op = nullptr;
  std::vector<Value> t_operands;
  for (int i = 0; i < num_devices; i++) {
    auto offset = i * slice_n;
    auto length = std::min(slice_n, N - offset);
    auto suffix = std::to_string(i);
    auto newFilter =
        module::opSliceAxis(mm.getOperand(1), num_dims - 1, offset, length);
    std::vector<Value> operands;
    operands.push_back(mm.getInput());
    operands.push_back(newFilter);
    if (a16_mm) {
      auto scale_op = mm.getOperand(2).template getDefiningOp<top::WeightOp>();
      operands.push_back(scale_op.clone(suffix));
    }
    if (has_bias) {
      auto new_bias =
          module::opSliceAxis(mm.getBias(), num_dims - 1, offset, length);
      operands.push_back(new_bias);
    } else {
      operands.push_back(mm.getBias());
    }
    auto new_loc = module::getLocLike(mm.getOutput(), suffix);
    std::vector<int64_t> new_shape = outputShape;
    new_shape[new_shape.size() - 1] = (weight_bits == 4 ? 2 : 1) * length;
    auto new_type = module::getTypeLike(mm.getOutput(), new_shape);
    rewriter.setInsertionPointAfter(mm);
    auto new_mm =
        rewriter.create<MatMulTy>(new_loc, new_type, operands, mm_attrs);
    Value cur_output = new_mm.getOutput();
    next_op = *mm->user_begin();
    while (!isa<tpu::TopKOp>(next_op)) {
      auto new_op = cloneOp(rewriter, next_op, new_shape, suffix);
      new_op->setOperand(0, cur_output);
      cur_output = new_op->getResult(0);
      next_op = *next_op->user_begin();
    }
    auto topk = cast<tpu::TopKOp>(next_op);
    auto new_op =
        cloneOp(rewriter, next_op, module::getShape(topk.getIndices()), suffix);
    new_op->setOperand(0, cur_output);
    auto new_topk = cast<tpu::TopKOp>(new_op);
    t_operands.push_back(new_topk.getValues());
    auto indices = new_topk.getIndices();
    if (i == 0) {
      t_operands.push_back(indices);
    } else {
      auto new_name = module::getName(indices);
      auto new_loc = NameLoc::get(
          rewriter.getStringAttr(new_name.str() + suffix + "_add"));
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr(
          "const_val",
          rewriter.getF64FloatAttr((weight_bits == 4 ? 2 : 1) * offset)));
      rewriter.setInsertionPointAfter(new_op);
      auto new_add = rewriter.create<tpu::AddConstOp>(
          new_loc, indices.getType(), ValueRange{indices}, attrs);
      t_operands.push_back(new_add.getOutput());
    }
    if (end_op == nullptr) {
      end_op = *topk.getIndices().user_begin();
    } else {
      assert(end_op == *topk.getIndices().user_begin());
    }
  }
  end_op->setOperands(t_operands);
  eraseForward(rewriter, mm);
}

template void topKSplit(tpu::MatMulOp mm, PatternRewriter &rewriter,
                        tpu::DistributionBeginOp op, int64_t num_devices);

template void topKSplit(tpu::A16MatMulOp mm, PatternRewriter &rewriter,
                        tpu::DistributionBeginOp op, int64_t num_devices);

} // namespace tpu
} // namespace tpu_mlir
